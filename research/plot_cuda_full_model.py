"""Render the nine-variant full-model CUDA comparison from benchmark CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = (
    "MHA",
    "GQA",
    "MQA",
    "MLA-style",
    "GDN",
    "KDA",
    "GDN + gated MHA",
    "KDA + MLA",
    "K3: KDA + gated MLA",
)
SHORT_LABELS = (
    "MHA",
    "GQA",
    "MQA",
    "MLA",
    "GDN",
    "KDA",
    "GDN\nhybrid",
    "KDA\nhybrid",
    "K3\nhybrid",
)
VERIFY_VARIANTS = (
    "GDN",
    "KDA",
    "GDN + gated MHA",
    "KDA + MLA",
    "K3: KDA + gated MLA",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_variants(
    rows: list[dict[str, str]], path: Path, expected: tuple[str, ...]
) -> None:
    observed = {row["variant"] for row in rows}
    if observed != set(expected):
        raise ValueError(
            f"{path} has variants {sorted(observed)!r}; expected {list(expected)!r}"
        )


def one_row_by_variant(
    rows: list[dict[str, str]], path: Path
) -> dict[str, dict[str, str]]:
    output = {}
    for row in rows:
        variant = row["variant"]
        if variant in output:
            raise ValueError(f"{path} has multiple rows for {variant!r}")
        output[variant] = row
    return output


def load_results(
    input_dir: Path,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, str],
]:
    training_path = input_dir / "cuda_full_model_training.csv"
    prefill_path = input_dir / "cuda_full_model_prefill.csv"
    decode_path = input_dir / "cuda_full_model_decode.csv"
    verification_path = input_dir / "cuda_full_model_verification.csv"
    environment_path = input_dir / "cuda_full_model_environment.csv"
    for path in (
        training_path,
        prefill_path,
        decode_path,
        verification_path,
        environment_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing CUDA full-model result: {path}")

    training = read_csv(training_path)
    prefill = read_csv(prefill_path)
    decode = read_csv(decode_path)
    verification = read_csv(verification_path)
    environment = read_csv(environment_path)
    if len(environment) != 1:
        raise ValueError(f"{environment_path} must contain exactly one row")
    if len(training) != len(VARIANTS) or len(decode) != len(VARIANTS):
        raise ValueError("training and decode CSVs must each contain nine variants")
    require_variants(training, training_path, VARIANTS)
    require_variants(decode, decode_path, VARIANTS)
    require_variants(verification, verification_path, VERIFY_VARIANTS)
    if any(row.get("status") != "passed" for row in verification):
        raise ValueError("refusing to plot: at least one FLA verification did not pass")

    prefill_by_variant: dict[str, set[int]] = {variant: set() for variant in VARIANTS}
    for row in prefill:
        variant = row["variant"]
        if variant not in prefill_by_variant:
            raise ValueError(f"unexpected prefill variant {variant!r}")
        prefill_by_variant[variant].add(int(row["tokens"]))
    lengths = next(iter(prefill_by_variant.values()))
    if not lengths or any(values != lengths for values in prefill_by_variant.values()):
        raise ValueError("each variant must have the same non-empty prefill lengths")
    return training, prefill, decode, verification, environment[0]


def write_summary(
    output_path: Path,
    training: list[dict[str, str]],
    prefill: list[dict[str, str]],
    decode: list[dict[str, str]],
    verification: list[dict[str, str]],
    environment: dict[str, str],
) -> None:
    top_training = max(training, key=lambda row: float(row["tokens_per_second_p50"]))
    longest = max(int(row["tokens"]) for row in prefill)
    longest_prefill = [row for row in prefill if int(row["tokens"]) == longest]
    top_prefill = max(
        longest_prefill, key=lambda row: float(row["tokens_per_second_p50"])
    )
    top_decode = min(decode, key=lambda row: float(row["p50_ms_per_token"]))
    max_full_error = max(
        float(row["max_abs_full_logits_error"]) for row in verification
    )
    max_cached_error = max(
        float(row["max_abs_cached_logits_error"]) for row in verification
    )
    output_path.write_text(
        f"""# CUDA full-model comparison — result summary

## Environment

| GPU | Compute capability | PyTorch | FLA | Dtype |
|---|---:|---|---|---|
| {environment["gpu"]} | {environment["compute_capability"]} | {environment["torch"]} | {environment["fla"]} | {environment["dtype"]} |

The benchmark configuration is recorded verbatim in
`cuda_full_model_environment.csv`. The recurrent path uses FLA chunkwise
kernels for prefill/training and FLA fused recurrent kernels for decode.
MHA/GQA/MQA/MLA use the repository's PyTorch SDPA implementation.

## Verification gate

FLA/reference checks passed for all five recurrent-containing schedules. The
largest absolute full-sequence-logit difference was `{max_full_error:.6g}`;
the largest cached-logit difference was `{max_cached_error:.6g}`. These are
reduced-precision kernel-equivalence checks, not a quality measurement.

## Observed winners

- Highest full-model train-step throughput: **{top_training["variant"]}** at
  **{float(top_training["tokens_per_second_p50"]):,.0f} tok/s** (p50).
- Highest {longest}-token full-model prefill throughput:
  **{top_prefill["variant"]}** at
  **{float(top_prefill["tokens_per_second_p50"]) / 1e6:.2f}M tok/s** (p50).
- Lowest cached decode p50 after a {top_decode["prompt_tokens"]}-token prefix:
  **{top_decode["variant"]}** at
  **{float(top_decode["p50_ms_per_token"]):.3f} ms/token**.

## Scope

This is one CUDA session on random token IDs. Training rows include the full
4-layer decoder, vocabulary loss, backward pass, and AdamW; prefill and decode
rows include the full model and cache updates. It is a direct performance
comparison of the nine implemented token mixers, but it does **not** retrain
the models, produce CUDA PPL values, match parameters, or measure a production
Qwen/Kimi/DeepSeek system. Do not merge its absolute throughput values with the
earlier CPU reference-path chart.
""",
        encoding="utf-8",
    )


def render(
    output_path: Path,
    training: list[dict[str, str]],
    prefill: list[dict[str, str]],
    decode: list[dict[str, str]],
    environment: dict[str, str],
) -> None:
    training_by_variant = one_row_by_variant(training, output_path)
    decode_by_variant = one_row_by_variant(decode, output_path)
    prefill_by_variant = {variant: [] for variant in VARIANTS}
    for row in prefill:
        prefill_by_variant[row["variant"]].append(row)
    for rows in prefill_by_variant.values():
        rows.sort(key=lambda row: int(row["tokens"]))
    training_lengths = {int(row["tokens"]) for row in training}
    decode_prompts = {int(row["prompt_tokens"]) for row in decode}
    if len(training_lengths) != 1 or len(decode_prompts) != 1:
        raise ValueError("training/decode rows must use one common sequence length")
    training_tokens = training_lengths.pop()
    decode_prompt_tokens = decode_prompts.pop()
    decode_prompt_label = (
        f"{decode_prompt_tokens // 1024}K"
        if decode_prompt_tokens >= 1024 and decode_prompt_tokens % 1024 == 0
        else str(decode_prompt_tokens)
    )

    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(1, 3, figsize=(20, 6.8), layout="constrained")
    positions = list(range(len(VARIANTS)))

    train_values = [
        float(training_by_variant[variant]["tokens_per_second_p50"]) / 1e3
        for variant in VARIANTS
    ]
    axes[0].bar(positions, train_values, color=colors[: len(VARIANTS)])
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions, SHORT_LABELS, rotation=35, ha="right")
    axes[0].set_ylabel("Thousand tokens/s (p50, higher is better)")
    axes[0].set_title(f"Full-model training step @ {training_tokens} tokens")
    axes[0].grid(axis="y", alpha=0.25, which="both")

    for index, variant in enumerate(VARIANTS):
        rows = prefill_by_variant[variant]
        axes[1].plot(
            [int(row["tokens"]) for row in rows],
            [float(row["tokens_per_second_p50"]) / 1e6 for row in rows],
            marker="o",
            linewidth=2,
            color=colors[index],
            label=SHORT_LABELS[index].replace("\n", " "),
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks(
        [int(row["tokens"]) for row in prefill_by_variant[VARIANTS[0]]],
        [
            "4K" if int(row["tokens"]) == 4096 else str(row["tokens"])
            for row in prefill_by_variant[VARIANTS[0]]
        ],
    )
    axes[1].set_ylabel("Million tokens/s (p50, higher is better)")
    axes[1].set_xlabel("Prefill length (tokens)")
    axes[1].set_title("Full-model cached prefill")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=7.5, ncol=2, loc="best")

    p50_values = [
        float(decode_by_variant[variant]["p50_ms_per_token"]) for variant in VARIANTS
    ]
    p95_values = [
        float(decode_by_variant[variant]["p95_ms_per_token"]) for variant in VARIANTS
    ]
    width = 0.38
    axes[2].bar(
        [position - width / 2 for position in positions],
        p50_values,
        width,
        label="p50",
        color=colors[: len(VARIANTS)],
    )
    axes[2].bar(
        [position + width / 2 for position in positions],
        p95_values,
        width,
        label="p95",
        color=colors[: len(VARIANTS)],
        alpha=0.45,
    )
    axes[2].set_xticks(positions, SHORT_LABELS, rotation=35, ha="right")
    axes[2].set_ylabel("ms/token (lower is better)")
    axes[2].set_title(f"Cached decode after {decode_prompt_label}-token prefix")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()

    figure.suptitle("CUDA full-model benchmark: all nine implemented token mixers")
    figure.text(
        0.5,
        0.01,
        f"{environment['gpu']} · {environment['dtype']} · full 4-layer GPT · "
        "random-token performance microbenchmark; no CUDA PPL claim",
        ha="center",
        color="dimgray",
    )
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    training, prefill, decode, verification, environment = load_results(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    graph = args.output_dir / "cuda_full_model_comparison.png"
    summary = args.output_dir / "CUDA_FULL_MODEL_SUMMARY.md"
    render(graph, training, prefill, decode, environment)
    write_summary(summary, training, prefill, decode, verification, environment)
    print(f"Wrote {graph} and {summary}")


if __name__ == "__main__":
    main()

"""Render the CUDA FLA chunkwise-kernel benchmark from Colab CSV exports.

Example:
    python -m research.plot_cuda_chunkwise --input-dir results --output-dir results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {
    "MHA SDPA": "#16324F",
    "GDN reference": "#B8B8B8",
    "GDN chunkwise": "#59A14F",
    "KDA reference": "#C9A1A1",
    "KDA chunkwise": "#E15759",
    "GDN fused recurrent": "#59A14F",
    "KDA fused recurrent": "#E15759",
}
PREFILL_ORDER = ("MHA SDPA", "GDN chunkwise", "KDA chunkwise")
TRAINING_ORDER = (
    "MHA SDPA",
    "GDN reference",
    "GDN chunkwise",
    "KDA reference",
    "KDA chunkwise",
)
DECODE_ORDER = ("MHA SDPA", "GDN fused recurrent", "KDA fused recurrent")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require_row(
    rows: list[dict[str, str]], operator: str, *, tokens: int | None = None
) -> dict[str, str]:
    for row in rows:
        if row["operator"] != operator:
            continue
        if tokens is None or int(row["tokens"]) == tokens:
            return row
    suffix = "" if tokens is None else f" at {tokens} tokens"
    raise ValueError(f"missing {operator}{suffix}")


def validate(
    prefill: list[dict[str, str]],
    training: list[dict[str, str]],
    decode: list[dict[str, str]],
    environment: list[dict[str, str]],
) -> None:
    if len(environment) != 1:
        raise ValueError("chunkwise_environment.csv must contain exactly one row")
    for operator in PREFILL_ORDER:
        for tokens in (64, 256, 1024, 4096):
            require_row(prefill, operator, tokens=tokens)
    for operator in TRAINING_ORDER:
        require_row(training, operator)
    for operator in DECODE_ORDER:
        if not any(row["operator"] == operator for row in decode):
            raise ValueError(f"missing {operator} from decode CSV")


def plot(
    prefill: list[dict[str, str]],
    training: list[dict[str, str]],
    decode: list[dict[str, str]],
    environment: dict[str, str],
    output_dir: Path,
) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(17.2, 5.4))
    prefill_axis, training_axis, decode_axis = axes
    lengths = (64, 256, 1024, 4096)

    for operator in PREFILL_ORDER:
        values = [
            float(
                require_row(prefill, operator, tokens=length)["tokens_per_second_p50"]
            )
            / 1_000_000
            for length in lengths
        ]
        prefill_axis.plot(
            lengths,
            values,
            marker="o",
            linewidth=2.2,
            label=operator,
            color=COLORS[operator],
        )
    for operator in ("GDN reference", "KDA reference"):
        row = require_row(prefill, operator, tokens=64)
        prefill_axis.scatter(
            [64],
            [float(row["tokens_per_second_p50"]) / 1_000_000],
            marker="X",
            s=70,
            color=COLORS[operator],
            label=f"{operator} @64",
            zorder=3,
        )
    prefill_axis.set_xscale("log", base=2)
    prefill_axis.set_yscale("log")
    prefill_axis.set_xticks(lengths, ("64", "256", "1K", "4K"))
    prefill_axis.set_title("CUDA prefill throughput")
    prefill_axis.set_xlabel("Sequence length (tokens)")
    prefill_axis.set_ylabel("Million tokens/s (p50, higher is better)")
    prefill_axis.grid(alpha=0.25, which="both")
    prefill_axis.legend(fontsize=8)

    training_values = [
        float(require_row(training, operator)["tokens_per_second_p50"]) / 1_000
        for operator in TRAINING_ORDER
    ]
    training_axis.bar(
        range(len(TRAINING_ORDER)),
        training_values,
        color=[COLORS[operator] for operator in TRAINING_ORDER],
    )
    training_axis.set_yscale("log")
    training_axis.set_xticks(
        range(len(TRAINING_ORDER)),
        ("MHA\nSDPA", "GDN\nref", "GDN\nchunk", "KDA\nref", "KDA\nchunk"),
    )
    training_axis.set_title("Attention forward + backward @64")
    training_axis.set_ylabel("Thousand tokens/s (p50, higher is better)")
    training_axis.grid(axis="y", alpha=0.25, which="both")

    p50 = [
        float(
            next(row for row in decode if row["operator"] == operator)[
                "p50_ms_per_token"
            ]
        )
        for operator in DECODE_ORDER
    ]
    p95 = [
        float(
            next(row for row in decode if row["operator"] == operator)[
                "p95_ms_per_token"
            ]
        )
        for operator in DECODE_ORDER
    ]
    positions = list(range(len(DECODE_ORDER)))
    decode_axis.bar(
        [position - 0.18 for position in positions],
        p50,
        width=0.36,
        color=[COLORS[operator] for operator in DECODE_ORDER],
        label="p50",
    )
    decode_axis.bar(
        [position + 0.18 for position in positions],
        p95,
        width=0.36,
        color=[COLORS[operator] for operator in DECODE_ORDER],
        alpha=0.45,
        label="p95",
    )
    decode_axis.set_xticks(positions, ("MHA\nSDPA", "GDN\nrecurrent", "KDA\nrecurrent"))
    decode_axis.set_title("Cached decode after 4K-token prefix")
    decode_axis.set_ylabel("ms/token (lower is better)")
    decode_axis.grid(axis="y", alpha=0.25)
    decode_axis.legend()

    figure.suptitle(
        "CUDA kernel benchmark: FLA chunkwise GDN/KDA versus fused SDPA", y=1.01
    )
    figure.text(
        0.5,
        -0.04,
        f"{environment['gpu']} · FP16 · batch {environment['batch']} · "
        f"{environment['heads']} heads × {environment['head_dim']} · "
        "operator-level timings; not the CPU full-GPT training chart",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout()
    output = output_dir / "cuda_chunkwise_operator_benchmark.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def write_summary(
    prefill: list[dict[str, str]],
    training: list[dict[str, str]],
    decode: list[dict[str, str]],
    environment: dict[str, str],
    output_dir: Path,
) -> Path:
    def throughput(row: dict[str, str]) -> float:
        return float(row["tokens_per_second_p50"])

    gdn_prefill_speedup = throughput(
        require_row(prefill, "GDN chunkwise", tokens=64)
    ) / throughput(require_row(prefill, "GDN reference", tokens=64))
    kda_prefill_speedup = throughput(
        require_row(prefill, "KDA chunkwise", tokens=64)
    ) / throughput(require_row(prefill, "KDA reference", tokens=64))
    gdn_training_speedup = throughput(
        require_row(training, "GDN chunkwise")
    ) / throughput(require_row(training, "GDN reference"))
    kda_training_speedup = throughput(
        require_row(training, "KDA chunkwise")
    ) / throughput(require_row(training, "KDA reference"))
    mha_4k = throughput(require_row(prefill, "MHA SDPA", tokens=4096))
    gdn_4k = throughput(require_row(prefill, "GDN chunkwise", tokens=4096))
    kda_4k = throughput(require_row(prefill, "KDA chunkwise", tokens=4096))
    mha_decode = float(
        next(row for row in decode if row["operator"] == "MHA SDPA")["p50_ms_per_token"]
    )
    gdn_decode = float(
        next(row for row in decode if row["operator"] == "GDN fused recurrent")[
            "p50_ms_per_token"
        ]
    )
    kda_decode = float(
        next(row for row in decode if row["operator"] == "KDA fused recurrent")[
            "p50_ms_per_token"
        ]
    )
    contents = f"""# CUDA chunkwise kernel benchmark — result summary

This is a separate operator-level CUDA session. It must not be numerically
merged with the earlier CPU full-GPT reference-path chart.

## Environment

| GPU | Compute capability | PyTorch | FLA | Dtype | Batch | Heads × head dim |
|---|---:|---|---|---|---:|---:|
| {environment["gpu"]} | {environment["compute_capability"]} | {environment["torch"]} | {environment["fla"]} | {environment["dtype"]} | {environment["batch"]} | {environment["heads"]} × {environment["head_dim"]} |

## Findings

- At 64 tokens, FLA chunkwise prefill is **{gdn_prefill_speedup:.1f}×** faster
  than the GDN reference scan and **{kda_prefill_speedup:.1f}×** faster than
  the KDA reference scan.
- At the 64-token attention `forward + backward` scope, chunkwise is
  **{gdn_training_speedup:.1f}×** faster for GDN and **{kda_training_speedup:.1f}×**
  faster for KDA than their reference paths.
- At 4K-token prefill, MHA SDPA reaches {mha_4k / 1_000_000:.2f}M tokens/s;
  GDN chunkwise reaches {gdn_4k / 1_000_000:.2f}M ({mha_4k / gdn_4k:.2f}× below MHA)
  and KDA chunkwise reaches {kda_4k / 1_000_000:.2f}M ({mha_4k / kda_4k:.2f}× below MHA).
- After a 4K-token prefix, GDN decode p50 is {gdn_decode:.3f} ms/token and
  KDA is {kda_decode:.3f} ms/token, versus {mha_decode:.3f} ms/token for MHA.
  This is {(1 - gdn_decode / mha_decode) * 100:.1f}% and
  {(1 - kda_decode / mha_decode) * 100:.1f}% lower latency, respectively.

## Scope and conclusion

The reference scan's large slowdown was implementation overhead, not a direct
architecture ranking: chunkwise kernels remove roughly 14–22× of that penalty
at the tested short window. MHA SDPA remains faster for prefill through 4K in
this T4 FP16 microbenchmark, while KDA approaches it at 4K and has the lowest
cached decode latency.

The training panel measures only the attention operator's forward and backward
passes. It excludes projections, convolution, MLP, embeddings, loss, AdamW,
and all full-model work; it is therefore not a replacement for the original
CPU end-to-end GPT training-throughput figure.
"""
    output = output_dir / "CUDA_CHUNKWISE_SUMMARY.md"
    output.write_text(contents, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefill = read_csv(args.input_dir / "chunkwise_prefill.csv")
    training = read_csv(args.input_dir / "chunkwise_training.csv")
    decode = read_csv(args.input_dir / "chunkwise_decode.csv")
    environment = read_csv(args.input_dir / "chunkwise_environment.csv")
    validate(prefill, training, decode, environment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = plot(prefill, training, decode, environment[0], args.output_dir)
    summary = write_summary(prefill, training, decode, environment[0], args.output_dir)
    print(f"Wrote {figure} and {summary}")


if __name__ == "__main__":
    main()

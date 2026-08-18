"""Create publication-ready plots and a compact Markdown summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {
    "MHA": "#16324F",
    "GQA": "#00A6A6",
    "MQA": "#7A5195",
    "MLA": "#F28E2B",
}
ORDER = ("MHA", "GQA", "MQA", "MLA")


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_latency(records: list[dict[str, Any]], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for variant in ORDER:
        rows = sorted(
            (row for row in records if row["variant"] == variant),
            key=lambda row: row["prompt_length"],
        )
        contexts = [row["prompt_length"] for row in rows]
        cached = [row["cached_decode_ms_per_token"]["p50"] for row in rows]
        recompute = [row["recompute_decode_ms_per_token"]["p50"] for row in rows]
        speedup = [row["decode_speedup_p50"] for row in rows]
        axes[0].plot(
            contexts,
            cached,
            marker="o",
            color=COLORS[variant],
            label=f"{variant} — KV cache",
        )
        axes[0].plot(
            contexts,
            recompute,
            marker="x",
            linestyle="--",
            color=COLORS[variant],
            alpha=0.85,
            label=f"{variant} — recompute",
        )
        axes[1].plot(
            contexts,
            speedup,
            marker="o",
            color=COLORS[variant],
            label=variant,
        )

    axes[0].set_title("Autoregressive decode latency")
    axes[0].set_xlabel("Prompt length (tokens)")
    axes[0].set_ylabel("p50 latency (ms/token, one attention layer)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_title("KV-cache decode speedup")
    axes[1].set_xlabel("Prompt length (tokens)")
    axes[1].set_ylabel("Speedup over full recomputation (×)")
    axes[1].axhline(1.0, color="#777777", linewidth=1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("MHA vs GQA vs MQA vs MLA — fixed model width and query-head count")
    figure.tight_layout()
    figure.savefig(output_dir / "latency_vs_context.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_memory(records: list[dict[str, Any]], output_dir: Path, dtype: str) -> None:
    maximum_context = max(row["total_sequence_length"] for row in records)
    rows = [row for row in records if row["total_sequence_length"] == maximum_context]
    rows.sort(key=lambda row: ORDER.index(row["variant"]))
    variants = [row["variant"] for row in rows]
    actual_mib = [row["kv_cache_bytes_model"] / 1024**2 for row in rows]
    dtype_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[dtype]
    fp16_mib = [value * 2 / dtype_bytes for value in actual_mib]

    figure, axis = plt.subplots(figsize=(7.5, 4.6))
    positions = list(range(len(variants)))
    width = 0.34
    axis.bar(
        [position - width / 2 for position in positions],
        actual_mib,
        width,
        color=[COLORS[variant] for variant in variants],
        label=f"Measured storage ({dtype})",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        fp16_mib,
        width,
        color=[COLORS[variant] for variant in variants],
        alpha=0.35,
        hatch="//",
        label="FP16 equivalent",
    )
    axis.set_xticks(positions, variants)
    axis.set_ylabel("KV-cache size (MiB)")
    axis.set_title(f"KV-cache memory at {maximum_context} tokens (all model layers)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for position, value in zip(positions, actual_mib, strict=True):
        axis.text(position - width / 2, value, f"{value:.2f}", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output_dir / "kv_cache_memory.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_summary(data: dict[str, Any], output_dir: Path) -> None:
    records = data["records"]
    maximum_prompt = max(row["prompt_length"] for row in records)
    rows = [row for row in records if row["prompt_length"] == maximum_prompt]
    rows.sort(key=lambda row: ORDER.index(row["variant"]))
    mha_memory = next(
        row["kv_cache_bytes_model"] for row in rows if row["variant"] == "MHA"
    )
    lines = [
        "# Benchmark summary",
        "",
        f"Configuration: `{data['configuration']['dim']}` hidden dimensions, "
        f"`{data['configuration']['n_heads']}` query heads, "
        f"`{data['configuration']['layers_for_cache']}` layers for cache accounting, "
        f"batch `{data['configuration']['batch_size']}`, "
        f"dtype `{data['configuration']['dtype']}`.",
        "",
        "| Attention | KV representation | Cached ms/token | Recompute ms/token | Decode speedup | Model KV cache | Memory vs MHA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        memory = row["kv_cache_bytes_model"]
        lines.append(
            f"| {row['variant']} | "
            f"{('rank ' + str(row['mla_latent_rank'])) if row['variant'] == 'MLA' else (str(row['n_kv_heads']) + ' heads')} | "
            f"{row['cached_decode_ms_per_token']['p50']:.3f} | "
            f"{row['recompute_decode_ms_per_token']['p50']:.3f} | "
            f"{row['decode_speedup_p50']:.2f}× | "
            f"{memory / 1024**2:.2f} MiB | {mha_memory / memory:.1f}× smaller |"
        )
    lines.extend(
        [
            "",
            f"The table uses the longest measured prompt (`{maximum_prompt}` tokens). "
            "Latency is a microbenchmark of one attention layer; cache memory is exact "
            "storage extrapolated across the configured model layers.",
            "",
            "![Latency versus context](latency_vs_context.png)",
            "",
            "![KV-cache memory](kv_cache_memory.png)",
            "",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_results(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_latency(data["records"], args.output_dir)
    plot_memory(data["records"], args.output_dir, data["configuration"]["dtype"])
    write_summary(data, args.output_dir)
    print(f"Wrote plots and summary to {args.output_dir}")


if __name__ == "__main__":
    main()

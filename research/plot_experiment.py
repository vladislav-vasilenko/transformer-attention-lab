"""Render the training, quality-efficiency, and inference figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402


COLORS = {
    "MHA": "#16324F",
    "GQA": "#00A6A6",
    "MQA": "#7A5195",
    "MLA": "#F28E2B",
}
ORDER = ("MHA", "GQA", "MQA", "MLA")
LONG_CONTEXTS = (128, 512, 2_048, 8_192, 32_768, 131_072)
CAPACITY_BUDGET_BYTES = 1024**3


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _architecture_box(
    axis: Any,
    x: float,
    y: float,
    label: str,
    *,
    width: float = 0.12,
    height: float = 0.17,
    cached: bool = False,
    transient: bool = False,
) -> None:
    edge = "#16324F"
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012",
        facecolor="#E9EFF6" if not transient else "#FFFFFF",
        edgecolor=edge,
        linewidth=1.25,
        linestyle="--" if transient else "-",
        hatch="////" if cached else None,
    )
    axis.add_patch(box)
    axis.text(x, y, label, ha="center", va="center", fontsize=8)


def _architecture_link(
    axis: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    arrow: bool = False,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->" if arrow else "-",
            "color": "#555555",
            "linewidth": 1.0,
            "linestyle": ":",
        },
    )


def plot_attention_architectures(output_dir: Path) -> None:
    """Show how query heads share the persistent inference state."""

    figure, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    query_x = (0.14, 0.38, 0.62, 0.86)
    definitions = {
        "MHA": ((0.14, 0.38, 0.62, 0.86), ((0,), (1,), (2,), (3,))),
        "GQA": ((0.26, 0.74), ((0, 1), (2, 3))),
        "MQA": ((0.50,), ((0, 1, 2, 3),)),
    }
    cache_labels = {
        "MHA": r"cache: $2H d_h$",
        "GQA": r"cache: $2H_{kv}d_h$",
        "MQA": r"cache: $2d_h$",
        "MLA": r"cache: $r$",
    }

    for axis, variant in zip(axes, ORDER, strict=True):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.set_title(variant, color=COLORS[variant], fontweight="bold", pad=10)
        for index, x in enumerate(query_x, start=1):
            _architecture_box(axis, x, 0.16, rf"$Q_{index}$", height=0.14)

        if variant != "MLA":
            kv_x, groups = definitions[variant]
            for kv_index, (x, query_group) in enumerate(
                zip(kv_x, groups, strict=True), start=1
            ):
                _architecture_box(axis, x, 0.49, rf"$K_{kv_index}$", cached=True)
                _architecture_box(axis, x, 0.76, rf"$V_{kv_index}$", cached=True)
                _architecture_link(axis, (x, 0.58), (x, 0.67))
                for query_index in query_group:
                    _architecture_link(axis, (query_x[query_index], 0.23), (x, 0.40))
        else:
            mla_x = (0.10, 0.29, 0.48, 0.67)
            for index, (query_position, x) in enumerate(
                zip(query_x, mla_x, strict=True), start=1
            ):
                _architecture_box(
                    axis, x, 0.49, rf"$K_{index}$", transient=True, width=0.10
                )
                _architecture_box(
                    axis, x, 0.76, rf"$V_{index}$", transient=True, width=0.10
                )
                _architecture_link(axis, (query_position, 0.23), (x, 0.40))
                _architecture_link(axis, (x, 0.58), (x, 0.67))
            _architecture_box(
                axis,
                0.91,
                0.625,
                r"$c^{KV}$",
                width=0.13,
                height=0.30,
                cached=True,
            )
            _architecture_link(axis, (0.84, 0.625), (0.70, 0.625), arrow=True)
            axis.text(
                0.78,
                0.69,
                "up-projection",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

        axis.text(
            0.5,
            -0.01,
            cache_labels[variant],
            ha="center",
            va="top",
            fontsize=10,
            color=COLORS[variant],
            fontweight="bold",
        )

    figure.legend(
        handles=[
            Patch(
                facecolor="#E9EFF6",
                edgecolor="#16324F",
                hatch="////",
                label="stored in persistent inference cache",
            ),
            Patch(
                facecolor="#FFFFFF",
                edgecolor="#16324F",
                linestyle="--",
                label="reconstructed transiently",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Attention state sharing (four query heads shown)", y=1.09, fontsize=16
    )
    figure.text(
        0.5,
        -0.02,
        "MQA is the Hₖᵥ = 1 endpoint of GQA; MLA-style attention jointly "
        "compresses K and V into one learned latent per token.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "attention_architectures.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def plot_training(data: dict[str, Any], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for variant in data["variants"]:
        name = variant["variant"]
        history = variant["training_history"]
        tokens = [row["tokens_seen"] for row in history]
        axes[0].plot(
            tokens,
            [row["train_loss"] for row in history],
            marker="o",
            color=COLORS[name],
            linestyle="--",
            alpha=0.65,
            label=f"{name} train",
        )
        axes[0].plot(
            tokens,
            [row["validation_loss"] for row in history],
            marker="o",
            color=COLORS[name],
            label=f"{name} validation",
        )
        axes[1].plot(
            tokens,
            [row["validation_perplexity"] for row in history],
            marker="o",
            color=COLORS[name],
            label=name,
        )
    axes[0].set_title("Learning curves under a fixed token budget")
    axes[0].set_xlabel("Training tokens seen")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_title("Validation perplexity")
    axes[1].set_xlabel("Training tokens seen")
    axes[1].set_ylabel("Perplexity (lower is better)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("MHA vs GQA vs MQA vs MLA — controlled nanoGPT training")
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_inference(data: dict[str, Any], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for variant in data["variants"]:
        name = variant["variant"]
        rows = variant["inference"]
        prompts = [row["prompt_length"] for row in rows]
        axes[0].plot(
            prompts,
            [row["cached_decode_ms_per_token"]["p50"] for row in rows],
            marker="o",
            color=COLORS[name],
            label=f"{name} — KV/latent cache",
        )
        axes[0].plot(
            prompts,
            [row["recompute_decode_ms_per_token"]["p50"] for row in rows],
            marker="x",
            linestyle="--",
            color=COLORS[name],
            alpha=0.8,
            label=f"{name} — recompute",
        )
        axes[1].plot(
            prompts,
            [row["decode_speedup_p50"] for row in rows],
            marker="o",
            color=COLORS[name],
            label=name,
        )
    axes[0].set_title("Full-model autoregressive decode")
    axes[0].set_xlabel("Prompt length (tokens)")
    axes[0].set_ylabel("p50 latency (ms/token)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_title("Speedup from caching")
    axes[1].set_xlabel("Prompt length (tokens)")
    axes[1].set_ylabel("Recompute / cached latency (×)")
    axes[1].axhline(1.0, color="#777777", linewidth=1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "inference_latency.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_tradeoffs(data: dict[str, Any], output_dir: Path) -> None:
    variants = sorted(data["variants"], key=lambda row: ORDER.index(row["variant"]))
    names = [row["variant"] for row in variants]
    maximum_rows = [row["inference"][-1] for row in variants]
    cache_mib = [row["cache_occupied_bytes"] / 1024**2 for row in maximum_rows]
    perplexity = [row["final_validation_perplexity"] for row in variants]
    throughput = [row["training_tokens_per_second"] for row in variants]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, memory, ppl in zip(names, cache_mib, perplexity, strict=True):
        axes[0].scatter(memory, ppl, s=110, color=COLORS[name])
        axes[0].annotate(name, (memory, ppl), xytext=(7, 5), textcoords="offset points")
    axes[0].set_title("Quality–cache Pareto view")
    axes[0].set_xlabel("Cache at longest sequence (MiB, all layers)")
    axes[0].set_ylabel("Final validation perplexity")
    axes[0].grid(alpha=0.25)

    axes[1].bar(names, throughput, color=[COLORS[name] for name in names])
    axes[1].set_title("Training throughput")
    axes[1].set_ylabel("Tokens/second")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(throughput):
        axes[1].text(index, value, f"{value:,.0f}", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output_dir / "quality_efficiency.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def long_context_projection(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Project cache-only capacity from the implemented storage layouts.

    This is an exact byte-count calculation for the lab's cache data structures,
    not a claim that the trained 192-token model can execute at these contexts.
    """

    config = data["experiment_config"]
    head_dim = config["n_embd"] // config["n_head"]
    # The completed experiment ran FP32; derive rather than hard-code the size.
    first_variant = data["variants"][0]
    dtype_bytes = first_variant["parameter_bytes"] // first_variant["parameter_count"]
    bytes_per_token = {
        "MHA": config["n_layer"] * 2 * config["n_head"] * head_dim * dtype_bytes,
        "GQA": config["n_layer"]
        * 2
        * config["gqa_n_kv_heads"]
        * head_dim
        * dtype_bytes,
        "MQA": config["n_layer"] * 2 * head_dim * dtype_bytes,
        "MLA": config["n_layer"] * config["mla_latent_rank"] * dtype_bytes,
    }

    records: list[dict[str, Any]] = []
    for context in LONG_CONTEXTS:
        for variant in ORDER:
            cache_bytes = bytes_per_token[variant] * context
            records.append(
                {
                    "variant": variant,
                    "context_tokens": context,
                    "cache_bytes": cache_bytes,
                    "cache_mib": cache_bytes / 1024**2,
                    "sequences_per_1_gib": CAPACITY_BUDGET_BYTES // cache_bytes,
                    "bytes_per_token_all_layers": bytes_per_token[variant],
                }
            )
    return records


def plot_long_context(data: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    records = long_context_projection(data)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.9))
    for variant in ORDER:
        rows = [row for row in records if row["variant"] == variant]
        contexts = [row["context_tokens"] for row in rows]
        memory = [row["cache_mib"] for row in rows]
        sequences = [row["sequences_per_1_gib"] for row in rows]
        axes[0].plot(
            contexts,
            memory,
            marker="o",
            linewidth=2.2,
            color=COLORS[variant],
            label=variant,
        )
        axes[1].plot(
            contexts,
            sequences,
            marker="o",
            linewidth=2.2,
            color=COLORS[variant],
            label=variant,
        )
        axes[0].annotate(
            f"{memory[-1]:.0f} MiB",
            (contexts[-1], memory[-1]),
            xytext=(-6, 7),
            textcoords="offset points",
            ha="right",
            color=COLORS[variant],
        )
        axes[1].annotate(
            f"{sequences[-1]} seq",
            (contexts[-1], sequences[-1]),
            xytext=(-6, 7),
            textcoords="offset points",
            ha="right",
            color=COLORS[variant],
        )

        measured = next(
            item
            for item in next(
                model for model in data["variants"] if model["variant"] == variant
            )["inference"]
            if item["total_sequence_length"]
            == max(
                row["total_sequence_length"]
                for model in data["variants"]
                for row in model["inference"]
            )
        )
        axes[0].scatter(
            measured["total_sequence_length"],
            measured["cache_occupied_bytes"] / 1024**2,
            marker="X",
            s=95,
            color=COLORS[variant],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )

    tick_labels = ("128", "512", "2K", "8K", "32K", "128K")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log", base=2)
        axis.set_xticks(LONG_CONTEXTS, tick_labels)
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    axes[0].scatter(
        [],
        [],
        marker="X",
        s=75,
        color="#555555",
        label="Measured tensor @ 160 tokens",
    )
    axes[0].legend()
    axes[0].set_title("Cache footprint per sequence")
    axes[0].set_xlabel("Context length (tokens)")
    axes[0].set_ylabel("Cache only (MiB, all layers)")
    axes[1].set_title("Concurrency under a fixed cache budget")
    axes[1].set_xlabel("Context length (tokens)")
    axes[1].set_ylabel("Sequences fitting in 1 GiB of cache")
    axes[0].text(
        0.03,
        0.08,
        "MLA: 8× less cache than MHA",
        transform=axes[0].transAxes,
        color=COLORS["MLA"],
        fontweight="bold",
    )
    axes[1].text(
        0.03,
        0.08,
        "MLA: 8× more cache-only slots",
        transform=axes[1].transAxes,
        color=COLORS["MLA"],
        fontweight="bold",
    )
    figure.suptitle("Long-context capacity from the implemented cache layouts")
    figure.text(
        0.5,
        -0.01,
        "X markers are measured cache tensors at 160 tokens; lines beyond the "
        "192-token model window are analytical. Cache only; excludes weights, "
        "activations, temporary K/V, allocator overhead, and latency.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "long_context_capacity.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)

    with (output_dir / "long_context_projection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def write_summary(data: dict[str, Any], output_dir: Path) -> None:
    longest_prompt = max(
        row["prompt_length"]
        for variant in data["variants"]
        for row in variant["inference"]
    )
    rows = []
    for variant in sorted(
        data["variants"], key=lambda row: ORDER.index(row["variant"])
    ):
        inference = next(
            row
            for row in variant["inference"]
            if row["prompt_length"] == longest_prompt
        )
        rows.append((variant, inference))
    mha_cache = rows[0][1]["cache_occupied_bytes"]
    lines = [
        "# Efficient Transformer Systems — result summary",
        "",
        f"Completed controlled run on `{data['environment']['device']}` with "
        f"PyTorch `{data['environment']['torch']}`. All variants saw "
        f"`{rows[0][0]['training_tokens']:,}` training tokens.",
        "",
        "| Variant | Parameters | Final val loss | Val perplexity | Train tokens/s | Cached ms/token | Recompute ms/token | Decode speedup | Cache | vs MHA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, inference in rows:
        cache = inference["cache_occupied_bytes"]
        lines.append(
            f"| {variant['variant']} | {variant['parameter_count']:,} | "
            f"{variant['final_validation_loss']:.3f} | "
            f"{variant['final_validation_perplexity']:.2f} | "
            f"{variant['training_tokens_per_second']:,.0f} | "
            f"{inference['cached_decode_ms_per_token']['p50']:.3f} | "
            f"{inference['recompute_decode_ms_per_token']['p50']:.3f} | "
            f"{inference['decode_speedup_p50']:.2f}× | "
            f"{cache / 1024:.1f} KiB | {mha_cache / cache:.1f}× smaller |"
        )
    lines.extend(
        [
            "",
            f"Inference columns use the longest measured prompt (`{longest_prompt}` tokens).",
            "",
            "![Attention architectures](attention_architectures.png)",
            "",
            "![Training curves](training_curves.png)",
            "",
            "![Inference latency](inference_latency.png)",
            "",
            "![Quality and efficiency](quality_efficiency.png)",
            "",
            "## Long-context cache projection",
            "",
            "The following figure is an analytical byte count for the implemented "
            "cache layouts, not a timed 128K-context run. It isolates the inference "
            "capacity benefit: MLA uses 8× less cache than MHA and therefore fits "
            "8× more sequences under the same cache-only memory budget.",
            "",
            "![Long-context cache capacity](long_context_capacity.png)",
            "",
        ]
    )
    (output_dir / "EXPERIMENT_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_attention_architectures(args.output_dir)
    plot_training(data, args.output_dir)
    plot_inference(data, args.output_dir)
    plot_tradeoffs(data, args.output_dir)
    plot_long_context(data, args.output_dir)
    write_summary(data, args.output_dir)
    print(f"Wrote figures and summary to {args.output_dir}")


if __name__ == "__main__":
    main()

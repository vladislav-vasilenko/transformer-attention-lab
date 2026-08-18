"""Render figures for the 4-layer GDN/KDA hybrid experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402


ORDER = (
    "MHA_4L",
    "GQA_4L",
    "MQA_4L",
    "MLA_4L",
    "GDN_4L",
    "KDA_4L",
    "GDN_GA_3TO1",
    "KDA_MLA_3TO1",
    "K3_KDA_GMLA_3TO1",
)
LABELS = {
    "MHA_4L": "MHA (4L)",
    "GQA_4L": "GQA (4L, 2 KV heads)",
    "MQA_4L": "MQA (4L, 1 KV head)",
    "MLA_4L": "MLA-style (DeepSeek-V2; Kimi K2/K2.5 lineage)",
    "GDN_4L": "Gated DeltaNet (4L pure)",
    "KDA_4L": "KDA (4L pure)",
    "GDN_GA_3TO1": "GDN + gated MHA (Qwen3-Next-inspired)",
    "KDA_MLA_3TO1": "KDA + MLA-style (Kimi Linear-inspired)",
    "K3_KDA_GMLA_3TO1": "KDA + gated MLA (Kimi K3-inspired*)",
}
SHORT_LABELS = {
    "MHA_4L": "MHA",
    "GQA_4L": "GQA",
    "MQA_4L": "MQA",
    "MLA_4L": "MLA",
    "GDN_4L": "GDN pure",
    "KDA_4L": "KDA pure",
    "GDN_GA_3TO1": "GDN hybrid",
    "KDA_MLA_3TO1": "KDA hybrid",
    "K3_KDA_GMLA_3TO1": "K3 gated hybrid",
}
COLORS = {
    "MHA_4L": "#16324F",
    "GQA_4L": "#4E79A7",
    "MQA_4L": "#B07AA1",
    "MLA_4L": "#F28E2B",
    "GDN_4L": "#59A14F",
    "KDA_4L": "#E15759",
    "GDN_GA_3TO1": "#00A6A6",
    "KDA_MLA_3TO1": "#7A5195",
    "K3_KDA_GMLA_3TO1": "#D45087",
}
LONG_CONTEXTS = (128, 512, 2_048, 8_192, 32_768, 131_072)
CAPACITY_BUDGET_BYTES = 1024**3
MECHANISM_LINEAGE = (
    "Mechanism lineage: MLA-style -> DeepSeek-V2 and Kimi K2/K2.5; "
    "GDN hybrid -> Qwen3-Next; KDA/MLA hybrid -> Kimi Linear; "
    "KDA/gated-MLA hybrid -> Kimi K3."
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rounded_box(
    axis: Any,
    center: tuple[float, float],
    text: str,
    color: str,
    *,
    width: float = 0.17,
    height: float = 0.18,
    hatch: str | None = None,
) -> None:
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.018",
        facecolor=color,
        edgecolor="#16324F",
        linewidth=1.2,
        alpha=0.18,
        hatch=hatch,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=9)


def plot_hybrid_architectures(output_dir: Path) -> None:
    """Original schematic of the schedules and scalar/channel-wise decay."""

    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    schedule_axis, gate_axis = axes

    schedule_axis.set_xlim(0, 1)
    schedule_axis.set_ylim(0, 1)
    schedule_axis.axis("off")
    schedule_axis.set_title("Whole-layer 3:1 hybrid schedules", pad=12)
    x_positions = (0.13, 0.36, 0.59, 0.82)
    schedules = (
        (
            0.94,
            "Qwen3-Next-inspired\nschedule",
            ("GDN", "GDN", "GDN", "gated\nMHA"),
            COLORS["GDN_GA_3TO1"],
        ),
        (
            0.64,
            "Kimi Linear research\narchitecture (not K2/K2.5)",
            ("KDA", "KDA", "KDA", "MLA-style"),
            COLORS["KDA_MLA_3TO1"],
        ),
        (
            0.34,
            "Kimi K3-inspired schedule*\n(launch disclosure)",
            ("KDA", "KDA", "KDA", "gated\nMLA"),
            COLORS["K3_KDA_GMLA_3TO1"],
        ),
    )
    for y, label, blocks, color in schedules:
        schedule_axis.text(0.01, y, label, ha="left", va="center", fontsize=10)
        for index, (x, block) in enumerate(zip(x_positions, blocks, strict=True)):
            _rounded_box(
                schedule_axis,
                (x, y - 0.12),
                block,
                color if index < 3 else COLORS["MHA_4L"],
                hatch="////" if index == 3 else None,
            )
            if index:
                schedule_axis.add_patch(
                    FancyArrowPatch(
                        (x_positions[index - 1] + 0.09, y - 0.12),
                        (x - 0.09, y - 0.12),
                        arrowstyle="->",
                        mutation_scale=10,
                        color="#555555",
                    )
                )
            schedule_axis.text(
                x,
                y - 0.25,
                f"layer {index + 1}",
                ha="center",
                va="top",
                fontsize=8,
                color="#555555",
            )
    gate_axis.set_xlim(0, 1)
    gate_axis.set_ylim(0, 1)
    gate_axis.axis("off")
    gate_axis.set_title("Same delta rule, different forgetting granularity", pad=12)
    for x, title, variant, channel_count in (
        (0.28, "Gated DeltaNet", "GDN_4L", 1),
        (0.73, "Kimi Delta Attention", "KDA_4L", 5),
    ):
        gate_axis.text(x, 0.87, title, ha="center", va="center", fontsize=11)
        gate_axis.add_patch(
            Rectangle(
                (x - 0.15, 0.37),
                0.30,
                0.30,
                facecolor=COLORS[variant],
                edgecolor="#16324F",
                alpha=0.16,
            )
        )
        for line in range(1, 5):
            gate_axis.plot(
                (x - 0.15, x + 0.15),
                (0.37 + 0.06 * line, 0.37 + 0.06 * line),
                color="#A0A0A0",
                linewidth=0.6,
            )
            gate_axis.plot(
                (x - 0.15 + 0.06 * line, x - 0.15 + 0.06 * line),
                (0.37, 0.67),
                color="#A0A0A0",
                linewidth=0.6,
            )
        gate_axis.text(x, 0.52, r"$S_t$", ha="center", va="center", fontsize=18)
        if channel_count == 1:
            labels = ["one αₜ,ₕ"]
            gate_x = [x]
        else:
            labels = ["α₁", "α₂", "…", "αdₖ"]
            gate_x = [x - 0.13, x - 0.045, x + 0.045, x + 0.13]
        for label_x, label in zip(gate_x, labels, strict=True):
            gate_axis.text(
                label_x,
                0.73,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=COLORS[variant],
            )
            gate_axis.add_patch(
                FancyArrowPatch(
                    (label_x, 0.70),
                    (label_x, 0.64),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    color=COLORS[variant],
                )
            )
        gate_axis.text(
            x,
            0.25,
            "scalar decay per head"
            if channel_count == 1
            else "channel-wise decay per head",
            ha="center",
            va="center",
            fontsize=9,
        )
        gate_axis.text(
            x,
            0.17,
            (
                r"$\bar S=\alpha S;\;S\leftarrow\bar S+\beta k(v-k^T\bar S)^T$"
                if channel_count == 1
                else r"$\bar S=\mathrm{Diag}(\alpha)S;\;S\leftarrow\bar S+\beta k(v-k^T\bar S)^T$"
            ),
            ha="center",
            va="center",
            fontsize=9,
        )
    figure.suptitle(
        "Linear-attention operators and recurrent/full hybrids in nanoGPT", y=1.02
    )
    figure.text(
        0.5,
        0.015,
        "MLA-style lineage: DeepSeek-V2; Kimi K2 and K2.5 also use MLA "
        "decoders. *K3 label follows the 22 Jul 2026 launch diagram.\n"
        "Token-mixer study only: all variants hold the dense GPT-2 MLP fixed; "
        "production MoE, AttnRes, positional details, and fused kernels are out of scope.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.98))
    figure.savefig(
        output_dir / "hybrid_architectures.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def plot_training(data: dict[str, Any], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for variant in data["variants"]:
        name = variant["variant"]
        history = variant["training_history"]
        tokens = [row["tokens_seen"] for row in history]
        axes[0].plot(
            tokens,
            [row["validation_loss"] for row in history],
            marker="o",
            color=COLORS[name],
            label=LABELS[name],
        )
        axes[1].plot(
            tokens,
            [row["validation_perplexity"] for row in history],
            marker="o",
            color=COLORS[name],
            label=LABELS[name],
        )
    axes[0].set_title("Validation loss")
    axes[0].set_ylabel("Cross-entropy (lower is better)")
    axes[1].set_title("Validation perplexity")
    axes[1].set_ylabel("Character-level PPL (lower is better)")
    for axis in axes:
        axis.set_xlabel("Training tokens seen")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Matched 4-layer training under one token budget")
    figure.tight_layout()
    figure.savefig(
        output_dir / "hybrid_training_curves.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def plot_inference(data: dict[str, Any], output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for variant in data["variants"]:
        name = variant["variant"]
        rows = variant["inference"]
        prompts = [row["prompt_length"] for row in rows]
        axes[0].plot(
            prompts,
            [row["cached_decode_ms_per_token"]["p50"] for row in rows],
            marker="o",
            color=COLORS[name],
            label=LABELS[name],
        )
        axes[1].plot(
            prompts,
            [row["decode_speedup_p50"] for row in rows],
            marker="o",
            color=COLORS[name],
            label=LABELS[name],
        )
    axes[0].set_title("Cached decode on the reference path")
    axes[0].set_ylabel("p50 latency (ms/token, log scale)")
    axes[0].set_yscale("log")
    axes[1].set_title("Caching vs full-prefix recomputation")
    axes[1].set_ylabel("Recompute / cached latency (×)")
    axes[1].axhline(1, color="#777777", linewidth=1)
    for axis in axes:
        axis.set_xlabel("Prompt length (tokens)")
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8)
    figure.text(
        0.5,
        -0.01,
        "GDN/KDA use an intentionally sequential PyTorch scan; these timings "
        "measure this prototype, not FLA/FlashKDA production kernels.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "hybrid_inference_latency.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def state_projection(data: dict[str, Any]) -> list[dict[str, Any]]:
    layouts = {row["variant"]: row["cache_layout"] for row in data["variants"]}
    records = []
    for context in LONG_CONTEXTS:
        for variant in ORDER:
            layout = layouts[variant]
            fixed = layout["fixed_bytes"]
            growing = layout["bytes_per_token"] * context
            total = fixed + growing
            records.append(
                {
                    "variant": variant,
                    "context_tokens": context,
                    "fixed_bytes": fixed,
                    "growing_bytes": growing,
                    "state_bytes": total,
                    "state_gb": total / 1_000_000_000,
                    "state_mib": total / 1024**2,
                    "sequences_per_1_gib": CAPACITY_BUDGET_BYTES // total,
                    "bytes_per_token": layout["bytes_per_token"],
                }
            )
    return records


def plot_total_state_gb(data: dict[str, Any], output_dir: Path) -> None:
    """Plot measured state tensors and their exact context-length storage law."""

    layouts = {row["variant"]: row["cache_layout"] for row in data["variants"]}
    measured_contexts = sorted(
        {
            row["total_sequence_length"]
            for variant in data["variants"]
            for row in variant["inference"]
        }
    )
    block_size = data["experiment_config"]["block_size"]
    projection_contexts = sorted(
        set(measured_contexts)
        | {block_size, 512, 2_048, 8_192, 32_768, 65_536, 131_072}
    )

    figure, axes = plt.subplots(1, 2, figsize=(15.2, 6.4), sharex=True)
    linear_axis, log_axis = axes
    for variant in ORDER:
        layout = layouts[variant]

        def total_gb(context: int) -> float:
            total = layout["fixed_bytes"] + layout["bytes_per_token"] * context
            return total / 1_000_000_000

        within_window = [value for value in projection_contexts if value <= block_size]
        projected = [value for value in projection_contexts if value >= block_size]
        result = next(row for row in data["variants"] if row["variant"] == variant)
        actual_contexts = [row["total_sequence_length"] for row in result["inference"]]
        actual_gb = [
            row["cache_occupied_bytes"] / 1_000_000_000 for row in result["inference"]
        ]
        for axis in axes:
            axis.plot(
                within_window,
                [total_gb(value) for value in within_window],
                color=COLORS[variant],
                linewidth=2.1,
                label=LABELS[variant] if axis is linear_axis else None,
            )
            axis.plot(
                projected,
                [total_gb(value) for value in projected],
                color=COLORS[variant],
                linewidth=2.1,
                linestyle="--",
            )
            axis.scatter(
                actual_contexts,
                actual_gb,
                marker="X",
                s=70,
                color=COLORS[variant],
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )

    linear_axis.set_title("Absolute GB scale")
    linear_axis.set_ylabel("Total persistent inference state / KV cache (GB)")
    linear_axis.set_ylim(bottom=0)
    log_axis.set_title("Log GB scale reveals fixed recurrent states")
    log_axis.set_yscale("log")
    log_axis.set_ylabel("Total persistent inference state / KV cache (GB, log scale)")
    pure_endpoint = (
        layouts["GDN_4L"]["fixed_bytes"]
        + layouts["GDN_4L"]["bytes_per_token"] * projection_contexts[-1]
    ) / 1_000_000_000
    log_axis.annotate(
        f"pure GDN = pure KDA\n{pure_endpoint:.6f} GB (constant)",
        (projection_contexts[-1], pure_endpoint),
        xytext=(-8, 9),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
        arrowprops={"arrowstyle": "->", "color": "#777777", "linewidth": 0.8},
    )
    gated_hybrid_endpoint = (
        layouts["K3_KDA_GMLA_3TO1"]["fixed_bytes"]
        + layouts["K3_KDA_GMLA_3TO1"]["bytes_per_token"] * projection_contexts[-1]
    ) / 1_000_000_000
    log_axis.annotate(
        "ungated and gated MLA anchors\nhave identical state law",
        (projection_contexts[-1], gated_hybrid_endpoint),
        xytext=(-8, -13),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
        arrowprops={"arrowstyle": "->", "color": "#777777", "linewidth": 0.8},
    )

    for axis in axes:
        axis.axvline(block_size, color="#777777", linewidth=1, linestyle=":")
        axis.set_xscale("log")
        axis.set_xlabel("Context length (tokens, log scale)")
        axis.grid(alpha=0.28, which="both")
    log_axis.text(
        block_size * 1.08,
        log_axis.get_ylim()[1] / 1.8,
        f"trained window = {block_size}",
        color="#555555",
        fontsize=8,
        rotation=90,
        va="top",
    )
    linear_axis.scatter(
        [],
        [],
        marker="X",
        s=70,
        color="#555555",
        label="measured occupied tensors",
    )
    linear_axis.plot(
        [],
        [],
        color="#555555",
        linestyle="--",
        label="exact layout projection",
    )
    handles, labels = linear_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        fontsize=8,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
    )
    figure.suptitle(
        "Inference state vs context length — measured 4-layer architectures\n"
        "n_heads=4, d_model=128, batch=1, dtype=FP32",
        y=1.01,
    )
    figure.text(
        0.5,
        0.01,
        f"{MECHANISM_LINEAGE}\n"
        "Solid/X = executed state inside 192 tokens; dashed = exact byte-law "
        "projection, not 128K latency or quality. Token mixer only; production MoE excluded.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.82))
    figure.savefig(
        output_dir / "total_state_vs_context_gb.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_state_scaling(data: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    records = state_projection(data)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    longest_total = max(
        row["total_sequence_length"]
        for variant in data["variants"]
        for row in variant["inference"]
    )
    for variant in ORDER:
        rows = [row for row in records if row["variant"] == variant]
        contexts = [row["context_tokens"] for row in rows]
        memory = [row["state_mib"] for row in rows]
        capacity = [row["sequences_per_1_gib"] for row in rows]
        axes[0].plot(
            contexts,
            memory,
            marker="o",
            linewidth=2.1,
            color=COLORS[variant],
            label=LABELS[variant],
        )
        axes[1].plot(
            contexts,
            capacity,
            marker="o",
            linewidth=2.1,
            color=COLORS[variant],
            label=LABELS[variant],
        )
        endpoint = rows[-1]
        if variant != "GDN_4L":
            memory_label = (
                f"GDN/KDA {endpoint['state_mib']:.2f} MiB"
                if variant == "KDA_4L"
                else f"{endpoint['state_mib']:.1f} MiB"
            )
            capacity_label = (
                f"GDN/KDA {endpoint['sequences_per_1_gib']} seq"
                if variant == "KDA_4L"
                else f"{endpoint['sequences_per_1_gib']} seq"
            )
            axes[0].annotate(
                memory_label,
                (contexts[-1], memory[-1]),
                xytext=(-6, 7),
                textcoords="offset points",
                ha="right",
                color=COLORS[variant],
                fontsize=8,
            )
            axes[1].annotate(
                capacity_label,
                (contexts[-1], capacity[-1]),
                xytext=(-6, 7),
                textcoords="offset points",
                ha="right",
                color=COLORS[variant],
                fontsize=8,
            )
        measured_row = next(
            item
            for item in next(
                model for model in data["variants"] if model["variant"] == variant
            )["inference"]
            if item["total_sequence_length"] == longest_total
        )
        axes[0].scatter(
            longest_total,
            measured_row["cache_occupied_bytes"] / 1024**2,
            marker="X",
            s=85,
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
        axis.set_xlabel("Context length (tokens)")
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8)
    axes[0].set_title("Persistent inference state per sequence")
    axes[0].set_ylabel("State/cache (MiB, all 4 layers)")
    axes[1].set_title("Concurrency under a 1 GiB state budget")
    axes[1].set_ylabel("Whole sequence states fitting in 1 GiB")
    figure.text(
        0.5,
        -0.01,
        f"X markers are measured tensors at {longest_total} tokens; longer points "
        "are exact storage-law projections, not long-context quality or latency runs.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "hybrid_state_scaling.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)

    with (output_dir / "hybrid_state_projection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def plot_state_composition(data: dict[str, Any], output_dir: Path) -> None:
    layouts = {row["variant"]: row["cache_layout"] for row in data["variants"]}
    contexts = (160, 131_072)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = [SHORT_LABELS[name] for name in ORDER]
    for axis, context in zip(axes, contexts, strict=True):
        fixed = [layouts[name]["fixed_bytes"] / 1024**2 for name in ORDER]
        growing = [
            layouts[name]["bytes_per_token"] * context / 1024**2 for name in ORDER
        ]
        axis.bar(
            names,
            growing,
            color=[COLORS[name] for name in ORDER],
            label="token-growing KV/latent cache",
        )
        axis.bar(
            names,
            fixed,
            bottom=growing,
            color="none",
            edgecolor=[COLORS[name] for name in ORDER],
            hatch="////",
            linewidth=1.3,
            label="fixed recurrent + conv state",
        )
        axis.set_title(f"State composition at {context:,} tokens")
        axis.set_ylabel("MiB, all 4 layers")
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
        for index, (base, extra) in enumerate(zip(growing, fixed, strict=True)):
            total = base + extra
            axis.text(
                index, total, f"{total:.2f}", ha="center", va="bottom", fontsize=8
            )
    axes[0].legend(fontsize=8)
    figure.suptitle("Why the hybrids are fixed + linear, not fully constant-memory")
    figure.tight_layout()
    figure.savefig(
        output_dir / "hybrid_state_composition.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def plot_quality_efficiency(data: dict[str, Any], output_dir: Path) -> None:
    variants = sorted(data["variants"], key=lambda row: ORDER.index(row["variant"]))
    names = [row["variant"] for row in variants]
    longest_rows = [row["inference"][-1] for row in variants]
    state_kib = [row["cache_occupied_bytes"] / 1024 for row in longest_rows]
    perplexity = [row["final_validation_perplexity"] for row in variants]
    throughput = [row["training_tokens_per_second"] for row in variants]
    cached_latency = [row["cached_decode_ms_per_token"]["p50"] for row in longest_rows]

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    label_offsets = {
        "MHA_4L": (7, 5),
        "GQA_4L": (7, 5),
        "MQA_4L": (7, -10),
        "MLA_4L": (7, 5),
        "GDN_4L": (12, 3),
        "KDA_4L": (12, -3),
        "GDN_GA_3TO1": (7, 5),
        "KDA_MLA_3TO1": (12, 30),
        "K3_KDA_GMLA_3TO1": (12, 13),
    }
    for name, state, ppl in zip(names, state_kib, perplexity, strict=True):
        axes[0].scatter(state, ppl, s=110, color=COLORS[name])
        annotation = {
            "xytext": label_offsets[name],
            "textcoords": "offset points",
        }
        if name in {"GDN_4L", "KDA_4L", "KDA_MLA_3TO1", "K3_KDA_GMLA_3TO1"}:
            annotation["arrowprops"] = {
                "arrowstyle": "-",
                "color": COLORS[name],
                "linewidth": 0.7,
            }
        axes[0].annotate(SHORT_LABELS[name], (state, ppl), **annotation)
    axes[0].set_title("Quality vs measured state at 160 tokens")
    axes[0].set_xlabel("Persistent state/cache (KiB, log scale)")
    axes[0].set_ylabel("Validation PPL (lower is better)")
    axes[0].set_xscale("log", base=2)
    axes[0].grid(alpha=0.25)

    for name, latency, ppl in zip(names, cached_latency, perplexity, strict=True):
        axes[1].scatter(
            latency,
            ppl,
            s=105,
            color=COLORS[name],
            label=SHORT_LABELS[name],
        )
    axes[1].set_title(
        "Quality vs cached reference latency\n128-token prompt + 32 decode"
    )
    axes[1].set_xlabel("Decode p50 (ms/token, log scale)")
    axes[1].set_ylabel("Validation PPL (lower is better)")
    axes[1].set_xscale("log")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=7, ncol=2, loc="upper right")

    axes[2].bar(
        [SHORT_LABELS[name] for name in names],
        throughput,
        color=[COLORS[name] for name in names],
    )
    axes[2].set_title(
        "Unoptimized reference-path training throughput\nCPU, 64-token windows"
    )
    axes[2].set_ylabel("Tokens/second (higher is better)")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", alpha=0.25)
    for index, value in enumerate(throughput):
        axes[2].text(
            index, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=8
        )
    axes[2].annotate(
        "token-by-token Python scan\n(no chunkwise kernel)",
        (5.5, max(throughput[4:])),
        xytext=(4.7, max(throughput) * 0.48),
        ha="center",
        fontsize=8,
        color="#555555",
        arrowprops={"arrowstyle": "->", "color": "#777777", "linewidth": 0.8},
    )
    figure.text(
        0.5,
        0.005,
        f"{MECHANISM_LINEAGE}\n"
        "Lower-left is better only for the two plotted metrics. PPL is next-token "
        "quality, not answer quality; the recurrent path is an unoptimized scan. "
        "Production MoE is held constant out of scope.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.11, 1, 1))
    figure.savefig(
        output_dir / "hybrid_quality_efficiency.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


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
    lines = [
        "# Recurrent/full attention hybrid experiment — result summary",
        "",
        f"Completed controlled reference-path run on `{data['environment']['device']}`. "
        f"Every variant used `{data['experiment_config']['n_layer']}` layers and "
        f"`{rows[0][0]['training_tokens']:,}` training tokens.",
        "",
        "| Variant | Parameters | Val PPL | Train tok/s | Cached ms/token | Recompute ms/token | State @160 | Fixed state | Growing bytes/token |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, inference in rows:
        layout = variant["cache_layout"]
        lines.append(
            f"| {LABELS[variant['variant']]} | {variant['parameter_count']:,} | "
            f"{variant['final_validation_perplexity']:.2f} | "
            f"{variant['training_tokens_per_second']:,.0f} | "
            f"{inference['cached_decode_ms_per_token']['p50']:.3f} | "
            f"{inference['recompute_decode_ms_per_token']['p50']:.3f} | "
            f"{inference['cache_occupied_bytes'] / 1024:.1f} KiB | "
            f"{layout['fixed_bytes'] / 1024:.1f} KiB | "
            f"{layout['bytes_per_token']:,} |"
        )
    by_name = {variant["variant"]: variant for variant, _ in rows}
    inference_by_name = {variant["variant"]: inference for variant, inference in rows}
    gdn = by_name["GDN_4L"]
    mha = by_name["MHA_4L"]
    gdn_state_kib = inference_by_name["GDN_4L"]["cache_occupied_bytes"] / 1024
    gdn_reference_slowdown = (
        mha["training_tokens_per_second"] / gdn["training_tokens_per_second"]
    )
    projections = state_projection(data)
    endpoints = {
        row["variant"]: row
        for row in projections
        if row["context_tokens"] == LONG_CONTEXTS[-1]
    }
    kda_ratio = (
        endpoints["MLA_4L"]["state_bytes"] / endpoints["KDA_MLA_3TO1"]["state_bytes"]
    )
    k3_ratio = (
        endpoints["MLA_4L"]["state_bytes"]
        / endpoints["K3_KDA_GMLA_3TO1"]["state_bytes"]
    )
    lines.extend(
        [
            "",
            "The measured state column uses the 128-token prompt plus 32 decoded "
            "tokens. GDN and KDA timings are for the sequential PyTorch reference "
            "scan, not the production FLA/FlashKDA kernels.",
            "",
            f"Pure GDN is the lower-left point in the two-metric PPL/state panel "
            f"(PPL `{gdn['final_validation_perplexity']:.2f}`, state "
            f"`{gdn_state_kib:.1f} KiB`), "
            f"but it is not the globally best system: it has more parameters than "
            f"MHA and its unoptimized training path is `{gdn_reference_slowdown:.1f}×` "
            "slower. PPL measures held-out next-token prediction at the trained "
            "window, not instruction-answer quality or long-range recall.",
            "",
            "![Hybrid architectures](hybrid_architectures.png)",
            "",
            "![Training curves](hybrid_training_curves.png)",
            "",
            "![Reference-path inference](hybrid_inference_latency.png)",
            "",
            "![Quality and efficiency](hybrid_quality_efficiency.png)",
            "",
            "## Long-context state projection",
            "",
            f"At 128K tokens, the KDA/MLA hybrid uses "
            f"`{endpoints['KDA_MLA_3TO1']['state_mib']:.2f} MiB` versus "
            f"`{endpoints['MLA_4L']['state_mib']:.2f} MiB` for four MLA-style "
            f"layers ({kda_ratio:.2f}× smaller, approaching the 75% asymptote). "
            f"The gated-MLA anchor has the same storage law and reaches "
            f"`{endpoints['K3_KDA_GMLA_3TO1']['state_mib']:.2f} MiB` "
            f"({k3_ratio:.2f}× smaller than four MLA-style layers). "
            "These are exact byte counts for the implemented states, not 128K "
            "latency or quality measurements.",
            "",
            "![Long-context state scaling](hybrid_state_scaling.png)",
            "",
            "![Total state versus context length](total_state_vs_context_gb.png)",
            "",
            "![State composition](hybrid_state_composition.png)",
            "",
            "## Evidence boundary",
            "",
            "- One seed, one character-level corpus, 12 deterministic validation "
            "batches per checkpoint, and unequal parameter counts.",
            "- GDN and KDA use their respective output gates as well as different "
            "decay granularity, so their PPL difference is not an isolated "
            "scalar-versus-channel decay ablation.",
            f"- {data['training_recipe_boundary']}",
            f"- {data['timing_session_boundary']}",
            "- Learned GPT-2 positions, dense MLPs, and the 192-token model window "
            "do not reproduce the full Qwen3-Next, Kimi Linear, or Kimi K3 systems. "
            "The K3-inspired row isolates only the KDA ×3 + sigmoid-gated MLA "
            "attention schedule; Stable LatentMoE and Attention Residuals are out "
            "of scope.",
            "",
        ]
    )
    (output_dir / "HYBRID_EXPERIMENT_SUMMARY.md").write_text(
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
    plot_hybrid_architectures(args.output_dir)
    plot_training(data, args.output_dir)
    plot_inference(data, args.output_dir)
    plot_total_state_gb(data, args.output_dir)
    plot_state_scaling(data, args.output_dir)
    plot_state_composition(data, args.output_dir)
    plot_quality_efficiency(data, args.output_dir)
    write_summary(data, args.output_dir)
    print(f"Wrote hybrid figures and summary to {args.output_dir}")


if __name__ == "__main__":
    main()

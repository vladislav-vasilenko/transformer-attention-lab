"""Controlled 4-layer experiment for GDN- and KDA-based hybrid decoders."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .attention import KVCache, MLACache
from .benchmark import select_device
from .experiment import (
    CharacterCorpus,
    ExperimentConfig,
    benchmark_model,
    default_data_path,
    environment_metadata,
    train_variant,
)
from .linear_attention import DeltaStateCache
from .model import GPT, GPTConfig


VARIANTS = (
    ("MHA_4L", "MHA"),
    ("GQA_4L", "GQA"),
    ("MQA_4L", "MQA"),
    ("MLA_4L", "MLA"),
    ("GDN_4L", "GDN"),
    ("KDA_4L", "KDA"),
    ("GDN_GA_3TO1", "GDN_HYBRID"),
    ("KDA_MLA_3TO1", "KDA_HYBRID"),
    ("K3_KDA_GMLA_3TO1", "KDA_GATED_MLA_HYBRID"),
)


def _gpt_config(
    corpus: CharacterCorpus, config: ExperimentConfig, attention_variant: str
) -> GPTConfig:
    return GPTConfig(
        vocab_size=corpus.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        attention_variant=attention_variant,
        gqa_n_kv_heads=config.gqa_n_kv_heads,
        mla_latent_rank=config.mla_latent_rank,
        delta_conv_kernel=config.delta_conv_kernel,
        kda_gate_rank=config.kda_gate_rank,
        delta_scan_backend=config.delta_scan_backend,
    )


def shared_backbone_initialization(
    corpus: CharacterCorpus, config: ExperimentConfig
) -> dict[str, torch.Tensor]:
    """Freeze identical non-attention initialization across all variants."""

    torch.manual_seed(config.seed)
    template = GPT(_gpt_config(corpus, config, "MHA"))
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in template.state_dict().items()
        if ".attention." not in name
    }


def cache_layout(model: GPT) -> dict[str, Any]:
    """Extract the exact fixed-plus-linear storage law from model caches."""

    dtype = model.token_embedding.weight.dtype
    caches = model.new_caches(1, device="cpu", dtype=dtype)
    layer_rows = []
    for layer_index, cache in enumerate(caches):
        if isinstance(cache, DeltaStateCache):
            cache_type = cache.__class__.__name__
            detail = {
                "recurrent_state_bytes": cache.state_bytes,
                "convolution_history_bytes": cache.convolution_bytes,
            }
        elif isinstance(cache, MLACache):
            attention = model.blocks[layer_index].attention
            cache_type = (
                "gated MLA latent cache"
                if getattr(attention.config, "output_gate", False)
                else "MLA latent cache"
            )
            detail = {"latent_rank": cache.latent.shape[-1]}
        elif isinstance(cache, KVCache):
            attention = model.blocks[layer_index].attention
            kv_heads = cache.key.shape[1]
            if getattr(attention.config, "output_gate", False):
                cache_type = "gated MHA KV cache"
            elif kv_heads == 1:
                cache_type = "MQA KV cache"
            elif kv_heads < attention.config.n_heads:
                cache_type = "GQA KV cache"
            else:
                cache_type = "MHA KV cache"
            detail = {"kv_heads": kv_heads}
        else:  # pragma: no cover - guarded by the cache union
            raise TypeError(f"unsupported cache type: {type(cache)!r}")
        layer_rows.append(
            {
                "layer": layer_index + 1,
                "type": cache_type,
                "fixed_bytes": cache.fixed_bytes,
                "bytes_per_token": cache.bytes_per_token,
                **detail,
            }
        )
    return {
        "law": "fixed_bytes + bytes_per_token * context_tokens",
        "fixed_bytes": sum(row["fixed_bytes"] for row in layer_rows),
        "bytes_per_token": sum(row["bytes_per_token"] for row in layer_rows),
        "layers": layer_rows,
    }


def run_hybrid_experiment(
    data_path: Path,
    output_path: Path,
    config: ExperimentConfig,
    device_name: str = "auto",
    *,
    resume: bool = False,
) -> dict[str, Any]:
    if config.n_layer % 4:
        raise ValueError("the 3:1 hybrid comparison requires n_layer divisible by 4")
    torch.set_num_threads(config.num_threads)
    device = select_device(device_name)
    corpus = CharacterCorpus(data_path, config.split_fraction)
    shared_state = shared_backbone_initialization(corpus, config)
    completed: dict[str, dict[str, Any]] = {}
    if resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        normalized_config = json.loads(json.dumps(asdict(config)))
        if previous.get("experiment_config") != normalized_config:
            raise ValueError("cannot resume: experiment configuration changed")
        if previous.get("dataset", {}).get("sha256") != corpus.sha256:
            raise ValueError("cannot resume: dataset hash changed")
        completed = {row["variant"]: row for row in previous["variants"]}

    variants = []
    for result_name, model_variant in VARIANTS:
        if result_name in completed:
            print(f"Reusing completed {result_name}...", flush=True)
            variants.append(completed[result_name])
            continue
        print(f"Training {result_name} ({model_variant})...", flush=True)
        model, result = train_variant(
            result_name,
            corpus,
            config,
            device,
            model_variant=model_variant,
            shared_backbone_state=shared_state,
        )
        result["shared_backbone_initialization"] = True
        result["cache_layout"] = cache_layout(model.to("cpu"))
        model = model.to(device)
        result["inference"] = benchmark_model(model, corpus, config, device)
        variants.append(result)
        longest = result["inference"][-1]
        print(
            f"{result_name}: val ppl={result['final_validation_perplexity']:.2f}, "
            f"cached={longest['cached_decode_ms_per_token']['p50']:.3f} ms/token, "
            f"state={longest['cache_occupied_bytes'] / 1024:.1f} KiB",
            flush=True,
        )
        del model

    output = {
        "schema_version": 1,
        "title": "Efficient Transformer Systems: linear attention and 3:1 hybrids",
        "status": "completed controlled reference-path experiment",
        "question": (
            "Under one 4-layer nanoGPT training budget, how do MHA, GQA, MQA, "
            "MLA, pure GDN/KDA, and Qwen3-Next-, Kimi-Linear-, and "
            "Kimi-K3-inspired 3:1 attention schedules trade validation "
            "quality, persistent-state memory, and reference-path latency against "
            "homogeneous MHA, GQA, MQA, and MLA baselines?"
        ),
        "hypotheses": [
            "At equal depth and width, GQA and MQA reduce the token-growing KV-cache slope in direct proportion to their two and one shared KV heads, versus four KV heads in MHA.",
            "Pure GDN and KDA keep a context-independent recurrent state; their difference is scalar versus channel-wise forgetting, not state size.",
            "Three recurrent layers make the hybrid state grow only through its one-in-four full-attention anchor.",
            "KDA and GDN have the same fixed matrix-state size; KDA's channel-wise decay targets quality and memory control, not a smaller state.",
            "The KDA/MLA hybrid approaches a 75% smaller long-context state than a four-layer MLA baseline, after a small fixed-state crossover.",
            "Adding an output gate to the one-in-four MLA anchor leaves the KDA hybrid storage law unchanged, so measured quality and latency isolate the gate's effect within this prototype.",
            "A sequential PyTorch scan may save persistent state without reproducing the optimized-kernel throughput reported for Qwen3-Next or Kimi Linear.",
        ],
        "environment": environment_metadata(device),
        "experiment_config": asdict(config),
        "dataset": {
            "path": str(Path("..") / "nanoGPT-lab" / data_path.name),
            "sha256": corpus.sha256,
            "vocab_size": corpus.vocab_size,
            "train_tokens": len(corpus.train),
            "validation_tokens": len(corpus.validation),
        },
        "controlled_variables": [
            "four decoder layers",
            "identical non-attention initial weights",
            "same corpus split and hash",
            "same seed and batch sequence",
            "same training-token budget, optimizer, width, and evaluation tokens",
        ],
        "training_protocol": {
            "mode": config.training_mode,
            "epochs": config.epochs if config.training_mode == "full_epoch" else None,
            "non_overlapping_windows_per_epoch": corpus.train_windows_per_epoch(
                config.train_context
            ),
            "batches_per_epoch": (
                (
                    corpus.train_windows_per_epoch(config.train_context)
                    + config.batch_size
                    - 1
                )
                // config.batch_size
            ),
            "tokens_per_full_epoch": corpus.train_windows_per_epoch(
                config.train_context
            )
            * config.train_context,
            "trailing_tokens_outside_complete_window": (
                len(corpus.train)
                - 1
                - corpus.train_windows_per_epoch(config.train_context)
                * config.train_context
            ),
            "coverage": (
                "Every complete non-overlapping train window is visited exactly "
                "once per epoch, including the final partial batch."
                if config.training_mode == "full_epoch"
                else "Random fixed-step sampling with replacement."
            ),
        },
        "implementation_boundary": (
            "Operator-level GPT-2/nanoGPT prototype with learned positions and dense "
            "MLPs. It reproduces the core recurrent equations, short convolutions, "
            "gates, persistent states, and 3:1 token-mixer schedules, but not MoE, "
            "Attention Residuals, partial RoPE/NoPE, large-model recipes, or "
            "optimized chunkwise kernels. The Kimi-K3-inspired label refers only "
            "to the KDA x3 + gated-MLA attention schedule."
        ),
        "training_recipe_boundary": (
            "One common AdamW parameter group applies weight decay uniformly, "
            "including to A_log and dt_bias. The official recurrent-layer recipes "
            "exclude those gate parameters from weight decay; this controlled "
            "prototype does not reproduce that optimizer grouping."
        ),
        "timing_session_boundary": (
            "GQA and MQA were appended with deterministic resume in a second local "
            "session, and the Kimi-K3-inspired variant in a later session, on the "
            "same recorded CPU configuration. Quality and storage results remain "
            "reproducible, but small timing differences across all nine variants "
            "should not be treated as a single-session benchmark."
        ),
        "sources": {
            "linear_transformer": "https://arxiv.org/abs/2006.16236",
            "gated_deltanet": "https://arxiv.org/abs/2412.06464",
            "gated_full_attention": "https://arxiv.org/abs/2505.06708",
            "qwen3_next": "https://qwen.ai/blog?id=e34c4305036ce60d55a0791b170337c2b70ae51d",
            "kimi_linear": "https://arxiv.org/abs/2510.26692",
            "kimi_code": "https://github.com/MoonshotAI/Kimi-Linear",
            "kimi_k3_launch": "https://www.kimi.com/blog/kimi-k3",
            "attention_residuals": "https://arxiv.org/abs/2603.15031",
        },
        "variants": variants,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hybrid_linear_attention_cpu.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-context", type=int, default=64)
    parser.add_argument(
        "--delta-scan-backend",
        choices=("reference", "compiled", "fla"),
        default="reference",
    )
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--inference-repeats", type=int, default=5)
    parser.add_argument(
        "--training-mode",
        choices=("full_epoch", "random_steps"),
        default="full_epoch",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        steps=args.steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        train_context=args.train_context,
        n_layer=4,
        delta_scan_backend=args.delta_scan_backend,
        num_threads=args.num_threads,
        inference_repeats=args.inference_repeats,
        training_mode=args.training_mode,
        epochs=args.epochs,
    )
    run_hybrid_experiment(
        args.data,
        args.output,
        config,
        args.device,
        resume=args.resume,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

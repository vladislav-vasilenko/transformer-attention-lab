"""Reproducible MHA/GQA/MQA/MLA and KV-cache microbenchmark.

Example:
    python -m research.benchmark --output results/cpu_benchmark.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch

from .attention import (
    AttentionConfig,
    CausalSelfAttention,
    MLAConfig,
    MultiLatentAttention,
)


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def elapsed_ms(device: torch.device, operation: Callable[[], Any]) -> float:
    synchronize(device)
    started = time.perf_counter_ns()
    operation()
    synchronize(device)
    return (time.perf_counter_ns() - started) / 1_000_000


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


@torch.inference_mode()
def run_cached_trial(
    attention: CausalSelfAttention | MultiLatentAttention,
    hidden: torch.Tensor,
    prompt_length: int,
    decode_tokens: int,
    device: torch.device,
) -> tuple[float, float, int, int]:
    cache = attention.new_cache(hidden.shape[0], device=device, dtype=hidden.dtype)
    prefill_ms = elapsed_ms(device, lambda: attention(hidden[:, :prompt_length], cache))

    def decode() -> None:
        for offset in range(decode_tokens):
            index = prompt_length + offset
            attention(hidden[:, index : index + 1], cache)

    decode_total_ms = elapsed_ms(device, decode)
    return (
        prefill_ms,
        decode_total_ms / decode_tokens,
        cache.occupied_bytes,
        cache.allocated_bytes,
    )


@torch.inference_mode()
def run_recompute_trial(
    attention: CausalSelfAttention | MultiLatentAttention,
    hidden: torch.Tensor,
    prompt_length: int,
    decode_tokens: int,
    device: torch.device,
) -> float:
    def decode() -> None:
        for offset in range(decode_tokens):
            prefix_end = prompt_length + offset + 1
            attention(hidden[:, :prefix_end])[:, -1:]

    return elapsed_ms(device, decode) / decode_tokens


def benchmark_configuration(
    attention: CausalSelfAttention | MultiLatentAttention,
    hidden: torch.Tensor,
    prompt_length: int,
    decode_tokens: int,
    device: torch.device,
    warmup: int,
    repeats: int,
    layers_for_cache: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        run_cached_trial(attention, hidden, prompt_length, decode_tokens, device)
        run_recompute_trial(attention, hidden, prompt_length, decode_tokens, device)

    prefill: list[float] = []
    cached_decode: list[float] = []
    recompute_decode: list[float] = []
    occupied_bytes = 0
    allocated_bytes = 0
    for repeat in range(repeats):
        # Alternating order reduces systematic thermal/order bias.
        if repeat % 2:
            recompute_decode.append(
                run_recompute_trial(
                    attention, hidden, prompt_length, decode_tokens, device
                )
            )
        cached = run_cached_trial(
            attention, hidden, prompt_length, decode_tokens, device
        )
        prefill.append(cached[0])
        cached_decode.append(cached[1])
        occupied_bytes, allocated_bytes = cached[2], cached[3]
        if not repeat % 2:
            recompute_decode.append(
                run_recompute_trial(
                    attention, hidden, prompt_length, decode_tokens, device
                )
            )

    cached_summary = summarize(cached_decode)
    recompute_summary = summarize(recompute_decode)
    prefill_summary = summarize(prefill)
    cached_total_p50 = prefill_summary["p50"] + cached_summary["p50"] * decode_tokens
    recompute_total_p50 = recompute_summary["p50"] * decode_tokens

    return {
        "prefill_ms": prefill_summary,
        "cached_decode_ms_per_token": cached_summary,
        "recompute_decode_ms_per_token": recompute_summary,
        "decode_speedup_p50": recompute_summary["p50"] / cached_summary["p50"],
        "end_to_end_speedup_p50": recompute_total_p50 / cached_total_p50,
        "kv_cache_bytes_per_layer": occupied_bytes,
        "kv_cache_bytes_model": occupied_bytes * layers_for_cache,
        "kv_cache_allocated_bytes_per_layer": allocated_bytes,
        "raw_timing_ms": {
            "prefill": prefill,
            "cached_decode_per_token": cached_decode,
            "recompute_decode_per_token": recompute_decode,
        },
    }


def environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "torch_num_threads": torch.get_num_threads(),
    }


def write_csv(output: Path, records: list[dict[str, Any]]) -> None:
    csv_path = output.with_suffix(".csv")
    fields = [
        "variant",
        "n_kv_heads",
        "mla_latent_rank",
        "prompt_length",
        "decode_tokens",
        "total_sequence_length",
        "attention_parameters",
        "prefill_ms_p50",
        "cached_decode_ms_per_token_p50",
        "recompute_decode_ms_per_token_p50",
        "decode_speedup_p50",
        "end_to_end_speedup_p50",
        "kv_cache_bytes_model",
        "kv_cache_mib_model",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "variant": record["variant"],
                    "n_kv_heads": record["n_kv_heads"],
                    "mla_latent_rank": record["mla_latent_rank"],
                    "prompt_length": record["prompt_length"],
                    "decode_tokens": record["decode_tokens"],
                    "total_sequence_length": record["total_sequence_length"],
                    "attention_parameters": record["attention_parameters"],
                    "prefill_ms_p50": record["prefill_ms"]["p50"],
                    "cached_decode_ms_per_token_p50": record[
                        "cached_decode_ms_per_token"
                    ]["p50"],
                    "recompute_decode_ms_per_token_p50": record[
                        "recompute_decode_ms_per_token"
                    ]["p50"],
                    "decode_speedup_p50": record["decode_speedup_p50"],
                    "end_to_end_speedup_p50": record["end_to_end_speedup_p50"],
                    "kv_cache_bytes_model": record["kv_cache_bytes_model"],
                    "kv_cache_mib_model": record["kv_cache_bytes_model"] / 1024**2,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--gqa-kv-heads", type=int, default=2)
    parser.add_argument("--mla-latent-rank", type=int, default=64)
    parser.add_argument("--layers-for-cache", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--prompt-lengths", type=int, nargs="+", default=[64, 128, 256, 512]
    )
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.decode_tokens <= 0 or args.repeats <= 0 or args.warmup < 0:
        raise ValueError(
            "decode-tokens/repeats must be positive; warmup cannot be negative"
        )
    if any(length <= 0 for length in args.prompt_lengths):
        raise ValueError("all prompt lengths must be positive")
    if args.layers_for_cache <= 0:
        raise ValueError("layers-for-cache must be positive")

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    dtype = DTYPES[args.dtype]
    maximum_length = max(args.prompt_lengths) + args.decode_tokens
    variants = ["MHA", "GQA", "MQA", "MLA"]

    config_summary = {
        "dim": args.dim,
        "n_heads": args.heads,
        "gqa_n_kv_heads": args.gqa_kv_heads,
        "mla_latent_rank": args.mla_latent_rank,
        "layers_for_cache": args.layers_for_cache,
        "batch_size": args.batch_size,
        "prompt_lengths": args.prompt_lengths,
        "decode_tokens": args.decode_tokens,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dtype": args.dtype,
        "seed": args.seed,
        "timing_scope": "one causal self-attention layer",
        "cache_scope": f"{args.layers_for_cache}-layer model extrapolation",
    }
    records: list[dict[str, Any]] = []

    was_gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for variant_name in variants:
            if variant_name == "MLA":
                config = MLAConfig(
                    dim=args.dim,
                    n_heads=args.heads,
                    latent_rank=args.mla_latent_rank,
                    max_seq_len=maximum_length,
                )
                attention = MultiLatentAttention(config)
                n_kv_heads = None
                mla_latent_rank = args.mla_latent_rank
            else:
                if variant_name == "MHA":
                    n_kv_heads = args.heads
                elif variant_name == "MQA":
                    n_kv_heads = 1
                else:
                    n_kv_heads = args.gqa_kv_heads
                config = AttentionConfig(
                    dim=args.dim,
                    n_heads=args.heads,
                    n_kv_heads=n_kv_heads,
                    max_seq_len=maximum_length,
                )
                attention = CausalSelfAttention(config)
                mla_latent_rank = None
            attention = attention.to(device=device, dtype=dtype).eval()
            parameters = sum(parameter.numel() for parameter in attention.parameters())

            for prompt_length in args.prompt_lengths:
                generator = torch.Generator().manual_seed(args.seed + prompt_length)
                hidden = torch.randn(
                    args.batch_size,
                    prompt_length + args.decode_tokens,
                    args.dim,
                    generator=generator,
                    dtype=dtype,
                ).to(device)
                measurements = benchmark_configuration(
                    attention,
                    hidden,
                    prompt_length,
                    args.decode_tokens,
                    device,
                    args.warmup,
                    args.repeats,
                    args.layers_for_cache,
                )
                records.append(
                    {
                        "variant": variant_name,
                        "n_kv_heads": n_kv_heads,
                        "mla_latent_rank": mla_latent_rank,
                        "prompt_length": prompt_length,
                        "decode_tokens": args.decode_tokens,
                        "total_sequence_length": prompt_length + args.decode_tokens,
                        "attention_parameters": parameters,
                        **measurements,
                    }
                )
                print(
                    f"{variant_name:>3} prompt={prompt_length:>4}: "
                    f"cache={measurements['cached_decode_ms_per_token']['p50']:.3f} ms/token, "
                    f"recompute={measurements['recompute_decode_ms_per_token']['p50']:.3f} ms/token, "
                    f"speedup={measurements['decode_speedup_p50']:.2f}x",
                    flush=True,
                )
    finally:
        if was_gc_enabled:
            gc.enable()

    result = {
        "schema_version": 1,
        "question": "How do KV caching and the number of KV heads affect autoregressive attention latency and cache memory?",
        "environment": environment_metadata(device),
        "configuration": config_summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(args.output, records)
    print(f"Wrote {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

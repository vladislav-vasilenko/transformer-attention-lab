"""Benchmark Gated DeltaNet and Kimi Delta Attention scan backends.

This is intentionally an attention-layer microbenchmark.  It separates the
execution cost of the recurrent scan from the GPT embedding and MLP stack, and
records whether ``torch.compile`` actually compiled or fell back to the exact
reference implementation.

Example:
    python -m research.linear_benchmark \\
        --scan-backend reference compiled \\
        --prompt-lengths 64 256 1024 4096 \\
        --output results/linear_scan_cpu.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch

from .benchmark import DTYPES, percentile, select_device, synchronize
from .linear_attention import (
    DeltaNetAttention,
    DeltaNetConfig,
    GatedDeltaNet,
    KimiDeltaAttention,
)


def elapsed_ms(device: torch.device, operation: Callable[[], Any]) -> float:
    synchronize(device)
    started = time.perf_counter_ns()
    operation()
    synchronize(device)
    return (time.perf_counter_ns() - started) / 1_000_000


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("at least one timing value is required")
    return {
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def build_attention(
    variant: str,
    *,
    dim: int,
    heads: int,
    max_seq_len: int,
    conv_kernel_size: int,
    gate_rank: int,
    scan_backend: str,
    device: torch.device,
    dtype: torch.dtype,
) -> DeltaNetAttention:
    config = DeltaNetConfig(
        dim=dim,
        n_heads=heads,
        max_seq_len=max_seq_len,
        conv_kernel_size=conv_kernel_size,
        gate_rank=gate_rank,
        channelwise_decay=variant == "KDA",
        scan_backend=scan_backend,
    )
    attention_cls = KimiDeltaAttention if variant == "KDA" else GatedDeltaNet
    return attention_cls(config).to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def cached_decode_trial(
    attention: DeltaNetAttention,
    hidden: torch.Tensor,
    prompt_length: int,
    decode_tokens: int,
    device: torch.device,
) -> tuple[float, float, int]:
    cache = attention.new_cache(hidden.shape[0], device=device, dtype=hidden.dtype)
    prefill_ms = elapsed_ms(device, lambda: attention(hidden[:, :prompt_length], cache))

    def decode() -> None:
        for offset in range(decode_tokens):
            index = prompt_length + offset
            attention(hidden[:, index : index + 1], cache)

    decode_ms_per_token = elapsed_ms(device, decode) / decode_tokens
    return prefill_ms, decode_ms_per_token, cache.occupied_bytes


def training_step_ms(
    attention: DeltaNetAttention,
    hidden: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    attention.train()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = attention(hidden).float().square().mean()
        loss.backward()
        optimizer.step()

    result = elapsed_ms(device, step)
    attention.eval()
    return result


def environment_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "torch_num_threads": torch.get_num_threads(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/linear_scan_benchmark.json")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--gate-rank", type=int, default=32)
    parser.add_argument("--conv-kernel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--train-length", type=int, default=64)
    parser.add_argument(
        "--prompt-lengths", type=int, nargs="+", default=(64, 256, 1024, 4096)
    )
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument(
        "--scan-backend",
        choices=("reference", "compiled"),
        nargs="+",
        default=("reference", "compiled"),
    )
    parser.add_argument(
        "--compiled-max-prompt-length",
        type=int,
        default=256,
        help=(
            "Skip longer compiled prefill points because torch.compile unrolls "
            "the Python recurrence. Set 0 to disable this safety limit."
        ),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dim <= 0 or args.heads <= 0 or args.dim % args.heads:
        raise ValueError("dim must be positive and divisible by heads")
    if any(length <= 0 for length in args.prompt_lengths):
        raise ValueError("all prompt lengths must be positive")
    if min(args.batch_size, args.train_batch_size, args.train_length) <= 0:
        raise ValueError("batch sizes and train length must be positive")
    if min(args.decode_tokens, args.repeats) <= 0 or args.warmup < 0:
        raise ValueError(
            "decode-tokens/repeats must be positive; warmup cannot be negative"
        )
    if args.compiled_max_prompt_length < 0:
        raise ValueError("compiled-max-prompt-length cannot be negative")

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    dtype = DTYPES[args.dtype]
    maximum_length = max(
        max(args.prompt_lengths) + args.decode_tokens, args.train_length
    )
    records: list[dict[str, Any]] = []
    was_gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for scan_backend in args.scan_backend:
            for variant in ("GDN", "KDA"):
                attention = build_attention(
                    variant,
                    dim=args.dim,
                    heads=args.heads,
                    max_seq_len=maximum_length,
                    conv_kernel_size=args.conv_kernel_size,
                    gate_rank=args.gate_rank,
                    scan_backend=scan_backend,
                    device=device,
                    dtype=dtype,
                )
                optimizer = torch.optim.AdamW(attention.parameters(), lr=3e-4)
                train_hidden = torch.randn(
                    args.train_batch_size,
                    args.train_length,
                    args.dim,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(args.warmup):
                    training_step_ms(attention, train_hidden, optimizer, device)

                train_times = [
                    training_step_ms(attention, train_hidden, optimizer, device)
                    for _ in range(args.repeats)
                ]
                train_summary = summarize(train_times)
                for prompt_length in args.prompt_lengths:
                    if (
                        scan_backend == "compiled"
                        and args.compiled_max_prompt_length
                        and prompt_length > args.compiled_max_prompt_length
                    ):
                        records.append(
                            {
                                "variant": variant,
                                "scan_backend": attention.scan_backend_metadata(),
                                "prompt_length": prompt_length,
                                "decode_tokens": args.decode_tokens,
                                "measurement_status": "skipped",
                                "skip_reason": (
                                    "torch.compile would unroll the token loop; "
                                    "use a chunkwise kernel or set "
                                    "--compiled-max-prompt-length 0 to force it"
                                ),
                            }
                        )
                        print(
                            f"{variant} requested=compiled prompt={prompt_length}: "
                            "skipped (Python scan would be unrolled)",
                            flush=True,
                        )
                        continue
                    hidden = torch.randn(
                        args.batch_size,
                        prompt_length + args.decode_tokens,
                        args.dim,
                        dtype=dtype,
                        device=device,
                    )
                    for _ in range(args.warmup):
                        cached_decode_trial(
                            attention,
                            hidden,
                            prompt_length,
                            args.decode_tokens,
                            device,
                        )
                    prefill: list[float] = []
                    cached_decode: list[float] = []
                    occupied_bytes = 0
                    for _ in range(args.repeats):
                        trial = cached_decode_trial(
                            attention,
                            hidden,
                            prompt_length,
                            args.decode_tokens,
                            device,
                        )
                        prefill.append(trial[0])
                        cached_decode.append(trial[1])
                        occupied_bytes = trial[2]
                    prefill_summary = summarize(prefill)
                    cached_summary = summarize(cached_decode)
                    records.append(
                        {
                            "variant": variant,
                            "scan_backend": attention.scan_backend_metadata(),
                            "measurement_status": "completed",
                            "prompt_length": prompt_length,
                            "decode_tokens": args.decode_tokens,
                            "prefill_ms": prefill_summary,
                            "prefill_tokens_per_second_p50": (
                                1000
                                * args.batch_size
                                * prompt_length
                                / prefill_summary["p50"]
                            ),
                            "cached_decode_ms_per_token": cached_summary,
                            "cache_occupied_bytes": occupied_bytes,
                            "training_step_ms": train_summary,
                            "training_tokens_per_second_p50": (
                                1000
                                * args.train_batch_size
                                * args.train_length
                                / train_summary["p50"]
                            ),
                            "raw_timing_ms": {
                                "prefill": prefill,
                                "cached_decode_per_token": cached_decode,
                                "training_step": train_times,
                            },
                        }
                    )
                    status = attention.scan_backend_metadata()
                    print(
                        f"{variant} requested={scan_backend} effective={status['effective']} "
                        f"prompt={prompt_length}: "
                        f"prefill={prefill_summary['p50']:.3f} ms, "
                        f"decode={cached_summary['p50']:.3f} ms/token",
                        flush=True,
                    )
    finally:
        if was_gc_enabled:
            gc.enable()

    output = {
        "schema_version": 1,
        "title": "GDN/KDA scan backend benchmark",
        "question": (
            "How do the exact reference and compiled recurrent scan backends affect "
            "attention-only training, prefill, and cached decode at increasing context?"
        ),
        "execution_boundary": (
            "This benchmark measures one GDN/KDA attention layer.  It reports an "
            "explicit fallback when torch.compile cannot produce an executable graph; "
            "it is not a production chunkwise FLA/FlashKDA benchmark."
        ),
        "environment": environment_metadata(device),
        "configuration": {
            "dim": args.dim,
            "heads": args.heads,
            "gate_rank": args.gate_rank,
            "conv_kernel_size": args.conv_kernel_size,
            "batch_size": args.batch_size,
            "train_batch_size": args.train_batch_size,
            "train_length": args.train_length,
            "prompt_lengths": args.prompt_lengths,
            "decode_tokens": args.decode_tokens,
            "dtype": args.dtype,
            "scan_backends": args.scan_backend,
            "compiled_max_prompt_length": args.compiled_max_prompt_length,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

"""End-to-end CUDA microbenchmark for all nine 4-layer token mixers.

The benchmark executes the actual GPT decoder: embeddings, positions, layer
norms, attention/recurrent mixers, MLPs, vocabulary loss, backward, and AdamW.
GDN/KDA layers use FLA's chunkwise CUDA kernels for full sequences and fused
recurrent kernels for one-token cached decode. It is a performance benchmark
with random token IDs, not a second quality-training run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import platform
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch

from .hybrid_experiment import VARIANTS
from .model import GPT, GPTConfig


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
DISPLAY_NAMES = {
    "MHA_4L": "MHA",
    "GQA_4L": "GQA",
    "MQA_4L": "MQA",
    "MLA_4L": "MLA-style",
    "GDN_4L": "GDN",
    "KDA_4L": "KDA",
    "GDN_GA_3TO1": "GDN + gated MHA",
    "KDA_MLA_3TO1": "KDA + MLA",
    "K3_KDA_GMLA_3TO1": "K3: KDA + gated MLA",
}
VERIFY_VARIANTS = (
    ("GDN", "GDN"),
    ("KDA", "KDA"),
    ("GDN + gated MHA", "GDN_HYBRID"),
    ("KDA + MLA", "KDA_HYBRID"),
    ("K3: KDA + gated MLA", "KDA_GATED_MLA_HYBRID"),
)


@dataclass(frozen=True)
class BenchmarkConfig:
    vocab_size: int = 65
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    gqa_n_kv_heads: int = 2
    mla_latent_rank: int = 32
    delta_conv_kernel: int = 4
    kda_gate_rank: int = 32
    training_batch_size: int = 16
    inference_batch_size: int = 8
    training_tokens: int = 64
    prefill_lengths: tuple[int, ...] = (64, 256, 1024, 4096)
    decode_prompt_tokens: int = 4096
    decode_tokens: int = 128
    warmup: int = 3
    repeats: int = 7
    verify_tokens: int = 64
    verify_batch_size: int = 2
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.vocab_size <= 1 or min(self.n_layer, self.n_head, self.n_embd) <= 0:
            raise ValueError("model dimensions and vocab_size must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.training_batch_size <= 0 or self.inference_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.training_tokens <= 0 or self.decode_tokens <= 0:
            raise ValueError("sequence lengths must be positive")
        if not self.prefill_lengths or min(self.prefill_lengths) <= 0:
            raise ValueError("prefill_lengths must contain positive values")
        if self.decode_prompt_tokens <= 0 or self.warmup < 0 or self.repeats <= 0:
            raise ValueError("prompt/repeats must be positive and warmup non-negative")

    @property
    def block_size(self) -> int:
        return max(
            self.training_tokens,
            max(self.prefill_lengths),
            self.decode_prompt_tokens + self.decode_tokens,
        )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize empty timing samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def select_dtype(name: str) -> torch.dtype:
    if name != "auto":
        return DTYPES[name]
    return (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )


def require_cuda_fla() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires a CUDA runtime (for example Colab T4)."
        )
    try:
        import fla
    except ImportError as error:
        raise RuntimeError(
            "flash-linear-attention[cuda] is not installed. Run the notebook "
            "installation cell, then restart the Colab runtime if requested."
        ) from error
    return getattr(fla, "__version__", "unknown")


def make_gpt_config(
    config: BenchmarkConfig, attention_variant: str, *, scan_backend: str
) -> GPTConfig:
    return GPTConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=0.0,
        attention_variant=attention_variant,
        gqa_n_kv_heads=config.gqa_n_kv_heads,
        mla_latent_rank=config.mla_latent_rank,
        delta_conv_kernel=config.delta_conv_kernel,
        kda_gate_rank=config.kda_gate_rank,
        delta_scan_backend=scan_backend,
    )


def build_model(
    config: BenchmarkConfig,
    attention_variant: str,
    *,
    scan_backend: str,
    device: torch.device,
) -> GPT:
    return GPT(
        make_gpt_config(config, attention_variant, scan_backend=scan_backend)
    ).to(device)


def model_parameter_count(model: GPT) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def recurrent_backend_label(model: GPT, *, decode: bool) -> str:
    """Describe the recurrent-kernel route without assigning FLA to MHA/MLA."""

    if not model.scan_backend_metadata():
        return "not applicable (SDPA/MLA only)"
    return "FLA fused recurrent" if decode else "FLA chunkwise"


def cuda_timed_samples(
    operation: Callable[[], None], *, warmup: int, repeats: int
) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return summarize(samples)


def verify_variant(
    label: str,
    attention_variant: str,
    config: BenchmarkConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Compare FLA and reference full-model logits before timing the variant."""

    torch.manual_seed(config.seed + sum(map(ord, attention_variant)))
    reference = build_model(
        config, attention_variant, scan_backend="reference", device=device
    ).to(dtype=dtype)
    fla = build_model(config, attention_variant, scan_backend="fla", device=device).to(
        dtype=dtype
    )
    fla.load_state_dict(reference.state_dict())
    reference.eval()
    fla.eval()
    tokens = torch.randint(
        config.vocab_size,
        (config.verify_batch_size, config.verify_tokens),
        device=device,
    )

    with torch.inference_mode():
        reference_logits, _ = reference(tokens)
        fla_logits, _ = fla(tokens)
        reference_caches = reference.new_caches(
            config.verify_batch_size, device=device, dtype=dtype
        )
        fla_caches = fla.new_caches(
            config.verify_batch_size, device=device, dtype=dtype
        )
        reference_cached = []
        fla_cached = []
        for index in range(config.verify_tokens):
            reference_step, _ = reference(
                tokens[:, index : index + 1], caches=reference_caches
            )
            fla_step, _ = fla(tokens[:, index : index + 1], caches=fla_caches)
            reference_cached.append(reference_step)
            fla_cached.append(fla_step)
        reference_cached_logits = torch.cat(reference_cached, dim=1)
        fla_cached_logits = torch.cat(fla_cached, dim=1)

    full_error = (fla_logits.float() - reference_logits.float()).abs().max().item()
    cached_error = (
        (fla_cached_logits.float() - reference_cached_logits.float()).abs().max().item()
    )
    torch.testing.assert_close(
        fla_logits.float(), reference_logits.float(), rtol=8e-2, atol=8e-2
    )
    torch.testing.assert_close(
        fla_cached_logits.float(),
        reference_cached_logits.float(),
        rtol=8e-2,
        atol=8e-2,
    )
    metadata = fla.scan_backend_metadata()
    del reference, fla
    torch.cuda.empty_cache()
    return {
        "variant": label,
        "model_variant": attention_variant,
        "verify_tokens": config.verify_tokens,
        "verify_batch": config.verify_batch_size,
        "dtype": str(dtype),
        "max_abs_full_logits_error": full_error,
        "max_abs_cached_logits_error": cached_error,
        "fla_backends": str(metadata),
        "status": "passed",
    }


def training_row(
    label: str,
    model_variant: str,
    model: GPT,
    config: BenchmarkConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model.train()
    inputs = torch.randint(
        config.vocab_size,
        (config.training_batch_size, config.training_tokens),
        device=device,
    )
    targets = torch.randint_like(inputs, high=config.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    def operation() -> None:
        optimizer.zero_grad(set_to_none=True)
        # The model itself is explicitly converted to FP16/BF16 in ``run``.
        # Do not wrap it in autocast: F.normalize in the delta projections can
        # otherwise promote Q/K/V to FP32, which FLA correctly rejects.
        _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

    timing = cuda_timed_samples(operation, warmup=config.warmup, repeats=config.repeats)
    model.eval()
    return {
        "variant": label,
        "model_variant": model_variant,
        "tokens": config.training_tokens,
        "batch": config.training_batch_size,
        "parameters": model_parameter_count(model),
        "scope": "full GPT forward + loss + backward + AdamW",
        "delta_scan_backend": recurrent_backend_label(model, decode=False),
        **timing,
        "tokens_per_second_p50": (
            1000
            * config.training_batch_size
            * config.training_tokens
            / timing["p50_ms"]
        ),
    }


@torch.inference_mode()
def prefill_once(model: GPT, tokens: torch.Tensor, dtype: torch.dtype) -> None:
    caches = model.new_caches(tokens.shape[0], device=tokens.device, dtype=dtype)
    model(tokens, caches=caches)


def prefill_rows(
    label: str,
    model_variant: str,
    model: GPT,
    config: BenchmarkConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    for tokens_count in config.prefill_lengths:
        tokens = torch.randint(
            config.vocab_size,
            (config.inference_batch_size, tokens_count),
            device=device,
        )
        timing = cuda_timed_samples(
            lambda current=tokens: prefill_once(model, current, dtype),
            warmup=config.warmup,
            repeats=config.repeats,
        )
        rows.append(
            {
                "variant": label,
                "model_variant": model_variant,
                "tokens": tokens_count,
                "batch": config.inference_batch_size,
                "parameters": model_parameter_count(model),
                "scope": "full GPT cached prefill",
                "delta_scan_backend": recurrent_backend_label(model, decode=False),
                **timing,
                "tokens_per_second_p50": (
                    1000 * config.inference_batch_size * tokens_count / timing["p50_ms"]
                ),
            }
        )
    return rows


@torch.inference_mode()
def prefill_caches(model: GPT, prefix: torch.Tensor, dtype: torch.dtype) -> list[Any]:
    caches = model.new_caches(prefix.shape[0], device=prefix.device, dtype=dtype)
    model(prefix, caches=caches)
    return caches


def decode_row(
    label: str,
    model_variant: str,
    model: GPT,
    config: BenchmarkConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    tokens = torch.randint(
        config.vocab_size,
        (
            config.inference_batch_size,
            config.decode_prompt_tokens + config.decode_tokens,
        ),
        device=device,
    )
    prefix = tokens[:, : config.decode_prompt_tokens]

    def timed_decode() -> float:
        caches = prefill_caches(model, prefix, dtype)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for index in range(config.decode_prompt_tokens, tokens.shape[1]):
                model(tokens[:, index : index + 1], caches=caches)
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for _ in range(config.warmup):
        timed_decode()
    torch.cuda.synchronize()
    samples = [timed_decode() / config.decode_tokens for _ in range(config.repeats)]
    timing = summarize(samples)
    final_caches = prefill_caches(model, prefix, dtype)
    with torch.inference_mode():
        for index in range(config.decode_prompt_tokens, tokens.shape[1]):
            model(tokens[:, index : index + 1], caches=final_caches)
    p50_ms_per_token = timing["p50_ms"]
    p95_ms_per_token = timing["p95_ms"]
    return {
        "variant": label,
        "model_variant": model_variant,
        "prompt_tokens": config.decode_prompt_tokens,
        "decode_tokens": config.decode_tokens,
        "batch": config.inference_batch_size,
        "parameters": model_parameter_count(model),
        "scope": "full GPT cached decode after prompt prefill",
        "delta_scan_backend": recurrent_backend_label(model, decode=True),
        "mean_ms_per_token": timing["mean_ms"],
        "min_ms_per_token": timing["min_ms"],
        "max_ms_per_token": timing["max_ms"],
        "p50_ms_per_token": p50_ms_per_token,
        "p95_ms_per_token": p95_ms_per_token,
        "cache_occupied_bytes": model.cache_occupied_bytes(final_caches),
        "tokens_per_second_p50": (
            1000 * config.inference_batch_size / p50_ms_per_token
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def environment_row(
    config: BenchmarkConfig, *, dtype: torch.dtype, fla_version: str
) -> dict[str, Any]:
    capability = torch.cuda.get_device_capability()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": ".".join(map(str, capability)),
        "torch": torch.__version__,
        "fla": fla_version,
        "dtype": str(dtype),
        "platform": platform.platform(),
        "benchmark": "full 4-layer GPT CUDA microbenchmark",
        "delta_backend": "FLA chunkwise prefill/training; FLA fused recurrent decode",
        "attention_backend": "PyTorch SDPA; native GQA when supported, explicit expansion fallback",
        "model_config": str(asdict(config)),
        "quality_boundary": "random-token performance microbenchmark; no retraining or PPL claim",
    }


def read_completed_verification(path: Path) -> list[dict[str, str]]:
    """Load a prior successful verification result for a resumed benchmark."""

    if not path.is_file():
        raise FileNotFoundError(
            f"--skip-verification requires the existing verification file: {path}"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_variants = {label for label, _ in VERIFY_VARIANTS}
    completed_variants = {
        row.get("variant") for row in rows if row.get("status") == "passed"
    }
    if completed_variants != expected_variants:
        raise ValueError(
            "--skip-verification requires successful rows for exactly these variants: "
            f"{sorted(expected_variants)}"
        )
    return rows


def run(
    config: BenchmarkConfig,
    output_dir: Path,
    dtype_name: str,
    *,
    skip_verification: bool = False,
) -> None:
    fla_version = require_cuda_fla()
    device = torch.device("cuda")
    dtype = select_dtype(dtype_name)
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    output_dir.mkdir(parents=True, exist_ok=True)

    verification_path = output_dir / "cuda_full_model_verification.csv"
    if skip_verification:
        read_completed_verification(verification_path)
        print(
            f"Using existing successful FLA verification: {verification_path}",
            flush=True,
        )
    else:
        verification = []
        for label, model_variant in VERIFY_VARIANTS:
            print(f"Verifying FLA equivalence: {label}...", flush=True)
            verification.append(
                verify_variant(label, model_variant, config, device=device, dtype=dtype)
            )
        write_csv(verification_path, verification)

    training, prefill, decode = [], [], []
    for result_name, model_variant in VARIANTS:
        label = DISPLAY_NAMES[result_name]
        print(f"Benchmarking {label} ({model_variant})...", flush=True)
        torch.manual_seed(config.seed)
        # FLA kernels require FP16/BF16 Q/K/V tensors. Keeping every benchmarked
        # model in the selected CUDA precision also gives MHA/GQA/MQA/MLA the
        # same numerical execution regime as the recurrent variants.
        model = build_model(
            config, model_variant, scan_backend="fla", device=device
        ).to(dtype=dtype)
        training.append(
            training_row(
                label, model_variant, model, config, device=device, dtype=dtype
            )
        )
        prefill.extend(
            prefill_rows(
                label, model_variant, model, config, device=device, dtype=dtype
            )
        )
        decode.append(
            decode_row(label, model_variant, model, config, device=device, dtype=dtype)
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    write_csv(output_dir / "cuda_full_model_training.csv", training)
    write_csv(output_dir / "cuda_full_model_prefill.csv", prefill)
    write_csv(output_dir / "cuda_full_model_decode.csv", decode)
    write_csv(
        output_dir / "cuda_full_model_environment.csv",
        [environment_row(config, dtype=dtype, fla_version=fla_version)],
    )
    print(f"Wrote CUDA full-model CSVs to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--dtype", choices=("auto", *DTYPES), default="auto")
    parser.add_argument("--vocab-size", type=int, default=65)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--train-tokens", type=int, default=64)
    parser.add_argument(
        "--prefill-lengths", type=int, nargs="+", default=(64, 256, 1024, 4096)
    )
    parser.add_argument("--decode-prompt-tokens", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--verify-tokens", type=int, default=64)
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="reuse the already-written successful FLA verification CSV",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        vocab_size=args.vocab_size,
        training_batch_size=args.train_batch_size,
        inference_batch_size=args.inference_batch_size,
        training_tokens=args.train_tokens,
        prefill_lengths=tuple(args.prefill_lengths),
        decode_prompt_tokens=args.decode_prompt_tokens,
        decode_tokens=args.decode_tokens,
        warmup=args.warmup,
        repeats=args.repeats,
        verify_tokens=args.verify_tokens,
        seed=args.seed,
    )
    run(
        config,
        args.output_dir,
        args.dtype,
        skip_verification=args.skip_verification,
    )


if __name__ == "__main__":
    main()

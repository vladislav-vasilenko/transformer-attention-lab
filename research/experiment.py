"""Controlled training and inference experiment for MHA, GQA, MQA, and MLA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch

from .benchmark import percentile, select_device, synchronize
from .model import GPT, GPTConfig


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    split_fraction: float = 0.9
    steps: int = 250
    eval_interval: int = 50
    eval_batches: int = 12
    batch_size: int = 16
    train_context: int = 64
    block_size: int = 192
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    gqa_n_kv_heads: int = 2
    mla_latent_rank: int = 32
    delta_conv_kernel: int = 4
    kda_gate_rank: int = 32
    delta_scan_backend: str = "reference"
    dropout: float = 0.0
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    inference_prompt_lengths: tuple[int, ...] = (32, 64, 128)
    inference_decode_tokens: int = 32
    inference_warmup: int = 1
    inference_repeats: int = 5
    num_threads: int = 1
    training_mode: str = "random_steps"
    epochs: int = 1

    def __post_init__(self) -> None:
        if self.train_context >= self.block_size:
            raise ValueError("train_context must be smaller than block_size")
        maximum = max(self.inference_prompt_lengths) + self.inference_decode_tokens
        if maximum > self.block_size:
            raise ValueError("inference prompt plus decode tokens exceeds block_size")
        if self.steps <= 0 or self.batch_size <= 0 or self.eval_batches <= 0:
            raise ValueError("steps, batch_size, and eval_batches must be positive")
        if self.training_mode not in {"random_steps", "full_epoch"}:
            raise ValueError("training_mode must be random_steps or full_epoch")
        if self.delta_scan_backend not in {"reference", "compiled", "fla"}:
            raise ValueError(
                "delta_scan_backend must be 'reference', 'compiled', or 'fla'"
            )
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")


class CharacterCorpus:
    """Deterministic character-level train/validation corpus."""

    def __init__(self, path: Path, split_fraction: float) -> None:
        self.path = path
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        self.characters = sorted(set(text))
        self.stoi = {
            character: index for index, character in enumerate(self.characters)
        }
        encoded = torch.tensor(
            [self.stoi[character] for character in text], dtype=torch.long
        )
        split = int(len(encoded) * split_fraction)
        self.train = encoded[:split]
        self.validation = encoded[split:]

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        context: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.train if split == "train" else self.validation
        starts = torch.randint(
            0,
            len(source) - context - 1,
            (batch_size,),
            generator=generator,
        )
        inputs = torch.stack([source[start : start + context] for start in starts])
        targets = torch.stack(
            [source[start + 1 : start + context + 1] for start in starts]
        )
        return inputs.to(device), targets.to(device)

    def train_windows_per_epoch(self, context: int) -> int:
        """Number of non-overlapping train windows that fit completely."""

        return (len(self.train) - 1) // context

    def iter_train_epoch(
        self,
        batch_size: int,
        context: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield every non-overlapping train window once in shuffled order."""

        window_count = self.train_windows_per_epoch(context)
        starts = torch.arange(window_count) * context
        starts = starts[torch.randperm(window_count, generator=generator)]
        for batch_start in range(0, window_count, batch_size):
            batch_indices = starts[batch_start : batch_start + batch_size]
            inputs = torch.stack(
                [self.train[start : start + context] for start in batch_indices]
            )
            targets = torch.stack(
                [self.train[start + 1 : start + context + 1] for start in batch_indices]
            )
            yield inputs.to(device), targets.to(device)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join(self.characters[index] for index in tokens.tolist())


def default_data_path() -> Path:
    project = Path(__file__).resolve().parents[1]
    return project.parent / "nanoGPT-lab" / "input.txt"


@torch.inference_mode()
def evaluate(
    model: GPT,
    corpus: CharacterCorpus,
    config: ExperimentConfig,
    device: torch.device,
    step: int,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for split_index, split in enumerate(("train", "validation")):
        generator = torch.Generator().manual_seed(
            config.seed + 10_000 + step * 17 + split_index
        )
        values = []
        for _ in range(config.eval_batches):
            inputs, targets = corpus.get_batch(
                split,
                config.batch_size,
                config.train_context,
                generator,
                device,
            )
            _, loss = model(inputs, targets)
            values.append(float(loss))
        losses[split] = statistics.fmean(values)
    model.train()
    return losses


def elapsed_ms(device: torch.device, operation: Callable[[], Any]) -> float:
    synchronize(device)
    started = time.perf_counter_ns()
    operation()
    synchronize(device)
    return (time.perf_counter_ns() - started) / 1_000_000


@torch.inference_mode()
def cached_model_trial(
    model: GPT,
    tokens: torch.Tensor,
    prompt_length: int,
    decode_tokens: int,
    device: torch.device,
) -> tuple[float, float, int, int]:
    caches = model.new_caches(
        tokens.shape[0],
        device=device,
        dtype=model.token_embedding.weight.dtype,
    )
    prefill_ms = elapsed_ms(
        device, lambda: model(tokens[:, :prompt_length], caches=caches)
    )

    def decode() -> None:
        for offset in range(decode_tokens):
            index = prompt_length + offset
            model(tokens[:, index : index + 1], caches=caches)

    decode_ms = elapsed_ms(device, decode) / decode_tokens
    return (
        prefill_ms,
        decode_ms,
        model.cache_occupied_bytes(caches),
        model.cache_allocated_bytes(caches),
    )


@torch.inference_mode()
def recompute_model_trial(
    model: GPT,
    tokens: torch.Tensor,
    prompt_length: int,
    decode_tokens: int,
    device: torch.device,
) -> float:
    def decode() -> None:
        for offset in range(decode_tokens):
            prefix_end = prompt_length + offset + 1
            model(tokens[:, :prefix_end])

    return elapsed_ms(device, decode) / decode_tokens


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def benchmark_model(
    model: GPT,
    corpus: CharacterCorpus,
    config: ExperimentConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []
    for prompt_length in config.inference_prompt_lengths:
        total = prompt_length + config.inference_decode_tokens
        tokens = corpus.validation[:total].unsqueeze(0).to(device)
        for _ in range(config.inference_warmup):
            cached_model_trial(
                model, tokens, prompt_length, config.inference_decode_tokens, device
            )
            recompute_model_trial(
                model, tokens, prompt_length, config.inference_decode_tokens, device
            )

        prefill: list[float] = []
        cached: list[float] = []
        recompute: list[float] = []
        occupied_bytes = 0
        allocated_bytes = 0
        for repeat in range(config.inference_repeats):
            if repeat % 2:
                recompute.append(
                    recompute_model_trial(
                        model,
                        tokens,
                        prompt_length,
                        config.inference_decode_tokens,
                        device,
                    )
                )
            cached_trial = cached_model_trial(
                model,
                tokens,
                prompt_length,
                config.inference_decode_tokens,
                device,
            )
            prefill.append(cached_trial[0])
            cached.append(cached_trial[1])
            occupied_bytes, allocated_bytes = cached_trial[2], cached_trial[3]
            if not repeat % 2:
                recompute.append(
                    recompute_model_trial(
                        model,
                        tokens,
                        prompt_length,
                        config.inference_decode_tokens,
                        device,
                    )
                )

        prefill_summary = summarize(prefill)
        cached_summary = summarize(cached)
        recompute_summary = summarize(recompute)
        cached_total = (
            prefill_summary["p50"]
            + cached_summary["p50"] * config.inference_decode_tokens
        )
        recompute_total = recompute_summary["p50"] * config.inference_decode_tokens
        results.append(
            {
                "prompt_length": prompt_length,
                "decode_tokens": config.inference_decode_tokens,
                "total_sequence_length": total,
                "prefill_ms": prefill_summary,
                "cached_decode_ms_per_token": cached_summary,
                "recompute_decode_ms_per_token": recompute_summary,
                "decode_tokens_per_second_p50": 1000 / cached_summary["p50"],
                "decode_speedup_p50": recompute_summary["p50"] / cached_summary["p50"],
                "end_to_end_speedup_p50": recompute_total / cached_total,
                "cache_occupied_bytes": occupied_bytes,
                "cache_allocated_bytes": allocated_bytes,
                "raw_timing_ms": {
                    "prefill": prefill,
                    "cached_decode_per_token": cached,
                    "recompute_decode_per_token": recompute,
                },
            }
        )
    return results


def train_variant(
    variant: str,
    corpus: CharacterCorpus,
    config: ExperimentConfig,
    device: torch.device,
    *,
    model_variant: str | None = None,
    shared_backbone_state: dict[str, torch.Tensor] | None = None,
) -> tuple[GPT, dict[str, Any]]:
    torch.manual_seed(config.seed)
    model_variant = variant if model_variant is None else model_variant
    model_config = GPTConfig(
        vocab_size=corpus.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        attention_variant=model_variant,
        gqa_n_kv_heads=config.gqa_n_kv_heads,
        mla_latent_rank=config.mla_latent_rank,
        delta_conv_kernel=config.delta_conv_kernel,
        kda_gate_rank=config.kda_gate_rank,
        delta_scan_backend=config.delta_scan_backend,
    )
    model = GPT(model_config).to(device)
    if shared_backbone_state is not None:
        state = model.state_dict()
        for name, value in shared_backbone_state.items():
            if name in state and state[name].shape == value.shape:
                state[name].copy_(value.to(device=state[name].device))
        model.load_state_dict(state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batch_generator = torch.Generator().manual_seed(config.seed + 1)
    history: list[dict[str, float | int]] = []
    step_times: list[float] = []
    step_token_counts: list[int] = []

    def record_evaluation(step: int, tokens_seen: int) -> None:
        losses = evaluate(model, corpus, config, device, step)
        history.append(
            {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": losses["train"],
                "validation_loss": losses["validation"],
                "validation_perplexity": math.exp(min(losses["validation"], 20)),
            }
        )

    if config.training_mode == "full_epoch":
        batches_per_epoch = math.ceil(
            corpus.train_windows_per_epoch(config.train_context) / config.batch_size
        )
        total_steps = config.epochs * batches_per_epoch

        def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for _ in range(config.epochs):
                yield from corpus.iter_train_epoch(
                    config.batch_size,
                    config.train_context,
                    batch_generator,
                    device,
                )

    else:
        total_steps = config.steps

        def batches() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for _ in range(config.steps):
                yield corpus.get_batch(
                    "train",
                    config.batch_size,
                    config.train_context,
                    batch_generator,
                    device,
                )

    tokens_seen = 0
    record_evaluation(0, tokens_seen)
    model.train()
    for step, (inputs, targets) in enumerate(batches(), start=1):
        synchronize(device)
        started = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip
        )
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise RuntimeError(
                f"non-finite training signal for {variant} at step {step}"
            )
        optimizer.step()
        synchronize(device)
        step_times.append((time.perf_counter_ns() - started) / 1_000_000)
        step_token_counts.append(inputs.numel())
        tokens_seen += inputs.numel()
        if step % config.eval_interval == 0 or step == total_steps:
            record_evaluation(step, tokens_seen)

    warmup_steps = min(5, len(step_times))
    timed_steps = step_times[warmup_steps:]
    timed_token_counts = step_token_counts[warmup_steps:]
    if not timed_steps:
        timed_steps = step_times
        timed_token_counts = step_token_counts
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    attention_parameter_count = sum(
        parameter.numel()
        for block in model.blocks
        for parameter in block.attention.parameters()
    )
    result = {
        "variant": variant,
        "model_variant": model_variant,
        "model_config": asdict(model_config),
        "parameter_count": parameter_count,
        "attention_parameter_count": attention_parameter_count,
        "delta_scan_backends": model.scan_backend_metadata(),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
        "training_mode": config.training_mode,
        "training_epochs": config.epochs
        if config.training_mode == "full_epoch"
        else None,
        "training_steps": total_steps,
        "training_tokens": tokens_seen,
        "training_ms_per_step": summarize(timed_steps),
        "training_tokens_per_second": 1000 * sum(timed_token_counts) / sum(timed_steps),
        "training_history": history,
        "final_validation_loss": history[-1]["validation_loss"],
        "final_validation_perplexity": history[-1]["validation_perplexity"],
    }
    return model, result


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


def run_experiment(
    data_path: Path,
    output_path: Path,
    config: ExperimentConfig,
    device_name: str = "auto",
) -> dict[str, Any]:
    torch.set_num_threads(config.num_threads)
    device = select_device(device_name)
    corpus = CharacterCorpus(data_path, config.split_fraction)
    variants = []
    for variant_name in ("MHA", "GQA", "MQA", "MLA"):
        print(f"Training {variant_name}...", flush=True)
        model, result = train_variant(variant_name, corpus, config, device)
        result["inference"] = benchmark_model(model, corpus, config, device)
        variants.append(result)
        longest = result["inference"][-1]
        print(
            f"{variant_name}: val ppl={result['final_validation_perplexity']:.2f}, "
            f"cached={longest['cached_decode_ms_per_token']['p50']:.3f} ms/token, "
            f"cache={longest['cache_occupied_bytes'] / 1024:.1f} KiB",
            flush=True,
        )
        del model

    output = {
        "schema_version": 2,
        "title": "Efficient Transformer Systems: MHA vs GQA vs MQA vs MLA",
        "status": "completed controlled prototype experiment",
        "question": (
            "Under a fixed nanoGPT/GPT-2 training budget, how do MHA, GQA, MQA, "
            "and MLA-style latent attention trade validation quality for training "
            "and autoregressive inference efficiency?"
        ),
        "hypotheses": [
            "GQA and MQA reduce cache memory in proportion to the number of KV heads; more aggressive sharing may trade away validation quality.",
            "MLA's latent cache yields the smallest cache, but without a fused kernel or weight absorption it may not have the lowest latency.",
            "KV caching produces larger decode speedups as the prompt grows because full recomputation repeatedly projects the prefix.",
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
        "variants": variants,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument(
        "--output", type=Path, default=Path("results/attention_systems.json")
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
        choices=("random_steps", "full_epoch"),
        default="random_steps",
    )
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        steps=args.steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        batch_size=args.batch_size,
        train_context=args.train_context,
        delta_scan_backend=args.delta_scan_backend,
        num_threads=args.num_threads,
        inference_repeats=args.inference_repeats,
        training_mode=args.training_mode,
        epochs=args.epochs,
    )
    run_experiment(args.data, args.output, config, args.device)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

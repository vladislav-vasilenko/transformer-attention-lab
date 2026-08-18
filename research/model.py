"""A compact GPT-2/nanoGPT decoder with pluggable attention mechanisms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import (
    AttentionConfig,
    CausalSelfAttention,
    KVCache,
    MLAConfig,
    MLACache,
    MultiLatentAttention,
)
from .linear_attention import (
    DeltaNetAttention,
    DeltaNetConfig,
    DeltaStateCache,
    GatedDeltaNet,
    KimiDeltaAttention,
)

AttentionCache = KVCache | MLACache | DeltaStateCache


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int = 192
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    attention_variant: str = "MHA"
    gqa_n_kv_heads: int = 2
    mla_latent_rank: int = 32
    delta_conv_kernel: int = 4
    kda_gate_rank: int = 32
    delta_scan_backend: str = "reference"

    def __post_init__(self) -> None:
        variant = self.attention_variant.upper()
        object.__setattr__(self, "attention_variant", variant)
        valid_variants = {
            "MHA",
            "GQA",
            "MQA",
            "MLA",
            "GDN",
            "KDA",
            "GDN_HYBRID",
            "KDA_HYBRID",
            "KDA_GATED_MLA_HYBRID",
        }
        if variant not in valid_variants:
            raise ValueError(
                "attention_variant must be MHA, GQA, MQA, MLA, GDN, KDA, "
                "GDN_HYBRID, KDA_HYBRID, or KDA_GATED_MLA_HYBRID"
            )
        if self.vocab_size <= 0 or self.block_size <= 0 or self.n_layer <= 0:
            raise ValueError("vocab_size, block_size, and n_layer must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.delta_conv_kernel <= 0 or self.kda_gate_rank <= 0:
            raise ValueError("delta_conv_kernel and kda_gate_rank must be positive")
        if self.delta_scan_backend not in {"reference", "compiled", "fla"}:
            raise ValueError(
                "delta_scan_backend must be 'reference', 'compiled', or 'fla'"
            )
        if variant.endswith("_HYBRID") and self.n_layer % 4:
            raise ValueError("3:1 hybrid variants require n_layer divisible by 4")


def build_attention(
    config: GPTConfig, layer_index: int = 0
) -> CausalSelfAttention | MultiLatentAttention | GatedDeltaNet | KimiDeltaAttention:
    if config.attention_variant in {"GDN", "KDA"}:
        delta_config = DeltaNetConfig(
            dim=config.n_embd,
            n_heads=config.n_head,
            max_seq_len=config.block_size,
            conv_kernel_size=config.delta_conv_kernel,
            gate_rank=config.kda_gate_rank,
            channelwise_decay=config.attention_variant == "KDA",
            bias=config.bias,
            scan_backend=config.delta_scan_backend,
        )
        if config.attention_variant == "KDA":
            return KimiDeltaAttention(delta_config)
        return GatedDeltaNet(delta_config)
    if config.attention_variant in {
        "GDN_HYBRID",
        "KDA_HYBRID",
        "KDA_GATED_MLA_HYBRID",
    }:
        full_anchor = (layer_index + 1) % 4 == 0
        if full_anchor and config.attention_variant == "GDN_HYBRID":
            return CausalSelfAttention(
                AttentionConfig(
                    dim=config.n_embd,
                    n_heads=config.n_head,
                    n_kv_heads=config.n_head,
                    max_seq_len=config.block_size,
                    bias=config.bias,
                    output_gate=True,
                )
            )
        if full_anchor:
            return MultiLatentAttention(
                MLAConfig(
                    dim=config.n_embd,
                    n_heads=config.n_head,
                    latent_rank=config.mla_latent_rank,
                    max_seq_len=config.block_size,
                    bias=config.bias,
                    output_gate=(config.attention_variant == "KDA_GATED_MLA_HYBRID"),
                )
            )
        delta_config = DeltaNetConfig(
            dim=config.n_embd,
            n_heads=config.n_head,
            max_seq_len=config.block_size,
            conv_kernel_size=config.delta_conv_kernel,
            gate_rank=config.kda_gate_rank,
            channelwise_decay=config.attention_variant
            in {"KDA_HYBRID", "KDA_GATED_MLA_HYBRID"},
            bias=config.bias,
            scan_backend=config.delta_scan_backend,
        )
        if config.attention_variant in {
            "KDA_HYBRID",
            "KDA_GATED_MLA_HYBRID",
        }:
            return KimiDeltaAttention(delta_config)
        return GatedDeltaNet(delta_config)
    if config.attention_variant == "MLA":
        return MultiLatentAttention(
            MLAConfig(
                dim=config.n_embd,
                n_heads=config.n_head,
                latent_rank=config.mla_latent_rank,
                max_seq_len=config.block_size,
                bias=config.bias,
            )
        )
    if config.attention_variant == "MHA":
        n_kv_heads = config.n_head
    elif config.attention_variant == "MQA":
        n_kv_heads = 1
    else:
        n_kv_heads = config.gqa_n_kv_heads
    return CausalSelfAttention(
        AttentionConfig(
            dim=config.n_embd,
            n_heads=config.n_head,
            n_kv_heads=n_kv_heads,
            max_seq_len=config.block_size,
            bias=config.bias,
        )
    )


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: GPTConfig, layer_index: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention = build_attention(config, layer_index)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: Tensor, cache: AttentionCache | None = None) -> Tensor:
        x = x + self.attention_dropout(self.attention(self.ln_1(x), cache))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """GPT-2-style decoder used for the controlled attention experiment."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(config, layer_index) for layer_index in range(config.n_layer)
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _position_offset(self, caches: list[AttentionCache] | None) -> int:
        if caches is None:
            return 0
        if len(caches) != len(self.blocks):
            raise ValueError("one cache is required per decoder block")
        lengths = {cache.length for cache in caches}
        if len(lengths) != 1:
            raise ValueError("all layer caches must have the same logical length")
        return lengths.pop()

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        caches: list[AttentionCache] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (B, T)")
        batch, sequence_length = tokens.shape
        del batch
        position_offset = self._position_offset(caches)
        if position_offset + sequence_length > self.config.block_size:
            raise ValueError("sequence exceeds the configured block_size")

        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=tokens.device,
        )
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        for index, block in enumerate(self.blocks):
            cache = None if caches is None else caches[index]
            hidden = block(hidden, cache)
        logits = self.lm_head(self.final_norm(hidden))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return logits, loss

    def new_caches(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> list[AttentionCache]:
        return [
            block.attention.new_cache(
                batch_size,
                device=device,
                dtype=dtype,
            )
            for block in self.blocks
        ]

    def scan_backend_metadata(self) -> list[dict[str, object]]:
        """Return actual recurrent scan backends after optional compilation."""

        records = []
        for layer_index, block in enumerate(self.blocks, start=1):
            if isinstance(block.attention, DeltaNetAttention):
                records.append(
                    {"layer": layer_index, **block.attention.scan_backend_metadata()}
                )
        return records

    @staticmethod
    def cache_occupied_bytes(caches: list[AttentionCache]) -> int:
        return sum(cache.occupied_bytes for cache in caches)

    @staticmethod
    def cache_allocated_bytes(caches: list[AttentionCache]) -> int:
        return sum(cache.allocated_bytes for cache in caches)

    @torch.inference_mode()
    def generate(
        self,
        tokens: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if tokens.shape[1] + max_new_tokens > self.config.block_size:
            raise ValueError("prompt plus generated tokens exceeds block_size")
        caches = self.new_caches(
            tokens.shape[0],
            device=tokens.device,
            dtype=self.token_embedding.weight.dtype,
        )
        logits, _ = self(tokens, caches=caches)
        generated = tokens
        for index in range(max_new_tokens):
            next_logits = logits[:, -1] / temperature
            if top_k is not None:
                threshold = torch.topk(
                    next_logits, min(top_k, next_logits.shape[-1])
                ).values[:, -1:]
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float("-inf")
                )
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            )
            generated = torch.cat((generated, next_token), dim=1)
            if index + 1 < max_new_tokens:
                logits, _ = self(next_token, caches=caches)
        return generated

"""Attention variants with an explicit, preallocated KV cache.

The implementation is intentionally compact and framework-native.  It keeps
only the non-repeated K/V heads in the cache, so the memory difference between
MHA, GQA, and MQA is observable rather than merely analytical.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AttentionConfig:
    """Shape configuration shared by MHA, GQA, and MQA."""

    dim: int = 384
    n_heads: int = 6
    n_kv_heads: int = 6
    max_seq_len: int = 576
    bias: bool = False
    output_gate: bool = False

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("dim, n_heads, and n_kv_heads must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def queries_per_kv(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def variant(self) -> str:
        if self.n_kv_heads == self.n_heads:
            return "MHA"
        if self.n_kv_heads == 1:
            return "MQA"
        return "GQA"


class KVCache:
    """A fixed-capacity cache that stores K/V without repeated query groups."""

    def __init__(
        self,
        config: AttentionConfig,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (
            batch_size,
            config.n_kv_heads,
            config.max_seq_len,
            config.head_dim,
        )
        self.key = torch.empty(shape, device=device, dtype=dtype)
        self.value = torch.empty_like(self.key)
        self.length = 0

    @property
    def capacity(self) -> int:
        return self.key.shape[2]

    @property
    def allocated_bytes(self) -> int:
        return 2 * self.key.numel() * self.key.element_size()

    @property
    def fixed_bytes(self) -> int:
        return 0

    @property
    def bytes_per_token(self) -> int:
        elements_per_token = self.key.shape[0] * self.key.shape[1] * self.key.shape[3]
        return 2 * elements_per_token * self.key.element_size()

    @property
    def occupied_bytes(self) -> int:
        return self.length * self.bytes_per_token

    def reset(self) -> None:
        """Reset the logical length without reallocating the buffers."""

        self.length = 0

    def append(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Append a contiguous chunk and return views of the populated prefix."""

        if key.shape != value.shape:
            raise ValueError("key and value must have identical shapes")
        if key.ndim != 4:
            raise ValueError("expected cache tensors with shape (B, H_kv, T, D)")
        expected = (self.key.shape[0], self.key.shape[1], self.key.shape[3])
        received = (key.shape[0], key.shape[1], key.shape[3])
        if received != expected:
            raise ValueError(
                f"cache shape mismatch: expected {expected}, got {received}"
            )

        end = self.length + key.shape[2]
        if end > self.capacity:
            raise ValueError(
                f"KV cache overflow: requested {end}, capacity {self.capacity}"
            )
        self.key[:, :, self.length : end].copy_(key)
        self.value[:, :, self.length : end].copy_(value)
        self.length = end
        return self.key[:, :, :end], self.value[:, :, :end]


class CausalSelfAttention(nn.Module):
    """Causal attention supporting MHA, GQA, MQA, and incremental decoding."""

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self._native_gqa_available: bool | None = None
        kv_dim = config.n_kv_heads * config.head_dim
        self.q_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.k_proj = nn.Linear(config.dim, kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.dim, kv_dim, bias=config.bias)
        self.output_gate_proj = (
            nn.Linear(config.dim, config.dim, bias=config.bias)
            if config.output_gate
            else None
        )
        self.out_proj = nn.Linear(config.dim, config.dim, bias=config.bias)

    def _queries(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return (
            self.q_proj(x)
            .view(batch, tokens, self.config.n_heads, self.config.head_dim)
            .transpose(1, 2)
        )

    def _keys_and_values(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, tokens, _ = x.shape
        shape = (batch, tokens, self.config.n_kv_heads, self.config.head_dim)
        key = self.k_proj(x).view(shape).transpose(1, 2)
        value = self.v_proj(x).view(shape).transpose(1, 2)
        return key, value

    def _expand_kv(self, tensor: Tensor) -> Tensor:
        if self.config.queries_per_kv == 1:
            return tensor
        # Expansion is performed only for the attention kernel.  The cache keeps
        # the compact H_kv representation and therefore realizes the memory gain.
        return tensor.repeat_interleave(self.config.queries_per_kv, dim=1)

    def _scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        attention_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        """Prefer PyTorch's native GQA route when the active backend supports it.

        The compact cache remains in ``H_kv`` form. Older CPU/CUDA SDPA builds
        may not implement ``enable_gqa`` for their selected kernel, in which
        case expansion is an explicit compatibility fallback rather than a
        hidden change in cache accounting.
        """

        if self.config.queries_per_kv == 1:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        if self._native_gqa_available is not False:
            try:
                output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                    enable_gqa=True,
                )
            except (RuntimeError, TypeError):
                self._native_gqa_available = False
            else:
                self._native_gqa_available = True
                return output
        return F.scaled_dot_product_attention(
            query,
            self._expand_kv(key),
            self._expand_kv(value),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

    @staticmethod
    def _offset_causal_mask(
        query_length: int,
        key_length: int,
        prefix_length: int,
        device: torch.device,
    ) -> Tensor:
        query_positions = torch.arange(
            prefix_length, prefix_length + query_length, device=device
        )
        key_positions = torch.arange(key_length, device=device)
        return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

    def forward(self, x: Tensor, cache: KVCache | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.dim:
            raise ValueError(f"expected x with shape (B, T, {self.config.dim})")
        if x.shape[1] == 0:
            raise ValueError("the token dimension must be non-empty")

        query = self._queries(x)
        key, value = self._keys_and_values(x)
        prefix_length = 0 if cache is None else cache.length

        if cache is not None:
            if cache.key.shape[0] != x.shape[0]:
                raise ValueError("cache batch size does not match x")
            key, value = cache.append(key, value)

        query_length = query.shape[2]
        key_length = key.shape[2]

        attention_mask: Tensor | None = None
        is_causal = cache is None or prefix_length == 0
        if cache is not None and prefix_length > 0:
            if query_length == 1:
                # A single newly appended token may attend to the entire prefix.
                is_causal = False
            else:
                attention_mask = self._offset_causal_mask(
                    query_length, key_length, prefix_length, query.device
                )
                is_causal = False

        output = self._scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask=attention_mask,
            is_causal=is_causal,
        )
        output = output.transpose(1, 2).contiguous().view(x.shape)
        if self.output_gate_proj is not None:
            output = output * self.output_gate_proj(x).sigmoid()
        return self.out_proj(output)

    def new_cache(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> KVCache:
        return KVCache(
            self.config,
            batch_size,
            device=device,
            dtype=dtype,
        )


@dataclass(frozen=True)
class MLAConfig:
    """Configuration for MLA-style latent K/V compression.

    The surrounding GPT-2 model supplies learned positional embeddings.  This
    controlled variant therefore isolates latent-cache compression and does not
    add DeepSeek's decoupled RoPE path.
    """

    dim: int = 384
    n_heads: int = 6
    latent_rank: int = 64
    max_seq_len: int = 576
    bias: bool = False
    output_gate: bool = False

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.n_heads <= 0 or self.latent_rank <= 0:
            raise ValueError("dim, n_heads, and latent_rank must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def variant(self) -> str:
        return "GATED_MLA" if self.output_gate else "MLA"


class MLACache:
    """Preallocated cache storing one shared latent vector per token."""

    def __init__(
        self,
        config: MLAConfig,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.latent = torch.empty(
            batch_size,
            config.max_seq_len,
            config.latent_rank,
            device=device,
            dtype=dtype,
        )
        self.length = 0

    @property
    def capacity(self) -> int:
        return self.latent.shape[1]

    @property
    def allocated_bytes(self) -> int:
        return self.latent.numel() * self.latent.element_size()

    @property
    def fixed_bytes(self) -> int:
        return 0

    @property
    def bytes_per_token(self) -> int:
        elements_per_token = self.latent.shape[0] * self.latent.shape[2]
        return elements_per_token * self.latent.element_size()

    @property
    def occupied_bytes(self) -> int:
        return self.length * self.bytes_per_token

    def reset(self) -> None:
        self.length = 0

    def append(self, latent: Tensor) -> Tensor:
        if latent.ndim != 3:
            raise ValueError("expected latent cache tensor with shape (B, T, R)")
        expected = (self.latent.shape[0], self.latent.shape[2])
        received = (latent.shape[0], latent.shape[2])
        if received != expected:
            raise ValueError(
                f"cache shape mismatch: expected {expected}, got {received}"
            )
        end = self.length + latent.shape[1]
        if end > self.capacity:
            raise ValueError(
                f"MLA cache overflow: requested {end}, capacity {self.capacity}"
            )
        self.latent[:, self.length : end].copy_(latent)
        self.length = end
        return self.latent[:, :end]


class MultiLatentAttention(nn.Module):
    """MLA-style attention that reconstructs K/V from a cached low-rank latent.

    This implementation makes cache storage measurable and keeps the math easy
    to inspect.  It intentionally does not implement deployment-only weight
    absorption or a fused MLA kernel, so lower cache memory may coexist with
    higher decode latency on general PyTorch kernels.  ``output_gate=True``
    adds a learned sigmoid gate to the reconstructed attention output without
    changing the latent-cache storage law; this is used only for the explicit
    Kimi-K3-inspired schedule, not claimed as a faithful K3 reproduction.
    """

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.kv_down_proj = nn.Linear(config.dim, config.latent_rank, bias=config.bias)
        self.kv_norm = nn.LayerNorm(config.latent_rank)
        self.k_up_proj = nn.Linear(config.latent_rank, config.dim, bias=config.bias)
        self.v_up_proj = nn.Linear(config.latent_rank, config.dim, bias=config.bias)
        # Keep all ungated MLA parameters in the same construction order so a
        # seeded gated/ungated ablation starts from identical shared weights.
        self.out_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.output_gate_proj = (
            nn.Linear(config.dim, config.dim, bias=config.bias)
            if config.output_gate
            else None
        )

    def _queries(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return (
            self.q_proj(x)
            .view(batch, tokens, self.config.n_heads, self.config.head_dim)
            .transpose(1, 2)
        )

    def _reconstruct(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        batch, tokens, _ = latent.shape
        shape = (batch, tokens, self.config.n_heads, self.config.head_dim)
        key = self.k_up_proj(latent).view(shape).transpose(1, 2)
        value = self.v_up_proj(latent).view(shape).transpose(1, 2)
        return key, value

    def forward(self, x: Tensor, cache: MLACache | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.dim:
            raise ValueError(f"expected x with shape (B, T, {self.config.dim})")
        if x.shape[1] == 0:
            raise ValueError("the token dimension must be non-empty")

        query = self._queries(x)
        latent = self.kv_norm(self.kv_down_proj(x))
        prefix_length = 0 if cache is None else cache.length
        if cache is not None:
            if cache.latent.shape[0] != x.shape[0]:
                raise ValueError("cache batch size does not match x")
            latent = cache.append(latent)
        key, value = self._reconstruct(latent)

        query_length = query.shape[2]
        key_length = key.shape[2]
        attention_mask: Tensor | None = None
        is_causal = cache is None or prefix_length == 0
        if cache is not None and prefix_length > 0:
            if query_length == 1:
                is_causal = False
            else:
                attention_mask = CausalSelfAttention._offset_causal_mask(
                    query_length, key_length, prefix_length, query.device
                )
                is_causal = False

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        output = output.transpose(1, 2).contiguous().view(x.shape)
        if self.output_gate_proj is not None:
            output = output * self.output_gate_proj(x).sigmoid()
        return self.out_proj(output)

    def new_cache(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> MLACache:
        return MLACache(
            self.config,
            batch_size,
            device=device,
            dtype=dtype,
        )

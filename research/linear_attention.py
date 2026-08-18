"""Readable recurrent references for Gated DeltaNet and Kimi Delta Attention.

The modules in this file implement the recurrent equations from the papers in
plain PyTorch. The default route deliberately uses a token scan, which keeps
correctness and cache accounting inspectable but does not reproduce production
throughput. The opt-in CUDA ``fla`` backend preserves those equations while
dispatching to FLA chunkwise and fused recurrent kernels for the full-model
benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DeltaNetConfig:
    """Shape and gate configuration shared by the GDN and KDA references."""

    dim: int = 384
    n_heads: int = 6
    max_seq_len: int = 576
    conv_kernel_size: int = 4
    gate_rank: int = 64
    channelwise_decay: bool = False
    bias: bool = False
    rms_norm_eps: float = 1e-5
    scan_backend: str = "reference"

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.n_heads <= 0 or self.max_seq_len <= 0:
            raise ValueError("dim, n_heads, and max_seq_len must be positive")
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be positive")
        if self.gate_rank <= 0:
            raise ValueError("gate_rank must be positive")
        if self.scan_backend not in {"reference", "compiled", "fla"}:
            raise ValueError("scan_backend must be 'reference', 'compiled', or 'fla'")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def variant(self) -> str:
        return "KDA" if self.channelwise_decay else "GDN"


class DeltaStateCache:
    """Fixed-size recurrent matrix plus rolling causal-convolution history."""

    def __init__(
        self,
        config: DeltaNetConfig,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        state_dtype = (
            torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype
        )
        self.state = torch.zeros(
            batch_size,
            config.n_heads,
            config.head_dim,
            config.head_dim,
            device=device,
            dtype=state_dtype,
        )
        history_shape = (
            batch_size,
            max(0, config.conv_kernel_size - 1),
            config.dim,
        )
        self.q_history = torch.zeros(history_shape, device=device, dtype=dtype)
        self.k_history = torch.zeros_like(self.q_history)
        self.v_history = torch.zeros_like(self.q_history)
        self.length = 0
        self._capacity = config.max_seq_len

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def state_bytes(self) -> int:
        return self.state.numel() * self.state.element_size()

    @property
    def convolution_bytes(self) -> int:
        return (
            self.q_history.numel() + self.k_history.numel() + self.v_history.numel()
        ) * self.q_history.element_size()

    @property
    def fixed_bytes(self) -> int:
        """Saturated recurrent storage; independent of context length."""

        return self.state_bytes + self.convolution_bytes

    @property
    def bytes_per_token(self) -> int:
        return 0

    @property
    def allocated_bytes(self) -> int:
        return self.fixed_bytes

    @property
    def occupied_bytes(self) -> int:
        if self.length == 0:
            return 0
        history_capacity = self.q_history.shape[1]
        if history_capacity == 0:
            return self.state_bytes
        valid_history = min(self.length, history_capacity)
        history_bytes = self.convolution_bytes * valid_history // history_capacity
        return self.state_bytes + history_bytes

    def reset(self) -> None:
        self.state.zero_()
        self.q_history.zero_()
        self.k_history.zero_()
        self.v_history.zero_()
        self.length = 0

    def update(
        self,
        state: Tensor,
        q_history: Tensor,
        k_history: Tensor,
        v_history: Tensor,
        token_count: int,
    ) -> None:
        if token_count <= 0:
            raise ValueError("token_count must be positive")
        end = self.length + token_count
        if end > self.capacity:
            raise ValueError(
                f"recurrent cache overflow: requested {end}, capacity {self.capacity}"
            )
        expected_state = self.state.shape
        if state.shape != expected_state:
            raise ValueError(
                f"state shape mismatch: expected {expected_state}, got {state.shape}"
            )
        expected_history = self.q_history.shape
        for name, history in (
            ("q", q_history),
            ("k", k_history),
            ("v", v_history),
        ):
            if history.shape != expected_history:
                raise ValueError(
                    f"{name} history shape mismatch: expected {expected_history}, "
                    f"got {history.shape}"
                )
        with torch.no_grad():
            self.state.copy_(state)
            self.q_history.copy_(q_history)
            self.k_history.copy_(k_history)
            self.v_history.copy_(v_history)
        self.length = end


class CausalDepthwiseConv1d(nn.Module):
    """Small causal depthwise convolution with explicit rolling history."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(
        self, x: Tensor, history: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"expected x with shape (B, T, {self.channels})")
        history_length = self.kernel_size - 1
        if history is None:
            history = x.new_zeros(x.shape[0], history_length, self.channels)
        expected = (x.shape[0], history_length, self.channels)
        if history.shape != expected:
            raise ValueError(
                f"convolution history mismatch: expected {expected}, got {history.shape}"
            )
        combined = torch.cat((history, x), dim=1)
        output = F.conv1d(
            combined.transpose(1, 2),
            self.weight.unsqueeze(1),
            groups=self.channels,
        ).transpose(1, 2)
        new_history = combined[:, -history_length:] if history_length else history
        return F.silu(output), new_history


def _inverse_softplus(value: Tensor) -> Tensor:
    return value + torch.log(-torch.expm1(-value))


class DeltaNetAttention(nn.Module):
    """Shared GDN/KDA block with an exact recurrent reference path.

    At the recurrent-rule level, GDN uses one scalar decay per head and KDA
    changes that forget gate to a channel-wise vector parameterized with a
    low-rank MLP.  The complete reference blocks also retain their respective
    output gates: full-rank SiLU for GDN and low-rank sigmoid for KDA.
    """

    # T4's fused recurrent GDN route is not finite for very short prefixes.
    # A completed FLA chunk establishes a stable state, while the intended
    # benchmark decode starts after a 4K-token prefill in any case.
    _fla_minimum_fused_cache_tokens = 64

    def __init__(self, config: DeltaNetConfig) -> None:
        super().__init__()
        self.config = config
        self._compiled_scan: Callable[..., tuple[Tensor, Tensor]] | None = None
        self._compiled_scan_failure: str | None = None
        self._last_fla_kernel: str | None = None
        self.q_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.q_conv = CausalDepthwiseConv1d(config.dim, config.conv_kernel_size)
        self.k_conv = CausalDepthwiseConv1d(config.dim, config.conv_kernel_size)
        self.v_conv = CausalDepthwiseConv1d(config.dim, config.conv_kernel_size)
        if config.channelwise_decay:
            self.alpha_down_proj = nn.Linear(
                config.dim, config.gate_rank, bias=config.bias
            )
            self.alpha_up_proj = nn.Linear(
                config.gate_rank, config.dim, bias=config.bias
            )
            gate_shape = (config.n_heads, config.head_dim)
        else:
            self.alpha_proj = nn.Linear(config.dim, config.n_heads, bias=config.bias)
            gate_shape = (config.n_heads,)
        self.beta_proj = nn.Linear(config.dim, config.n_heads, bias=config.bias)

        # The official layers initialize long time constants through an inverse
        # softplus parameterization.  Keeping it here avoids an unstable, naked
        # decay scalar while remaining fully inspectable.
        a = torch.empty(config.n_heads).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(a.log())
        log_dt = torch.empty(gate_shape).uniform_(math.log(1e-3), math.log(1e-1))
        self.dt_bias = nn.Parameter(_inverse_softplus(log_dt.exp()))

        if config.channelwise_decay:
            self.output_gate_down_proj = nn.Linear(
                config.dim, config.gate_rank, bias=config.bias
            )
            self.output_gate_up_proj = nn.Linear(
                config.gate_rank, config.dim, bias=config.bias
            )
        else:
            self.output_gate_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.output_norm_weight = nn.Parameter(
            torch.ones(config.n_heads, config.head_dim)
        )
        self.out_proj = nn.Linear(config.dim, config.dim, bias=config.bias)

    def _project_decay(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        if self.config.channelwise_decay:
            raw = self.alpha_up_proj(self.alpha_down_proj(x)).view(
                batch,
                tokens,
                self.config.n_heads,
                self.config.head_dim,
            )
            rate = self.A_log.exp().view(1, 1, self.config.n_heads, 1)
            bias = self.dt_bias.view(1, 1, self.config.n_heads, self.config.head_dim)
        else:
            raw = self.alpha_proj(x)
            rate = self.A_log.exp().view(1, 1, self.config.n_heads)
            bias = self.dt_bias.view(1, 1, self.config.n_heads)
        return -rate.float() * F.softplus(raw.float() + bias.float())

    def _output_gate(self, x: Tensor) -> Tensor:
        if self.config.channelwise_decay:
            gate = self.output_gate_up_proj(self.output_gate_down_proj(x)).sigmoid()
        else:
            gate = F.silu(self.output_gate_proj(x))
        return gate.view(
            x.shape[0], x.shape[1], self.config.n_heads, self.config.head_dim
        )

    def _headwise_rms_norm(self, x: Tensor) -> Tensor:
        normalized = x * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
        ).to(x.dtype)
        return normalized * self.output_norm_weight

    def _scan_reference(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        log_decay: Tensor,
        beta: Tensor,
        initial_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # The reference kernels accumulate the recurrent matrix in FP32.  This
        # matters for long scans and also makes the implementation comparable
        # across FP32/BF16 model configurations.
        query = query.float()
        key = key.float()
        value = value.float()
        state = initial_state.float()
        outputs = []
        scale = self.config.head_dim**-0.5
        for token_index in range(query.shape[1]):
            if self.config.channelwise_decay:
                decay = log_decay[:, token_index].exp().unsqueeze(-1)
            else:
                decay = log_decay[:, token_index].exp().unsqueeze(-1).unsqueeze(-1)
            state = state * decay
            current_key = key[:, token_index]
            prediction = torch.einsum("bhk,bhkv->bhv", current_key, state)
            error = value[:, token_index] - prediction
            write = beta[:, token_index].unsqueeze(-1).unsqueeze(-1)
            state = state + write * current_key.unsqueeze(-1) * error.unsqueeze(-2)
            output = torch.einsum("bhk,bhkv->bhv", query[:, token_index] * scale, state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), state

    def _scan(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        log_decay: Tensor,
        beta: Tensor,
        initial_state: Tensor,
        *,
        use_fused_recurrent: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Run the exact scan using the selected execution backend.

        ``compiled`` preserves the recurrence and autograd semantics of the
        reference implementation.  Compilation is deliberately best-effort:
        some CPU builds cannot compile Inductor graphs (for example when the
        compiler cannot consume an installation path containing spaces).  In
        that case the module visibly falls back to ``reference`` instead of
        reporting an invalid performance result as compiled.
        """

        if self.config.scan_backend == "fla":
            return self._scan_fla(
                query,
                key,
                value,
                log_decay,
                beta,
                initial_state,
                use_fused_recurrent=use_fused_recurrent,
            )
        if self.config.scan_backend == "reference" or self._compiled_scan_failure:
            return self._scan_reference(
                query, key, value, log_decay, beta, initial_state
            )
        if self._compiled_scan is None:
            self._compiled_scan = torch.compile(
                self._scan_reference, fullgraph=True, dynamic=False
            )
        try:
            return self._compiled_scan(
                query, key, value, log_decay, beta, initial_state
            )
        except Exception as error:  # pragma: no cover - toolchain-dependent
            detail = str(error).splitlines()[0][:500]
            self._compiled_scan_failure = f"{type(error).__name__}: {detail}"
            return self._scan_reference(
                query, key, value, log_decay, beta, initial_state
            )

    def _scan_fla(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        log_decay: Tensor,
        beta: Tensor,
        initial_state: Tensor,
        *,
        use_fused_recurrent: bool,
    ) -> tuple[Tensor, Tensor]:
        """Use FLA's CUDA kernels while preserving the reference equations.

        FLA is deliberately opt-in rather than an automatic fallback: a missing
        CUDA extension must fail loudly so a reference scan is never reported as
        an optimized-kernel measurement. Full sequences use chunkwise kernels;
        one cached token after a sufficiently long state initialization uses
        FLA's fused recurrent decode kernel. On some T4 builds the fused GDN
        kernel is not finite for short prefixes, so cached tokens before one
        full 64-token chunk use the numerically equivalent chunkwise route.
        """

        if not query.is_cuda:
            raise RuntimeError("scan_backend='fla' requires CUDA tensors")
        if query.dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError(
                "scan_backend='fla' requires FP16 or BF16 query/key/value tensors"
            )
        # FLA's CUDA interfaces expect the gates in the activation dtype. The
        # readable reference keeps them in FP32, so this is the only intentional
        # precision boundary validated by the CUDA equivalence gate.
        log_decay = log_decay.to(dtype=query.dtype)
        beta = beta.to(dtype=query.dtype)
        try:
            if self.config.channelwise_decay:
                from fla.ops.kda import chunk_kda, fused_recurrent_kda
            else:
                from fla.ops.gated_delta_rule import (
                    chunk_gated_delta_rule,
                    fused_recurrent_gdn,
                )
        except ImportError as error:  # pragma: no cover - CUDA-only dependency
            raise RuntimeError(
                "scan_backend='fla' requires flash-linear-attention[cuda]; "
                "install it in the CUDA environment before benchmarking"
            ) from error

        if query.shape[1] == 1 and use_fused_recurrent:
            self._last_fla_kernel = "fused_recurrent"
            if self.config.channelwise_decay:
                return fused_recurrent_kda(
                    query,
                    key,
                    value,
                    log_decay,
                    beta,
                    initial_state=initial_state,
                    output_final_state=True,
                )
            return fused_recurrent_gdn(
                query,
                key,
                value,
                log_decay,
                beta,
                initial_state=initial_state,
                output_final_state=True,
            )

        self._last_fla_kernel = "chunkwise"
        if self.config.channelwise_decay:
            return chunk_kda(
                query,
                key,
                value,
                log_decay,
                beta,
                initial_state=initial_state,
                output_final_state=True,
            )
        return chunk_gated_delta_rule(
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state=initial_state,
            output_final_state=True,
        )

    def scan_backend_metadata(self) -> dict[str, Any]:
        """Report the requested and actual scan backend for benchmark records."""

        return {
            "requested": self.config.scan_backend,
            "effective": (
                "reference"
                if self.config.scan_backend == "reference"
                or self._compiled_scan_failure
                else self.config.scan_backend
            ),
            "fallback_reason": self._compiled_scan_failure,
            "fla_kernel_last_used": self._last_fla_kernel,
        }

    def forward(self, x: Tensor, cache: DeltaStateCache | None = None) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.dim:
            raise ValueError(f"expected x with shape (B, T, {self.config.dim})")
        if x.shape[1] == 0:
            raise ValueError("the token dimension must be non-empty")
        if cache is not None and cache.state.shape[0] != x.shape[0]:
            raise ValueError("cache batch size does not match x")

        q_history = None if cache is None else cache.q_history
        k_history = None if cache is None else cache.k_history
        v_history = None if cache is None else cache.v_history
        query, new_q_history = self.q_conv(self.q_proj(x), q_history)
        key, new_k_history = self.k_conv(self.k_proj(x), k_history)
        value, new_v_history = self.v_conv(self.v_proj(x), v_history)

        shape = (
            x.shape[0],
            x.shape[1],
            self.config.n_heads,
            self.config.head_dim,
        )
        query = F.normalize(query.view(shape), dim=-1).contiguous()
        key = F.normalize(key.view(shape), dim=-1).contiguous()
        value = value.view(shape).contiguous()
        log_decay = self._project_decay(x)
        beta = self.beta_proj(x).float().sigmoid()
        if cache is None:
            initial_state = x.new_zeros(
                x.shape[0],
                self.config.n_heads,
                self.config.head_dim,
                self.config.head_dim,
                dtype=torch.float32,
            )
        else:
            initial_state = cache.state
        output, final_state = self._scan(
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
            use_fused_recurrent=(
                cache is not None
                and cache.length >= self._fla_minimum_fused_cache_tokens
            ),
        )
        output = output.to(x.dtype)
        output = self._headwise_rms_norm(output) * self._output_gate(x)
        output = output.reshape(x.shape)
        if cache is not None:
            cache.update(
                final_state,
                new_q_history,
                new_k_history,
                new_v_history,
                x.shape[1],
            )
        return self.out_proj(output)

    def new_cache(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> DeltaStateCache:
        return DeltaStateCache(
            self.config,
            batch_size,
            device=device,
            dtype=dtype,
        )


class GatedDeltaNet(DeltaNetAttention):
    """Gated DeltaNet with one scalar state decay per head and token."""

    def __init__(self, config: DeltaNetConfig) -> None:
        if config.channelwise_decay:
            raise ValueError("GatedDeltaNet requires channelwise_decay=False")
        super().__init__(config)


class KimiDeltaAttention(DeltaNetAttention):
    """KDA with one state decay per key channel, head, and token."""

    def __init__(self, config: DeltaNetConfig) -> None:
        if not config.channelwise_decay:
            raise ValueError("KimiDeltaAttention requires channelwise_decay=True")
        super().__init__(config)

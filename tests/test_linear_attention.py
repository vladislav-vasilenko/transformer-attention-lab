from __future__ import annotations

import unittest

import torch

from research.linear_attention import (
    DeltaNetConfig,
    DeltaStateCache,
    GatedDeltaNet,
    KimiDeltaAttention,
)


class DeltaAttentionCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.hidden = torch.randn(2, 11, 32)

    @staticmethod
    def build(channelwise_decay: bool, scan_backend: str = "reference"):
        config = DeltaNetConfig(
            dim=32,
            n_heads=4,
            max_seq_len=16,
            conv_kernel_size=4,
            gate_rank=8,
            channelwise_decay=channelwise_decay,
            scan_backend=scan_backend,
        )
        cls = KimiDeltaAttention if channelwise_decay else GatedDeltaNet
        return cls(config).eval()

    def test_tokenwise_cache_matches_full_scan(self) -> None:
        for channelwise in (False, True):
            with self.subTest(channelwise=channelwise):
                attention = self.build(channelwise)
                full = attention(self.hidden)
                cache = attention.new_cache(
                    self.hidden.shape[0], device="cpu", dtype=self.hidden.dtype
                )
                pieces = [
                    attention(self.hidden[:, index : index + 1], cache)
                    for index in range(self.hidden.shape[1])
                ]
                cached = torch.cat(pieces, dim=1)
                torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)

    def test_arbitrary_cached_chunks_match_full_scan(self) -> None:
        for channelwise in (False, True):
            with self.subTest(channelwise=channelwise):
                attention = self.build(channelwise)
                cache = attention.new_cache(2, device="cpu", dtype=torch.float32)
                pieces = [
                    attention(self.hidden[:, :3], cache),
                    attention(self.hidden[:, 3:7], cache),
                    attention(self.hidden[:, 7:], cache),
                ]
                torch.testing.assert_close(
                    torch.cat(pieces, dim=1),
                    attention(self.hidden),
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_future_perturbation_cannot_change_prefix(self) -> None:
        attention = self.build(channelwise_decay=True)
        changed = self.hidden.clone()
        changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 20
        torch.testing.assert_close(
            attention(self.hidden)[:, :7],
            attention(changed)[:, :7],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_decay_and_write_gates_are_bounded(self) -> None:
        for channelwise in (False, True):
            attention = self.build(channelwise)
            decay = attention._project_decay(self.hidden).exp()
            beta = attention.beta_proj(self.hidden).sigmoid()
            self.assertTrue(torch.all((decay > 0) & (decay < 1)))
            self.assertTrue(torch.all((beta > 0) & (beta < 1)))

    def test_gradients_are_finite(self) -> None:
        for channelwise in (False, True):
            attention = self.build(channelwise).train()
            loss = attention(self.hidden).square().mean()
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in attention.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in gradients)
            )

    def test_compiled_backend_matches_reference_outputs_and_gradients(self) -> None:
        reference = self.build(channelwise_decay=False).train()
        compiled = self.build(channelwise_decay=False, scan_backend="compiled").train()
        compiled.load_state_dict(reference.state_dict())
        reference_hidden = self.hidden.detach().clone().requires_grad_()
        compiled_hidden = self.hidden.detach().clone().requires_grad_()

        reference_loss = reference(reference_hidden).square().mean()
        compiled_loss = compiled(compiled_hidden).square().mean()
        reference_loss.backward()
        compiled_loss.backward()

        torch.testing.assert_close(compiled_loss, reference_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            compiled_hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6
        )
        for (_, reference_parameter), (_, compiled_parameter) in zip(
            reference.named_parameters(), compiled.named_parameters(), strict=True
        ):
            torch.testing.assert_close(
                compiled_parameter.grad,
                reference_parameter.grad,
                rtol=1e-5,
                atol=1e-6,
            )
        metadata = compiled.scan_backend_metadata()
        self.assertEqual(metadata["requested"], "compiled")
        self.assertIn(metadata["effective"], {"compiled", "reference"})
        if metadata["fallback_reason"] is not None:
            self.assertEqual(metadata["effective"], "reference")

    def test_fla_backend_is_an_explicit_valid_configuration(self) -> None:
        attention = self.build(channelwise_decay=False, scan_backend="fla")
        metadata = attention.scan_backend_metadata()
        self.assertEqual(metadata["requested"], "fla")
        self.assertEqual(metadata["effective"], "fla")

    def test_fla_cached_decode_initializes_with_chunkwise_then_fused(self) -> None:
        attention = self.build(channelwise_decay=False, scan_backend="fla")
        requests: list[bool] = []

        def fake_fla_scan(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            log_decay: torch.Tensor,
            beta: torch.Tensor,
            initial_state: torch.Tensor,
            *,
            use_fused_recurrent: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del key, value, log_decay, beta
            requests.append(use_fused_recurrent)
            return torch.zeros_like(query), initial_state

        attention._scan_fla = fake_fla_scan  # type: ignore[method-assign]
        cache = attention.new_cache(2, device="cpu", dtype=torch.float32)
        attention(self.hidden[:, :1], cache)
        cache._capacity = 65
        cache.length = attention._fla_minimum_fused_cache_tokens
        attention(self.hidden[:, 1:2], cache)
        self.assertEqual(requests, [False, True])


class DeltaStateCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DeltaNetConfig(
            dim=32,
            n_heads=4,
            max_seq_len=16,
            conv_kernel_size=4,
            gate_rank=8,
        )

    def test_state_is_fixed_after_convolution_history_saturates(self) -> None:
        attention = GatedDeltaNet(self.config).eval()
        cache = attention.new_cache(1, device="cpu", dtype=torch.float32)
        attention(torch.randn(1, 4, 32), cache)
        four_token_bytes = cache.occupied_bytes
        attention(torch.randn(1, 7, 32), cache)
        self.assertEqual(cache.occupied_bytes, four_token_bytes)
        self.assertEqual(cache.bytes_per_token, 0)
        self.assertEqual(cache.occupied_bytes, cache.fixed_bytes)

    def test_reset_clears_state_history_and_length(self) -> None:
        attention = GatedDeltaNet(self.config).eval()
        cache = attention.new_cache(1, device="cpu", dtype=torch.float32)
        hidden = torch.randn(1, 6, 32)
        expected = attention(hidden)
        attention(hidden, cache)
        cache.reset()
        self.assertEqual(cache.length, 0)
        self.assertEqual(cache.occupied_bytes, 0)
        self.assertEqual(torch.count_nonzero(cache.state), 0)
        cached = attention(hidden, cache)
        torch.testing.assert_close(cached, expected, rtol=1e-5, atol=1e-6)

    def test_cache_shape_and_memory_formula(self) -> None:
        cache = DeltaStateCache(
            self.config, batch_size=2, device="cpu", dtype=torch.float32
        )
        expected_state = 2 * 4 * 8 * 8 * 4
        expected_history = 3 * 2 * 3 * 32 * 4
        self.assertEqual(cache.state_bytes, expected_state)
        self.assertEqual(cache.convolution_bytes, expected_history)
        self.assertEqual(cache.allocated_bytes, expected_state + expected_history)


if __name__ == "__main__":
    unittest.main()

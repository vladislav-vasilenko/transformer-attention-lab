from __future__ import annotations

import unittest

import torch

from research.attention import (
    AttentionConfig,
    CausalSelfAttention,
    KVCache,
    MLAConfig,
    MLACache,
    MultiLatentAttention,
)


class AttentionConfigTests(unittest.TestCase):
    def test_rejects_incompatible_head_counts(self) -> None:
        with self.assertRaises(ValueError):
            AttentionConfig(dim=64, n_heads=8, n_kv_heads=3)

    def test_variant_names(self) -> None:
        self.assertEqual(AttentionConfig(64, 8, 8).variant, "MHA")
        self.assertEqual(AttentionConfig(64, 8, 2).variant, "GQA")
        self.assertEqual(AttentionConfig(64, 8, 1).variant, "MQA")


class KVCacheTests(unittest.TestCase):
    def test_occupied_and_allocated_bytes(self) -> None:
        config = AttentionConfig(dim=64, n_heads=8, n_kv_heads=2, max_seq_len=10)
        cache = KVCache(config, batch_size=2, device="cpu", dtype=torch.float32)
        key = torch.randn(2, 2, 3, 8)
        cache.append(key, key)
        expected_occupied = 2 * 2 * 2 * 3 * 8 * 4
        expected_allocated = 2 * 2 * 2 * 10 * 8 * 4
        self.assertEqual(cache.occupied_bytes, expected_occupied)
        self.assertEqual(cache.allocated_bytes, expected_allocated)

    def test_overflow_is_rejected(self) -> None:
        config = AttentionConfig(dim=16, n_heads=2, n_kv_heads=1, max_seq_len=2)
        cache = KVCache(config, batch_size=1, device="cpu", dtype=torch.float32)
        key = torch.randn(1, 1, 3, 8)
        with self.assertRaises(ValueError):
            cache.append(key, key)

    def test_mla_cache_stores_one_latent_per_token(self) -> None:
        config = MLAConfig(dim=64, n_heads=8, latent_rank=12, max_seq_len=10)
        cache = MLACache(config, batch_size=2, device="cpu", dtype=torch.float32)
        cache.append(torch.randn(2, 3, 12))
        self.assertEqual(cache.occupied_bytes, 2 * 3 * 12 * 4)
        self.assertEqual(cache.allocated_bytes, 2 * 10 * 12 * 4)

    def test_gated_mla_keeps_the_same_cache_and_adds_output_gate(self) -> None:
        plain = MultiLatentAttention(
            MLAConfig(dim=64, n_heads=8, latent_rank=16, max_seq_len=9)
        )
        gated = MultiLatentAttention(
            MLAConfig(
                dim=64,
                n_heads=8,
                latent_rank=16,
                max_seq_len=9,
                output_gate=True,
            )
        )
        self.assertIsNone(plain.output_gate_proj)
        self.assertIsNotNone(gated.output_gate_proj)
        plain_cache = plain.new_cache(1, device="cpu", dtype=torch.float32)
        gated_cache = gated.new_cache(1, device="cpu", dtype=torch.float32)
        self.assertEqual(plain_cache.bytes_per_token, gated_cache.bytes_per_token)
        plain_parameters = sum(parameter.numel() for parameter in plain.parameters())
        gated_parameters = sum(parameter.numel() for parameter in gated.parameters())
        self.assertEqual(gated_parameters - plain_parameters, 64 * 64)


class AttentionCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_cached_decode_matches_full_recomputation(self) -> None:
        hidden = torch.randn(2, 9, 64)
        for n_kv_heads in (8, 2, 1):
            with self.subTest(n_kv_heads=n_kv_heads):
                config = AttentionConfig(
                    dim=64,
                    n_heads=8,
                    n_kv_heads=n_kv_heads,
                    max_seq_len=9,
                )
                attention = CausalSelfAttention(config).eval()
                cache = attention.new_cache(2, device="cpu", dtype=hidden.dtype)

                for token_index in range(hidden.shape[1]):
                    cached = attention(hidden[:, token_index : token_index + 1], cache)
                    full = attention(hidden[:, : token_index + 1])[:, -1:]
                    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)

    def test_chunked_cached_decode_uses_offset_causal_mask(self) -> None:
        config = AttentionConfig(dim=32, n_heads=4, n_kv_heads=2, max_seq_len=8)
        attention = CausalSelfAttention(config).eval()
        hidden = torch.randn(1, 8, 32)
        cache = attention.new_cache(1, device="cpu", dtype=hidden.dtype)

        first = attention(hidden[:, :3], cache)
        second = attention(hidden[:, 3:], cache)
        combined = torch.cat((first, second), dim=1)
        full = attention(hidden)
        torch.testing.assert_close(combined, full, rtol=1e-5, atol=1e-6)

    def test_mla_cached_decode_matches_full_recomputation(self) -> None:
        for output_gate in (False, True):
            with self.subTest(output_gate=output_gate):
                config = MLAConfig(
                    dim=64,
                    n_heads=8,
                    latent_rank=16,
                    max_seq_len=9,
                    output_gate=output_gate,
                )
                attention = MultiLatentAttention(config).eval()
                hidden = torch.randn(2, 9, 64)
                cache = attention.new_cache(2, device="cpu", dtype=hidden.dtype)
                for token_index in range(hidden.shape[1]):
                    cached = attention(hidden[:, token_index : token_index + 1], cache)
                    full = attention(hidden[:, : token_index + 1])[:, -1:]
                    torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

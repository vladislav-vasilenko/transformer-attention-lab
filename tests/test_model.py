from __future__ import annotations

import unittest

import torch

from research.attention import CausalSelfAttention, MultiLatentAttention
from research.linear_attention import GatedDeltaNet, KimiDeltaAttention
from research.model import GPT, GPTConfig


class GPTAttentionVariantTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_all_variants_produce_logits_and_loss(self) -> None:
        tokens = torch.randint(0, 32, (2, 12))
        for variant in ("MHA", "GQA", "MQA", "MLA"):
            with self.subTest(variant=variant):
                model = GPT(
                    GPTConfig(
                        vocab_size=32,
                        block_size=16,
                        n_layer=2,
                        n_head=4,
                        n_embd=32,
                        attention_variant=variant,
                        gqa_n_kv_heads=2,
                        mla_latent_rank=8,
                    )
                )
                logits, loss = model(tokens, targets=tokens)
                self.assertEqual(logits.shape, (2, 12, 32))
                self.assertTrue(torch.isfinite(loss))

    def test_cached_model_logits_match_full_recomputation(self) -> None:
        tokens = torch.randint(0, 32, (1, 12))
        for variant in ("MHA", "GQA", "MQA", "MLA"):
            with self.subTest(variant=variant):
                model = GPT(
                    GPTConfig(
                        vocab_size=32,
                        block_size=16,
                        n_layer=2,
                        n_head=4,
                        n_embd=32,
                        attention_variant=variant,
                        gqa_n_kv_heads=2,
                        mla_latent_rank=8,
                    )
                ).eval()
                caches = model.new_caches(1, device="cpu", dtype=torch.float32)
                for index in range(tokens.shape[1]):
                    cached_logits, _ = model(
                        tokens[:, index : index + 1], caches=caches
                    )
                    full_logits, _ = model(tokens[:, : index + 1])
                    torch.testing.assert_close(
                        cached_logits, full_logits[:, -1:], rtol=1e-5, atol=1e-6
                    )

    def test_cache_storage_order_matches_hypothesis(self) -> None:
        occupied: dict[str, int] = {}
        for variant in ("MHA", "GQA", "MQA", "MLA"):
            model = GPT(
                GPTConfig(
                    vocab_size=32,
                    block_size=16,
                    n_layer=2,
                    n_head=4,
                    n_embd=32,
                    attention_variant=variant,
                    gqa_n_kv_heads=2,
                    mla_latent_rank=4,
                )
            ).eval()
            caches = model.new_caches(1, device="cpu", dtype=torch.float32)
            model(torch.randint(0, 32, (1, 10)), caches=caches)
            occupied[variant] = model.cache_occupied_bytes(caches)
        self.assertGreater(occupied["MHA"], occupied["GQA"])
        self.assertGreater(occupied["GQA"], occupied["MQA"])
        self.assertGreater(occupied["MQA"], occupied["MLA"])

    def test_mqa_uses_one_shared_kv_head(self) -> None:
        model = GPT(
            GPTConfig(
                vocab_size=32,
                block_size=16,
                n_layer=2,
                n_head=4,
                n_embd=32,
                attention_variant="MQA",
                gqa_n_kv_heads=2,
            )
        )
        for block in model.blocks:
            self.assertEqual(block.attention.config.n_kv_heads, 1)

    def test_hybrid_layer_schedules(self) -> None:
        gdn = GPT(
            GPTConfig(
                vocab_size=32,
                block_size=16,
                n_layer=4,
                n_head=4,
                n_embd=32,
                attention_variant="GDN_HYBRID",
            )
        )
        self.assertEqual(
            [type(block.attention) for block in gdn.blocks],
            [GatedDeltaNet, GatedDeltaNet, GatedDeltaNet, CausalSelfAttention],
        )
        self.assertTrue(gdn.blocks[-1].attention.config.output_gate)

        kda = GPT(
            GPTConfig(
                vocab_size=32,
                block_size=16,
                n_layer=4,
                n_head=4,
                n_embd=32,
                attention_variant="KDA_HYBRID",
                mla_latent_rank=8,
            )
        )
        self.assertEqual(
            [type(block.attention) for block in kda.blocks],
            [
                KimiDeltaAttention,
                KimiDeltaAttention,
                KimiDeltaAttention,
                MultiLatentAttention,
            ],
        )
        self.assertFalse(kda.blocks[-1].attention.config.output_gate)

        k3 = GPT(
            GPTConfig(
                vocab_size=32,
                block_size=16,
                n_layer=4,
                n_head=4,
                n_embd=32,
                attention_variant="KDA_GATED_MLA_HYBRID",
                mla_latent_rank=8,
            )
        )
        self.assertEqual(
            [type(block.attention) for block in k3.blocks],
            [
                KimiDeltaAttention,
                KimiDeltaAttention,
                KimiDeltaAttention,
                MultiLatentAttention,
            ],
        )
        self.assertTrue(k3.blocks[-1].attention.config.output_gate)

    def test_pure_linear_attention_schedules(self) -> None:
        for variant, expected in (("GDN", GatedDeltaNet), ("KDA", KimiDeltaAttention)):
            with self.subTest(variant=variant):
                model = GPT(
                    GPTConfig(
                        vocab_size=32,
                        block_size=16,
                        n_layer=4,
                        n_head=4,
                        n_embd=32,
                        attention_variant=variant,
                        kda_gate_rank=8,
                    )
                )
                self.assertTrue(
                    all(isinstance(block.attention, expected) for block in model.blocks)
                )

    def test_linear_and_hybrid_cached_logits_match_full_recomputation(self) -> None:
        tokens = torch.randint(0, 32, (1, 10))
        for variant in (
            "GDN",
            "KDA",
            "GDN_HYBRID",
            "KDA_HYBRID",
            "KDA_GATED_MLA_HYBRID",
        ):
            with self.subTest(variant=variant):
                model = GPT(
                    GPTConfig(
                        vocab_size=32,
                        block_size=16,
                        n_layer=4,
                        n_head=4,
                        n_embd=32,
                        attention_variant=variant,
                        mla_latent_rank=8,
                        kda_gate_rank=8,
                    )
                ).eval()
                caches = model.new_caches(1, device="cpu", dtype=torch.float32)
                for index in range(tokens.shape[1]):
                    cached_logits, _ = model(
                        tokens[:, index : index + 1], caches=caches
                    )
                    full_logits, _ = model(tokens[:, : index + 1])
                    torch.testing.assert_close(
                        cached_logits,
                        full_logits[:, -1:],
                        rtol=1e-5,
                        atol=1e-6,
                    )

    def test_hybrid_requires_complete_three_to_one_cycle(self) -> None:
        with self.assertRaises(ValueError):
            GPTConfig(
                vocab_size=32,
                n_layer=2,
                n_head=4,
                n_embd=32,
                attention_variant="GDN_HYBRID",
            )


if __name__ == "__main__":
    unittest.main()

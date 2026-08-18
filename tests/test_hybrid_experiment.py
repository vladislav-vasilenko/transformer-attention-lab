from __future__ import annotations

import unittest

from research.hybrid_experiment import cache_layout
from research.model import GPT, GPTConfig


class HybridCacheLayoutTests(unittest.TestCase):
    def build(self, variant: str) -> GPT:
        return GPT(
            GPTConfig(
                vocab_size=32,
                block_size=16,
                n_layer=4,
                n_head=4,
                n_embd=32,
                attention_variant=variant,
                mla_latent_rank=8,
                delta_conv_kernel=4,
                kda_gate_rank=8,
            )
        )

    def test_fixed_plus_linear_storage_laws(self) -> None:
        state_per_recurrent_layer = (4 * 8 * 8 + 3 * 3 * 32) * 4
        expected = {
            "MHA": (0, 4 * 2 * 4 * 8 * 4),
            "GQA": (0, 4 * 2 * 2 * 8 * 4),
            "MQA": (0, 4 * 2 * 1 * 8 * 4),
            "MLA": (0, 4 * 8 * 4),
            "GDN": (4 * state_per_recurrent_layer, 0),
            "KDA": (4 * state_per_recurrent_layer, 0),
            "GDN_HYBRID": (3 * state_per_recurrent_layer, 2 * 4 * 8 * 4),
            "KDA_HYBRID": (3 * state_per_recurrent_layer, 8 * 4),
            "KDA_GATED_MLA_HYBRID": (3 * state_per_recurrent_layer, 8 * 4),
        }
        for variant, law in expected.items():
            with self.subTest(variant=variant):
                layout = cache_layout(self.build(variant))
                self.assertEqual(
                    (layout["fixed_bytes"], layout["bytes_per_token"]), law
                )


if __name__ == "__main__":
    unittest.main()

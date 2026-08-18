from __future__ import annotations

import unittest

import torch

from research.experiment import CharacterCorpus


class FullEpochBatchingTests(unittest.TestCase):
    def test_every_non_overlapping_window_is_seen_once(self) -> None:
        corpus = CharacterCorpus.__new__(CharacterCorpus)
        corpus.train = torch.arange(66)
        batches = list(
            corpus.iter_train_epoch(
                batch_size=3,
                context=8,
                generator=torch.Generator().manual_seed(99),
                device=torch.device("cpu"),
            )
        )
        starts = sorted(int(row[0]) for inputs, _ in batches for row in inputs)
        self.assertEqual(starts, list(range(0, 64, 8)))
        self.assertEqual(sum(inputs.numel() for inputs, _ in batches), 64)


if __name__ == "__main__":
    unittest.main()

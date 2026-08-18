from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from research.plot_cuda_full_model import (
    VARIANTS,
    VERIFY_VARIANTS,
    load_results,
    render,
    write_summary,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class CudaFullModelPlotTests(unittest.TestCase):
    def test_valid_full_model_result_set_renders(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            training = [
                {
                    "variant": variant,
                    "tokens": 64,
                    "tokens_per_second_p50": 100_000 + index * 1_000,
                }
                for index, variant in enumerate(VARIANTS)
            ]
            prefill = [
                {
                    "variant": variant,
                    "tokens": tokens,
                    "tokens_per_second_p50": 100_000 * (index + 1) * tokens,
                }
                for index, variant in enumerate(VARIANTS)
                for tokens in (64, 256)
            ]
            decode = [
                {
                    "variant": variant,
                    "prompt_tokens": 256,
                    "p50_ms_per_token": 1.0 + index / 10,
                    "p95_ms_per_token": 1.2 + index / 10,
                }
                for index, variant in enumerate(VARIANTS)
            ]
            verification = [
                {
                    "variant": variant,
                    "status": "passed",
                    "max_abs_full_logits_error": 0.01,
                    "max_abs_cached_logits_error": 0.02,
                }
                for variant in VERIFY_VARIANTS
            ]
            environment = [
                {
                    "gpu": "Synthetic CUDA",
                    "compute_capability": "9.0",
                    "torch": "test",
                    "fla": "test",
                    "dtype": "torch.bfloat16",
                }
            ]
            write_rows(directory / "cuda_full_model_training.csv", training)
            write_rows(directory / "cuda_full_model_prefill.csv", prefill)
            write_rows(directory / "cuda_full_model_decode.csv", decode)
            write_rows(directory / "cuda_full_model_verification.csv", verification)
            write_rows(directory / "cuda_full_model_environment.csv", environment)

            loaded = load_results(directory)
            graph = directory / "cuda_full_model_comparison.png"
            summary = directory / "CUDA_FULL_MODEL_SUMMARY.md"
            render(graph, loaded[0], loaded[1], loaded[2], loaded[4])
            write_summary(summary, *loaded)

            self.assertGreater(graph.stat().st_size, 0)
            self.assertIn("Synthetic CUDA", summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

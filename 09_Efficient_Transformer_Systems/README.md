# Efficient Transformer Systems laboratories

MQA was proposed in
[Fast Transformer Decoding](https://arxiv.org/abs/1911.02150). It keeps
multiple query heads but shares one K/V pair and is implemented here as the
`H_kv = 1` endpoint of GQA.

MLA was introduced in the
[DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434). The implementation in
this laboratory is explicitly MLA-style rather than a full DeepSeek-V2
reproduction.

The executed notebook
[`01_mha_gqa_mla_comparison.ipynb`](01_mha_gqa_mla_comparison.ipynb) is the
completed cache-attention baseline of the Efficient Transformer Systems project.

It follows the sequence:

1. research question and motivation;
2. controlled variables and scope;
3. architecture comparison and falsifiable hypotheses;
4. quality and systems metrics;
5. fixed-budget training curves;
6. cached versus full-recomputation inference;
7. measured-cache validation and long-context capacity projection;
8. findings against each hypothesis;
9. limitations and next experiments;
10. personal contribution and interview summary.

Run `make admission-demo` from the project root to verify code/tests and rebuild
the plots from the committed raw result.

The extension
[`02_gated_deltanet_kda_hybrids.ipynb`](02_gated_deltanet_kda_hybrids.ipynb)
adds a separate matched-depth experiment for recurrent linear attention:

- homogeneous 4-layer MHA, GQA, MQA, and MLA-style baselines;
- pure 4-layer Gated DeltaNet and Kimi Delta Attention;
- `GDN → GDN → GDN → gated MHA`, inspired by Qwen3-Next;
- `KDA → KDA → KDA → MLA-style`, inspired by Kimi Linear;
- `KDA → KDA → KDA → gated MLA-style`, a controlled
  Kimi-K3-inspired gate ablation with the same persistent-state law as the
  ungated KDA/MLA hybrid.

This second experiment uses one deterministic full pass over every complete,
non-overlapping train window rather than random fixed-step sampling. It measures
validation PPL, training throughput, prefill and decode latency, cached versus
recomputed generation, and the exact persistent-state storage law.

The recurrent equations, short causal convolutions, gates, and cache states are
implemented directly in PyTorch. The experiment does not claim to reproduce
Qwen3-Next, Kimi Linear, or Kimi K3 as complete models: it retains the nanoGPT
backbone, learned positional embeddings, and dense MLPs, and it omits MoE,
Attention Residuals, SiTU, production training recipes, and optimized
FLA/FlashKDA kernels.

## CUDA chunkwise follow-up

[`03_cuda_chunkwise_kernels.ipynb`](03_cuda_chunkwise_kernels.ipynb) is a
separate Google Colab CUDA benchmark for the throughput claim. It verifies the
FLA chunked GDN/KDA outputs and final states against the exact recurrence,
then benchmarks chunkwise prefill, fused-recurrent decode, and fused SDPA.
It does not overwrite the CPU reference result or the quality experiment.

Run it in a Colab GPU runtime, download its four CSV files, and add them to
`results/` as a new CUDA timing session. Do not mix those timings with the CPU
reference chart: record GPU model, PyTorch/FLA versions, dtype, batch size, and
warmups with the exported tables.

After placing the CSV files in `results/`, run `make cuda-chunkwise-plots`.
This creates `cuda_chunkwise_operator_benchmark.png` and
`CUDA_CHUNKWISE_SUMMARY.md` without modifying the original CPU figures.

## CUDA full-model follow-up

[`04_cuda_full_model_all_variants.ipynb`](04_cuda_full_model_all_variants.ipynb)
extends the comparison to all nine implemented 4-layer systems. It runs the
full decoder for train steps (forward + loss + backward + AdamW), cached
prefill, and cached decode—not just a standalone attention operator. GDN/KDA
layers use the new explicit `fla` backend: chunkwise FLA kernels for
prefill/training and fused recurrent FLA kernels for one-token decode.

The notebook first verifies FLA logits against the reference scan for pure and
hybrid recurrent schedules. A successful standard run writes five files:

- `cuda_full_model_verification.csv`;
- `cuda_full_model_training.csv`;
- `cuda_full_model_prefill.csv`;
- `cuda_full_model_decode.csv`;
- `cuda_full_model_environment.csv`.

After copying them into `results/`, run `make cuda-full-model-plots`. This
creates `cuda_full_model_comparison.png` and `CUDA_FULL_MODEL_SUMMARY.md`.
These measurements use random token IDs for systems timing; they are a
full-model CUDA performance comparison, not retraining or CUDA PPL evidence.

## Proposed Experiment C

The next top-tier comparison is deliberately separate from the completed
nine-variant study. DeepSeek-V3/R1 is treated as the MLA control; a
DeepSeek-V4-style compressed/sparse-attention track would compare `MLA vs
CSA/HCA`, followed by the crossed ablation `plain residual vs mhC`.

Its metrics must include MQAR / sparse-retrieval recall, selected-KV-block
fraction, indexer memory and latency, actual KV bytes read, peak device memory,
and a dedicated mhC ablation. Cache capacity alone is insufficient because it
does not account for search cost or the possibility of omitting the context
block that contains the needed evidence.

# CUDA full-model comparison — result summary

## Environment

| GPU | Compute capability | PyTorch | FLA | Dtype |
|---|---:|---|---|---|
| Tesla T4 | 7.5 | 2.11.0+cu128 | 0.5.1 | torch.float16 |

The benchmark configuration is recorded verbatim in
`cuda_full_model_environment.csv`. The recurrent path uses FLA chunkwise
kernels for prefill/training and FLA fused recurrent kernels for decode.
MHA/GQA/MQA/MLA use the repository's PyTorch SDPA implementation.

## Verification gate

FLA/reference checks passed for all five recurrent-containing schedules. The
largest absolute full-sequence-logit difference was `0.00128174`;
the largest cached-logit difference was `0.0012207`. These are
reduced-precision kernel-equivalence checks, not a quality measurement.

## Observed winners

- Highest full-model train-step throughput: **MHA** at
  **138,007 tok/s** (p50).
- Highest 4096-token full-model prefill throughput:
  **MLA-style** at
  **1.25M tok/s** (p50).
- Lowest cached decode p50 after a 4096-token prefix:
  **MLA-style** at
  **2.140 ms/token**.

## Scope

This is one CUDA session on random token IDs. Training rows include the full
4-layer decoder, vocabulary loss, backward pass, and AdamW; prefill and decode
rows include the full model and cache updates. It is a direct performance
comparison of the nine implemented token mixers, but it does **not** retrain
the models, produce CUDA PPL values, match parameters, or measure a production
Qwen/Kimi/DeepSeek system. Do not merge its absolute throughput values with the
earlier CPU reference-path chart.

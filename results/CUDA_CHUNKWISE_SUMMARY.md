# CUDA chunkwise kernel benchmark — result summary

This is a separate operator-level CUDA session. It must not be numerically
merged with the earlier CPU full-GPT reference-path chart.

## Environment

| GPU | Compute capability | PyTorch | FLA | Dtype | Batch | Heads × head dim |
|---|---:|---|---|---|---:|---:|
| Tesla T4 | 7.5 | 2.11.0+cu128 | 0.5.1 | torch.float16 | 8 | 4 × 32 |

## Findings

- At 64 tokens, FLA chunkwise prefill is **21.1×** faster
  than the GDN reference scan and **13.9×** faster than
  the KDA reference scan.
- At the 64-token attention `forward + backward` scope, chunkwise is
  **21.9×** faster for GDN and **21.1×**
  faster for KDA than their reference paths.
- At 4K-token prefill, MHA SDPA reaches 7.64M tokens/s;
  GDN chunkwise reaches 4.19M (1.82× below MHA)
  and KDA chunkwise reaches 5.96M (1.28× below MHA).
- After a 4K-token prefix, GDN decode p50 is 0.278 ms/token and
  KDA is 0.240 ms/token, versus 0.288 ms/token for MHA.
  This is 3.3% and
  16.5% lower latency, respectively.

## Scope and conclusion

The reference scan's large slowdown was implementation overhead, not a direct
architecture ranking: chunkwise kernels remove roughly 14–22× of that penalty
at the tested short window. MHA SDPA remains faster for prefill through 4K in
this T4 FP16 microbenchmark, while KDA approaches it at 4K and has the lowest
cached decode latency.

The training panel measures only the attention operator's forward and backward
passes. It excludes projections, convolution, MLP, embeddings, loss, AdamW,
and all full-model work; it is therefore not a replacement for the original
CPU end-to-end GPT training-throughput figure.

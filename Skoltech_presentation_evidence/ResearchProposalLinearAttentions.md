# Presentation brief — Efficient Linear Attention Systems

## Mission

Create an English 16:9 research presentation from this repository. The talk
should make one precise argument:

> Recurrent linear-attention mechanisms can drastically reduce persistent
> inference state, but their realized speed is a property of the execution
> kernel and the surrounding model. A Python token-by-token reference scan
> exaggerates their slowdown; optimized CUDA kernels expose the remaining
> full-model prefill-versus-decode-versus-state trade-off.

This is an engineering research project and a controlled prototype study, not a
new architecture paper, a production benchmark, or a reproduction of full
Qwen3-Next, Kimi Linear, DeepSeek, or Kimi K3 models.

## Non-negotiable evidence boundary

There are **three separate evidence sets**. Keep their hardware, scope, graphs,
and numerical claims separate on every slide.

| Evidence set | What was measured | Correct use | Never claim |
|---|---|---|---|
| **CPU full-model reference experiment** | 4-layer nanoGPT training, validation PPL, measured recurrent/KV state, and cached/recomputed decode timing | Quality–memory trade-off; diagnosis of the unoptimized reference implementation | That GDN/KDA architectures are intrinsically 9× slower, or that this is a CUDA benchmark |
| **CUDA kernel microbenchmark** | Attention operator prefill, forward + backward at 64 tokens, and cached decode after a 4K prefix | Demonstrates the effect of FLA chunkwise/recurrent kernels | Full-GPT training throughput, PPL, AdamW step time, or a direct CPU-versus-GPU comparison |
| **CUDA full-model microbenchmark** | The nine 4-layer models: full train step, cached prefill, and cached decode; GDN/KDA run with FLA | Direct CUDA performance comparison of the implemented token mixers | CUDA PPL, retraining, matched parameters, or production-model performance |

Use the labels **“CPU full-model reference path”** and **“CUDA attention-operator microbenchmark”** verbatim. Do not merge their bars, axes, or tables.

## Source of truth and visual assets

This hand-off folder contains the completed Tesla T4 **full-model** CUDA
benchmark. All paths below are relative to this Markdown file.

| Asset | Role in deck | Evidence class |
|---|---|---|
| [`cuda_full_model_comparison.png`](cuda_full_model_comparison.png) | Main CUDA performance visual; all nine full 4-layer models | Measured Tesla T4 full-model microbenchmark |
| [`CUDA_FULL_MODEL_SUMMARY.md`](CUDA_FULL_MODEL_SUMMARY.md) | Exact winners, environment, verification status, and scope | Primary result summary |
| [`cuda_full_model_training.csv`](cuda_full_model_training.csv) | Full forward + loss + backward + AdamW at 64 tokens | Raw T4 measurements |
| [`cuda_full_model_prefill.csv`](cuda_full_model_prefill.csv) | Cached full-model prefill from 64 to 4096 tokens | Raw T4 measurements |
| [`cuda_full_model_decode.csv`](cuda_full_model_decode.csv) | Cached decode after a 4096-token prefix | Raw T4 measurements |
| [`cuda_full_model_verification.csv`](cuda_full_model_verification.csv) | FLA/reference full and cached-logit checks | Equivalence gate; all five rows passed |
| [`cuda_full_model_environment.csv`](cuda_full_model_environment.csv) | GPU, precision, library versions, and full configuration | Reproducibility provenance |

The CPU reference and CUDA attention-operator assets are **not included in
this hand-off directory**. If slides 4–6 are retained, supply their charts and
summaries from the repository's `results/` directory separately. Do not use
the old CPU throughput panel as the headline visual.

`cuda_full_model_comparison.png` is the **main performance visual**. Keep the
operator chart in the appendix or use it briefly to explain why the reference
path was misleading.

## Completed T4 full-model CUDA result

Environment: **Tesla T4** (compute capability 7.5), PyTorch
**2.11.0+cu128**, FLA **0.5.1**, **FP16**. The benchmark uses a 4-layer,
128-dimensional GPT with random token IDs; training uses batch 16 × 64 tokens,
and prefill/decode use batch 8.

All five FLA/reference gates passed: pure GDN, pure KDA, and three recurrent
hybrids. The maximum absolute full-logit difference was **0.00128174** and
cached-logit difference **0.00122070**. This validates reduced-precision kernel
equivalence only; it is not a quality result.

- **Training:** MHA leads at **138,007 tok/s** p50; MLA-style is **129,696
  tok/s**. GDN and KDA are **32,946** and **21,303 tok/s**.
- **4K cached prefill:** MLA-style leads at **1.25M tok/s**, MHA reaches
  **1.17M**, KDA **660,611**, and GDN **584,688 tok/s**.
- **Cached decode after 4K:** MLA-style is lowest at **2.140 ms/token**;
  MHA is **2.410**, GDN **6.726**, and KDA **6.888 ms/token**.
- **Measured decode cache:** GDN/KDA use **598,016 bytes**, versus MHA's
  **69,206,016 bytes** — about **116× smaller**. MLA-style uses **8,650,752
  bytes**.

The appropriate conclusion is not that recurrent attention is universally
faster. On this complete small decoder and T4, MHA/MLA win the measured speed
metrics; GDN/KDA retain the state-memory advantage. This is one random-token
performance session, not CUDA PPL, matched-parameter evidence, or a production
Qwen/Kimi/DeepSeek result.

## Facts and numbers that may be stated

### CPU full-model reference experiment

- Controlled 4-layer nanoGPT run on CPU; each of nine variants saw
  **1,003,840 training tokens**.
- At 160 tokens, pure Gated DeltaNet (GDN) reached validation PPL **5.91** with
  **82.0 KiB** persistent state; pure KDA reached PPL **6.15** with the same
  **82.0 KiB** state. MHA used **640.0 KiB** and reached PPL **8.44**.
- The reported recurrent training rates were GDN **2,954 tok/s** and KDA
  **2,857 tok/s**, against MHA **26,939 tok/s**. This gap is caused by the
  sequential PyTorch/Python reference scan, which has no fused chunkwise
  kernel.
- At 128K context, the KDA + MLA-style hybrid storage law gives **16.06 MiB**
  state versus **64.00 MiB** for four MLA-style layers: **3.99× smaller**.
  This is an exact storage-layout projection, **not** a 128K quality or latency
  measurement.

### CUDA chunkwise kernel microbenchmark

Environment: **Tesla T4**, compute capability 7.5, PyTorch 2.11.0+cu128,
Flash Linear Attention (FLA) 0.5.1, FP16, batch 8, 4 heads × head dimension 32.

- At 64 tokens of prefill, FLA chunkwise is **21.1×** faster than the GDN
  reference scan and **13.9×** faster than the KDA reference scan.
- At 64 tokens for the attention operator’s forward + backward pass,
  chunkwise is **21.9×** faster for GDN and **21.1×** faster for KDA than their
  reference paths.
- At 4K prefill, MHA SDPA is still fastest at **7.64M tok/s**. GDN chunkwise
  reaches **4.19M tok/s** (1.82× below MHA); KDA chunkwise reaches
  **5.96M tok/s** (1.28× below MHA).
- After a 4K-token prefix, p50 decode latency is **0.288 ms/token** for MHA,
  **0.278 ms/token** for GDN, and **0.240 ms/token** for KDA. KDA is therefore
  **16.5% lower latency** than MHA (not “16.5 percentage points” and not
  “19.8% lower”).

## Recommended 8-slide story

Keep the main deck to 7–8 minutes. One conclusion per slide. Use short text,
large diagrams/plots, and a small evidence footer on every result slide.

| # | Slide title | Main message | Required visual and footer |
|---:|---|---|---|
| 1 | **Efficient Transformer Systems** | I studied the quality–state–speed trade-off of attention and recurrent token mixers in one controlled decoder. | Minimal title slide. Subtitle: “Memory efficiency is measurable; realized speed depends on the kernel.” |
| 2 | **Why the KV cache becomes a systems problem** | Standard attention keeps per-token K/V state; shared, latent, and recurrent alternatives alter that scaling. | Simple cache-growth diagram or `hybrid_architectures.png`. Mark architecture facts as literature/background. |
| 3 | **A controlled implementation study** | I implemented nine token mixers/schedules in the same 4-layer nanoGPT backbone and held the data schedule fixed. | `hybrid_architectures.png` or a clean redraw. Footer: “CPU full-model reference experiment; 1,003,840 tokens/variant.” |
| 4 | **Memory efficiency is real** | Recurrent GDN/KDA state is fixed; the KDA+MLA hybrid reaches 16.06 MiB at 128K versus 64.00 MiB for MLA-style. | Left half of `hybrid_state_scaling.png`; retain its “projection” disclaimer. |
| 5 | **The first speed result was an implementation warning** | The CPU chart shows the reference scan overhead, not an architecture verdict. | Right throughput panel of `hybrid_quality_efficiency.png`, visibly labelled “CPU full-model reference path / token-by-token Python scan.” Mention PPL/state from the first panel only if legible. |
| 6 | **Chunkwise CUDA kernels explain the CPU anomaly** | The operator-level FLA result removes 14–22× of the short-window reference-scan penalty. | `cuda_chunkwise_operator_benchmark.png`. Footer: “Tesla T4, FP16, B=8, H=4, D=32; attention operator only.” |
| 7 | **All nine models: CUDA full-model comparison** | On this T4, MHA leads training while MLA-style leads 4K prefill and decode; recurrent GDN/KDA retain much smaller decode state. | `cuda_full_model_comparison.png`. Footer: “Tesla T4, FP16; 5/5 FLA checks passed; random-token performance microbenchmark, no CUDA PPL claim.” |
| 8 | **Conclusions, limitations, next test** | State savings and kernel sensitivity are supported; multi-seed quality and long-context quality remain next work. | Three evidence boxes: measured / projection / next test. Add programme/research fit only if this is an admissions deck. |

For a shorter six-slide technical deck, combine slides 2+3 and slides 7+8; do
not remove slide 6 or the evidence boundary.

## Slide-specific speaker guidance

### Slide 4 — memory

Say: “The x-markers are measured states at 160 tokens. The longer-context lines
are exact byte-count projections of the implemented cache layout, not an
assertion that we measured 128K quality or latency.”

### Slide 5 — CPU reference path

Say: “This surprising 9× gap is the result that changed the experiment. GDN
and KDA are evaluated by a sequential Python reference recurrence, while MHA
uses optimized attention primitives. It diagnoses an execution-path mismatch.”

Do **not** say: “Linear attention is 9× slower than attention.”

### Slide 6 — CUDA kernels

Say: “With the appropriate CUDA kernels, the short-window reference-scan
penalty is mostly removed: roughly 14–22×. MHA remains faster for 4K prefill on
this T4 microbenchmark, but KDA is close and has the lowest cached decode
latency.”

Do **not** say: “KDA is universally faster,” “this measures a full LLM,” or
“CUDA proves the CPU quality result.”

### Slide 7 — full-model CUDA comparison

Say: “All nine implemented mixers now share one full-decoder CUDA timing scope.
On this Tesla T4, MHA leads the 64-token training step, while MLA-style leads
4K prefill and cached decode. GDN and KDA instead demonstrate the smallest
recurrent state. This measures random-token performance, not a new quality
run.” The slide must carry **Tesla T4, FP16, train batch 16, inference batch
8**, and **5/5 FLA/reference checks passed**.

## Visual and editorial rules

- Deck language: **English**. Explain abbreviations at first appearance:
  multi-head attention (MHA), Gated DeltaNet (GDN), Kimi Delta Attention (KDA),
  and multi-head latent attention (MLA-style).
- Preserve units exactly: `tok/s`, `ms/token`, `KiB`, `MiB`, `128K tokens`.
- If redrawing charts, preserve log axes where present and copy the hardware and
  scope footnotes. Never interpolate extra data points.
- Do not use the word “fused” for the reference scan. The relevant terms are
  **chunkwise kernel** for GDN/KDA prefill/training and **fused recurrent
  kernel** for cached decode.
- Do not place CPU and CUDA numeric bars on a shared axis. Do not compare their
  absolute tok/s values.
- Make “measured”, “exact storage projection”, and “proposed next work” visibly
  distinct (for example: filled marks, dashed lines, and outlined boxes).
- Do not claim matched parameter counts, production throughput, long-context
  accuracy, MoE behavior, or results for the full external architectures.

## Final conclusion slide copy

Use this wording, or a faithful shorter version:

> Recurrent linear attention offers a genuine state-memory advantage. The first
> CPU benchmark revealed that an unoptimized reference scan can obscure that
> advantage. The completed CUDA experiment shows the more nuanced result:
> optimized kernels enable a fair comparison, but on this T4 the full decoder
> still favors MHA/MLA for speed while GDN/KDA retain a much smaller cache. The
> next test is multi-seed quality training and scaling in one backbone.

## Appendix candidates

1. Full `hybrid_quality_efficiency.png` with all three CPU panels.
2. Full `hybrid_state_scaling.png`, including the 1 GiB concurrency projection.
3. Method details: shared corpus/seed/optimizer and the nine variants.
4. CUDA environment and raw CSV provenance:
   `chunkwise_environment.csv`, `chunkwise_prefill.csv`,
   `chunkwise_training.csv`, `chunkwise_decode.csv`.
5. Full-model CUDA provenance: all five `cuda_full_model_*.csv` files,
   including verification, environment, training, prefill, and decode.

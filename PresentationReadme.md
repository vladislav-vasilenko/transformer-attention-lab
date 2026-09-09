# Skoltech Admission presentation guide

This file is the source-of-truth brief for an agent assembling the final
Skoltech MSc **Artificial Intelligence and Machine Learning** admission deck.
The final deliverable must be a PDF, prepared before **07:00 GMT+3 on 23 July
2026**.

## 1. Verdict: is this research suitable?

**Yes.** Use it as the main “past work” project. It is a completed engineering
research experiment rather than a claimed academic publication. It directly
demonstrates the qualities the committee asks for:

- a concrete systems problem and its practical importance;
- falsifiable hypotheses;
- an implementation built personally rather than only a library comparison;
- controlled training and inference measurements;
- exact figures and reproducible evidence;
- a useful negative result about unoptimized kernels;
- limitations and a credible next experiment.

The safe positioning is:

> This is an engineering research project rather than a completed academic
> study. I implemented modern Transformer and recurrent token-mixer components
> from scratch, placed them in one controlled nanoGPT backbone, and measured
> their quality–memory–latency trade-offs. The reference experiment is complete;
> multi-seed long-context GPU benchmarking is the next stage.

Do **not** describe this work as a new attention architecture, a full
DeepSeek-V2/Qwen3-Next/Kimi Linear/Kimi K3 reproduction, a publication, or a production
128K benchmark.

## 2. Presentation constraints

- Target duration: **7:00–7:30**. Never exceed 8 minutes.
- Main deck: **8 slides maximum**, including the title and program-fit slide.
- Appendix: up to 5 backup slides for questions; do not present them by default.
- Language: English, unless the interview instructions explicitly change.
- Format: editable 16:9 source plus exported PDF.
- Audience: scientifically literate but not necessarily specialized in LLM
  inference.
- One conclusion per slide; prefer one graph over a dense table.
- The project section should occupy about 4.5–5 minutes.
- Achievements/CV summary must fit one slide and 45–60 seconds.
- Do not repeat the transcript, job history, or all projects already visible in
  the application.
- Every remote repository URL, employer result, product metric, award, and
  entrepreneurship claim must be verified against the submitted application.
  Never invent missing numbers. Use `{VERIFY ...}` placeholders while drafting
  and remove them before export.

The invitation contains both a 10–20 minute total-interview estimate and a
15–25 minute recommendation. A 7–7.5 minute deck is safe under either version
and leaves time for technical questions.

## 3. One-sentence narrative

Use this story throughout the deck:

> Autoregressive LLMs trade model quality for a token-growing inference cache;
> I tested whether KV sharing, latent compression, recurrent state, and hybrid
> layers change that trade-off inside the same small decoder—and found that
> memory efficiency is measurable, while speed depends on the execution kernel.

The presentation is not “I studied several attention mechanisms.” It is:

> I asked a systems question, implemented nine alternatives, controlled the
> comparison, measured the result, found both an advantage and a bottleneck,
> and designed the next experiment.

## 4. Evidence classes that must remain visually distinct

Every slide must label evidence as one of the following:

1. **Measured experiment** — training, PPL, latency, or occupied state obtained
   during an executed run.
2. **Exact storage-law projection** — arithmetic applied to the measured cache
   object layout: `fixed_bytes + bytes_per_token × context`.
3. **Literature/background** — architecture facts taken from primary papers or
   official implementation pages.
4. **Proposed next work** — not yet completed.

Recommended slide notation:

- filled point or `X`: measured;
- dashed line: exact storage projection;
- outline box: proposed next experiment;
- small source footer: local evidence file or primary external source.

Never use the word “measured” for a dashed 128K point.

## 5. Recommended 8-slide structure

| # | Slide title | Time | Main admissions question answered | Primary visual |
|---:|---|---:|---|---|
| 1 | Efficient Transformer Systems | 0:20 | What is the project and the result? | One clean title statement |
| 2 | One-minute profile | 0:45 | What distinguishes me? Programming/innovation? | Three verified achievements |
| 3 | Problem and hypotheses | 0:55 | What problem, why important, what did I expect? | Cache-growth concept |
| 4 | What I built and how I tested it | 1:05 | What did I do personally? Why this method? | Architecture/schedule diagram |
| 5 | Quality–memory–speed result | 1:15 | What happened in the completed experiment? | Three-panel result graph |
| 6 | What changes at long context? | 0:55 | Where is the systems advantage? | Linear + log state graph |
| 7 | Conclusions, limitations, next test | 0:55 | What did I learn and what remains unknown? | Three conclusion boxes |
| 8 | Why Skoltech and what I will research | 1:00 | Educational fit, groups, project, why accept me? | Courses + research path |
|  | **Total** | **7:10** |  |  |

### Slide 1 — Efficient Transformer Systems

**Title:** `Efficient Transformer Systems: quality, memory, and realized speed`

**Subtitle:**

> A controlled nanoGPT study of MHA, GQA, MQA, MLA, Gated DeltaNet, Kimi Delta
> Attention, and 3:1 hybrid decoders

Add name, `Skoltech MSc Artificial Intelligence and Machine Learning`, and one
verified repository/QR link. If the public GitHub URL is not known, leave the QR
out rather than showing a local filesystem path.

Opening sentence:

> My question was not which attention equation looks most efficient, but which
> quality–memory–latency point its actual implementation realizes.

Do not spend time defining every abbreviation on this slide.

### Slide 2 — One-minute profile

This is the only biography/achievements slide. Use at most three bullets:

1. one verified engineering/product achievement with a measured outcome;
2. one sentence covering commercial LLM/RAG/voice/generative-audio experience;
3. one sentence on open code, Transformer implementation, and distributed
   training experience (DDP plus ongoing FSDP/DeepSpeed work).

Add one compact line for entrepreneurship/innovation. Prefer a verified product
decision, user result, revenue/cost improvement, or prototype-to-production
story from the submitted CV. If none is defensible, say “product engineering
experience” and do not manufacture a startup claim.

Suggested strongest-suit sentence:

> My strongest suit is connecting first-principles ML understanding with
> measurable systems engineering and product constraints.

The slide agent must obtain exact names, dates, metrics, and the public code URL
from the submitted CV/application; this repository does not contain enough
evidence to invent them.

### Slide 3 — Problem and hypotheses

Explain the problem for a broad engineering audience:

> During autoregressive generation, standard attention stores keys and values
> for every previous token in every layer. That avoids recomputation, but memory
> grows with context and limits sequence length or concurrent requests.

Present only four hypotheses:

- **H1 — KV sharing:** GQA and MQA should reduce the token-growing cache slope
  in proportion to the number of KV heads.
- **H2 — Latent compression:** MLA-style attention should reduce per-token
  state further, with a possible quality cost.
- **H3 — Recurrent state:** pure GDN/KDA should keep a fixed persistent state
  independent of context length.
- **H4 — Hybrid compromise:** a 3:1 recurrent/full-attention schedule should be
  `fixed + one-quarter token-growing cache`, but memory savings do not
  automatically imply speed.

Add a small mechanism-lineage strip under the diagram:

- `MLA-style -> DeepSeek-V2; Kimi K2/K2.5 decoder lineage`;
- `GDN x3 -> gated attention -> Qwen3-Next-inspired`;
- `KDA x3 -> MLA -> Kimi Linear-inspired`;
- `KDA x3 -> gated MLA -> Kimi K3-inspired*`.

The asterisk must say that the K3 label follows the launch diagram available on
22 July 2026; the exact full technical specification was not yet the basis of
this prototype.

Use a simple architecture ladder, not paper equations. If a diagram is needed,
crop or reuse:

- [`results/attention_architectures.png`](results/attention_architectures.png)
  for MHA/GQA/MQA/MLA;
- [`results/hybrid_architectures.png`](results/hybrid_architectures.png) for GDN,
  KDA, and three 3:1 schedules, including a gated-versus-ungated MLA-anchor
  ablation.

Literature footnotes:

- [MQA](https://arxiv.org/abs/1911.02150)
- [MLA / DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [Linear Attention](https://arxiv.org/abs/2006.16236)
- [Gated DeltaNet](https://arxiv.org/abs/2412.06464)
- [Gated full attention](https://arxiv.org/abs/2505.06708)
- [Kimi Linear / KDA](https://arxiv.org/abs/2510.26692)
- [Official Kimi K3 launch architecture](https://www.kimi.com/blog/kimi-k3)
- [Official Qwen3-Next architecture description](https://qwen.ai/blog?id=e34c4305036ce60d55a0791b170337c2b70ae51d)

### Slide 4 — What I built and how I tested it

The slide must emphasize personal contribution, not paper summaries.

Use first-person verbs:

- implemented MHA, GQA, MQA, and MLA-style caches in one nanoGPT decoder;
- implemented GDN/KDA recurrence, causal short convolutions, decay/write/output
  gates, and explicit recurrent inference states;
- assembled `GDN ×3 → gated MHA`, `KDA ×3 → MLA-style`, and
  Kimi-K3-inspired `KDA ×3 → gated MLA-style` schedules;
- built deterministic full-epoch training and cached/recomputed inference
  measurement;
- added numerical equivalence, causality, cache-layout, and gradient tests;
- recorded configuration, corpus hash, raw timing repeats, and curves in JSON.

Show the 4-layer hybrid diagram from
[`results/hybrid_architectures.png`](results/hybrid_architectures.png). Do not
show a screenshot of source code in the main deck.

Controlled Experiment B protocol:

- 9 variants;
- 4 layers, width 128, 4 query heads;
- training context 64, model window 192, batch 16;
- same corpus split/hash, seed, optimizer, batch order, evaluation tokens, and
  non-attention initialization;
- 15,685 complete non-overlapping windows;
- 981 batches, including the final partial batch;
- **1,003,840 training tokens per variant**;
- parameter counts reported but not matched.

State the FFN control explicitly:

> Every variant uses the same dense GPT-2 MLP. Holding MoE out of scope isolates
> the token mixer and its persistent state; the measured PPL and throughput are
> therefore prototype results, not full-model Qwen/Kimi/DeepSeek performance.

Say explicitly:

> Every complete training window was visited exactly once. This is a full-epoch
> comparison, not a few randomly selected batches.

Detailed sources:

- protocol implementation:
  [`research/experiment.py`](research/experiment.py);
- nine-variant orchestration and metadata:
  [`research/hybrid_experiment.py`](research/hybrid_experiment.py);
- decoder/schedules: [`research/model.py`](research/model.py);
- recurrent operators:
  [`research/linear_attention.py`](research/linear_attention.py).

### Slide 5 — Quality–memory–speed result

Use the complete three-panel figure:

[`results/hybrid_quality_efficiency.png`](results/hybrid_quality_efficiency.png)

The spoken interpretation must follow this order:

1. **Left panel:** pure GDN is the lower-left PPL/state point in this run:
   validation PPL `5.91`, measured persistent state `82.0 KiB` at 160 tokens.
2. **Middle panel:** it is not the fastest implementation: cached latency is
   `3.701 ms/token`, versus `0.388 ms/token` for MHA.
3. **Right panel:** the readable recurrent Python scan trains at `2,954 tok/s`,
   about `9.1×` below MHA's `26,939 tok/s`.

Use this conclusion:

> GDN won the two-metric PPL/state view in this single run, while losing realized
> CPU speed. The experiment therefore separates representation efficiency from
> kernel efficiency.

Add the gated-MLA ablation as a compact callout, not a new headline: ungated
`KDA x3 + MLA` versus K3-inspired `KDA x3 + gated MLA` produced PPL
`6.24 -> 6.23`, cached latency `2.763 -> 2.838 ms/token`, and exactly the same
`81.5 KiB` state at 160 tokens. The gate adds `16,384` current-token projection
parameters but no persistent-state bytes. With one seed, this is evidence of
near-equivalence in this run, not proof that gating never helps.

Do not call PPL “answer quality.” It is held-out next-character predictive
quality on the common Shakespeare split. It supports comparison inside this
experiment, not a claim about instruction following or human preference.

Do not attribute the GDN/KDA PPL gap only to scalar versus channel-wise decay:
the complete blocks also use different output gates, and their parameter counts
are not matched.

The full nine-row table belongs in an appendix, not on the main slide:

| Variant | Parameters | Val PPL ↓ | Train tok/s ↑ | Cached p50 ↓ | State @160 ↓ |
|---|---:|---:|---:|---:|---:|
| MHA | 821,632 | 8.44 | 26,939 | 0.388 ms/token | 640.0 KiB |
| GQA | 756,096 | 8.48 | 26,310 | 0.426 ms/token | 320.0 KiB |
| MQA | 723,328 | 8.84 | 27,057 | 0.415 ms/token | 160.0 KiB |
| MLA-style | 739,968 | 9.36 | 26,739 | 0.429 ms/token | 80.0 KiB |
| GDN pure | 897,952 | 5.91 | 2,954 | 3.701 ms/token | 82.0 KiB |
| KDA pure | 896,400 | 6.15 | 2,857 | 3.696 ms/token | 82.0 KiB |
| GDN + gated MHA | 895,256 | 6.11 | 3,772 | 2.723 ms/token | 221.5 KiB |
| KDA + MLA-style | 857,292 | 6.24 | 3,723 | 2.763 ms/token | 81.5 KiB |
| K3-inspired KDA + gated MLA-style | 873,676 | 6.23 | 3,683 | 2.838 ms/token | 81.5 KiB |

Machine-readable evidence:
[`results/hybrid_linear_attention_cpu.json`](results/hybrid_linear_attention_cpu.json).

### Slide 6 — What changes at long context?

Use:

[`results/total_state_vs_context_gb.png`](results/total_state_vs_context_gb.png)

Explain the two panels:

- the left panel preserves absolute GB scale and shows capacity-relevant gaps;
- the right log-GB panel reveals the constant pure GDN/KDA line that otherwise
  looks like zero.

Evidence boundary:

- `X` markers at total contexts **64, 96, and 160** are measured occupied state
  tensors;
- the trained model window ends at 192 tokens;
- dashed points through 128K are exact storage-law projections, not executed
  128K latency, PPL, retrieval accuracy, or peak-device-memory measurements.

Use only three callouts:

- MHA at 128K: `512 MiB` (`0.537 GB` decimal);
- KDA/MLA hybrid at 128K: `16.06 MiB`;
- pure GDN/KDA: `0.080 MiB`, constant in this layout.

Optional spoken comparison:

> The KDA/MLA hybrid is 3.99× smaller than four MLA-style layers at 128K and
> approaches a 75% asymptotic reduction, because only one of four layers retains
> a token-growing latent cache.

Immediately add:

> Fixed bytes are a capacity result, not proof that the fixed state preserves
> arbitrary information at long range; collisions require a retrieval test.

Exact projection evidence:

- [`results/hybrid_state_projection.csv`](results/hybrid_state_projection.csv)
- [`research/plot_hybrid.py`](research/plot_hybrid.py)

### Slide 7 — Conclusions, limitations, and next test

Use three columns or three horizontal blocks.

**What the experiment established**

- KV sharing and latent compression reduce the per-token cache slope exactly as
  expected.
- Pure recurrent layers replace token-growing cache with fixed state.
- Hybrids are fixed-plus-linear, not constant-memory.
- memory savings did not produce speed in the unoptimized reference path.

**What it did not establish**

- no universal architecture-quality ranking;
- no 128K model-quality or latency result;
- no production Qwen3-Next/Kimi/DeepSeek reproduction;
- no instruction-answer quality result;
- no matched-parameter, multi-seed, or optimized-GPU benchmark.

**Next decisive experiment**

1. MQAR/associative recall and needle-style retrieval versus context length;
2. 3–5 seeds and matched-parameter controls;
3. optimized FLA/FlashKDA and MLA CUDA BF16 kernels;
4. synchronized prefill/decode p50/p95 plus actual peak GPU memory;
5. hybrid-ratio sweep: 1:1, 3:1, 7:1.

**Proposed Experiment C — compressed/sparse attention and residual routing**

Keep the production-lineage labels precise: DeepSeek-V3/R1 is the MLA control;
CSA/HCA and mhC belong to a separate DeepSeek-V4-style future track and were
not implemented in the completed experiment. Use a two-factor design rather
than adding one more bar to the current cache plot:

1. token-mixer factor: `MLA vs CSA/HCA`, with identical decoder depth, width,
   MLP, optimizer, data exposure, and positional treatment where possible;
2. residual-routing factor: `plain residual vs mhC`, crossed with the
   token-mixer factor;
3. sparse-retrieval quality: MQAR / associative-recall accuracy and recall of
   the required context item versus sequence length and distractor count;
4. selection behavior: fraction of KV blocks selected, miss rate, and recall
   conditional on retrieval distance;
5. search-system cost: indexer memory, index construction/update latency, and
   lookup latency;
6. realized I/O and capacity: KV bytes actually read, peak device memory, and
   maximum feasible batch/context;
7. mhC ablation: quality, optimization stability, communication/activation
   cost, and latency with identical attention settings.

The fairness rule is explicit: CSA/HCA must not be judged only by a smaller
resident cache. The comparison must include the price of search and the risk of
omitting the KV block that contains the needed evidence.

The key transition sentence is:

> The completed experiment answers the storage question. The next experiment
> must answer whether the compressed state retains the right information and
> whether an optimized kernel converts the algorithmic advantage into speed.

### Slide 8 — Why Skoltech and what I will research

Use verified 2026 sources:

- [AIML program page](https://msc.skoltech.ru/data-science)
- [Official 2026 curriculum PDF](https://back.skoltech.ru/storage/app/media/EDUCATION/Curriculum%20msc/AY%202026-2027/AI%20and%20ML/Cohort%202026/msc-curriculum-ds-ai-ml-cohort-2026.pdf)
- [Skoltech Artificial Intelligence Center](https://www.skoltech.ru/en/center/ai/)

Recommended courses and the reason for each:

- **MA030556 — Transformers and Large Language Models:** connect this
  operator-level prototype to current LLM training and deployment research.
- **MA060433 — Models of Sequential Data:** develop a principled view of
  recurrent state, attention, and state-space alternatives.
- **MA060024 — Numerical Linear Algebra:** deepen the low-rank/tensor and
  numerical-stability foundations behind MLA and recurrent state.
- **MA060776 — Optimization Methods in Artificial Intelligence:** design
  matched, stable, reproducible training comparisons.
- Optional engineering link: **MA030406 — Foundations of Software Engineering
  for AI** for production-quality experimental systems.

Do not list all courses. Show three or four connected by arrows to the proposed
project.

Relevant laboratories/directions:

- [Laboratory of Computational Intelligence](https://www.skoltech.ru/en/laboratories/computational-intelligence-lab):
  numerical and tensor methods, computational complexity, robust and fast neural
  architectures;
- [Natural Language Processing Laboratory](https://www.skoltech.ru/en/laboratories/natural-language-processing-lab):
  neural language technology and applied NLP evaluation;
- program research directions **Transformers in large language models**,
  **Parallel algorithms for AI**, **AI & supercomputing**, and **Generative AI**.

Frame these as interests, not promises of supervision. Do not claim contact or
agreement with a professor unless it actually happened.

Potential MSc project:

> **Long-context hybrid decoders under a fixed memory budget.** Compare
> attention, recurrent, and hybrid token mixers with matched parameters and
> optimized kernels; measure retrieval accuracy, PPL, latency, energy, and peak
> memory across context and batch size; then derive an adaptive layer schedule
> for a deployment constraint.

Practical relevance:

- more concurrent requests on fixed hardware;
- lower inference memory and potentially lower energy;
- explicit quality/capacity trade-offs instead of architecture marketing;
- applicable to RAG, long-document assistants, and on-device/private inference.

Closing “why accept me” statement:

> I bring production experience, an open and reproducible implementation, and
> the habit of turning an architecture claim into a controlled systems
> experiment. Skoltech gives me the mathematical depth, research supervision,
> and high-performance computing environment needed to turn this prototype into
> rigorous long-context research.

## 6. Experiment-stage map for the slide-building agent

### Stage A — Research question

Primary wording:

> Under one shared 4-layer nanoGPT training protocol, how do MHA, GQA, MQA,
> MLA-style attention, pure GDN/KDA, and three 3:1 hybrids trade validation PPL,
> persistent inference-state memory, and realized reference-path latency?

Sources:

- [`README.md`](README.md), “Experiment B”;
- [`results/hybrid_linear_attention_cpu.json`](results/hybrid_linear_attention_cpu.json),
  `question` field;
- executed
  [`02_gated_deltanet_kda_hybrids.ipynb`](09_Efficient_Transformer_Systems/02_gated_deltanet_kda_hybrids.ipynb).

### Stage B — Hypotheses

| Hypothesis | Test | Outcome |
|---|---|---|
| GQA/MQA reduce KV slope with fewer KV heads | inspect measured cache and bytes/token | supported: 4,096 → 2,048 → 1,024 bytes/token |
| MLA reduces per-token state further | measured latent cache | supported: 512 bytes/token in 4-layer Experiment B |
| pure GDN/KDA state is independent of context | cache object layout + 64/96/160 measurements | supported: fixed 83,968 bytes |
| 3:1 hybrid grows only through its anchor | mixed-cache layout | supported: `62,976 + slope × T` |
| gating the MLA anchor changes compute/quality, not state law | ungated/gated KDA hybrid ablation | same `62,976 + 128T` bytes; PPL 6.24 vs 6.23 |
| lower asymptotic memory means immediate speed | CPU training/inference timing | not supported by the unoptimized path |

### Stage C — Chosen metrics

| Dimension | Metric | Why it was chosen | Interpretation boundary |
|---|---|---|---|
| Predictive quality | validation loss and PPL | same tokenizer/split enables controlled comparison | character-level next-token quality, not answer quality |
| Capacity | occupied persistent-state bytes | measures the actual cache/state tensors | excludes weights, activations, workspace, allocator overhead |
| Scaling | fixed bytes + bytes/token | exposes asymptotic context dependence | projections are not model executions |
| Inference | prefill, cached decode p50/p95, recompute p50/p95 | separates cache benefit from full-prefix recomputation | recurrent path is an unoptimized reference kernel |
| Training systems | tokens/second | exposes implementation cost | not a production GDN/KDA speed claim |
| Fairness/accounting | total and token-mixer parameters | makes unmatched capacity visible | not parameter-matched |
| Correctness | cached/full logits, chunk equivalence, causality, gradients | rejects invalid speed/memory comparisons | numerical unit tests, not statistical confidence |

### Stage D — Method

The agent should summarize, not reproduce code:

1. common corpus, tokenizer, split hash, seed, and batches;
2. shared four-layer decoder and identical non-attention initialization;
3. nine token-mixer variants;
4. one exact full epoch for each variant;
5. held-out PPL checkpoints;
6. cached and full-recompute inference on the same tokens;
7. direct inspection of active state tensors;
8. tests before accepting results.

### Stage E — Completed measurements

Use the saved evidence, not manually retyped values:

- raw Experiment B:
  [`results/hybrid_linear_attention_cpu.json`](results/hybrid_linear_attention_cpu.json);
- generated summary:
  [`results/HYBRID_EXPERIMENT_SUMMARY.md`](results/HYBRID_EXPERIMENT_SUMMARY.md);
- training curves:
  [`results/hybrid_training_curves.png`](results/hybrid_training_curves.png);
- cached/recomputed latency:
  [`results/hybrid_inference_latency.png`](results/hybrid_inference_latency.png);
- quality/memory/reference-speed synthesis:
  [`results/hybrid_quality_efficiency.png`](results/hybrid_quality_efficiency.png).

### Stage F — Exact projection

Use only for the storage/capacity claim:

- [`results/total_state_vs_context_gb.png`](results/total_state_vs_context_gb.png);
- [`results/hybrid_state_scaling.png`](results/hybrid_state_scaling.png);
- [`results/hybrid_state_composition.png`](results/hybrid_state_composition.png);
- [`results/hybrid_state_projection.csv`](results/hybrid_state_projection.csv).

### Stage G — Conclusions

Safe conclusions:

1. compact cache/state layouts produced the expected memory laws;
2. pure recurrent state stayed fixed across the executed contexts;
3. hybrid layers retained a smaller but nonzero token-growing component;
4. the generic/reference implementation did not convert memory savings into
   speed;
5. long-context retrieval and optimized-kernel performance remain open.

Unsafe conclusions:

- “GDN is universally better than attention”;
- “KDA is worse than GDN”;
- “the model was benchmarked at 128K”;
- “the experiment measured total GPU memory”;
- “this is Qwen3-Next/Kimi Linear/Kimi K3/DeepSeek-V2”;
- “PPL proves better answers”;
- “linear attention is slow.”

## 7. Source navigation order

An agent building the deck should inspect files in this order:

1. [`docs/ADMISSION_BRIEF.md`](docs/ADMISSION_BRIEF.md) — concise narrative and
   exact admission-safe wording.
2. [`README.md`](README.md) — complete project story, both experiments, and
   evidence boundaries.
3. Executed
   [`02_gated_deltanet_kda_hybrids.ipynb`](09_Efficient_Transformer_Systems/02_gated_deltanet_kda_hybrids.ipynb)
   — full Experiment B walkthrough; default presentation source.
4. Executed
   [`01_mha_gqa_mla_comparison.ipynb`](09_Efficient_Transformer_Systems/01_mha_gqa_mla_comparison.ipynb)
   — completed Experiment A baseline; use mainly for backup.
5. [`results/hybrid_linear_attention_cpu.json`](results/hybrid_linear_attention_cpu.json)
   — numerical source of truth.
6. [`results/hybrid_state_projection.csv`](results/hybrid_state_projection.csv)
   — exact long-context state table.
7. [`research/`](research/) and [`tests/`](tests/) — implementation and
   verification evidence for technical questions.

Do not use `smoke*.json`, `smoke_plots/`, `.cache/`, or temporary files in the
presentation.

## 8. Code and test evidence for Q&A

| Question | Local evidence |
|---|---|
| Where are MHA/GQA/MQA/MLA implemented? | [`research/attention.py`](research/attention.py) |
| Where are GDN/KDA implemented? | [`research/linear_attention.py`](research/linear_attention.py) |
| Where are hybrid layer schedules assembled? | [`research/model.py`](research/model.py) |
| Where is full-epoch coverage implemented? | [`research/experiment.py`](research/experiment.py) |
| Where is Experiment B orchestrated? | [`research/hybrid_experiment.py`](research/hybrid_experiment.py) |
| How are plots/projections generated? | [`research/plot_hybrid.py`](research/plot_hybrid.py) |
| How is cache correctness tested? | [`tests/test_attention.py`](tests/test_attention.py), [`tests/test_model.py`](tests/test_model.py) |
| How is recurrent correctness tested? | [`tests/test_linear_attention.py`](tests/test_linear_attention.py) |
| How is full-epoch/state-law correctness tested? | [`tests/test_experiment.py`](tests/test_experiment.py), [`tests/test_hybrid_experiment.py`](tests/test_hybrid_experiment.py) |

Current verification status: formatting and static checks pass; **27/27 tests
pass**; Experiment B notebook has **6/6 executed code cells and zero error
outputs**.

## 9. Likely committee questions and defensible answers

### “Why did GDN get much better PPL?”

> It is an observation from one small run, not an architecture theorem. The
> recurrent variants have different parameter counts and output gates, and only
> one seed was run. I report the result but do not attribute it solely to the
> decay rule. A matched-parameter multi-seed ablation is required.

### “Why is linear attention much slower on your graph?”

> My implementation uses a transparent token-by-token Python recurrence so I
> could verify the equations and cache state. MHA/MLA use vectorized PyTorch
> matrix operations. Therefore the result exposes a reference-kernel bottleneck,
> not the optimized chunkwise performance claimed by the papers. The next
> systems experiment uses FLA/FlashKDA CUDA kernels.

### “Did you really run 128K?”

> No. I executed state tensors at 64, 96, and 160 total tokens. The 128K points
> are exact byte projections from those implemented cache layouts and are shown
> dashed. I do not claim 128K latency, PPL, or retrieval accuracy.

### “Does fixed memory mean unlimited context?”

> It means persistent-state bytes do not grow with context. Information capacity
> remains finite, so collisions or forgetting may reduce retrieval quality. That
> is why MQAR is my next quality experiment.

### “Why use Shakespeare and character PPL?”

> It makes a small from-scratch experiment reproducible and lets all variants use
> the same tokenizer and held-out data. It is sufficient for a controlled
> engineering prototype, but not for answer quality or long-range retrieval.

### “Did you train on the whole dataset?”

> For Experiment B, yes in the precise epoch sense: every complete
> non-overlapping 64-token training window was used exactly once—15,685 windows,
> 981 batches, and 1,003,840 training tokens per model. Thirteen trailing tokens
> could not form a full input/target window.

### “Why are parameter counts unequal?”

> I held depth, width, data, optimizer, token budget, and non-attention
> initialization fixed and reported architecture-dependent parameters. That
> isolates the implemented systems at common dimensions, not a matched-capacity
> scientific ranking. Matched-parameter controls are a planned follow-up.

### “Is your MLA the DeepSeek implementation?”

> It is an MLA-style latent KV-cache experiment inside GPT-2. It intentionally
> omits decoupled RoPE, absorbed projections, MoE, and fused deployment kernels.

### “What was personally yours?”

> The interchangeable decoder modules, recurrent equations, mixed cache API,
> training/evaluation harness, inference benchmark, storage accounting,
> correctness tests, plots, and interpretation. The papers supplied the
> mechanisms; my contribution was implementing and testing them in one
> reproducible comparison.

### “What is the strongest result?”

> The most defensible result is not the single-seed PPL ranking. It is the
> verified persistent-state law across nine implemented systems and the finding
> that an algorithmic memory advantage did not automatically become a speed
> advantage on the available execution path.

### “What would you do with Skoltech resources?”

> Convert the reference prototype into a matched, multi-seed, long-context GPU
> study with optimized kernels and retrieval tasks, connecting the Transformers
> and LLM, sequential-model, numerical-linear-algebra, and parallel-AI parts of
> the AIML curriculum.

### “What is the difference between MQA and MQAR?”

> MQA is Multi-Query Attention, an architecture with one shared K/V head. MQAR is
> Multi-Query Associative Recall, an evaluation task for retrieving multiple
> key–value associations. They must not be conflated.

## 10. Appendix slide plan

Keep these after the closing slide and open only when asked:

1. **Full nine-variant result table** — use the table from Slide 5.
2. **Mechanism-to-model mapping** — redraw a simple three-column schematic:
   MLA-style (DeepSeek-V2, Kimi K2/K2.5), GDN/gated-attention hybrid
   (Qwen3-Next), and KDA/gated-MLA hybrid (Kimi K3). Put MoE, AttnRes, vision,
   and production kernels in a separate `outside this experiment` band. Do not
   paste dense third-party/watermarked architecture images into the main deck.
3. **Recurrent equations** — use the equation section in the Experiment B
   notebook; show scalar GDN versus channel-wise KDA decay.
4. **Correctness gates** — cached/full equivalence, arbitrary chunks, causality,
   finite gradients, fixed state, full-epoch coverage; show `27/27 tests`.
5. **Evidence boundary** — measured versus projected, optimizer deviation,
   parameter mismatch, one seed, and resumed timing-session caveat.

## 11. Design instructions

- Use the Skoltech visual identity if an approved template is available.
- Prefer a light background, large graph labels, and one accent color for the
  candidate's contribution.
- Preserve original graph colors and legends across slides.
- Never stretch or crop away footnotes that distinguish measured data from
  projections.
- Do not paste the paper screenshots supplied during exploration into the final
  deck; use original project diagrams and cite primary papers.
- Do not add DeepSeek-V4-style CSA/HCA or mhC architecture art as completed
  evidence: those mechanisms were not implemented or measured here. Name them
  on the limitations/next-work slide as **Proposed Experiment C** and keep the
  dense third-party architecture images out of the main narrative.
- Put sources in a small footer, not a bibliography block over the graph.
- Use at least 24 pt body text and at most six short lines per text region.
- Avoid animations required to understand the result; the exported PDF must be
  self-contained.
- Use rounded values on the slide, exact values in speaker notes/appendix.
- If a graph is too dense, crop one complete panel and retain its axes, units,
  legend, and evidence-boundary caption.

## 12. Admissions requirement coverage matrix

| Required point | Slide |
|---|---:|
| Academic/professional achievements | 2, no more than 45–60 seconds |
| One past-work project | 3–7 |
| Problem and importance | 3 |
| Method and personal contribution | 4 |
| Exact figures and proved facts | 5–6 |
| Outcome and limitations | 7 |
| Programming experience and repository | 2 and 4 |
| Entrepreneurship/innovation | 2, one verified line |
| Why this program fits | 8 |
| Desired courses | 8 |
| Potential project and practical relevance | 7–8 |
| Research groups/directions | 8 |
| Strongest suit / why accept me | 2 and final sentence of 8 |

## 13. Final assembly checklist

- [ ] 8 main slides or fewer.
- [ ] Rehearsed duration between 7:00 and 7:30.
- [ ] Past-work section receives at least 4 minutes.
- [ ] Achievements slide is under one minute and does not reproduce the CV.
- [ ] Every personal/company/entrepreneurship metric is verified.
- [ ] Public code link works without authentication; otherwise omit it.
- [ ] PPL is never called answer quality.
- [ ] 128K points are labeled exact storage projections.
- [ ] GDN/KDA speed is labeled unoptimized reference-path speed.
- [ ] Pure GDN is not described as universally superior.
- [ ] MLA is called MLA-style; hybrids are called inspired schedules.
- [ ] The next MQAR/GPU study is labeled proposed work.
- [ ] Courses are copied from the official 2026 curriculum.
- [ ] Lab/group interest is phrased as interest, not prior agreement.
- [ ] All graphs have readable axes and units in the exported PDF.
- [ ] Source footers and repository link are clickable.
- [ ] PDF opens correctly on another device.
- [ ] Backup copy is available offline before 07:00 GMT+3, 23 July 2026.

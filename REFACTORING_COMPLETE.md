# Research refactoring complete

The original compression notebooks remain available as an exploration log.
The presentation-ready research path is now separated into reusable modules,
tests, raw results, figures, and one executed notebook:

```text
Lesson 3. LLMCompression/
├── 09_Efficient_Transformer_Systems/
│   └── 01_mha_gqa_mla_comparison.ipynb
├── research/
│   ├── attention.py
│   ├── model.py
│   ├── experiment.py
│   ├── benchmark.py
│   └── plot_experiment.py
├── tests/
├── results/
└── docs/ADMISSION_BRIEF.md
```

The refactoring establishes:

- one shared GPT-2/nanoGPT skeleton for MHA, GQA, MQA, and MLA;
- compact preallocated cache implementations;
- correctness tests for cached decoding;
- deterministic training and inference protocols;
- machine-readable results and rendered plots;
- a clear boundary between completed evidence and planned experiments.

Run `make admission-demo` for the non-training verification path or
`make attention-experiment` to repeat the full controlled run.

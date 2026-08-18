# Contributing

Contributions that improve correctness, reproducibility, or experimental scope
are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

## Before opening a pull request

Run formatting, linting, and the correctness suite:

```bash
ruff format --check research tests
ruff check research tests
python -m unittest discover -s tests -v
```

## Experimental evidence

New benchmark results should include:

- the exact command and configuration;
- hardware, software versions, and numerical precision;
- warm-up and repeated timing samples;
- raw machine-readable output;
- a correctness comparison against the relevant reference path;
- a clear distinction between measured values and analytical projections.

Do not combine CPU reference, CUDA operator-only, and CUDA full-model timings
on one numerical axis. Quality claims require an executed training/evaluation
run; random-token systems benchmarks do not produce quality evidence.

## Scope and naming

Architecture names describe mechanism lineage. A contribution must not present
a small controlled component as a reproduction of a complete production model.

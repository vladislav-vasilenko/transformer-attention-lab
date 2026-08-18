# Transformer Attention Lab workflow

.PHONY: help install install-cuda lab format check test attention-experiment attention-plots hybrid-experiment hybrid-plots linear-scan-benchmark cuda-chunkwise-plots cuda-full-model-benchmark cuda-full-model-plots demo

VENV ?= .venv
BASE_PYTHON ?= python3
PYTHON ?= $(VENV)/bin/python
RUFF ?= $(VENV)/bin/ruff
DATA ?= data/input.txt
ATTENTION_RESULTS ?= results/attention_systems_cpu.json
HYBRID_RESULTS ?= results/hybrid_linear_attention_cpu.json
CUDA_RESULTS ?= results/cuda_full_model_run
CACHE_DIR ?= .cache

help:
	@echo "Available targets:"
	@echo "  install                    Create the CPU/dev environment"
	@echo "  install-cuda               Add the optional FLA CUDA dependency"
	@echo "  check                      Check formatting and lint"
	@echo "  test                       Run the correctness suite"
	@echo "  attention-experiment       Run the MHA/GQA/MQA/MLA experiment"
	@echo "  hybrid-experiment          Run the nine-variant CPU experiment"
	@echo "  cuda-full-model-benchmark  Run the CUDA full-model benchmark"
	@echo "  demo                       Verify code and rebuild committed CPU plots"

install:
	"$(BASE_PYTHON)" -m venv "$(VENV)"
	"$(PYTHON)" -m pip install --upgrade pip
	"$(PYTHON)" -m pip install -e ".[dev,notebooks]"

install-cuda: install
	"$(PYTHON)" -m pip install -e ".[cuda]"

lab:
	"$(PYTHON)" -m jupyterlab

format:
	"$(RUFF)" format research tests
	"$(RUFF)" check --fix research tests

check:
	"$(RUFF)" format --check research tests
	"$(RUFF)" check research tests

test:
	"$(PYTHON)" -m unittest discover -s tests -v

attention-experiment:
	"$(PYTHON)" -m research.experiment --data "$(DATA)" --output "$(ATTENTION_RESULTS)" --steps 250 --eval-interval 50 --eval-batches 12 --batch-size 16 --train-context 64 --num-threads 1 --inference-repeats 7

attention-plots:
	mkdir -p "$(CACHE_DIR)/matplotlib"
	MPLCONFIGDIR="$(CACHE_DIR)/matplotlib" XDG_CACHE_HOME="$(CACHE_DIR)" "$(PYTHON)" -m research.plot_experiment "$(ATTENTION_RESULTS)" --output-dir results

hybrid-experiment:
	"$(PYTHON)" -m research.hybrid_experiment --data "$(DATA)" --output "$(HYBRID_RESULTS)" --training-mode full_epoch --epochs 1 --eval-interval 200 --eval-batches 12 --batch-size 16 --train-context 64 --num-threads 1 --inference-repeats 7

hybrid-plots:
	mkdir -p "$(CACHE_DIR)/matplotlib"
	MPLCONFIGDIR="$(CACHE_DIR)/matplotlib" XDG_CACHE_HOME="$(CACHE_DIR)" "$(PYTHON)" -m research.plot_hybrid "$(HYBRID_RESULTS)" --output-dir results

linear-scan-benchmark:
	"$(PYTHON)" -m research.linear_benchmark --output results/linear_scan_benchmark.json --prompt-lengths 64 256 1024 4096 --scan-backend reference compiled --num-threads 1

cuda-chunkwise-plots:
	mkdir -p "$(CACHE_DIR)/matplotlib"
	MPLCONFIGDIR="$(CACHE_DIR)/matplotlib" XDG_CACHE_HOME="$(CACHE_DIR)" "$(PYTHON)" -m research.plot_cuda_chunkwise --input-dir results --output-dir results

cuda-full-model-benchmark:
	"$(PYTHON)" -m research.cuda_full_model_benchmark --output-dir "$(CUDA_RESULTS)" --dtype auto

cuda-full-model-plots:
	mkdir -p "$(CACHE_DIR)/matplotlib"
	MPLCONFIGDIR="$(CACHE_DIR)/matplotlib" XDG_CACHE_HOME="$(CACHE_DIR)" "$(PYTHON)" -m research.plot_cuda_full_model --input-dir "$(CUDA_RESULTS)" --output-dir "$(CUDA_RESULTS)"

demo: check test attention-plots hybrid-plots cuda-chunkwise-plots
	@echo "Verified code, tests, and committed CPU/operator result figures."

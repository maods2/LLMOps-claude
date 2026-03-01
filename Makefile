# ============================================================
# LLM Stack – Makefile
# ============================================================
# Provides ergonomic entry points for all pipeline stages.
# All Python commands set PYTHONPATH=. for module resolution.
# ============================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON := python3
PYTEST := pytest
RUFF := ruff
MYPY := mypy

# Configurable paths
CONFIG_DIR := configs
CHECKPOINT_DIR := checkpoints
DATA_DIR := data
LOG_DIR := logs

# GPU / precision settings
DTYPE ?= bfloat16
DEVICE ?= cuda

.PHONY: help install lint format typecheck test preprocess pretrain sft lora dpo \
        evaluate merge serve serve-dev docker-build docker-up docker-down \
        clean clean-checkpoints setup-data

# ──────────────────────────────────────────────────────────────────────────
# HELP
# ──────────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  LLM Stack – Available Commands"
	@echo "  ================================"
	@echo ""
	@echo "  Setup"
	@echo "    make install          Install Python dependencies (dev mode)"
	@echo "    make setup-data       Generate sample dataset files"
	@echo ""
	@echo "  Quality"
	@echo "    make lint             Ruff lint check"
	@echo "    make format           Auto-format with ruff"
	@echo "    make typecheck        Mypy type checking"
	@echo "    make test             Run all tests (no GPU required)"
	@echo ""
	@echo "  Data"
	@echo "    make preprocess       Preprocess and cache training data"
	@echo ""
	@echo "  Training"
	@echo "    make pretrain         Pretrain from scratch"
	@echo "    make sft              Supervised fine-tuning"
	@echo "    make lora             LoRA fine-tuning"
	@echo "    make dpo              DPO preference optimisation"
	@echo "    make merge            Merge LoRA adapters into base model"
	@echo ""
	@echo "  Evaluation"
	@echo "    make evaluate         Run evaluation benchmarks"
	@echo ""
	@echo "  Serving"
	@echo "    make serve            Start production API server (GPU, vLLM)"
	@echo "    make serve-dev        Start development API server (GPU, hot-reload)"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-build     Build training + serving Docker images"
	@echo "    make docker-up        docker compose up (dev profile)"
	@echo "    make docker-down      docker compose down"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean            Remove build artefacts"
	@echo "    make clean-checkpoints Remove all checkpoints"
	@echo ""

# ──────────────────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────────────────

install:
	@echo "[install] Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	@echo "[install] Done."

setup-data:
	@echo "[setup-data] Creating sample data files..."
	@mkdir -p $(DATA_DIR)
	@$(PYTHON) -c "
import json, random, pathlib
# SFT data
sft = [
    {'instruction': 'What is the capital of France?', 'response': 'The capital of France is Paris.'},
    {'instruction': 'Explain photosynthesis.', 'response': 'Photosynthesis is the process by which plants convert sunlight into energy.'},
    {'instruction': 'Write a haiku about the sea.', 'response': 'Ocean waves crash down\nSalt spray fills the morning air\nEndless blue expanse'},
]
with open('$(DATA_DIR)/sft_train.jsonl', 'w') as f:
    for r in sft: f.write(json.dumps(r) + '\n')
with open('$(DATA_DIR)/sft_val.jsonl', 'w') as f:
    f.write(json.dumps(sft[0]) + '\n')

# DPO data
dpo = [
    {'prompt': 'Describe the sky.', 'chosen': 'The sky is a vast blue expanse.', 'rejected': 'Sky.'},
]
with open('$(DATA_DIR)/dpo_train.jsonl', 'w') as f:
    for r in dpo: f.write(json.dumps(r) + '\n')
with open('$(DATA_DIR)/dpo_val.jsonl', 'w') as f:
    f.write(json.dumps(dpo[0]) + '\n')
print('[setup-data] Sample data files created in $(DATA_DIR)/')
"

# ──────────────────────────────────────────────────────────────────────────
# QUALITY
# ──────────────────────────────────────────────────────────────────────────

lint:
	@echo "[lint] Running ruff..."
	PYTHONPATH=. $(RUFF) check . --output-format=concise

format:
	@echo "[format] Running ruff format..."
	$(RUFF) format .
	$(RUFF) check . --fix

typecheck:
	@echo "[typecheck] Running mypy..."
	PYTHONPATH=. $(MYPY) services/training/core_model/ services/data/ observability/ \
		--ignore-missing-imports --no-strict-optional

test:
	@echo "[test] Running all tests..."
	PYTHONPATH=. $(PYTEST) tests/ \
		-v --tb=short \
		-m "not slow and not gpu" \
		--cov=services --cov=evaluation --cov=observability \
		--cov-report=term-missing \
		--timeout=120

test-unit:
	@echo "[test] Running unit tests only..."
	PYTHONPATH=. $(PYTEST) tests/unit/ -v --tb=short --timeout=60

test-integration:
	@echo "[test] Running integration tests only..."
	PYTHONPATH=. $(PYTEST) tests/integration/ -v --tb=short --timeout=180

test-serving:
	@echo "[test] Running serving tests only..."
	PYTHONPATH=. $(PYTEST) tests/serving/ -v --tb=short --timeout=60

# ──────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────

preprocess:
	@echo "[preprocess] Tokenising and caching datasets..."
	PYTHONPATH=. $(PYTHON) -c "
from services.data.ingestion.loader import DataSourceConfig, load_source
from services.data.tokenization.tokenizer import get_default_tokenizer
tokenizer = get_default_tokenizer(cache_dir='./data_cache/tokenizer')
cfg = DataSourceConfig(name='wikitext2', split='train', streaming=False, cache_dir='./data_cache')
ds = load_source(cfg)
print(f'[preprocess] Loaded {len(ds) if hasattr(ds, \"__len__\") else \"streaming\"} examples')
print('[preprocess] Done.')
"

# ──────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────

pretrain:
	@echo "[pretrain] Starting pretraining..."
	@mkdir -p $(CHECKPOINT_DIR) $(LOG_DIR)
	PYTHONPATH=. $(PYTHON) -m services.training.pretrain.train \
		--config $(CONFIG_DIR)/pretrain.yaml

sft:
	@echo "[sft] Starting supervised fine-tuning..."
	@mkdir -p $(CHECKPOINT_DIR) $(LOG_DIR)
	PYTHONPATH=. $(PYTHON) -m services.training.sft.train \
		--config $(CONFIG_DIR)/sft.yaml

lora:
	@echo "[lora] Starting LoRA fine-tuning..."
	@mkdir -p $(CHECKPOINT_DIR) $(LOG_DIR)
	PYTHONPATH=. $(PYTHON) -m services.training.lora.train \
		--config $(CONFIG_DIR)/lora.yaml

dpo:
	@echo "[dpo] Starting DPO training..."
	@mkdir -p $(CHECKPOINT_DIR) $(LOG_DIR)
	PYTHONPATH=. $(PYTHON) -m services.training.dpo.train \
		--config $(CONFIG_DIR)/dpo.yaml

merge:
	@echo "[merge] Merging LoRA adapters into base model..."
	PYTHONPATH=. $(PYTHON) -c "
from services.training.lora.trainer import merge_and_save
import sys, os
adapter_path = os.environ.get('ADAPTER_PATH', '$(CHECKPOINT_DIR)/lora/final')
save_path = os.environ.get('MERGED_PATH', '$(CHECKPOINT_DIR)/merged')
from peft import PeftModel
from services.training.core_model.model import LLMModel
base_path = os.environ.get('BASE_PATH', '$(CHECKPOINT_DIR)/sft/best')
print(f'Loading base model from {base_path}')
model = LLMModel.from_pretrained(base_path)
print(f'Loading adapter from {adapter_path}')
peft_model = PeftModel.from_pretrained(model, adapter_path)
print(f'Merging and saving to {save_path}')
merge_and_save(peft_model, save_path)
print('[merge] Done.')
"

# ──────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────

evaluate:
	@echo "[evaluate] Running evaluation benchmarks..."
	@mkdir -p eval_results
	PYTHONPATH=. $(PYTHON) -c "
import os, json
from services.training.core_model.model import LLMModel
from services.data.tokenization.tokenizer import get_default_tokenizer
from evaluation.benchmarks.runner import BenchmarkRunner, BenchmarkConfig
from evaluation.perplexity.eval import PerplexityConfig
from evaluation.generation_quality.eval import GenerationConfig

model_path = os.environ.get('MODEL_PATH', '$(CHECKPOINT_DIR)/pretrain/final')
print(f'Loading model from {model_path}')
model = LLMModel.from_pretrained(model_path)
tokenizer = get_default_tokenizer()
runner = BenchmarkRunner(model, tokenizer)
benchmarks = [
    BenchmarkConfig(
        name='perplexity',
        type='perplexity',
        perplexity_config=PerplexityConfig(max_seq_len=512, batch_size=4, max_batches=20)
    ),
    BenchmarkConfig(
        name='generation',
        type='generation',
        prompts=['Once upon a time,', 'The scientist discovered that'],
        generation_config=GenerationConfig(max_new_tokens=64, do_sample=True)
    ),
]
results = runner.run_all(benchmarks, output_dir='./eval_results')
all_passed = all(r.passed for r in results)
print(f'Evaluation complete. All passed: {all_passed}')
"

# ──────────────────────────────────────────────────────────────────────────
# SERVING
# ──────────────────────────────────────────────────────────────────────────

serve:
	@echo "[serve] Starting production API server on :8000..."
	PYTHONPATH=. MODEL_PATH=$(CHECKPOINT_DIR)/pretrain/final \
		USE_VLLM=true DTYPE=$(DTYPE) \
		uvicorn services.serving.api.app:app \
		--host 0.0.0.0 --port 8000 --workers 1 --loop uvloop

serve-dev:
	@echo "[serve-dev] Starting development API server on :8000 (GPU, hot-reload)..."
	PYTHONPATH=. MODEL_PATH=$(CHECKPOINT_DIR)/pretrain/final \
		USE_VLLM=${USE_VLLM:-true} DTYPE=$(DTYPE) JSON_LOGS=false \
		uvicorn services.serving.api.app:app \
		--host 0.0.0.0 --port 8000 --reload

# ──────────────────────────────────────────────────────────────────────────
# DOCKER
# ──────────────────────────────────────────────────────────────────────────

docker-build:
	@echo "[docker] Building images..."
	docker build -f infra/docker/Dockerfile.training -t llm-stack-training:latest .
	docker build -f infra/docker/Dockerfile.serving -t llm-stack-serving:latest .
	@echo "[docker] Images built successfully."

docker-up:
	@echo "[docker] Starting dev GPU stack..."
	docker compose -f infra/compose/docker-compose.yml --profile dev up -d

docker-up-gpu:
	@echo "[docker] Starting production GPU stack..."
	docker compose -f infra/compose/docker-compose.yml --profile production up -d

docker-down:
	docker compose -f infra/compose/docker-compose.yml down

docker-logs:
	docker compose -f infra/compose/docker-compose.yml logs -f

# ──────────────────────────────────────────────────────────────────────────
# CLEANUP
# ──────────────────────────────────────────────────────────────────────────

clean:
	@echo "[clean] Removing build artefacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "[clean] Done."

clean-checkpoints:
	@echo "[clean] Removing checkpoints (this is destructive!)..."
	@read -p "Are you sure? [y/N] " CONFIRM && [ "$$CONFIRM" = "y" ] || exit 1
	rm -rf $(CHECKPOINT_DIR)
	@echo "[clean] Checkpoints removed."

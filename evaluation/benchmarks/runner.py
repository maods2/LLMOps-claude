"""
Config-driven benchmark runner.

Each benchmark is defined in a YAML config and produces a scalar metric.
The runner orchestrates multiple benchmarks and writes a summary report.

Usage:
    runner = BenchmarkRunner(model, tokenizer)
    results = runner.run_all(benchmark_configs)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast

from pydantic import BaseModel, ConfigDict, Field

from evaluation.perplexity.eval import PerplexityConfig, evaluate_perplexity
from evaluation.generation_quality.eval import GenerationConfig, generate_samples


class BenchmarkConfig(BaseModel):
    """Configuration for a single benchmark task."""

    name: str = Field(description="Benchmark identifier")
    type: str = Field(description="Benchmark type: perplexity | generation | regression")

    # Perplexity settings
    dataset_path: Optional[str] = Field(None)
    perplexity_config: Optional[PerplexityConfig] = Field(None)

    # Generation settings
    prompts: Optional[list[str]] = Field(None)
    generation_config: Optional[GenerationConfig] = Field(None)
    expected_substrings: Optional[list[str]] = Field(
        None, description="Strings expected in at least one generation"
    )

    model_config = ConfigDict(frozen=True)


@dataclass
class BenchmarkResult:
    """Result for a single benchmark."""

    name: str
    type: str
    metrics: dict[str, Any]
    passed: bool
    duration_s: float


class BenchmarkRunner:
    """Runs a collection of benchmarks against a model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerFast,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_all(
        self,
        configs: list[BenchmarkConfig],
        output_dir: Optional[str] = None,
    ) -> list[BenchmarkResult]:
        """Run all benchmarks and optionally write a JSON report.

        Args:
            configs:    List of BenchmarkConfig objects.
            output_dir: If provided, writes benchmark_results.json here.
        Returns:
            List of BenchmarkResult.
        """
        results = []
        for cfg in configs:
            result = self._run_one(cfg)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {cfg.name}: {result.metrics}")

        if output_dir:
            self._write_report(results, output_dir)

        return results

    def _run_one(self, cfg: BenchmarkConfig) -> BenchmarkResult:
        t0 = time.perf_counter()
        try:
            if cfg.type == "perplexity":
                metrics, passed = self._run_perplexity(cfg)
            elif cfg.type == "generation":
                metrics, passed = self._run_generation(cfg)
            elif cfg.type == "regression":
                metrics, passed = self._run_regression(cfg)
            else:
                raise ValueError(f"Unknown benchmark type: {cfg.type!r}")
        except Exception as exc:
            metrics = {"error": str(exc)}
            passed = False

        return BenchmarkResult(
            name=cfg.name,
            type=cfg.type,
            metrics=metrics,
            passed=passed,
            duration_s=round(time.perf_counter() - t0, 2),
        )

    def _run_perplexity(self, cfg: BenchmarkConfig) -> tuple[dict, bool]:
        from torch.utils.data import DataLoader
        from services.data.ingestion.loader import load_local_jsonl
        from services.data.preprocessing.processor import PretrainDataset, PreprocessingConfig

        ppl_cfg = cfg.perplexity_config or PerplexityConfig()

        if cfg.dataset_path:
            hf_ds = load_local_jsonl(cfg.dataset_path, streaming=False)
            pp_cfg = PreprocessingConfig(max_seq_len=ppl_cfg.max_seq_len)
            ds = PretrainDataset(hf_ds, self.tokenizer, pp_cfg)
            loader = DataLoader(ds, batch_size=ppl_cfg.batch_size, num_workers=0)
        else:
            # Tiny synthetic fallback
            dummy = _make_dummy_loader(self.tokenizer, ppl_cfg.max_seq_len, ppl_cfg.batch_size)
            loader = dummy

        result = evaluate_perplexity(self.model, loader, ppl_cfg, self.device)
        return result.to_dict(), True  # Always passes (it's a measurement)

    def _run_generation(self, cfg: BenchmarkConfig) -> tuple[dict, bool]:
        prompts = cfg.prompts or ["Once upon a time,"]
        gen_cfg = cfg.generation_config or GenerationConfig()
        samples = generate_samples(self.model, self.tokenizer, prompts, gen_cfg, self.device)
        avg_diversity = sum(s.unique_bigrams_ratio for s in samples) / len(samples)
        metrics = {
            "samples": [{"prompt": s.prompt, "generated": s.generated_text} for s in samples],
            "avg_bigram_diversity": round(avg_diversity, 4),
        }
        passed = True
        if cfg.expected_substrings:
            all_generated = " ".join(s.generated_text for s in samples)
            passed = all(sub.lower() in all_generated.lower() for sub in cfg.expected_substrings)
        return metrics, passed

    def _run_regression(self, cfg: BenchmarkConfig) -> tuple[dict, bool]:
        """Run prompt → expected-substring regression tests."""
        if not cfg.prompts:
            return {"error": "No prompts defined"}, False
        gen_cfg = cfg.generation_config or GenerationConfig(do_sample=False)
        samples = generate_samples(self.model, self.tokenizer, cfg.prompts, gen_cfg, self.device)
        metrics = {"samples": [{"prompt": s.prompt, "generated": s.generated_text} for s in samples]}
        passed = True
        if cfg.expected_substrings:
            for sample, expected in zip(samples, cfg.expected_substrings):
                if expected.lower() not in sample.generated_text.lower():
                    passed = False
        return metrics, passed

    @staticmethod
    def _write_report(results: list[BenchmarkResult], output_dir: str) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        report = [
            {
                "name": r.name,
                "type": r.type,
                "passed": r.passed,
                "duration_s": r.duration_s,
                "metrics": r.metrics,
            }
            for r in results
        ]
        out_path = Path(output_dir) / "benchmark_results.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {out_path}")


def _make_dummy_loader(
    tokenizer: PreTrainedTokenizerFast, seq_len: int, batch_size: int
) -> torch.utils.data.DataLoader:
    """Create a tiny dummy DataLoader for testing without a real dataset."""
    n = batch_size * 4
    vocab = tokenizer.vocab_size or 32000
    input_ids = torch.randint(3, vocab, (n, seq_len))
    labels = input_ids.clone()

    class _TinyDS(torch.utils.data.Dataset):
        def __len__(self):
            return n
        def __getitem__(self, i):
            return {"input_ids": input_ids[i], "labels": labels[i]}

    return torch.utils.data.DataLoader(_TinyDS(), batch_size=batch_size)

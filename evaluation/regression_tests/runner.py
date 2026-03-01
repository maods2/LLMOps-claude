"""
Regression test runner: verifies model outputs against known-good expectations.

Loads test cases from YAML/JSON and fails if any expectation is not met.
Used in CI to catch regressions after fine-tuning or merging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast

from pydantic import BaseModel, Field

from evaluation.generation_quality.eval import GenerationConfig, generate_samples


class RegressionTestCase(BaseModel):
    """A single regression test case."""

    name: str
    prompt: str
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    min_tokens: int = Field(default=10, description="Minimum generated token count")


@dataclass
class RegressionTestResult:
    name: str
    passed: bool
    prompt: str
    generated: str
    failures: list[str]


class RegressionTestRunner:
    """Runs regression test suites against a model."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerFast,
        device: Optional[torch.device] = None,
        gen_config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen_config = gen_config or GenerationConfig(do_sample=False, max_new_tokens=64)

    def run(self, test_cases: list[RegressionTestCase]) -> list[RegressionTestResult]:
        """Execute all regression test cases.

        Returns:
            List of RegressionTestResult; all failures have passed=False.
        """
        results = []
        for tc in test_cases:
            samples = generate_samples(
                self.model, self.tokenizer, [tc.prompt], self.gen_config, self.device
            )
            generated = samples[0].generated_text
            failures = []

            for substr in tc.must_contain:
                if substr.lower() not in generated.lower():
                    failures.append(f"must_contain {substr!r} not found")
            for substr in tc.must_not_contain:
                if substr.lower() in generated.lower():
                    failures.append(f"must_not_contain {substr!r} was found")
            if samples[0].n_tokens < tc.min_tokens:
                failures.append(
                    f"min_tokens {tc.min_tokens} not met (got {samples[0].n_tokens})"
                )

            results.append(
                RegressionTestResult(
                    name=tc.name,
                    passed=len(failures) == 0,
                    prompt=tc.prompt,
                    generated=generated,
                    failures=failures,
                )
            )

        return results

    def run_from_file(self, path: str) -> list[RegressionTestResult]:
        """Load test cases from a JSON file and run them."""
        with open(path) as f:
            raw = json.load(f)
        test_cases = [RegressionTestCase(**tc) for tc in raw]
        return self.run(test_cases)

    @staticmethod
    def print_summary(results: list[RegressionTestResult]) -> bool:
        """Print results table. Returns True if all passed."""
        all_passed = True
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}")
            if not r.passed:
                all_passed = False
                for f in r.failures:
                    print(f"         ✗ {f}")
        return all_passed

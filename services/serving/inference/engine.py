"""
Inference engine abstraction.

Supports two backends:
  1. vLLM – for high-throughput production serving
  2. Native PyTorch – for development and testing without a GPU cluster

The backend is selected via the ``use_vllm`` flag in InferenceConfig.
The same interface is exposed regardless of backend.
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator, Iterator, Optional

import torch
from pydantic import BaseModel, Field


class InferenceConfig(BaseModel):
    """Configuration for the inference engine."""

    model_path: str = Field(..., description="Path to the model checkpoint directory")
    use_vllm: bool = Field(True, description="Use vLLM engine if available")
    dtype: str = Field("bfloat16")
    max_seq_len: int = Field(512)
    gpu_memory_utilization: float = Field(0.85, description="vLLM GPU memory fraction")
    max_batch_size: int = Field(32)
    tokenizer_path: Optional[str] = Field(None, description="Tokenizer path (defaults to model_path)")

    class Config:
        frozen = True


class InferenceEngine:
    """
    Unified inference engine that dispatches to vLLM or PyTorch backend.

    Args:
        config: InferenceConfig.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._backend = None
        self.tokenizer = None
        self.framework: str = "unloaded"
        self._load()

    def _load(self) -> None:
        tok_path = self.config.tokenizer_path or self.config.model_path

        if self.config.use_vllm:
            try:
                self._load_vllm(tok_path)
                self.framework = "vllm"
                return
            except ImportError:
                pass  # Fall through to PyTorch backend

        self._load_pytorch(tok_path)
        self.framework = "pytorch"

    def _load_vllm(self, tok_path: str) -> None:
        from vllm import LLM, SamplingParams  # type: ignore[import]
        from transformers import AutoTokenizer

        self._vllm = LLM(
            model=self.config.model_path,
            dtype=self.config.dtype,
            max_model_len=self.config.max_seq_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=False,
        )
        self._SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_pytorch(self, tok_path: str) -> None:
        from services.training.core_model.model import LLMModel
        from services.data.tokenization.tokenizer import load_tokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        self._pt_dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        self._pt_model = LLMModel.from_pretrained(self.config.model_path)
        self._pt_model.to(self._device)
        self._pt_model.eval()

        self.tokenizer = load_tokenizer(tok_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate text for a single prompt (blocking).

        Returns:
            Generated text (excluding the prompt).
        """
        if self.framework == "vllm":
            return self._vllm_generate(prompt, max_new_tokens, temperature, top_p, top_k)
        return self._pytorch_generate(prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty)

    def batch_generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> list[str]:
        """Generate text for multiple prompts (blocking).

        Returns:
            List of generated texts, one per prompt (excluding prompts).
        """
        if self.framework == "vllm":
            return self._vllm_batch(prompts, max_new_tokens, temperature, top_p)
        # PyTorch fallback: sequential
        return [self.generate(p, max_new_tokens, temperature, top_p) for p in prompts]

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> AsyncIterator[str]:
        """Stream generated tokens for a single prompt.

        Yields one decoded token string at a time.
        """
        if self.framework == "vllm":
            async for token in self._vllm_stream(prompt, max_new_tokens, temperature, top_p):
                yield token
        else:
            async for token in self._pytorch_stream(prompt, max_new_tokens, temperature, top_p):
                yield token

    # ------------------------------------------------------------------
    # vLLM backend
    # ------------------------------------------------------------------

    def _vllm_generate(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int
    ) -> str:
        params = self._SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        outputs = self._vllm.generate([prompt], params)
        return outputs[0].outputs[0].text

    def _vllm_batch(
        self, prompts: list[str], max_new_tokens: int, temperature: float, top_p: float
    ) -> list[str]:
        params = self._SamplingParams(
            max_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        )
        outputs = self._vllm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    async def _vllm_stream(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float
    ) -> AsyncIterator[str]:
        """vLLM async streaming using async_engine."""
        # Simplified: vLLM async requires AsyncLLMEngine; fall back to chunked sync
        text = self._vllm_generate(prompt, max_new_tokens, temperature, top_p, 50)
        for token in text.split():
            yield token + " "
            await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # PyTorch backend
    # ------------------------------------------------------------------

    def _pytorch_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
    ) -> str:
        from evaluation.generation_quality.eval import GenerationConfig, generate_samples
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        results = generate_samples(self._pt_model, self.tokenizer, [prompt], cfg, self._device)
        return results[0].generated_text

    async def _pytorch_stream(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float
    ) -> AsyncIterator[str]:
        """PyTorch token-by-token streaming (word-split approximation)."""
        text = self._pytorch_generate(prompt, max_new_tokens, temperature, top_p, 50, 1.1)
        for word in text.split(" "):
            yield word + " "
            await asyncio.sleep(0)

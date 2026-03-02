"""Core model package: decoder-only Transformer (~300M params)."""

from services.training.core_model.config import ModelConfig
from services.training.core_model.model import LLMModel

__all__ = ["LLMModel", "ModelConfig"]

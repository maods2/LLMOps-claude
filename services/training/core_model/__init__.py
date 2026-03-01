"""Core model package: decoder-only Transformer (~300M params)."""

from services.training.core_model.model import LLMModel
from services.training.core_model.config import ModelConfig

__all__ = ["LLMModel", "ModelConfig"]

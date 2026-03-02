"""Model configuration using Pydantic v2."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    """
    Configuration for the decoder-only Transformer model.

    Default parameters target ~300M total parameters:
      24 layers × (4×1024² attn + 2×1024×4096 MLP) + 32000×1024 embed ≈ 302M
    """

    # Vocabulary / embedding
    vocab_size: int = Field(32000, description="Vocabulary size (BPE tokenizer)")
    max_seq_len: int = Field(512, description="Maximum context length")

    # Transformer dimensions
    n_layers: int = Field(24, description="Number of transformer layers")
    hidden_size: int = Field(1024, description="Model hidden dimension")
    intermediate_size: int = Field(4096, description="MLP intermediate dimension (≈4×hidden)")
    n_heads: int = Field(16, description="Number of attention heads")
    n_kv_heads: int | None = Field(
        None, description="Number of KV heads for GQA (None = full MHA)"
    )

    # Regularisation
    dropout: float = Field(0.0, description="Dropout probability (0 = disabled for inference)")
    attention_dropout: float = Field(0.0, description="Attention dropout probability")

    # Architecture flags
    use_rope: bool = Field(True, description="Use RoPE positional encoding")
    rope_theta: float = Field(10000.0, description="RoPE base frequency")
    use_flash_attention: bool = Field(False, description="Use Flash Attention 2 if installed")
    use_gradient_checkpointing: bool = Field(
        False, description="Enable gradient checkpointing to save VRAM"
    )
    tie_word_embeddings: bool = Field(True, description="Tie input/output embedding weights")

    # Precision
    dtype: str = Field("bfloat16", description="Model dtype: float32, float16, bfloat16")

    # Initialisation
    init_std: float = Field(0.02, description="Weight initialisation standard deviation")
    rms_norm_eps: float = Field(1e-6, description="RMSNorm epsilon")

    # HuggingFace compatibility (required by PEFT)
    model_type: str = Field("custom", description="Model type identifier for PEFT compatibility")

    # Pad token
    pad_token_id: int = Field(0, description="Padding token ID")
    bos_token_id: int = Field(1, description="Beginning-of-sequence token ID")
    eos_token_id: int = Field(2, description="End-of-sequence token ID")

    @model_validator(mode="after")
    def validate_heads(self) -> ModelConfig:
        assert self.hidden_size % self.n_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by n_heads ({self.n_heads})"
        )
        if self.n_kv_heads is not None:
            assert self.n_heads % self.n_kv_heads == 0, (
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        return self

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def effective_kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    def to_dict(self) -> dict:
        """HuggingFace-compatible config serialisation (called by PEFT ≥ 0.10)."""
        return self.model_dump()

    def get(self, key: str, default=None):
        """Dict-like access required by PEFT for tie_word_embeddings etc."""
        return self.model_dump().get(key, default)

    def estimate_params(self) -> int:
        """Estimate total parameter count."""
        embed = self.vocab_size * self.hidden_size
        attn = 4 * self.hidden_size * self.hidden_size  # Q, K, V, O
        mlp = 2 * self.hidden_size * self.intermediate_size  # up, down
        norm = 2 * self.hidden_size  # pre-attn + post-attn RMSNorm
        per_layer = attn + mlp + norm
        # Output head shares weights with embedding if tied
        output = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size
        final_norm = self.hidden_size
        return embed + self.n_layers * per_layer + output + final_norm

    model_config = ConfigDict(frozen=True)

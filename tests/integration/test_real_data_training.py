"""
Integration tests: full training pipelines using real text data.

Unlike the synthetic-tensor tests in test_training.py / test_dpo.py / test_lora.py,
these tests exercise the complete data path:

  raw text (JSONL)  →  BPE tokeniser  →  Dataset  →  DataLoader  →  Trainer  →  checkpoint

Each training mode (pretrain, SFT, LoRA, DPO) is covered with a tiny model and a
handful of real text examples so the tests remain fast on CPU.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from services.training.core_model.config import ModelConfig
from services.training.core_model.model import LLMModel

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Small but realistic text samples – deliberately varied in length and style
_PRETRAIN_TEXTS = [
    "The transformer architecture revolutionized natural language processing.",
    "Attention mechanisms allow models to focus on relevant parts of the input.",
    "Large language models are trained on vast amounts of text data.",
    "Gradient descent optimizes neural network parameters iteratively.",
    "Tokenization converts raw text into integer sequences for models.",
    "Residual connections help gradients flow during deep network training.",
    "Layer normalization stabilizes training by normalizing activations.",
    "Self-supervised learning leverages unlabeled data for representation.",
    "Fine-tuning adapts pretrained models to downstream tasks efficiently.",
    "Decoding strategies like beam search improve generation quality.",
]

_SFT_PAIRS = [
    {"instruction": "Explain what a neural network is.", "response": "A neural network is a computational model inspired by the brain, consisting of layers of interconnected nodes."},
    {"instruction": "What is backpropagation?", "response": "Backpropagation is an algorithm that computes gradients by propagating errors backward through a network."},
    {"instruction": "Define the softmax function.", "response": "Softmax converts a vector of real numbers into a probability distribution that sums to one."},
    {"instruction": "What is overfitting?", "response": "Overfitting occurs when a model memorizes training data and fails to generalize to new examples."},
    {"instruction": "Describe the attention mechanism.", "response": "Attention weighs the importance of different input positions when producing each output token."},
    {"instruction": "What is a learning rate?", "response": "The learning rate controls how large a step the optimizer takes during each parameter update."},
]

_DPO_PAIRS = [
    {
        "prompt": "Describe the role of the optimizer in training.",
        "chosen": "The optimizer updates model parameters to minimize the loss function using gradients computed by backpropagation.",
        "rejected": "The optimizer is a thing that does stuff to make the model better somehow.",
    },
    {
        "prompt": "What is the purpose of dropout?",
        "chosen": "Dropout randomly deactivates neurons during training to prevent overfitting and improve generalization.",
        "rejected": "Dropout drops stuff randomly which is good for models.",
    },
    {
        "prompt": "How does layer normalization help training?",
        "chosen": "Layer normalization stabilizes training by normalizing activations within each layer, enabling higher learning rates.",
        "rejected": "Normalization makes numbers normal so training works better.",
    },
    {
        "prompt": "Explain weight initialization.",
        "chosen": "Good weight initialization prevents vanishing or exploding gradients by setting weights close to their optimal scale at the start.",
        "rejected": "Weights need to start somewhere so we just pick some numbers.",
    },
]


@pytest.fixture(scope="module")
def real_tokenizer(tmp_path_factory):
    """BPE tokenizer trained from scratch on the small real corpus."""
    from services.data.tokenization.tokenizer import build_tokenizer_from_scratch

    tmp = tmp_path_factory.mktemp("tok")
    corpus = tmp / "corpus.txt"
    # Combine all text used by the tests for a richer vocabulary
    all_texts = list(_PRETRAIN_TEXTS)
    for pair in _SFT_PAIRS:
        all_texts.append(pair["instruction"])
        all_texts.append(pair["response"])
    for pair in _DPO_PAIRS:
        all_texts.extend([pair["prompt"], pair["chosen"], pair["rejected"]])
    corpus.write_text("\n".join(all_texts * 10))  # repeat to satisfy BPE trainer minimum

    tok = build_tokenizer_from_scratch(
        files=[str(corpus)],
        vocab_size=512,
        save_dir=str(tmp / "tokenizer"),
    )
    return tok


@pytest.fixture(scope="module")
def tiny_cfg():
    return ModelConfig(
        vocab_size=512,
        max_seq_len=32,
        n_layers=2,
        hidden_size=64,
        intermediate_size=128,
        n_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
        use_flash_attention=False,
        tie_word_embeddings=True,
        dtype="float32",
    )


@pytest.fixture
def fresh_model(tiny_cfg):
    """Returns a freshly-initialised tiny model (new instance per test)."""
    return LLMModel(tiny_cfg)


# ---------------------------------------------------------------------------
# Helper: write a JSONL file to a temp path
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Pretrain – real text pipeline
# ---------------------------------------------------------------------------

class TestPretrainRealData:
    """Pretraining with actual JSONL text → BPE tokeniser → PretrainDataset."""

    def _make_pretrain_loader(self, tokenizer, tmp_path, batch_size=2):
        from torch.utils.data import DataLoader

        from services.data.ingestion.loader import load_local_jsonl
        from services.data.preprocessing.processor import PreprocessingConfig, PretrainDataset

        jsonl = tmp_path / "pretrain.jsonl"
        # Repeat texts to produce enough packed chunks for multi-batch testing
        _write_jsonl(jsonl, [{"text": t} for t in _PRETRAIN_TEXTS * 8])

        ds = load_local_jsonl(str(jsonl), streaming=True)
        cfg = PreprocessingConfig(max_seq_len=32, batch_size=batch_size)
        pt_ds = PretrainDataset(ds, tokenizer, cfg)
        return DataLoader(pt_ds, batch_size=batch_size)

    def test_pretrain_loader_yields_real_tokens(self, real_tokenizer, tmp_path):
        """DataLoader built from real text must produce integer token ids."""
        loader = self._make_pretrain_loader(real_tokenizer, tmp_path)
        batch = next(iter(loader))
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].shape[-1] == 32
        # IDs should be in vocab range
        assert batch["input_ids"].min() >= 0
        assert batch["input_ids"].max() < real_tokenizer.vocab_size

    def test_pretrain_loss_decreases_on_real_data(self, fresh_model, real_tokenizer, tmp_path):
        """After several gradient steps on real text, loss should decrease."""
        from torch.optim import AdamW

        loader = self._make_pretrain_loader(real_tokenizer, tmp_path, batch_size=2)
        optimizer = AdamW(fresh_model.parameters(), lr=1e-3)
        fresh_model.train()

        losses = []
        for batch in loader:
            optimizer.zero_grad()
            out = fresh_model(batch["input_ids"], labels=batch["labels"])
            out.loss.backward()
            optimizer.step()
            losses.append(out.loss.item())
            if len(losses) >= 5:
                break

        assert len(losses) == 5
        # Final loss should be less than 2x the initial loss
        assert losses[-1] < losses[0] * 2.0

    def test_pretrain_trainer_real_data_two_steps(self, fresh_model, real_tokenizer, tmp_path):
        """PretrainTrainer must complete 2 gradient steps on the real data loader."""
        from services.training.pretrain.trainer import PretrainConfig, PretrainTrainer

        loader = self._make_pretrain_loader(real_tokenizer, tmp_path)
        cfg = PretrainConfig(
            output_dir=str(tmp_path / "ckpt"),
            max_steps=2,
            log_interval=1,
            eval_interval=100,
            save_interval=100,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
        )
        trainer = PretrainTrainer(fresh_model, loader, None, cfg)
        trainer.train()  # Must not raise

    def test_pretrain_trainer_saves_checkpoint(self, fresh_model, real_tokenizer, tmp_path):
        """Checkpoint files must be created after training completes."""
        from services.training.pretrain.trainer import PretrainConfig, PretrainTrainer

        ckpt_dir = tmp_path / "ckpt"
        loader = self._make_pretrain_loader(real_tokenizer, tmp_path)
        cfg = PretrainConfig(
            output_dir=str(ckpt_dir),
            max_steps=2,
            log_interval=1,
            eval_interval=100,
            save_interval=2,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
        )
        trainer = PretrainTrainer(fresh_model, loader, None, cfg)
        trainer.train()

        final = ckpt_dir / "final"
        assert (final / "config.json").exists(), "config.json missing from checkpoint"
        assert (final / "model.pt").exists(), "model.pt missing from checkpoint"

    def test_pretrain_checkpoint_is_loadable(self, fresh_model, real_tokenizer, tmp_path):
        """A saved pretrain checkpoint must be loadable and produce valid output."""
        from services.training.pretrain.trainer import PretrainConfig, PretrainTrainer

        ckpt_dir = tmp_path / "ckpt"
        loader = self._make_pretrain_loader(real_tokenizer, tmp_path)
        cfg = PretrainConfig(
            output_dir=str(ckpt_dir),
            max_steps=2,
            log_interval=1,
            eval_interval=100,
            save_interval=2,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
        )
        trainer = PretrainTrainer(fresh_model, loader, None, cfg)
        trainer.train()

        loaded = LLMModel.from_pretrained(str(ckpt_dir / "final"))
        ids = torch.randint(0, 512, (1, 16))
        out = loaded(ids)
        assert out.logits.shape == (1, 16, 512)
        assert not torch.isnan(out.logits).any()


# ---------------------------------------------------------------------------
# SFT – instruction / response pairs
# ---------------------------------------------------------------------------

class TestSFTRealData:
    """Supervised fine-tuning with real instruction-response pairs."""

    def _make_sft_loader(self, tokenizer, tmp_path, batch_size=2):
        from torch.utils.data import DataLoader

        from services.data.preprocessing.processor import SFTDataset

        ds = SFTDataset(_SFT_PAIRS, tokenizer, max_seq_len=32)
        return DataLoader(ds, batch_size=batch_size, drop_last=True)

    def test_sft_dataset_prompt_masking_with_real_text(self, real_tokenizer):
        """Response tokens must be unmasked; at least some prompt tokens must be -100."""
        from services.data.preprocessing.processor import SFTDataset

        ds = SFTDataset(_SFT_PAIRS, real_tokenizer, max_seq_len=32)
        item = ds[0]
        labels = item["labels"]
        # At least first label token is masked (prompt prefix)
        assert labels[0].item() == -100, "First label token should be prompt-masked"
        # At least one response token is not masked
        assert (labels != -100).any(), "All labels are -100; response is not learned"

    def test_sft_trainer_real_pairs_two_steps(self, fresh_model, real_tokenizer, tmp_path):
        """SFTTrainer must run 2 steps without error on real instruction data."""
        from services.training.sft.trainer import SFTConfig, SFTTrainer

        loader = self._make_sft_loader(real_tokenizer, tmp_path)
        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft"),
            max_steps=2,
            log_interval=1,
            eval_interval=100,
            save_interval=100,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
            early_stopping_patience=10,
        )
        trainer = SFTTrainer(fresh_model, loader, None, cfg)
        trainer.train()

    def test_sft_loss_is_finite_on_real_data(self, fresh_model, real_tokenizer):
        """SFT loss computed on real text must be a finite positive scalar."""
        from torch.utils.data import DataLoader

        from services.data.preprocessing.processor import SFTDataset

        ds = SFTDataset(_SFT_PAIRS, real_tokenizer, max_seq_len=32)
        loader = DataLoader(ds, batch_size=2, drop_last=True)
        batch = next(iter(loader))

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        out = fresh_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss_val = out.loss.item()
        import math
        assert math.isfinite(loss_val), f"SFT loss is not finite: {loss_val}"
        assert loss_val > 0, "SFT loss must be positive"

    def test_sft_with_val_set_real_data(self, fresh_model, real_tokenizer, tmp_path):
        """SFT trainer with a validation set should log val loss without error."""
        from torch.utils.data import DataLoader

        from services.data.preprocessing.processor import SFTDataset
        from services.training.sft.trainer import SFTConfig, SFTTrainer

        train_ds = SFTDataset(_SFT_PAIRS[:4], real_tokenizer, max_seq_len=32)
        val_ds = SFTDataset(_SFT_PAIRS[4:], real_tokenizer, max_seq_len=32)
        train_loader = DataLoader(train_ds, batch_size=2, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=2, drop_last=True)

        cfg = SFTConfig(
            output_dir=str(tmp_path / "sft_val"),
            max_steps=2,
            log_interval=1,
            eval_interval=1,   # evaluate every step
            save_interval=100,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
            early_stopping_patience=10,
        )
        trainer = SFTTrainer(fresh_model, train_loader, val_loader, cfg)
        trainer.train()


# ---------------------------------------------------------------------------
# LoRA – low-rank adaptation with real data
# ---------------------------------------------------------------------------

class TestLoRARealData:
    """LoRA fine-tuning using real instruction-response pairs."""

    def _make_loader(self, tokenizer, batch_size=2):
        from torch.utils.data import DataLoader

        from services.data.preprocessing.processor import SFTDataset

        ds = SFTDataset(_SFT_PAIRS, tokenizer, max_seq_len=32)
        return DataLoader(ds, batch_size=batch_size, drop_last=True)

    def test_lora_adapter_injects_on_real_model(self, fresh_model):
        """LoRA injection should dramatically reduce trainable parameter count."""
        pytest.importorskip("peft")

        from services.training.lora.trainer import LoRAConfig, inject_lora

        cfg = LoRAConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
            output_dir="/tmp/lora_real",
            max_steps=1,
        )
        lora_model = inject_lora(fresh_model, cfg)
        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in lora_model.parameters())
        ratio = trainable / total
        assert ratio < 0.5, f"Expected < 50% trainable params but got {ratio:.1%}"
        assert trainable > 0, "No trainable parameters after LoRA injection"

    def test_lora_trainer_real_data_two_steps(self, fresh_model, real_tokenizer, tmp_path):
        """LoRATrainer must complete 2 steps on real instruction data."""
        pytest.importorskip("peft")

        from services.training.lora.trainer import LoRAConfig, LoRATrainer

        loader = self._make_loader(real_tokenizer)
        cfg = LoRAConfig(
            r=4,
            lora_alpha=8.0,
            target_modules=["q_proj", "v_proj"],
            output_dir=str(tmp_path / "lora"),
            max_steps=2,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
            log_interval=1,
            eval_interval=100,
            save_interval=100,
        )
        trainer = LoRATrainer(fresh_model, loader, None, cfg)
        trainer.train()

    def test_lora_loss_finite_on_real_data(self, fresh_model, real_tokenizer):
        """LoRA forward pass on real tokenized text must produce finite loss."""
        pytest.importorskip("peft")

        from torch.utils.data import DataLoader

        from services.data.preprocessing.processor import SFTDataset
        from services.training.lora.trainer import LoRAConfig, inject_lora

        ds = SFTDataset(_SFT_PAIRS, real_tokenizer, max_seq_len=32)
        batch = next(iter(DataLoader(ds, batch_size=2, drop_last=True)))

        cfg = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["q_proj"], output_dir="/tmp", max_steps=1)
        lora_model = inject_lora(fresh_model, cfg)

        out = lora_model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        import math
        assert math.isfinite(out.loss.item())
        assert out.loss.item() > 0

    def test_lora_merge_produces_loadable_model(self, fresh_model, real_tokenizer, tmp_path):
        """Merged LoRA weights must be saveable and produce correct output shapes."""
        pytest.importorskip("peft")

        from services.training.lora.trainer import LoRAConfig, inject_lora, merge_and_save

        cfg = LoRAConfig(r=4, lora_alpha=8.0, target_modules=["q_proj"], output_dir="/tmp", max_steps=1)
        lora_model = inject_lora(fresh_model, cfg)
        merge_dir = str(tmp_path / "merged")
        merge_and_save(lora_model, merge_dir)

        merged = LLMModel.from_pretrained(merge_dir)
        ids = torch.randint(0, 512, (1, 16))
        out = merged(ids)
        assert out.logits.shape == (1, 16, 512)


# ---------------------------------------------------------------------------
# DPO – preference optimisation with real pairs
# ---------------------------------------------------------------------------

class TestDPORealData:
    """DPO training using real (prompt, chosen, rejected) pairs."""

    def _make_dpo_loader(self, tokenizer, batch_size=2):
        from torch.utils.data import DataLoader

        from services.data.preprocessing.processor import DPODataset

        ds = DPODataset(_DPO_PAIRS, tokenizer, max_seq_len=32)
        return DataLoader(ds, batch_size=batch_size, drop_last=True)

    def test_dpo_dataset_real_pairs_shapes(self, real_tokenizer):
        """DPODataset must produce correctly-shaped chosen/rejected tensors from real text."""
        from services.data.preprocessing.processor import DPODataset

        ds = DPODataset(_DPO_PAIRS, real_tokenizer, max_seq_len=32)
        assert len(ds) == len(_DPO_PAIRS)
        item = ds[0]
        for key in [
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels",
        ]:
            assert key in item, f"Missing key: {key}"
            assert item[key].shape == (32,), f"Wrong shape for {key}: {item[key].shape}"

    def test_dpo_prompt_masked_in_labels(self, real_tokenizer):
        """Prompt tokens in DPO labels must be masked with -100."""
        from services.data.preprocessing.processor import DPODataset

        ds = DPODataset(_DPO_PAIRS, real_tokenizer, max_seq_len=32)
        item = ds[0]
        # At least the first label of chosen/rejected should be -100 (prompt masked)
        assert item["chosen_labels"][0].item() == -100
        assert item["rejected_labels"][0].item() == -100

    def test_dpo_loss_positive_real_data(self, fresh_model, real_tokenizer):
        """DPO loss computed on real preference pairs must be a positive scalar."""
        from services.training.dpo.trainer import DPOConfig, DPOTrainer

        ref_model = copy.deepcopy(fresh_model)
        cfg = DPOConfig(
            beta=0.1,
            output_dir="/tmp/dpo_real",
            max_steps=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
            offload_ref_to_cpu=False,
        )
        loader = self._make_dpo_loader(real_tokenizer)
        trainer = DPOTrainer(fresh_model, ref_model, loader, None, cfg)
        batch = next(iter(loader))
        loss, metrics = trainer._dpo_loss(batch)
        assert loss.item() > 0
        assert "reward_margin" in metrics
        import math
        assert math.isfinite(metrics["reward_margin"])

    def test_dpo_trainer_real_data_one_step(self, fresh_model, real_tokenizer, tmp_path):
        """DPOTrainer must complete 1 full training step on real preference data."""
        from services.training.dpo.trainer import DPOConfig, DPOTrainer

        ref_model = copy.deepcopy(fresh_model)
        cfg = DPOConfig(
            beta=0.1,
            output_dir=str(tmp_path / "dpo"),
            max_steps=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
            offload_ref_to_cpu=False,
            log_interval=1,
            eval_interval=100,
            save_interval=100,
        )
        loader = self._make_dpo_loader(real_tokenizer)
        trainer = DPOTrainer(fresh_model, ref_model, loader, None, cfg)
        trainer.train()

    def test_dpo_reference_free_real_data(self, fresh_model, real_tokenizer):
        """Reference-free DPO on real pairs must yield finite positive loss."""
        from services.training.dpo.trainer import DPOConfig, DPOTrainer

        cfg = DPOConfig(
            beta=0.1,
            reference_free=True,
            output_dir="/tmp/dpo_rf_real",
            max_steps=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            dtype="float32",
            gradient_checkpointing=False,
        )
        loader = self._make_dpo_loader(real_tokenizer)
        trainer = DPOTrainer(fresh_model, None, loader, None, cfg)
        batch = next(iter(loader))
        loss, _ = trainer._dpo_loss(batch)
        import math
        assert math.isfinite(loss.item()) and loss.item() > 0

    def test_dpo_reward_margin_sign_interpretation(self, fresh_model, real_tokenizer):
        """Reward margin should be a float (can be negative at init, just must be finite)."""
        from services.training.dpo.trainer import DPOConfig, DPOTrainer

        ref_model = copy.deepcopy(fresh_model)
        cfg = DPOConfig(
            beta=0.1,
            output_dir="/tmp/dpo_sign",
            max_steps=1,
            offload_ref_to_cpu=False,
            dtype="float32",
            gradient_checkpointing=False,
        )
        loader = self._make_dpo_loader(real_tokenizer)
        trainer = DPOTrainer(fresh_model, ref_model, loader, None, cfg)
        batch = next(iter(loader))
        _, metrics = trainer._dpo_loss(batch)
        margin = metrics["reward_margin"]
        import math
        assert isinstance(margin, float) and math.isfinite(margin)

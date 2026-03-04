"""
SFT training entry point.

Usage:
    python -m services.training.sft.train --config configs/sft.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from observability.logging.logger import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        config_dir = config_path.parent
        merged = OmegaConf.create({})
        for entry in cfg.defaults:
            name = entry if isinstance(entry, str) else list(entry.values())[0]
            merged = OmegaConf.merge(merged, OmegaConf.load(config_dir / f"{name}.yaml"))
        cfg = OmegaConf.merge(merged, OmegaConf.masked_copy(cfg, [k for k in cfg if k != "defaults"]))

    if args.output_dir:
        cfg.sft.output_dir = args.output_dir
    if args.base_model_path:
        cfg.sft.base_model_path = args.base_model_path
    if args.max_steps:
        cfg.sft.max_steps = args.max_steps

    configure_logging(service_name="sft", json_logs=False, environment="training")
    logger = get_logger(__name__)

    from torch.utils.data import DataLoader

    from services.data.preprocessing.processor import SFTDataset
    from services.data.tokenization.tokenizer import get_default_tokenizer
    from services.training.core_model.config import ModelConfig
    from services.training.core_model.model import LLMModel
    from services.training.sft.trainer import SFTConfig, SFTTrainer

    base_model_path = cfg.sft.base_model_path
    checkpoint_config = Path(base_model_path) / "config.json"
    if checkpoint_config.exists():
        model = LLMModel.from_pretrained(base_model_path)
        logger.info("base_model_loaded", path=base_model_path)
    else:
        logger.warning(
            "base_model_not_found",
            path=base_model_path,
            message="Checkpoint not found; initialising model from scratch using model config",
        )
        model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
        model_cfg = ModelConfig(**model_cfg_dict)
        model = LLMModel(model_cfg)
        logger.info("base_model_initialised_from_scratch", n_params=model.num_parameters())

    tokenizer = get_default_tokenizer()

    with open(cfg.data.train_path) as f:
        train_records = [json.loads(l) for l in f]
    with open(cfg.data.val_path) as f:
        val_records = [json.loads(l) for l in f]

    train_ds = SFTDataset(
        train_records, tokenizer,
        max_seq_len=cfg.data.max_seq_len,
        template=cfg.data.get("template"),
    )
    val_ds = SFTDataset(
        val_records, tokenizer,
        max_seq_len=cfg.data.max_seq_len,
        template=cfg.data.get("template"),
    )

    sft_cfg = SFTConfig(**OmegaConf.to_container(cfg.sft, resolve=True))
    train_loader = DataLoader(train_ds, batch_size=sft_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=sft_cfg.batch_size, shuffle=False)

    trainer = SFTTrainer(model, train_loader, val_loader, sft_cfg, logger)
    trainer.train()


if __name__ == "__main__":
    main()

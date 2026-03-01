"""
SFT training entry point.

Usage:
    python -m services.training.sft.train --config configs/sft.yaml
"""

from __future__ import annotations

import argparse
import json

from omegaconf import OmegaConf

from observability.logging.logger import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.output_dir:
        cfg.sft.output_dir = args.output_dir
    if args.base_model_path:
        cfg.sft.base_model_path = args.base_model_path
    if args.max_steps:
        cfg.sft.max_steps = args.max_steps

    configure_logging(service_name="sft", json_logs=False, environment="training")
    logger = get_logger(__name__)

    from services.training.core_model.model import LLMModel
    from services.training.sft.trainer import SFTConfig, SFTTrainer
    from services.data.preprocessing.processor import SFTDataset
    from services.data.tokenization.tokenizer import get_default_tokenizer
    from torch.utils.data import DataLoader

    model = LLMModel.from_pretrained(cfg.sft.base_model_path)
    logger.info("base_model_loaded", path=cfg.sft.base_model_path)

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

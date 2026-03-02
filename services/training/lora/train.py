"""
LoRA fine-tuning entry point.

Usage:
    python -m services.training.lora.train --config configs/lora.yaml
    python -m services.training.lora.train --config configs/lora.yaml --merge  # auto-merge on finish
"""

from __future__ import annotations

import argparse
import json

from omegaconf import OmegaConf

from observability.logging.logger import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--merge", action="store_true", help="Merge adapter after training")
    parser.add_argument("--merge_save_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.output_dir:
        cfg.lora.output_dir = args.output_dir
    if args.base_model_path:
        cfg.lora.base_model_path = args.base_model_path
    if args.max_steps:
        cfg.lora.max_steps = args.max_steps

    configure_logging(service_name="lora", json_logs=False, environment="training")
    logger = get_logger(__name__)

    from torch.utils.data import DataLoader

    from services.data.preprocessing.processor import SFTDataset
    from services.data.tokenization.tokenizer import get_default_tokenizer
    from services.training.core_model.model import LLMModel
    from services.training.lora.trainer import LoRAConfig, LoRATrainer

    model = LLMModel.from_pretrained(cfg.lora.base_model_path)
    tokenizer = get_default_tokenizer()

    with open(cfg.data.train_path) as f:
        train_records = [json.loads(l) for l in f]
    with open(cfg.data.val_path) as f:
        val_records = [json.loads(l) for l in f]

    train_ds = SFTDataset(train_records, tokenizer, max_seq_len=cfg.data.max_seq_len)
    val_ds = SFTDataset(val_records, tokenizer, max_seq_len=cfg.data.max_seq_len)

    lora_cfg = LoRAConfig(**OmegaConf.to_container(cfg.lora, resolve=True))
    train_loader = DataLoader(train_ds, batch_size=lora_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=lora_cfg.batch_size, shuffle=False)

    trainer = LoRATrainer(model, train_loader, val_loader, lora_cfg, logger)
    trainer.train()

    if args.merge:
        merge_dir = args.merge_save_dir or f"{cfg.lora.output_dir}/merged"
        trainer.merge_and_save(merge_dir)
        logger.info("adapter_merged", path=merge_dir)


if __name__ == "__main__":
    main()

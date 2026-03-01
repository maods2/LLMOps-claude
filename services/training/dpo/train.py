"""
DPO training entry point.

Usage:
    python -m services.training.dpo.train --config configs/dpo.yaml
"""

from __future__ import annotations

import argparse
import copy
import json

from omegaconf import OmegaConf

from observability.logging.logger import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct Preference Optimisation (DPO)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.output_dir:
        cfg.dpo.output_dir = args.output_dir
    if args.base_model_path:
        cfg.dpo.base_model_path = args.base_model_path
    if args.max_steps:
        cfg.dpo.max_steps = args.max_steps

    configure_logging(service_name="dpo", json_logs=False, environment="training")
    logger = get_logger(__name__)

    from services.training.core_model.model import LLMModel
    from services.training.dpo.trainer import DPOConfig, DPOTrainer
    from services.data.preprocessing.processor import DPODataset
    from services.data.tokenization.tokenizer import get_default_tokenizer
    from torch.utils.data import DataLoader

    # Policy model (to be optimised)
    policy = LLMModel.from_pretrained(cfg.dpo.base_model_path)
    # Reference model (frozen copy of policy)
    ref_model = copy.deepcopy(policy)
    logger.info("models_loaded", path=cfg.dpo.base_model_path)

    tokenizer = get_default_tokenizer()

    with open(cfg.data.train_path) as f:
        train_records = [json.loads(l) for l in f]
    with open(cfg.data.val_path) as f:
        val_records = [json.loads(l) for l in f]

    train_ds = DPODataset(train_records, tokenizer, max_seq_len=cfg.data.max_seq_len)
    val_ds = DPODataset(val_records, tokenizer, max_seq_len=cfg.data.max_seq_len)

    dpo_cfg = DPOConfig(**OmegaConf.to_container(cfg.dpo, resolve=True))
    train_loader = DataLoader(train_ds, batch_size=dpo_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=dpo_cfg.batch_size, shuffle=False)

    trainer = DPOTrainer(policy, ref_model, train_loader, val_loader, dpo_cfg, logger)
    trainer.train()


if __name__ == "__main__":
    main()

"""
Pretraining entry point.

Usage:
    python -m services.training.pretrain.train --config configs/pretrain.yaml
    python -m services.training.pretrain.train --config configs/pretrain.yaml --max_steps 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from observability.logging.logger import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain LLM from scratch")
    parser.add_argument("--config", type=str, required=True, help="Path to pretrain config YAML")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    # Load base config, resolving any Hydra-style `defaults` lists manually
    # (plain OmegaConf.load does not process `defaults`)
    config_path = Path(args.config)
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        config_dir = config_path.parent
        merged = OmegaConf.create({})
        for entry in cfg.defaults:
            name = entry if isinstance(entry, str) else list(entry.values())[0]
            merged = OmegaConf.merge(merged, OmegaConf.load(config_dir / f"{name}.yaml"))
        cfg = OmegaConf.merge(merged, OmegaConf.masked_copy(cfg, [k for k in cfg if k != "defaults"]))

    # Apply CLI overrides
    if args.output_dir:
        cfg.pretrain.output_dir = args.output_dir
    if args.max_steps:
        cfg.pretrain.max_steps = args.max_steps
    if args.batch_size:
        cfg.pretrain.batch_size = args.batch_size
    if args.learning_rate:
        cfg.pretrain.learning_rate = args.learning_rate
    if args.dtype:
        cfg.pretrain.dtype = args.dtype
    if args.resume_from:
        cfg.pretrain.resume_from = args.resume_from

    configure_logging(
        service_name="pretrain",
        log_level="INFO",
        json_logs=False,
        environment="training",
    )
    logger = get_logger(__name__)

    # Build tokenizer first — model vocab_size must match the actual tokenizer
    from services.data.tokenization.tokenizer import get_default_tokenizer

    tokenizer = get_default_tokenizer(cache_dir=cfg.data.get("cache_dir"))

    # Build model with vocab_size derived from the tokenizer to avoid
    # embedding index-out-of-bounds errors at runtime
    from services.training.core_model.config import ModelConfig
    from services.training.core_model.model import LLMModel
    from services.training.pretrain.trainer import PretrainConfig, PretrainTrainer

    model_cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg_dict["vocab_size"] = len(tokenizer)
    model_cfg_dict["pad_token_id"] = tokenizer.pad_token_id or 0
    model_cfg_dict["bos_token_id"] = tokenizer.bos_token_id or tokenizer.eos_token_id
    model_cfg_dict["eos_token_id"] = tokenizer.eos_token_id
    model_cfg = ModelConfig(**model_cfg_dict)
    model = LLMModel(model_cfg)

    n_params = model.num_parameters()
    logger.info(
        "model_created",
        n_params=n_params,
        n_params_m=f"{n_params / 1e6:.1f}M",
        config=OmegaConf.to_yaml(cfg.model),
    )

    # Build data

    from services.data.ingestion.loader import DataSourceConfig, load_source
    from services.data.preprocessing.processor import (
        PreprocessingConfig,
        build_pretrain_dataloader,
    )
    pp_cfg = PreprocessingConfig(
        max_seq_len=cfg.data.max_seq_len,
        batch_size=cfg.pretrain.batch_size,
    )

    data_cfg = DataSourceConfig(**OmegaConf.to_container(cfg.data.sources[0], resolve=True))
    train_ds = load_source(data_cfg)
    train_loader = build_pretrain_dataloader(train_ds, tokenizer, pp_cfg)

    val_loader = None
    if "val_sources" in cfg.data:
        val_data_cfg = DataSourceConfig(
            **OmegaConf.to_container(cfg.data.val_sources[0], resolve=True)
        )
        val_ds = load_source(val_data_cfg)
        val_loader = build_pretrain_dataloader(val_ds, tokenizer, pp_cfg)

    # Train
    train_cfg = PretrainConfig(**OmegaConf.to_container(cfg.pretrain, resolve=True))
    trainer = PretrainTrainer(model, train_loader, val_loader, train_cfg, logger)
    trainer.train()


if __name__ == "__main__":
    main()

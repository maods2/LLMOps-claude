"""
Dataset ingestion: streaming loaders for supported datasets.

Supported sources:
  - wikitext-2-raw-v1  (HuggingFace datasets)
  - openwebtext        (HuggingFace datasets, filtered subset)
  - tinystories        (HuggingFace datasets)
  - local .jsonl files (custom data)

All loaders return HuggingFace Dataset / IterableDataset objects so they can
be composed uniformly downstream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional, Union

import datasets
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset

from pydantic import BaseModel, Field


class DataSourceConfig(BaseModel):
    """Configuration for a dataset source."""

    name: str = Field(description="Dataset name: wikitext2 | openwebtext | tinystories | local")
    split: str = Field(default="train", description="Dataset split")
    streaming: bool = Field(default=True, description="Use streaming mode (saves disk space)")
    subset_size: Optional[int] = Field(
        default=None, description="Limit to N examples (for dev/testing)"
    )
    local_path: Optional[str] = Field(
        default=None, description="Path to local .jsonl file (for 'local' source)"
    )
    text_column: str = Field(default="text", description="Column containing raw text")
    cache_dir: Optional[str] = Field(default=None, description="Cache directory for downloads")

    class Config:
        frozen = True


def load_wikitext2(
    split: str = "train",
    streaming: bool = True,
    cache_dir: Optional[str] = None,
) -> Union[Dataset, IterableDataset]:
    """Load WikiText-2 dataset."""
    return load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )


def load_openwebtext(
    split: str = "train",
    streaming: bool = True,
    subset_size: Optional[int] = 50_000,
    cache_dir: Optional[str] = None,
) -> Union[Dataset, IterableDataset]:
    """Load a filtered subset of OpenWebText."""
    ds = load_dataset(
        "openwebtext",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    if subset_size is not None and streaming:
        ds = ds.take(subset_size)  # type: ignore[union-attr]
    elif subset_size is not None:
        ds = ds.select(range(min(subset_size, len(ds))))  # type: ignore[arg-type]
    return ds


def load_tinystories(
    split: str = "train",
    streaming: bool = True,
    cache_dir: Optional[str] = None,
) -> Union[Dataset, IterableDataset]:
    """Load TinyStories dataset."""
    return load_dataset(
        "roneneldan/TinyStories",
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )


def load_local_jsonl(
    path: str,
    text_column: str = "text",
    streaming: bool = True,
) -> Union[Dataset, IterableDataset]:
    """Load a local .jsonl file.

    Each line must be valid JSON with at least a ``text_column`` key.
    Example line: {"text": "Hello world", "source": "custom"}
    """
    if streaming:
        def _generator() -> Iterator[dict]:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        yield {text_column: record.get(text_column, "")}

        return IterableDataset.from_generator(_generator)
    else:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    records.append({text_column: record.get(text_column, "")})
        return Dataset.from_list(records)


def load_source(config: DataSourceConfig) -> Union[Dataset, IterableDataset]:
    """Dispatch to the appropriate loader based on config.name."""
    name = config.name.lower()
    if name == "wikitext2":
        ds = load_wikitext2(config.split, config.streaming, config.cache_dir)
    elif name == "openwebtext":
        ds = load_openwebtext(
            config.split, config.streaming, config.subset_size, config.cache_dir
        )
    elif name == "tinystories":
        ds = load_tinystories(config.split, config.streaming, config.cache_dir)
    elif name == "local":
        assert config.local_path, "local_path must be set for 'local' source"
        ds = load_local_jsonl(config.local_path, config.text_column, config.streaming)
    else:
        raise ValueError(f"Unknown dataset source: {config.name!r}")

    # Apply subset limit for non-streaming datasets
    if (
        config.subset_size is not None
        and not config.streaming
        and isinstance(ds, Dataset)
    ):
        ds = ds.select(range(min(config.subset_size, len(ds))))

    return ds


def load_combined(
    sources: list[DataSourceConfig],
) -> Union[Dataset, IterableDataset]:
    """Load and concatenate multiple dataset sources."""
    datasets_list = [load_source(cfg) for cfg in sources]
    if not datasets_list:
        raise ValueError("No sources provided")
    if len(datasets_list) == 1:
        return datasets_list[0]

    # All streaming or all non-streaming
    if all(isinstance(d, IterableDataset) for d in datasets_list):
        return datasets.concatenate_datasets(datasets_list)  # type: ignore[arg-type]
    else:
        return datasets.concatenate_datasets(datasets_list)  # type: ignore[arg-type]

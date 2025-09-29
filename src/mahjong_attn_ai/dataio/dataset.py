"""Dataset utilities wrapping parsed mahjong samples."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import random

import torch
from torch.utils.data import DataLoader, Dataset

from .parser import SyntheticKifuParser
from .schema import DatasetConfig, MahjongBatch, MahjongSample


class MahjongDataset(Dataset[MahjongSample]):
    """Thin ``Dataset`` wrapper around in-memory samples."""

    def __init__(self, samples: Sequence[MahjongSample]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self._samples)

    def __getitem__(self, index: int) -> MahjongSample:
        return self._samples[index]


def _collate(batch: Sequence[MahjongSample]) -> MahjongBatch:
    board_tokens = torch.stack([item["board_tokens"] for item in batch])
    board_positions = torch.stack([item["board_positions"] for item in batch])
    action_tokens = torch.stack([item["action_tokens"] for item in batch])
    action_positions = torch.stack([item["action_positions"] for item in batch])
    legal_mask = torch.stack([item["legal_mask"] for item in batch])
    label_actions = torch.tensor([item["label_action"] for item in batch], dtype=torch.long)
    value_targets = torch.tensor([item["value_target"] for item in batch], dtype=torch.float)
    aux_targets = torch.tensor([item["aux_target"] for item in batch], dtype=torch.float)
    metadata = [item["metadata"] for item in batch]
    return MahjongBatch(
        board_tokens=board_tokens,
        board_positions=board_positions,
        action_tokens=action_tokens,
        action_positions=action_positions,
        legal_mask=legal_mask,
        label_actions=label_actions,
        value_targets=value_targets,
        aux_targets=aux_targets,
        metadata=metadata,
    )


def _split_samples(
    samples: Sequence[MahjongSample], val_ratio: float, seed: int
) -> Tuple[List[MahjongSample], List[MahjongSample]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must lie in (0, 1)")

    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    split = int(len(indices) * (1.0 - val_ratio))
    train_ids, val_ids = indices[:split], indices[split:]
    return [samples[i] for i in train_ids], [samples[i] for i in val_ids]


@dataclass
class DataLoaderBundle:
    """Container for training/validation data loaders."""

    train_loader: DataLoader[MahjongBatch]
    val_loader: DataLoader[MahjongBatch]


def build_dataloaders(
    data_dir: Path,
    batch_size: int = 16,
    val_ratio: float = 0.2,
    seed: int = 7,
    dataset_config: DatasetConfig | None = None,
) -> DataLoaderBundle:
    """Load synthetic kifus and return train/validation loaders."""

    parser = SyntheticKifuParser(data_dir, dataset_config)
    samples = parser.load()
    train_samples, val_samples = _split_samples(samples, val_ratio, seed)

    train_dataset = MahjongDataset(train_samples)
    val_dataset = MahjongDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    return DataLoaderBundle(train_loader=train_loader, val_loader=val_loader)


__all__ = [
    "MahjongDataset",
    "MahjongBatch",
    "MahjongSample",
    "DataLoaderBundle",
    "build_dataloaders",
]


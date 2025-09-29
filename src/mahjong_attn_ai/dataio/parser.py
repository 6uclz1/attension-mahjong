"""Utilities for turning raw (synthetic) kifu files into model-ready samples."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import json
import random

import torch
import yaml

from .schema import DatasetConfig, MahjongSample


def _pad_sequence(values: List[int], length: int, pad_value: int = 0) -> List[int]:
    if len(values) >= length:
        return values[:length]
    return values + [pad_value] * (length - len(values))


class SyntheticKifuParser:
    """Parse synthetic kifu files shipped with the template repository."""

    def __init__(self, data_dir: Path, config: DatasetConfig | None = None) -> None:
        self.data_dir = data_dir
        self.config = config or DatasetConfig()

    def load(self) -> List[MahjongSample]:
        """Load every supported file in ``data_dir`` and produce training samples."""

        raw_samples: List[Dict[str, Any]] = []
        for path in sorted(self.data_dir.glob("**/*")):
            if path.suffix.lower() in {".yaml", ".yml"}:
                raw_samples.extend(self._load_yaml(path))
            elif path.suffix.lower() == ".json":
                raw_samples.extend(self._load_json(path))
            elif path.suffix.lower() == ".jsonl":
                raw_samples.extend(self._load_jsonl(path))

        if not raw_samples:
            raise FileNotFoundError(
                f"No synthetic kifu files found in {self.data_dir!s}. "
                "Add files or disable data loading."
            )

        samples: List[MahjongSample] = []
        for record in raw_samples:
            samples.extend(self._materialise_samples(record))

        random.shuffle(samples)
        return samples

    def _load_yaml(self, path: Path) -> List[Dict[str, Any]]:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return list(data.get("samples", []))

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("samples", [])) if isinstance(data, dict) else list(data)

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def _materialise_samples(self, record: Dict[str, Any]) -> Iterable[MahjongSample]:
        board_tokens: List[int] = record.get("board_tokens", [])
        action_tokens: List[int] = record.get("action_tokens", [])
        legal_mask: List[int] = record.get("legal_mask", [1] * len(action_tokens))
        board_padded = _pad_sequence(board_tokens, self.config.board_seq_len)
        action_padded = _pad_sequence(action_tokens, self.config.action_seq_len)
        legal_padded = _pad_sequence(legal_mask, self.config.action_seq_len, pad_value=0)

        label_action = int(record.get("label_action", 0))
        label_action = max(0, min(label_action, self.config.action_seq_len - 1))
        value_target = float(record.get("value_target", 0.0))
        aux_target = float(record.get("aux_target", 0.0))

        metadata: Dict[str, Any] = {
            "game_id": record.get("game_id", "synthetic"),
            "round_id": record.get("round_id", 0),
            "turn_id": record.get("turn_id", 0),
        }

        rotations = range(4) if self.config.include_seat_rotation else range(1)
        for rotation in rotations:
            rotated_metadata = metadata | {"seat": rotation}
            yield {
                "board_tokens": torch.tensor(board_padded, dtype=torch.long),
                "board_positions": torch.arange(
                    self.config.board_seq_len, dtype=torch.long
                ),
                "action_tokens": torch.tensor(action_padded, dtype=torch.long),
                "action_positions": torch.arange(
                    self.config.action_seq_len, dtype=torch.long
                ),
                "legal_mask": torch.tensor(legal_padded, dtype=torch.bool),
                "label_action": label_action,
                "value_target": value_target,
                "aux_target": aux_target,
                "metadata": rotated_metadata,
            }


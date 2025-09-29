"""Utilities for turning raw (synthetic) kifu files into model-ready samples."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import json
import random

import torch
import yaml

from .schema import DatasetConfig, MahjongSample
from ..features.utils import MahjongVocabulary, pad_or_trim


class SyntheticKifuParser:
    """Parse synthetic kifu files shipped with the template repository."""

    def __init__(
        self,
        data_dir: Path,
        config: DatasetConfig | None = None,
        vocab: MahjongVocabulary | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.config = config or DatasetConfig()
        self.vocab = vocab or MahjongVocabulary.build_default()

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

        if self.config.auto_generate > 0:
            samples.extend(self._generate_synthetic(self.config.auto_generate))

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
        board_tokens_raw: Sequence[int | str] = record.get(
            "board_tokens", record.get("board_tiles", [])
        )
        action_tokens_raw: Sequence[int | str] = record.get(
            "action_tokens", record.get("action_labels", [])
        )

        board_tokens = [self.vocab.resolve_board_token(token) for token in board_tokens_raw]
        action_tokens = [self.vocab.resolve_action_token(token) for token in action_tokens_raw]
        legal_mask: List[int] = record.get("legal_mask", [1] * len(action_tokens))
        board_padded = pad_or_trim(board_tokens, self.config.board_seq_len, self.vocab.pad_id)
        action_padded = pad_or_trim(
            action_tokens, self.config.action_seq_len, self.vocab.action_pad_id
        )
        legal_padded = pad_or_trim(legal_mask, self.config.action_seq_len, 0)

        label_raw = record.get("label_action", 0)
        if isinstance(label_raw, str):
            target_token = self.vocab.resolve_action_token(label_raw)
            try:
                label_action = action_padded.index(target_token)
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError("label_action not present in action_tokens") from exc
        else:
            label_action = int(label_raw)
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

    def _generate_synthetic(self, num_games: int) -> List[MahjongSample]:
        rng = random.Random(self.config.auto_generate_seed)
        samples: List[MahjongSample] = []
        valid_board_tokens = list(range(1, self.vocab.num_board_tokens))
        valid_action_tokens = [
            token for token in range(1, self.vocab.num_action_tokens)
        ]
        num_actions = min(self.config.auto_generate_actions, len(valid_action_tokens))

        for game_idx in range(num_games):
            board = rng.choices(valid_board_tokens, k=self.config.board_seq_len)
            actions = rng.sample(valid_action_tokens, k=num_actions)
            action_sequence = pad_or_trim(actions, self.config.action_seq_len, self.vocab.action_pad_id)
            legal_mask = [1] * num_actions
            legal_mask = pad_or_trim(legal_mask, self.config.action_seq_len, 0)
            label_action = rng.randint(0, num_actions - 1)
            template = {
                "board_tokens": board,
                "action_tokens": action_sequence,
                "legal_mask": legal_mask,
                "label_action": label_action,
                "value_target": rng.uniform(-1.0, 1.0),
                "aux_target": rng.uniform(0.0, 1.0),
                "game_id": f"auto-{game_idx}",
                "round_id": rng.randint(0, 7),
                "turn_id": rng.randint(0, 17),
            }
            samples.extend(self._materialise_samples(template))
        return samples

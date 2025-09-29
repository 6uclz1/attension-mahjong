"""Data ingestion helpers for synthetic and real mahjong logs."""
from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
import yaml

from .schema import DatasetConfig, MahjongSample
from ..features.utils import MahjongVocabulary, pad_or_trim

ConverterFn = Callable[[Path, DatasetConfig, MahjongVocabulary, Dict[str, Any]], Iterable[Dict[str, Any]]]


class BaseKifuParser:
    """Common interface shared by all kifu parsers."""

    def __init__(
        self,
        data_dir: Path,
        config: DatasetConfig,
        vocab: Optional[MahjongVocabulary] = None,
    ) -> None:
        self.data_dir = data_dir
        self.config = config
        self.vocab = vocab or MahjongVocabulary.build_default()

    def load(self) -> List[MahjongSample]:  # pragma: no cover - overridden
        raise NotImplementedError


class SyntheticKifuParser(BaseKifuParser):
    """Parse synthetic kifu files shipped with the repository."""

    def load(self) -> List[MahjongSample]:
        raw_records: List[Dict[str, Any]] = []
        for path in sorted(self.data_dir.glob("**/*")):
            loader = _pick_loader(path.suffix.lower())
            if loader is None:
                continue
            raw_records.extend(loader(path))

        if not raw_records:
            raise FileNotFoundError(
                f"No synthetic kifu files found in {self.data_dir!s}. "
                "Add files or adjust dataset.parser settings."
            )

        samples: List[MahjongSample] = []
        for record in raw_records:
            samples.extend(_materialise_samples(record, self.config, self.vocab))

        if self.config.auto_generate > 0:
            samples.extend(self._generate_synthetic(self.config.auto_generate))

        random.shuffle(samples)
        return samples

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
            samples.extend(_materialise_samples(template, self.config, self.vocab))
        return samples


class MjaiKifuParser(BaseKifuParser):
    """Parse logs exported in the mjai JSONL format.

    The converter is pluggable so downstream users can implement a richer feature
    builder while leveraging the shared materialisation helpers.
    """

    def __init__(
        self,
        data_dir: Path,
        config: DatasetConfig,
        vocab: Optional[MahjongVocabulary] = None,
        converter: ConverterFn | None = None,
        options: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(data_dir, config, vocab)
        self.options = options or {}
        converter_path = self.options.get("converter")
        if converter is not None:
            self.converter = converter
        elif converter_path:
            self.converter = _resolve_converter(converter_path)
        else:
            self.converter = default_mjai_converter

    def load(self) -> List[MahjongSample]:
        glob_pattern = self.options.get("glob", "**/*.jsonl")
        paths = sorted(self.data_dir.glob(glob_pattern))
        if not paths:
            raise FileNotFoundError(
                f"No mjai logs matching '{glob_pattern}' found under {self.data_dir!s}."
            )

        samples: List[MahjongSample] = []
        for path in paths:
            records = self.converter(path, self.config, self.vocab, self.options)
            for record in records:
                samples.extend(_materialise_samples(record, self.config, self.vocab))
        if not samples:
            raise RuntimeError(
                "MjaiKifuParser produced no samples. Provide a custom converter via "
                "dataset.parser_options.converter"
            )
        random.shuffle(samples)
        return samples


def build_kifu_parser(
    data_dir: Path,
    config: DatasetConfig,
    vocab: MahjongVocabulary,
) -> BaseKifuParser:
    parser_type = config.parser.lower()
    options = config.parser_options or {}
    if parser_type == "synthetic":
        return SyntheticKifuParser(data_dir=data_dir, config=config, vocab=vocab)
    if parser_type == "mjai":
        return MjaiKifuParser(
            data_dir=data_dir,
            config=config,
            vocab=vocab,
            options=options,
        )
    raise ValueError(f"Unsupported dataset.parser '{config.parser}'.")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pick_loader(suffix: str) -> Callable[[Path], List[Dict[str, Any]]] | None:
    if suffix in {".yaml", ".yml"}:
        return _load_yaml
    if suffix == ".json":
        return _load_json
    if suffix == ".jsonl":
        return _load_jsonl
    return None


def _load_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "samples" in data:
        return list(data.get("samples", []))
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unsupported JSON structure in {path!s}. Expected list or dict with 'samples'.")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _materialise_samples(
    record: Dict[str, Any],
    config: DatasetConfig,
    vocab: MahjongVocabulary,
) -> Iterator[MahjongSample]:
    board_tokens_raw: Sequence[int | str] = record.get("board_tokens", record.get("board_tiles", []))
    action_tokens_raw: Sequence[int | str] = record.get("action_tokens", record.get("action_labels", []))

    board_tokens = [vocab.resolve_board_token(token) for token in board_tokens_raw]
    action_tokens = [vocab.resolve_action_token(token) for token in action_tokens_raw]
    legal_mask: List[int] = record.get("legal_mask", [1] * len(action_tokens))
    board_padded = pad_or_trim(board_tokens, config.board_seq_len, vocab.pad_id)
    action_padded = pad_or_trim(action_tokens, config.action_seq_len, vocab.action_pad_id)
    legal_padded = pad_or_trim(legal_mask, config.action_seq_len, 0)

    label_raw = record.get("label_action", 0)
    if isinstance(label_raw, str):
        target_token = vocab.resolve_action_token(label_raw)
        try:
            label_action = action_padded.index(target_token)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("label_action not present in action_tokens") from exc
    else:
        label_action = int(label_raw)
    label_action = max(0, min(label_action, config.action_seq_len - 1))
    value_target = float(record.get("value_target", 0.0))
    aux_target = float(record.get("aux_target", 0.0))

    metadata: Dict[str, Any] = {
        "game_id": record.get("game_id", "synthetic"),
        "round_id": record.get("round_id", 0),
        "turn_id": record.get("turn_id", 0),
    }

    rotations = range(4) if config.include_seat_rotation else range(1)
    position_indices = torch.arange(config.board_seq_len, dtype=torch.long)
    action_indices = torch.arange(config.action_seq_len, dtype=torch.long)
    for rotation in rotations:
        rotated_metadata = metadata | {"seat": rotation}
        yield {
            "board_tokens": torch.tensor(board_padded, dtype=torch.long),
            "board_positions": position_indices.clone(),
            "action_tokens": torch.tensor(action_padded, dtype=torch.long),
            "action_positions": action_indices.clone(),
            "legal_mask": torch.tensor(legal_padded, dtype=torch.bool),
            "label_action": label_action,
            "value_target": value_target,
            "aux_target": aux_target,
            "metadata": rotated_metadata,
        }


def _resolve_converter(path: str) -> ConverterFn:
    module_name, _, attribute = path.rpartition(".")
    if not module_name:
        raise ValueError("converter must be a dotted path to a callable, e.g. 'pkg.module.fn'")
    module = importlib.import_module(module_name)
    try:
        func = getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(f"Converter '{attribute}' not found in module '{module_name}'.") from exc
    if not callable(func):
        raise TypeError(f"Resolved converter '{path}' is not callable.")
    return func  # type: ignore[return-value]


def default_mjai_converter(
    path: Path,
    config: DatasetConfig,
    vocab: MahjongVocabulary,
    options: Dict[str, Any],
) -> Iterable[Dict[str, Any]]:
    """Fallback converter for mjai logs.

    The default implementation expects each JSON/JSONL record to already provide
    ``board_tokens``/``action_tokens``/``label_action`` fields. This mirrors the
    synthetic data schema and lets teams plug in lightweight preprocessing scripts
    that dump intermediate tensors from real mjai replays.
    """

    suffix = path.suffix.lower()
    loader = _pick_loader(suffix)
    if loader is None:
        raise ValueError(f"Unsupported mjai file extension '{suffix}' for {path!s}.")
    return loader(path)


__all__ = [
    "BaseKifuParser",
    "SyntheticKifuParser",
    "MjaiKifuParser",
    "build_kifu_parser",
    "default_mjai_converter",
]
def _load_yaml(path: Path) -> List[Dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "samples" in data:
        return list(data.get("samples", []))
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unsupported YAML structure in {path!s}. Expected list or dict with 'samples'.")

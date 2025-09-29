"""Schema definitions for mahjong transformer training samples."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, TypedDict

import torch

DEFAULT_BOARD_SEQ_LEN = 96
DEFAULT_ACTION_SEQ_LEN = 16


class MahjongSample(TypedDict):
    """Single training example extracted from a kifu."""

    board_tokens: torch.LongTensor
    board_positions: torch.LongTensor
    action_tokens: torch.LongTensor
    action_positions: torch.LongTensor
    legal_mask: torch.BoolTensor
    label_action: int
    value_target: float
    aux_target: float
    metadata: Dict[str, Any]


@dataclass
class MahjongBatch:
    """Mini-batch container used across training and evaluation."""

    board_tokens: torch.LongTensor
    board_positions: torch.LongTensor
    action_tokens: torch.LongTensor
    action_positions: torch.LongTensor
    legal_mask: torch.BoolTensor
    label_actions: torch.LongTensor
    value_targets: torch.FloatTensor
    aux_targets: torch.FloatTensor
    metadata: List[Dict[str, Any]]

    def to_device(self, device: torch.device) -> "MahjongBatch":
        """Move tensor fields to a device and return self."""

        self.board_tokens = self.board_tokens.to(device)
        self.board_positions = self.board_positions.to(device)
        self.action_tokens = self.action_tokens.to(device)
        self.action_positions = self.action_positions.to(device)
        self.legal_mask = self.legal_mask.to(device)
        self.label_actions = self.label_actions.to(device)
        self.value_targets = self.value_targets.to(device)
        self.aux_targets = self.aux_targets.to(device)
        return self


@dataclass
class DatasetConfig:
    """Config controlling dataset slicing behaviour."""

    board_seq_len: int = DEFAULT_BOARD_SEQ_LEN
    action_seq_len: int = DEFAULT_ACTION_SEQ_LEN
    include_seat_rotation: bool = True
    auto_generate: int = 0
    auto_generate_actions: int = 6
    auto_generate_seed: int = 1234
    parser: str = "synthetic"
    parser_options: Dict[str, Any] = field(default_factory=dict)

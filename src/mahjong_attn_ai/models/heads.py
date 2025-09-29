"""Output heads for the mahjong transformer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from ..features.legal_mask import apply_legal_mask


@dataclass
class HeadOutputs:
    policy_logits: torch.Tensor
    policy_log_probs: torch.Tensor
    value: torch.Tensor
    aux: Dict[str, torch.Tensor]


class PolicyHead(nn.Module):
    """Scalar head projecting action embeddings to logits."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, action_states: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        logits = self.proj(action_states).squeeze(-1)
        return apply_legal_mask(logits, legal_mask)


class ValueHead(nn.Module):
    """Predict expected value of the full hand (per hand or game)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, board_state: torch.Tensor) -> torch.Tensor:
        return self.mlp(board_state).squeeze(-1)


class AuxiliaryHead(nn.Module):
    """Predict auxiliary supervision targets such as danger rate."""

    def __init__(self, d_model: int, name: str = "danger") -> None:
        super().__init__()
        self.name = name
        self.layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, board_state: torch.Tensor) -> torch.Tensor:
        return self.layer(board_state).squeeze(-1)


__all__ = ["PolicyHead", "ValueHead", "AuxiliaryHead", "HeadOutputs"]


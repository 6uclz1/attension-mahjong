"""Transformer-friendly feature encoders for board state and action candidates."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .utils import MahjongVocabulary


class BoardFeatureEncoder(nn.Module):
    """Embed board token and position ids into a dense representation."""

    def __init__(
        self,
        vocab: MahjongVocabulary,
        d_model: int = 128,
        max_position: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.token_embedding = nn.Embedding(len(vocab.token_to_id), d_model)
        self.position_embedding = nn.Embedding(max_position, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, board_tokens: torch.LongTensor, board_positions: torch.LongTensor
    ) -> torch.Tensor:
        token_emb = self.token_embedding(board_tokens)
        pos_emb = self.position_embedding(board_positions)
        hidden = token_emb + pos_emb
        return self.dropout(self.layer_norm(hidden))


class ActionFeatureEncoder(nn.Module):
    """Embed action candidate ids before cross-attention."""

    def __init__(
        self,
        vocab: MahjongVocabulary,
        d_model: int = 128,
        max_position: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.token_embedding = nn.Embedding(len(vocab.action_to_id), d_model)
        self.position_embedding = nn.Embedding(max_position, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, action_tokens: torch.LongTensor, action_positions: torch.LongTensor
    ) -> torch.Tensor:
        token_emb = self.token_embedding(action_tokens)
        pos_emb = self.position_embedding(action_positions)
        hidden = token_emb + pos_emb
        return self.dropout(self.layer_norm(hidden))


__all__ = ["BoardFeatureEncoder", "ActionFeatureEncoder"]


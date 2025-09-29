"""Transformer encoder with cross-attention over action candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from ..dataio.schema import MahjongBatch
from ..features.encoders import ActionFeatureEncoder, BoardFeatureEncoder
from ..features.legal_mask import masked_log_softmax
from ..features.utils import MahjongVocabulary
from .heads import AuxiliaryHead, HeadOutputs, PolicyHead, ValueHead


@dataclass
class TransformerConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1


class MahjongTransformerModel(nn.Module):
    """Core model that produces policy/value/aux predictions."""

    def __init__(self, vocab: MahjongVocabulary, config: TransformerConfig | None = None) -> None:
        super().__init__()
        self.vocab = vocab
        self.config = config or TransformerConfig()
        self.board_encoder = BoardFeatureEncoder(
            vocab=vocab,
            d_model=self.config.d_model,
            max_position=256,
            dropout=self.config.dropout,
        )
        self.action_encoder = ActionFeatureEncoder(
            vocab=vocab,
            d_model=self.config.d_model,
            max_position=64,
            dropout=self.config.dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.config.d_model,
            num_heads=self.config.nhead,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.policy_head = PolicyHead(d_model=self.config.d_model, dropout=self.config.dropout)
        self.value_head = ValueHead(d_model=self.config.d_model)
        self.aux_head = AuxiliaryHead(d_model=self.config.d_model, name="danger")

    def encode_board(self, board_tokens: torch.Tensor, board_positions: torch.Tensor) -> torch.Tensor:
        board_hidden = self.board_encoder(board_tokens, board_positions)
        return self.transformer(board_hidden)

    def encode_actions(self, action_tokens: torch.Tensor, action_positions: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(action_tokens, action_positions)

    def forward(self, batch: MahjongBatch) -> HeadOutputs:
        board_states = self.encode_board(batch.board_tokens, batch.board_positions)
        action_states = self.encode_actions(batch.action_tokens, batch.action_positions)
        attn_output, _ = self.cross_attention(query=action_states, key=board_states, value=board_states)

        policy_logits = self.policy_head(attn_output, batch.legal_mask)
        policy_log_probs = masked_log_softmax(policy_logits, batch.legal_mask)
        board_summary = board_states[:, 0, :]
        value = self.value_head(board_summary)
        aux = {self.aux_head.name: self.aux_head(board_summary)}
        return HeadOutputs(
            policy_logits=policy_logits,
            policy_log_probs=policy_log_probs,
            value=value,
            aux=aux,
        )

    def loss(
        self,
        outputs: HeadOutputs,
        batch: MahjongBatch,
        lambda_value: float = 0.1,
        lambda_aux: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss terms."""

        ce_loss = nn.functional.nll_loss(outputs.policy_log_probs, batch.label_actions)
        value_loss = nn.functional.mse_loss(outputs.value, batch.value_targets)
        aux_pred = outputs.aux[self.aux_head.name]
        aux_loss = nn.functional.mse_loss(aux_pred, batch.aux_targets)
        total = ce_loss + lambda_value * value_loss + lambda_aux * aux_loss
        return {
            "total": total,
            "policy": ce_loss,
            "value": value_loss,
            "aux": aux_loss,
        }


__all__ = ["MahjongTransformerModel", "TransformerConfig"]


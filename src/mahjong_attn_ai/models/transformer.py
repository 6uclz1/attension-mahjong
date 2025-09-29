"""Transformer encoder with rotary-position attention over action candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..dataio.schema import MahjongBatch
from ..features.encoders import ActionFeatureEncoder, BoardFeatureEncoder
from ..features.legal_mask import masked_log_softmax
from ..features.utils import MahjongVocabulary
from .heads import AuxiliaryHead, HeadOutputs, PolicyHead, ValueHead


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate tensor halves used inside rotary position embedding."""

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Compute rotary cos/sin caches for supplied position ids."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Rotary embedding dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions.dtype not in (torch.int32, torch.int64):  # defensive, should remain long
            raise TypeError("positions tensor must be integral for rotary embedding")
        pos = positions.float()
        freqs = torch.einsum("b s, d -> b s d", pos, self.inv_freq)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        return cos, sin


def apply_rotary_pos_emb(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Mix projected Q/K tensors with rotary cos/sin values."""

    return (tensor * cos) + (_rotate_half(tensor) * sin)


class RotaryMultiHeadAttention(nn.Module):
    """Multi-head attention that injects rotary embeddings into Q/K projections."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embedding")
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim)

    def _shape(self, tensor: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_positions: torch.LongTensor,
        key_positions: torch.LongTensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        batch, q_len, _ = query.shape
        _, k_len, _ = key.shape

        q = self._shape(self.q_proj(query), batch, q_len)
        k = self._shape(self.k_proj(key), batch, k_len)
        v = self._shape(self.v_proj(value), batch, k_len)

        cos_q, sin_q = self.rotary(query_positions)
        cos_k, sin_k = self.rotary(key_positions)

        cos_q = cos_q.unsqueeze(1)
        sin_q = sin_q.unsqueeze(1)
        cos_k = cos_k.unsqueeze(1)
        sin_k = sin_k.unsqueeze(1)

        q = apply_rotary_pos_emb(q, cos_q, sin_q)
        k = apply_rotary_pos_emb(k, cos_k, sin_k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, q_len, self.embed_dim)
        return self.out_proj(attn_output)


class RotaryTransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder block using rotary self-attention."""

    def __init__(self, config: "TransformerConfig") -> None:
        super().__init__()
        self.self_attn = RotaryMultiHeadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, config.d_model)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        positions: torch.LongTensor,
        padding_mask: Optional[torch.BoolTensor],
    ) -> torch.Tensor:
        residual = src
        src_norm = self.norm1(src)
        attn_out = self.self_attn(src_norm, src_norm, src_norm, positions, positions, padding_mask)
        src = residual + self.dropout(attn_out)

        residual = src
        src_norm = self.norm2(src)
        ff = self.linear2(self.activation(self.linear1(src_norm)))
        return residual + self.dropout(ff)


class AttentionPooling(nn.Module):
    """Learned attention pooling over the board sequence."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = RotaryMultiHeadAttention(d_model, num_heads, dropout)

    def forward(
        self,
        states: torch.Tensor,
        positions: torch.LongTensor,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        batch = states.size(0)
        query = self.query.expand(batch, -1, -1)
        query_positions = torch.zeros(
            batch,
            1,
            dtype=positions.dtype,
            device=positions.device,
        )
        pooled = self.attn(
            query=query,
            key=states,
            value=states,
            query_positions=query_positions,
            key_positions=positions,
            key_padding_mask=padding_mask,
        )
        return pooled.squeeze(1)


def upgrade_legacy_state_dict(
    state_dict: Dict[str, torch.Tensor], config: "TransformerConfig"
) -> Tuple[Dict[str, torch.Tensor], bool]:
    """Map legacy transformer weights to the rotary architecture layout."""

    # Fast path: new checkpoints already store encoder_layers and q/k/v projections.
    if not any(key.startswith("transformer.layers") for key in state_dict):
        return state_dict, False

    upgraded = dict(state_dict)
    converted = False

    def _rewrite_layer(layer_idx: int) -> None:
        nonlocal converted
        old_prefix = f"transformer.layers.{layer_idx}"
        new_prefix = f"encoder_layers.{layer_idx}"
        attn_weight_key = f"{old_prefix}.self_attn.in_proj_weight"
        attn_bias_key = f"{old_prefix}.self_attn.in_proj_bias"
        if attn_weight_key in state_dict:
            converted = True
            q_weight, k_weight, v_weight = torch.chunk(state_dict[attn_weight_key], 3, dim=0)
            upgraded[f"{new_prefix}.self_attn.q_proj.weight"] = q_weight
            upgraded[f"{new_prefix}.self_attn.k_proj.weight"] = k_weight
            upgraded[f"{new_prefix}.self_attn.v_proj.weight"] = v_weight
            upgraded.pop(attn_weight_key, None)
        if attn_bias_key in state_dict:
            q_bias, k_bias, v_bias = torch.chunk(state_dict[attn_bias_key], 3, dim=0)
            upgraded[f"{new_prefix}.self_attn.q_proj.bias"] = q_bias
            upgraded[f"{new_prefix}.self_attn.k_proj.bias"] = k_bias
            upgraded[f"{new_prefix}.self_attn.v_proj.bias"] = v_bias
            upgraded.pop(attn_bias_key, None)

        mapping = [
            ("self_attn.out_proj.weight", "self_attn.out_proj.weight"),
            ("self_attn.out_proj.bias", "self_attn.out_proj.bias"),
            ("linear1.weight", "linear1.weight"),
            ("linear1.bias", "linear1.bias"),
            ("linear2.weight", "linear2.weight"),
            ("linear2.bias", "linear2.bias"),
            ("norm1.weight", "norm1.weight"),
            ("norm1.bias", "norm1.bias"),
            ("norm2.weight", "norm2.weight"),
            ("norm2.bias", "norm2.bias"),
        ]
        for old_suffix, new_suffix in mapping:
            old_key = f"{old_prefix}.{old_suffix}"
            if old_key in state_dict:
                converted = True
                upgraded[f"{new_prefix}.{new_suffix}"] = state_dict[old_key]
                upgraded.pop(old_key, None)

    for idx in range(config.num_layers):
        _rewrite_layer(idx)

    cross_weight_key = "cross_attention.in_proj_weight"
    cross_bias_key = "cross_attention.in_proj_bias"
    if cross_weight_key in state_dict:
        converted = True
        q_weight, k_weight, v_weight = torch.chunk(state_dict[cross_weight_key], 3, dim=0)
        upgraded["cross_attention.q_proj.weight"] = q_weight
        upgraded["cross_attention.k_proj.weight"] = k_weight
        upgraded["cross_attention.v_proj.weight"] = v_weight
        upgraded.pop(cross_weight_key, None)
    if cross_bias_key in state_dict:
        q_bias, k_bias, v_bias = torch.chunk(state_dict[cross_bias_key], 3, dim=0)
        upgraded["cross_attention.q_proj.bias"] = q_bias
        upgraded["cross_attention.k_proj.bias"] = k_bias
        upgraded["cross_attention.v_proj.bias"] = v_bias
        upgraded.pop(cross_bias_key, None)

    return upgraded, converted


@dataclass
class TransformerConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    num_action_layers: int = 1
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
        self.encoder_layers = nn.ModuleList(
            [RotaryTransformerEncoderLayer(self.config) for _ in range(self.config.num_layers)]
        )
        self.action_layers = nn.ModuleList(
            [RotaryTransformerEncoderLayer(self.config) for _ in range(self.config.num_action_layers)]
        )
        self.cross_attention = RotaryMultiHeadAttention(
            embed_dim=self.config.d_model,
            num_heads=self.config.nhead,
            dropout=self.config.dropout,
        )
        self.policy_head = PolicyHead(d_model=self.config.d_model, dropout=self.config.dropout)
        self.value_head = ValueHead(d_model=self.config.d_model)
        self.aux_head = AuxiliaryHead(d_model=self.config.d_model, name="danger")
        self.board_pool = AttentionPooling(
            d_model=self.config.d_model,
            num_heads=self.config.nhead,
            dropout=self.config.dropout,
        )

    def encode_board(self, board_tokens: torch.Tensor, board_positions: torch.Tensor) -> torch.Tensor:
        board_hidden = self.board_encoder(board_tokens, board_positions)
        padding_mask = board_tokens.eq(self.vocab.pad_id)
        output = board_hidden
        for layer in self.encoder_layers:
            output = layer(output, board_positions, padding_mask)
        return output

    def encode_actions(self, action_tokens: torch.Tensor, action_positions: torch.Tensor) -> torch.Tensor:
        action_hidden = self.action_encoder(action_tokens, action_positions)
        if not self.action_layers:
            return action_hidden
        padding_mask = action_tokens.eq(self.vocab.action_pad_id)
        output = action_hidden
        for layer in self.action_layers:
            output = layer(output, action_positions, padding_mask)
        return output

    def forward(self, batch: MahjongBatch) -> HeadOutputs:
        board_states = self.encode_board(batch.board_tokens, batch.board_positions)
        action_states = self.encode_actions(batch.action_tokens, batch.action_positions)
        board_padding = batch.board_tokens.eq(self.vocab.pad_id)
        attn_output = self.cross_attention(
            query=action_states,
            key=board_states,
            value=board_states,
            query_positions=batch.action_positions,
            key_positions=batch.board_positions,
            key_padding_mask=board_padding,
        )

        policy_logits = self.policy_head(attn_output, batch.legal_mask)
        policy_log_probs = masked_log_softmax(policy_logits, batch.legal_mask)
        board_summary = self.board_pool(board_states, batch.board_positions, board_padding)
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


__all__ = ["MahjongTransformerModel", "TransformerConfig", "upgrade_legacy_state_dict"]

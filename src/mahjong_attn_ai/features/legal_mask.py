"""Utilities for masking invalid mahjong actions."""
from __future__ import annotations

import torch

INVALID_LOGIT = -1e9


def apply_legal_mask(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Mask invalid entries in ``logits`` using ``legal_mask``.

    Args:
        logits: Tensor shaped ``(batch, actions)``.
        legal_mask: Boolean tensor of the same shape where ``False`` entries are invalid.
    """

    if logits.shape != legal_mask.shape:
        raise ValueError("logits and legal_mask must share the same shape")
    return logits.masked_fill(~legal_mask, INVALID_LOGIT)


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Apply log-softmax while respecting the legal action mask."""

    masked_logits = apply_legal_mask(logits, legal_mask)
    return torch.nn.functional.log_softmax(masked_logits, dim=-1)


def pick_greedy_action(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Return argmax actions among legal candidates."""

    masked_logits = apply_legal_mask(logits, legal_mask)
    return masked_logits.argmax(dim=-1)


__all__ = ["apply_legal_mask", "masked_log_softmax", "pick_greedy_action", "INVALID_LOGIT"]


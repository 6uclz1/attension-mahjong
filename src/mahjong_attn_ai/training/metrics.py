"""Metrics for supervised mahjong training."""
from __future__ import annotations

from typing import Dict

import torch


def topk_accuracy(log_probs: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        raise ValueError("k must be positive")
    topk = log_probs.topk(k, dim=-1).indices
    correct = topk.eq(labels.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean()


def basic_metrics(log_probs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    return {
        "top1": float(topk_accuracy(log_probs, labels, 1).item()),
        "top3": float(topk_accuracy(log_probs, labels, min(3, log_probs.size(-1))).item()),
    }


def aggregate_losses(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {name: float(value.detach().cpu().item()) for name, value in loss_dict.items()}


__all__ = ["topk_accuracy", "basic_metrics", "aggregate_losses"]


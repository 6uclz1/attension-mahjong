"""KPI aggregation helpers for simulated backtests."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import math
import random

from ..env.simulator_stub import GameLog


def aggregate_logs(
    logs: Sequence[GameLog],
    target_names: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Aggregate scalar KPIs from simulator logs.

    Args:
        logs: Sequence of simulator game logs.
        target_names: Optional iterable of player names to keep. When omitted the
            aggregation covers every player.
    """

    if not logs:
        return {}

    name_filter = set(target_names) if target_names else None
    ranks: List[float] = []
    scores: List[float] = []
    kpi_store: Dict[str, List[float]] = defaultdict(list)
    for log in logs:
        for name, rank, score, metrics in zip(
            log.player_names, log.ranks, log.scores, log.per_player_kpi
        ):
            if name_filter and name not in name_filter:
                continue
            ranks.append(float(rank))
            scores.append(float(score))
            for key, value in metrics.items():
                kpi_store[key].append(float(value))
    if not ranks:
        return {}
    aggregated = {
        "average_rank": mean(ranks),
        "score_ev": mean(scores) - 25000.0,
    }
    for key, values in kpi_store.items():
        aggregated[f"kpi_{key}"] = mean(values)
    return aggregated


def bootstrap_ci(
    values: Sequence[float],
    num_samples: int = 500,
    alpha: float = 0.05,
    seed: int = 13,
) -> Tuple[float, float]:
    """Return percentile bootstrap confidence intervals."""

    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    samples = []
    for _ in range(num_samples):
        resample = [rng.choice(values) for _ in values]
        samples.append(mean(resample))
    samples.sort()
    lower_idx = int(math.floor((alpha / 2) * len(samples)))
    upper_idx = int(math.ceil((1 - alpha / 2) * len(samples)) - 1)
    lower = samples[max(0, min(lower_idx, len(samples) - 1))]
    upper = samples[max(0, min(upper_idx, len(samples) - 1))]
    return (lower, upper)


def bootstrap_mean_difference(
    values_a: Sequence[float],
    values_b: Sequence[float],
    num_samples: int = 500,
    alpha: float = 0.05,
    seed: int = 17,
) -> Tuple[float, float, float]:
    """Return mean difference and bootstrap CI."""

    if not values_a or not values_b:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    samples: List[float] = []
    paired = list(zip(values_a, values_b))
    for _ in range(num_samples):
        resample = [rng.choice(paired) for _ in paired]
        diffs = [a - b for a, b in resample]
        samples.append(mean(diffs))
    samples.sort()
    point = mean([a - b for a, b in paired])
    lower_idx = int(math.floor((alpha / 2) * len(samples)))
    upper_idx = int(math.ceil((1 - alpha / 2) * len(samples)) - 1)
    lower = samples[max(0, min(lower_idx, len(samples) - 1))]
    upper = samples[max(0, min(upper_idx, len(samples) - 1))]
    return (point, lower, upper)


__all__ = ["aggregate_logs", "bootstrap_ci", "bootstrap_mean_difference"]


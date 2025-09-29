"""Lightweight stochastic simulator placeholder for backtests."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class SimulatorConfig:
    base_score: float = 25000.0
    variance: float = 4500.0
    skill_scale: float = 1200.0
    kpi_noise: float = 0.05


@dataclass
class GameLog:
    game_id: str
    scores: List[float]
    ranks: List[int]
    per_player_kpi: List[Dict[str, float]]
    player_names: List[str]
    seed: int


class Simulator:
    """Generate reproducible synthetic mahjong results driven by bot skill."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()

    def play_game(self, bots: Sequence["BaseMahjongBot"], seed: int) -> GameLog:
        rng = random.Random(seed)
        scores: List[float] = []
        per_player_kpi: List[Dict[str, float]] = []
        player_names: List[str] = []
        for bot in bots:
            skill = getattr(bot, "skill", 0.0)
            mean_shift = skill * self.config.skill_scale
            score = self.config.base_score + rng.gauss(mean_shift, self.config.variance)
            scores.append(score)
            per_player_kpi.append(self._sample_kpis(rng, skill))
            player_names.append(bot.name)
        ranks = _scores_to_ranks(scores)
        return GameLog(
            game_id=f"sim-{seed}",
            scores=scores,
            ranks=ranks,
            per_player_kpi=per_player_kpi,
            player_names=player_names,
            seed=seed,
        )

    def _sample_kpis(self, rng: random.Random, skill: float) -> Dict[str, float]:
        base = max(0.05, 0.3 - 0.02 * skill)
        win_base = min(0.4, 0.25 + 0.03 * skill)
        return {
            "deal_in_rate": max(0.0, min(1.0, rng.gauss(base, self.config.kpi_noise))),
            "win_rate": max(0.0, min(1.0, rng.gauss(win_base, self.config.kpi_noise))),
            "riichi_rate": max(0.0, min(1.0, rng.gauss(0.2 + 0.02 * skill, self.config.kpi_noise))),
            "average_win_points": max(500.0, rng.gauss(5500 + 300 * skill, 500.0)),
            "average_deal_in_points": max(500.0, rng.gauss(6000 - 300 * skill, 500.0)),
        }


def _scores_to_ranks(scores: Sequence[float]) -> List[int]:
    ranking = sorted(((score, idx) for idx, score in enumerate(scores)), reverse=True)
    ranks = [0] * len(scores)
    for position, (_, idx) in enumerate(ranking, start=1):
        ranks[idx] = position
    return ranks


class BaseMahjongBot:
    """Interface the simulator cares about."""

    def __init__(self, name: str, skill: float = 0.0) -> None:
        self.name = name
        self.skill = skill

    def select_action(self, observation, legal_mask, rng):  # pragma: no cover - stub
        raise NotImplementedError


__all__ = ["Simulator", "SimulatorConfig", "GameLog", "BaseMahjongBot"]


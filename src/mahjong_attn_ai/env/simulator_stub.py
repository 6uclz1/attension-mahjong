"""Data-driven simulator that evaluates policies on synthetic mahjong turns."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from ..dataio.parser import SyntheticKifuParser
from ..dataio.schema import (
    DEFAULT_ACTION_SEQ_LEN,
    DEFAULT_BOARD_SEQ_LEN,
    DatasetConfig,
    MahjongBatch,
    MahjongSample,
)
from ..features.utils import MahjongVocabulary


@dataclass
class SimulatorConfig:
    base_score: float = 25000.0
    variance: float = 45.0
    turns_per_game: int = 32
    reward_correct: float = 120.0
    penalty_incorrect: float = -60.0
    value_scale: float = 40.0
    kifu_dir: str = "data/sample_kifus"
    board_seq_len: int = DEFAULT_BOARD_SEQ_LEN
    action_seq_len: int = DEFAULT_ACTION_SEQ_LEN


@dataclass
class GameLog:
    game_id: str
    scores: List[float]
    ranks: List[int]
    per_player_kpi: List[Dict[str, float]]
    player_names: List[str]
    seed: int


class Simulator:
    """Generate reproducible results by rolling synthetic decision problems."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        self.vocab = MahjongVocabulary.build_default()
        dataset_cfg = DatasetConfig(
            board_seq_len=self.config.board_seq_len,
            action_seq_len=self.config.action_seq_len,
            include_seat_rotation=True,
        )
        data_dir = Path(self.config.kifu_dir).expanduser()
        parser = SyntheticKifuParser(data_dir=data_dir, config=dataset_cfg, vocab=self.vocab)
        self.samples = parser.load()
        if not self.samples:
            raise RuntimeError(
                "Simulator requires at least one synthetic kifu sample; "
                f"none found under {data_dir!s}."
            )

    def play_game(self, bots: Sequence["BaseMahjongBot"], seed: int) -> GameLog:
        rng = random.Random(seed)
        scores = [self.config.base_score for _ in bots]
        correct_counts = [0] * len(bots)
        decision_counts = [0] * len(bots)
        reward_accumulators = [0.0] * len(bots)
        value_accumulators = [0.0] * len(bots)

        for turn in range(self.config.turns_per_game):
            for idx, bot in enumerate(bots):
                sample = rng.choice(self.samples)
                batch = self._sample_to_batch(sample)
                legal = batch.legal_mask[0].tolist()
                action_idx = bot.select_action(batch, legal, rng)
                label_idx = int(batch.label_actions.item())

                decision_counts[idx] += 1
                is_correct = int(action_idx == label_idx)
                correct_counts[idx] += is_correct

                reward = self.config.reward_correct if is_correct else self.config.penalty_incorrect
                value_bonus = float(batch.value_targets.item()) * self.config.value_scale
                noise = rng.gauss(0.0, self.config.variance)
                scores[idx] += reward + value_bonus + noise
                reward_accumulators[idx] += reward
                value_accumulators[idx] += value_bonus

        per_player_kpi: List[Dict[str, float]] = []
        player_names: List[str] = []
        for idx, bot in enumerate(bots):
            total_turns = max(1, decision_counts[idx])
            accuracy = correct_counts[idx] / total_turns
            avg_reward = reward_accumulators[idx] / total_turns
            avg_value = value_accumulators[idx] / total_turns
            per_player_kpi.append(
                {
                    "decision_accuracy": accuracy,
                    "average_reward": avg_reward,
                    "value_bonus": avg_value,
                }
            )
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

    def _sample_to_batch(self, sample: MahjongSample) -> MahjongBatch:
        board_tokens = sample["board_tokens"].unsqueeze(0).clone()
        board_positions = sample["board_positions"].unsqueeze(0).clone()
        action_tokens = sample["action_tokens"].unsqueeze(0).clone()
        action_positions = sample["action_positions"].unsqueeze(0).clone()
        legal_mask = sample["legal_mask"].unsqueeze(0).clone().bool()
        label_action = torch.tensor([sample["label_action"]], dtype=torch.long)
        value_target = torch.tensor([sample["value_target"]], dtype=torch.float)
        aux_target = torch.tensor([sample["aux_target"]], dtype=torch.float)
        return MahjongBatch(
            board_tokens=board_tokens,
            board_positions=board_positions,
            action_tokens=action_tokens,
            action_positions=action_positions,
            legal_mask=legal_mask,
            label_actions=label_action,
            value_targets=value_target,
            aux_targets=aux_target,
            metadata=[sample["metadata"]],
        )


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

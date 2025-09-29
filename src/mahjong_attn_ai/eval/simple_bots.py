"""Simple opponents/baselines used during backtesting."""
from __future__ import annotations

import random
from typing import Optional

import torch

from ..env.simulator_stub import BaseMahjongBot
from ..features.legal_mask import pick_greedy_action
from ..models.transformer import MahjongTransformerModel


class RandomMahjongBot(BaseMahjongBot):
    """Choose a random legal action."""

    def __init__(self, name: str = "random", skill: float = 0.0) -> None:
        super().__init__(name=name, skill=skill)

    def select_action(self, observation, legal_mask, rng: random.Random) -> int:
        indices = [idx for idx, legal in enumerate(legal_mask) if legal]
        return rng.choice(indices) if indices else 0


class HeuristicMahjongBot(BaseMahjongBot):
    """Very small heuristic baseline emphasising riichi/win actions."""

    def __init__(self, name: str = "heuristic", skill: float = 0.3) -> None:
        super().__init__(name=name, skill=skill)

    def select_action(self, observation, legal_mask, rng: random.Random) -> int:
        prioritized = [i for i, legal in enumerate(legal_mask) if legal and i % 3 == 1]
        if prioritized:
            return prioritized[0]
        return RandomMahjongBot().select_action(observation, legal_mask, rng)


class ModelPolicyBot(BaseMahjongBot):
    """Wrap ``MahjongTransformerModel`` into the simulator bot interface."""

    def __init__(
        self,
        model: MahjongTransformerModel,
        device: torch.device | str = "cpu",
        name: str = "model",
        skill: float = 0.6,
    ) -> None:
        super().__init__(name=name, skill=skill)
        self.model = model.eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def select_action(self, batch, legal_mask, rng: random.Random) -> int:
        batch = batch.to_device(self.device)
        outputs = self.model(batch)
        action = pick_greedy_action(outputs.policy_logits, batch.legal_mask)
        return int(action.item())


__all__ = ["RandomMahjongBot", "HeuristicMahjongBot", "ModelPolicyBot"]


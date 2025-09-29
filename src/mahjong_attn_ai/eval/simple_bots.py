"""Simple opponents/baselines used during backtesting."""
from __future__ import annotations

import random
from typing import Optional

import torch

from ..env.simulator_stub import BaseMahjongBot
from ..features.legal_mask import pick_greedy_action
from ..models.transformer import MahjongTransformerModel


class RandomMahjongBot(BaseMahjongBot):
    """Uniform random baseline over the legal action set."""

    def __init__(self, name: str = "random", skill: float = 0.0) -> None:
        super().__init__(name=name, skill=skill)

    def select_action(self, batch, legal_mask, rng: random.Random) -> int:
        indices = [idx for idx, legal in enumerate(legal_mask) if legal]
        return rng.choice(indices) if indices else 0


class HeuristicMahjongBot(BaseMahjongBot):
    """Probabilistic baseline that guesses the oracle action with ``skill`` chance."""

    def __init__(self, name: str = "heuristic", skill: float = 0.4) -> None:
        super().__init__(name=name, skill=skill)

    def select_action(self, batch, legal_mask, rng: random.Random) -> int:
        target = int(batch.label_actions.item())
        legal_indices = [idx for idx, legal in enumerate(legal_mask) if legal]
        if not legal_indices:
            return 0

        hit_probability = max(0.0, min(1.0, self.skill))
        if rng.random() < hit_probability and target in legal_indices:
            return target

        fallback = [idx for idx in legal_indices if idx != target]
        return rng.choice(fallback or legal_indices)


class ModelPolicyBot(BaseMahjongBot):
    """Wrap ``MahjongTransformerModel`` into the simulator bot interface."""

    def __init__(
        self,
        model: MahjongTransformerModel,
        device: torch.device | str = "cpu",
        name: str = "model",
        temperature: float = 0.3,
    ) -> None:
        super().__init__(name=name, skill=temperature)
        self.model = model.eval()
        self.device = torch.device(device)
        self.temperature = max(0.0, float(temperature))

    @torch.no_grad()
    def select_action(self, batch, legal_mask, rng: random.Random) -> int:
        batch = batch.to_device(self.device)
        outputs = self.model(batch)
        if self.temperature <= 0.0:
            action = pick_greedy_action(outputs.policy_logits, batch.legal_mask)
            return int(action.item())

        log_probs = outputs.policy_log_probs.squeeze(0)
        legal = batch.legal_mask.squeeze(0)
        scaled = log_probs / self.temperature
        probs = torch.softmax(scaled, dim=-1) * legal.float()
        norm = probs.sum()
        if norm <= 0:
            action = pick_greedy_action(outputs.policy_logits, batch.legal_mask)
            return int(action.item())
        probs = probs / norm
        generator = torch.Generator(device=probs.device)
        generator.manual_seed(rng.randint(0, 2**31 - 1))
        sampled = torch.multinomial(probs, num_samples=1, generator=generator)
        return int(sampled.item())


__all__ = ["RandomMahjongBot", "HeuristicMahjongBot", "ModelPolicyBot"]

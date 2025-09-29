"""Placeholder PPO trainer interface for future reinforcement learning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Protocol


class PolicyModule(Protocol):
    """Minimal protocol for the policy/value network required by PPO."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - interface only
        ...

    def loss(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - interface only
        ...


@dataclass
class PPOConfig:
    rollout_episodes: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    epochs: int = 4
    minibatch_size: int = 256


class PPOTrainer:
    """Documented stub showing how PPO integration could look.

    The intention is to reuse ``MahjongTransformerModel`` as a shared policy/value
    backbone. Collect trajectories from the simulator, compute advantages, and then
    optimise using clipped surrogate objectives.
    """

    def __init__(self, policy: PolicyModule, config: PPOConfig) -> None:
        self.policy = policy
        self.config = config

    def collect_rollouts(self) -> Iterable[Dict[str, Any]]:  # pragma: no cover - stub
        """Collect trajectories from the mahjong simulator.

        Expected to interact with ``env.simulator_stub.Simulator`` (or future mjai/Rust
        bindings). Each trajectory contains observations, actions, log-probabilities,
        rewards, and done flags enabling GAE computations.
        """

        raise NotImplementedError("Implement simulator rollouts when RL becomes active.")

    def update_policy(self, rollouts: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        """Run the PPO optimisation steps on stored trajectories."""

        raise NotImplementedError(
            "Implement PPO optimisation loop using clipped objectives, value loss, "
            "and entropy regularisation."
        )


__all__ = ["PPOTrainer", "PPOConfig", "PolicyModule"]


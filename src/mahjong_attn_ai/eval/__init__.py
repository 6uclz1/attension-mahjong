"""Evaluation utilities."""
from .backtest import BacktestConfig, BacktestResult, BacktestRunner
from .kpi import aggregate_logs, bootstrap_ci, bootstrap_mean_difference
from .simple_bots import HeuristicMahjongBot, ModelPolicyBot, RandomMahjongBot

__all__ = [
    "BacktestRunner",
    "BacktestConfig",
    "BacktestResult",
    "aggregate_logs",
    "bootstrap_ci",
    "bootstrap_mean_difference",
    "RandomMahjongBot",
    "HeuristicMahjongBot",
    "ModelPolicyBot",
]


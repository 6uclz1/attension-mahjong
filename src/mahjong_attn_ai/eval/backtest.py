"""Backtesting utilities with duplicate deals and bootstrap statistics."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

from rich.console import Console
from rich.table import Table

from ..env.simulator_stub import BaseMahjongBot, GameLog, Simulator
from .kpi import aggregate_logs, bootstrap_ci, bootstrap_mean_difference


BotFactory = Callable[[], BaseMahjongBot]


@dataclass
class BacktestConfig:
    num_games: int = 16
    duplicate: bool = True
    seeds: List[int] | None = None
    bootstrap_samples: int = 500
    alpha: float = 0.05
    output_dir: Path = Path("runs/eval")
    sprt_min_effect: float = 0.05
    sprt_accept_prob: float = 0.8


@dataclass
class BacktestResult:
    summary: Dict[str, float]
    table_rows: List[Dict[str, float | int | str]]
    logs: List[GameLog]
    target_name: str


class BacktestRunner:
    """Drive simulator games and summarise analytics."""

    def __init__(self, simulator: Simulator, config: BacktestConfig | None = None) -> None:
        self.simulator = simulator
        self.config = config or BacktestConfig()
        self.console = Console()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run_policy_eval(
        self,
        policy_factory: BotFactory,
        opponent_factories: Sequence[BotFactory],
    ) -> BacktestResult:
        target_name = policy_factory().name
        logs, table_rows = self._play_series(policy_factory, opponent_factories)
        summary = self._summarise(logs, table_rows, target_name)
        self._render_summary(summary, title=f"Backtest ({target_name})")
        return BacktestResult(summary=summary, table_rows=table_rows, logs=logs, target_name=target_name)

    def run_ab_test(
        self,
        policy_a: BotFactory,
        policy_b: BotFactory,
        opponent_factories: Sequence[BotFactory],
    ) -> Dict[str, Dict[str, float]]:
        logs_a, rows_a = self._play_series(policy_a, opponent_factories)
        logs_b, rows_b = self._play_series(policy_b, opponent_factories)
        name_a = policy_a().name
        name_b = policy_b().name
        summary_a = self._summarise(logs_a, rows_a, name_a)
        summary_b = self._summarise(logs_b, rows_b, name_b)
        diff = self._compare(summary_a, summary_b, rows_a, rows_b, name_a, name_b)
        self._render_summary(summary_a, title=f"A: {name_a}")
        self._render_summary(summary_b, title=f"B: {name_b}")
        self._render_summary(diff, title="A-B Delta")
        return {"policy_a": summary_a, "policy_b": summary_b, "difference": diff}

    def save_result(self, result: BacktestResult) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out_dir = self.config.output_dir / timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "summary.json"
        table_path = out_dir / "table.csv"
        config_payload = dict(self.config.__dict__)
        if isinstance(config_payload.get("output_dir"), Path):
            config_payload["output_dir"] = str(config_payload["output_dir"])
        summary_payload = {
            "target": result.target_name,
            "metrics": result.summary,
            "config": config_payload,
            "num_logs": len(result.logs),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        with table_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=result.table_rows[0].keys())
            writer.writeheader()
            writer.writerows(result.table_rows)
        return out_dir

    def _play_series(
        self,
        policy_factory: BotFactory,
        opponent_factories: Sequence[BotFactory],
    ) -> tuple[List[GameLog], List[Dict[str, float | int | str]]]:
        seeds = self.config.seeds or list(range(self.config.num_games))
        logs: List[GameLog] = []
        rows: List[Dict[str, float | int | str]] = []
        base_factories: List[BotFactory] = [policy_factory, *opponent_factories]
        num_players = len(base_factories)
        rotations = num_players if self.config.duplicate else 1
        for game_index, seed in enumerate(seeds):
            for rotation in range(rotations):
                ordered_factories = base_factories[rotation:] + base_factories[:rotation]
                bots = [factory() for factory in ordered_factories]
                log = self.simulator.play_game(bots, seed=seed * 11 + rotation)
                logs.append(log)
                for seat, (name, rank, score) in enumerate(
                    zip(log.player_names, log.ranks, log.scores)
                ):
                    rows.append(
                        {
                            "game_index": game_index,
                            "seed": seed,
                            "rotation": rotation,
                            "seat": seat,
                            "player": name,
                            "rank": rank,
                            "score": score,
                        }
                    )
        return logs, rows

    def _summarise(
        self,
        logs: Sequence[GameLog],
        rows: Sequence[Dict[str, float | int | str]],
        target_name: str,
    ) -> Dict[str, float]:
        agg = aggregate_logs(logs, target_names=[target_name])
        ranks = [float(row["rank"]) for row in rows if row["player"] == target_name]
        scores = [float(row["score"]) - 25000.0 for row in rows if row["player"] == target_name]
        rank_ci = bootstrap_ci(ranks, num_samples=self.config.bootstrap_samples, alpha=self.config.alpha)
        score_ci = bootstrap_ci(scores, num_samples=self.config.bootstrap_samples, alpha=self.config.alpha)
        summary = {
            **agg,
            "rank_ci_low": rank_ci[0],
            "rank_ci_high": rank_ci[1],
            "score_ci_low": score_ci[0],
            "score_ci_high": score_ci[1],
            "num_rounds": float(len(ranks)),
        }
        return summary

    def _compare(
        self,
        summary_a: Dict[str, float],
        summary_b: Dict[str, float],
        rows_a: Sequence[Dict[str, float | int | str]],
        rows_b: Sequence[Dict[str, float | int | str]],
        name_a: str,
        name_b: str,
    ) -> Dict[str, float]:
        ranks_a = [float(row["rank"]) for row in rows_a if row["player"] == name_a]
        ranks_b = [float(row["rank"]) for row in rows_b if row["player"] == name_b]
        scores_a = [float(row["score"]) - 25000.0 for row in rows_a if row["player"] == name_a]
        scores_b = [float(row["score"]) - 25000.0 for row in rows_b if row["player"] == name_b]
        rank_delta, rank_low, rank_high = bootstrap_mean_difference(
            ranks_a,
            ranks_b,
            num_samples=self.config.bootstrap_samples,
            alpha=self.config.alpha,
        )
        score_delta, score_low, score_high = bootstrap_mean_difference(
            scores_a,
            scores_b,
            num_samples=self.config.bootstrap_samples,
            alpha=self.config.alpha,
        )
        accept = float(rank_low > 0 or rank_high < 0 or score_low > 0 or score_high < 0)
        return {
            "rank_delta": rank_delta,
            "rank_ci_low": rank_low,
            "rank_ci_high": rank_high,
            "score_delta": score_delta,
            "score_ci_low": score_low,
            "score_ci_high": score_high,
            "sprt_accept": accept,
        }

    def _render_summary(self, summary: Dict[str, float], title: str) -> None:
        table = Table(title=title)
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        for key, value in summary.items():
            table.add_row(key, f"{value:.4f}")
        self.console.print(table)


__all__ = ["BacktestRunner", "BacktestConfig", "BacktestResult"]


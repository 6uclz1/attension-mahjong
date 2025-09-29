# Mahjong Attention AI

Mahjong Attention AI is a transformer-centric MVP for Japanese (riichi) mahjong that starts from supervised imitation learning and is architected for future PPO-based reinforcement learning. The repository ships with synthetic data, evaluation utilities, and a reproducible uv-based workflow for encoding kifu, training the model, and running duplicate backtests with statistical confidence intervals.

## Features

- **Transformer encoder + cross-attention** over board state and explicit action candidates with policy/value/auxiliary heads.
- **Supervised learning pipeline** (encode → train → eval) on synthetic kifu data with Top-1/Top-3 reporting.
- **Backtest harness** with duplicate deals, bootstrap confidence intervals, and A/B comparison hooks.
- **Task automation** via taskipy and best-practice tooling (ruff, mypy, pytest).
- **RL-ready stubs** (PPO trainer & simulator bridge) to extend into self-play reinforcement learning.

## Quick Start (uv)

```bash
uv venv
uv sync
uv run python -m mahjong_attn_ai.cli encode --config configs/default.yaml
uv run python -m mahjong_attn_ai.cli train  --config configs/default.yaml
uv run python -m mahjong_attn_ai.cli eval   --config configs/default.yaml --ckpt runs/latest/best.ckpt
uv run python -m mahjong_attn_ai.cli backtest --config configs/eval_backtest.yaml --ckpt runs/latest/best.ckpt
uv run python -m mahjong_attn_ai.cli abtest   --config configs/eval_backtest.yaml --ckpt-a runs/latest/best.ckpt --ckpt-b runs/latest/baseline.ckpt
```

Taskipy mirrors the same flow:

```bash
uv run task setup
uv run task encode
uv run task train
uv run task eval
uv run task backtest
uv run task abtest
uv run task lint
uv run task typecheck
uv run task test
```

## Configuration Overview

- `configs/default.yaml`: primary training configuration (data paths, transformer hyperparameters, simulator defaults, backtest opponents).
- `configs/eval_backtest.yaml`: recommended settings for evaluation and A/B testing (more seeds, denser bootstrap sampling).

Each config follows the same schema used by `mahjong_attn_ai.cli`: `data`, `dataset`, `model`, `training`, `simulator`, and `backtest` sections.

## Synthetic Data

Synthetic kifu records live in `data/sample_kifus/` and are parsed via `SyntheticKifuParser`. The parser now accepts both numeric ids and human-readable tile/action labels; enabling `dataset.auto_generate` seeds additional random samples for smoke tests when you lack real logs. Real kifu ingestion can replace this layer by swapping the parser implementation.

## Model Architecture

- **Board encoder**: learned token + positional embeddings feeding a Pre-LN `nn.TransformerEncoder`.
- **Candidate encoder**: action token embeddings + positional embeddings.
- **Cross-attention**: candidate queries attend to the board sequence.
- **Heads**:
  - Policy head masks illegal moves and emits log-probabilities.
  - Value head regresses half-game EV.
  - Auxiliary head predicts danger rate (deal-in probability proxy).
- Loss: `L = CE(policy) + λ_v * MSE(value) + λ_aux * MSE(aux)` with coefficients configurable in YAML.

## Evaluation & Backtesting

`mahjong_attn_ai.eval.backtest.BacktestRunner` provides:

- Duplicate-deal self-play loops with configurable seeds.
- KPI logging (average rank, score EV, win/deal-in/riichi rates, average deal-in/win points).
- Bootstrap confidence intervals (default 95%) and mean-difference A/B deltas.
- CSV + JSON artefacts stored under `runs/eval/<timestamp>/` for dashboards.

Sample Rich console output:

```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric             ┃ Value  ┃
┣━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━┫
┃ average_rank       ┃ 2.1000 ┃
┃ score_ev           ┃ 320.0000 ┃
┃ kpi_win_rate       ┃ 0.2450 ┃
┃ rank_ci_low        ┃ 1.9500 ┃
┃ rank_ci_high       ┃ 2.2500 ┃
┃ score_ci_low       ┃ 180.0000 ┃
┃ score_ci_high      ┃ 460.0000 ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━┛
```

The `abtest` subcommand prints both policy summaries and their difference (with CI) while writing `summary.json` for automation.

## Tests

Pytest covers encoding shapes, legal masking, forward pass dimensions, and the backtest API. Run `uv run task test` for the full suite.

## Expected Outputs

- **Training**: `runs/latest/best.ckpt` and console metrics (loss decreasing, Top-1/Top-3 above random chance on synthetic data).
- **Evaluation**: JSON metrics with Top-1/Top-3 accuracy.
- **Backtest**: `runs/eval/<timestamp>/summary.json` & `table.csv` containing KPIs, CIs, and duplicate metadata.
- **A/B Test**: `runs/eval/abtest-*/summary.json` summarising both policies and their delta.

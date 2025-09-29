# Repository Guidelines

## Project Structure & Module Organization
Source lives in `src/mahjong_attn_ai`, split into focused packages (`features`, `models`, `training`, `eval`, `env`, `dataio`). Command-line entry points are in `cli.py`. YAML configs sit under `configs/` (`default.yaml` powers the encode→train→eval loop). Synthetic assets stay in `data/`; experiment artefacts land in `runs/`. Tests mirror the package layout inside `tests/` for quick traceability.

## Build, Test, and Development Commands
Bootstrap the workspace with `uv run task setup` (creates the venv and installs dependencies). Run individual pipeline stages with `uv run python -m mahjong_attn_ai.cli encode|train|eval --config configs/default.yaml`. Prefer the taskipy aliases: `uv run task encode`, `train`, `eval`, `backtest`, and `abtest`. Quality gates live behind `uv run task lint`, `typecheck`, `test`, and the aggregated `all` target before opening a PR.

## Coding Style & Naming Conventions
Ruff enforces formatting: 100-character lines, double quotes, sorted imports, and LF endings. Use 4-space indentation, comprehensive type hints, and avoid wildcard imports. Modules stay lowercase with underscores, classes use PascalCase, and functions plus variables use snake_case. Keep CLI command strings and config keys lowercase and hyphen-free to match current usage.

## Testing Guidelines
Pytest drives validation (`uv run task test`). Place new suites under `tests/` using `test_<subject>.py`, mirroring the runtime path. Targeted runs can use `uv run pytest tests/test_training.py -k smoke`. Exercise new data loaders, masking logic, and CLI flags with both success and failure assertions. Prefer fixtures that rely on synthetic kifus instead of committing generated checkpoints.

## Commit & Pull Request Guidelines
Write imperative, present-tense summaries (`Add attention mask validation`) and use optional body bullets wrapped near 72 characters. Reference issues with `Refs #123` only after local checks pass. PRs should state purpose, highlight risky areas, include command output (e.g., `uv run task all`), and point reviewers to relevant artefacts within `runs/`. Flag configuration edits that change training defaults so downstream agents can retrain with intent.

## Environment & Data Notes
Keep proprietary logs out of `data/`; rely on `data/sample_kifus/` for smoke tests. Large evaluation folders remain ignored—mention noteworthy checkpoints or reports in PR notes instead. When touching YAML configs, document disruptive overrides inline to preserve reproducibility across agents.

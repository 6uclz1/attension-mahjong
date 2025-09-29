"""Command line entry point for encoding, training, evaluation, and backtesting."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from rich.console import Console

from .dataio.dataset import build_dataloaders
from .dataio.schema import DatasetConfig
from .env.simulator_stub import Simulator, SimulatorConfig
from .eval.backtest import BacktestConfig, BacktestRunner
from .eval.simple_bots import HeuristicMahjongBot, ModelPolicyBot, RandomMahjongBot
from .features.utils import MahjongVocabulary
from .models.transformer import (
    MahjongTransformerModel,
    TransformerConfig,
    upgrade_legacy_state_dict,
)
from .training.sl_trainer import SupervisedTrainer, TrainerConfig

console = Console()


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_encode(config: Dict[str, Any]) -> None:
    from .dataio.parser import SyntheticKifuParser

    dataset_cfg = DatasetConfig(**config.get("dataset", {}))
    data_dir = Path(config["data"]["kifu_dir"]).expanduser()
    parser = SyntheticKifuParser(data_dir, dataset_cfg)
    samples = parser.load()
    out_dir = Path(config.get("output", {}).get("encode_dir", "runs/encoded")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = out_dir / "samples.pt"
    torch.save(samples, payload_path)
    console.log(f"Encoded {len(samples)} samples -> {payload_path}")


def _build_model(config: Dict[str, Any]) -> MahjongTransformerModel:
    vocab = MahjongVocabulary.build_default()
    model_cfg = TransformerConfig(**config.get("model", {}))
    return MahjongTransformerModel(vocab=vocab, config=model_cfg)


def _build_dataloaders(config: Dict[str, Any]):
    dataset_cfg = DatasetConfig(**config.get("dataset", {}))
    training_cfg = config.get("training", {})
    batch_size = training_cfg.get("batch_size", 16)
    val_ratio = training_cfg.get("val_ratio", 0.2)
    seed = training_cfg.get("seed", 7)
    return build_dataloaders(
        data_dir=Path(config["data"]["kifu_dir"]).expanduser(),
        batch_size=batch_size,
        val_ratio=val_ratio,
        seed=seed,
        dataset_config=dataset_cfg,
    )


def _build_trainer(model: MahjongTransformerModel, dataloaders, config: Dict[str, Any]) -> SupervisedTrainer:
    training_cfg = config.get("training", {})
    trainer_cfg = TrainerConfig(
        epochs=training_cfg.get("epochs", 3),
        lr=training_cfg.get("lr", 3e-4),
        weight_decay=training_cfg.get("weight_decay", 1e-2),
        grad_clip=training_cfg.get("grad_clip", 1.0),
        lambda_value=training_cfg.get("lambda_value", 0.1),
        lambda_aux=training_cfg.get("lambda_aux", 0.1),
        device=training_cfg.get("device", "cpu"),
        output_dir=Path(training_cfg.get("output_dir", "runs/latest")).expanduser(),
        use_wandb=training_cfg.get("use_wandb", False),
    )
    return SupervisedTrainer(model=model, dataloaders=dataloaders, config=trainer_cfg)


def run_train(config: Dict[str, Any]) -> None:
    dataloaders = _build_dataloaders(config)
    model = _build_model(config)
    trainer = _build_trainer(model, dataloaders, config)
    history = trainer.fit()
    console.log(f"Training complete. Best checkpoint stored at {trainer.best_path}")
    console.log(json.dumps(history, indent=2))


def run_eval(config: Dict[str, Any], ckpt: Path) -> None:
    dataloaders = _build_dataloaders(config)
    model = _build_model(config)
    trainer = _build_trainer(model, dataloaders, config)
    state = torch.load(ckpt, map_location=trainer.device)
    trainer.model.load_state_dict(state["model_state"])
    result = trainer._run_validation(epoch=0)
    console.log(f"Eval results: {json.dumps(result, indent=2)}")


def _opponent_factory(kind: str):
    kind = kind.lower()
    if kind == "random":
        return RandomMahjongBot
    if kind == "heuristic":
        return HeuristicMahjongBot
    raise ValueError(f"Unknown opponent kind: {kind}")


def _build_opponents(config: Dict[str, Any]) -> List:
    specs = config.get("opponents", ["heuristic", "random", "random"])
    factories = []
    for spec in specs:
        if isinstance(spec, str):
            factories.append(lambda s=spec: _opponent_factory(s)())
        elif isinstance(spec, dict):
            kind = spec.get("type", "random")
            skill = spec.get("skill", 0.0)
            name = spec.get("name", kind)

            def _factory(kind=kind, skill=skill, name=name):
                cls = _opponent_factory(kind)
                return cls(name=name, skill=skill)

            factories.append(_factory)
        else:
            raise ValueError(f"Unsupported opponent spec: {spec}")
    return factories


def _load_checkpoint(path: Path, config: Dict[str, Any]) -> MahjongTransformerModel:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = _build_model(config)
    state = torch.load(path, map_location="cpu")
    model_state = state["model_state"]
    upgraded_state, converted = upgrade_legacy_state_dict(model_state, model.config)
    load_result = model.load_state_dict(upgraded_state, strict=False)
    if converted:
        console.log("Upgraded legacy checkpoint weights for rotary attention model")
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        if len(missing) > 5:
            preview += ", ..."
        console.log(f"[yellow]Missing checkpoint parameters: {preview}")
    if unexpected:
        preview = ", ".join(sorted(unexpected)[:5])
        if len(unexpected) > 5:
            preview += ", ..."
        console.log(f"[yellow]Unexpected checkpoint parameters discarded: {preview}")
    return model


def run_backtest(config: Dict[str, Any], ckpt: Path) -> None:
    model = _load_checkpoint(ckpt, config)
    device = torch.device(config.get("training", {}).get("device", "cpu"))
    simulator_cfg = SimulatorConfig(**config.get("simulator", {}))
    simulator = Simulator(config=simulator_cfg)
    backtest_section = config.get("backtest", {})
    backtest_cfg = BacktestConfig(
        num_games=backtest_section.get("num_games", 16),
        duplicate=backtest_section.get("duplicate", True),
        seeds=backtest_section.get("seeds"),
        bootstrap_samples=backtest_section.get("bootstrap_samples", 500),
        alpha=backtest_section.get("alpha", 0.05),
        output_dir=Path(backtest_section.get("output_dir", "runs/eval")),
        sprt_min_effect=backtest_section.get("sprt_min_effect", 0.05),
        sprt_accept_prob=backtest_section.get("sprt_accept_prob", 0.8),
    )
    opponents = _build_opponents(backtest_section)
    runner = BacktestRunner(simulator=simulator, config=backtest_cfg)

    def policy_factory() -> ModelPolicyBot:
        return ModelPolicyBot(model=model, device=device, name="policy")

    result = runner.run_policy_eval(policy_factory, opponents)
    out_dir = runner.save_result(result)
    console.log(f"Backtest artefacts stored in {out_dir}")


def run_abtest(config: Dict[str, Any], ckpt_a: Path, ckpt_b: Path) -> None:
    model_a = _load_checkpoint(ckpt_a, config)
    model_b = _load_checkpoint(ckpt_b, config)
    device = torch.device(config.get("training", {}).get("device", "cpu"))
    simulator_cfg = SimulatorConfig(**config.get("simulator", {}))
    simulator = Simulator(config=simulator_cfg)
    backtest_section = config.get("backtest", {})
    backtest_cfg = BacktestConfig(
        num_games=backtest_section.get("num_games", 16),
        duplicate=backtest_section.get("duplicate", True),
        seeds=backtest_section.get("seeds"),
        bootstrap_samples=backtest_section.get("bootstrap_samples", 500),
        alpha=backtest_section.get("alpha", 0.05),
        output_dir=Path(backtest_section.get("output_dir", "runs/eval")),
        sprt_min_effect=backtest_section.get("sprt_min_effect", 0.05),
        sprt_accept_prob=backtest_section.get("sprt_accept_prob", 0.8),
    )
    opponents = _build_opponents(backtest_section)
    runner = BacktestRunner(simulator=simulator, config=backtest_cfg)

    def policy_a() -> ModelPolicyBot:
        return ModelPolicyBot(model=model_a, device=device, name="policy_a")

    def policy_b() -> ModelPolicyBot:
        return ModelPolicyBot(model=model_b, device=device, name="policy_b")

    summary = runner.run_ab_test(policy_a, policy_b, opponents)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = runner.config.output_dir / f"abtest-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.log(f"A/B test summary stored in {out_dir}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mahjong attention AI CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config_argument(sp):
        sp.add_argument("--config", type=Path, required=True, help="YAML config file")

    encode_parser = sub.add_parser("encode", help="Parse sample kifus into tensors")
    add_config_argument(encode_parser)

    train_parser = sub.add_parser("train", help="Run supervised training")
    add_config_argument(train_parser)

    eval_parser = sub.add_parser("eval", help="Evaluate on validation split")
    add_config_argument(eval_parser)
    eval_parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path")

    backtest_parser = sub.add_parser("backtest", help="Run stochastic backtests")
    add_config_argument(backtest_parser)
    backtest_parser.add_argument("--ckpt", type=Path, required=True)

    abtest_parser = sub.add_parser("abtest", help="Compare two checkpoints")
    add_config_argument(abtest_parser)
    abtest_parser.add_argument("--ckpt-a", type=Path, required=True)
    abtest_parser.add_argument("--ckpt-b", type=Path, required=True)

    args = parser.parse_args(argv)
    config = load_yaml(args.config)

    if args.command == "encode":
        run_encode(config)
    elif args.command == "train":
        run_train(config)
    elif args.command == "eval":
        run_eval(config, args.ckpt)
    elif args.command == "backtest":
        run_backtest(config, args.ckpt)
    elif args.command == "abtest":
        run_abtest(config, args.ckpt_a, args.ckpt_b)
    else:  # pragma: no cover - defensive
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()

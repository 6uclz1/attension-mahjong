"""Supervised imitation trainer for mahjong policy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from rich.console import Console
from rich.progress import Progress

from ..dataio.dataset import DataLoaderBundle
from ..dataio.schema import MahjongBatch
from ..models.transformer import MahjongTransformerModel
from .metrics import aggregate_losses, basic_metrics

try:  # pragma: no cover - optional dependency
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


@dataclass
class TrainerConfig:
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    lambda_value: float = 0.1
    lambda_aux: float = 0.1
    device: str = "cpu"
    output_dir: Path = Path("runs/latest")
    use_wandb: bool = False


class SupervisedTrainer:
    """Simple training loop for supervised imitation."""

    def __init__(
        self,
        model: MahjongTransformerModel,
        dataloaders: DataLoaderBundle,
        config: TrainerConfig,
    ) -> None:
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.console = Console()
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_top1 = float("-inf")
        self.best_path = self.output_dir / "best.ckpt"
        self.use_wandb = bool(wandb) and config.use_wandb
        self.wandb_run = None
        if self.use_wandb:
            self.wandb_run = wandb.init(project="mahjong-attn-ai", config=config.__dict__)

    def fit(self) -> Dict[str, float]:
        history: Dict[str, float] = {}
        progress = Progress(console=self.console)
        with progress:
            for epoch in progress.track(range(1, self.config.epochs + 1), description="Training"):
                train_log = self._run_epoch(epoch)
                val_log = self._run_validation(epoch)
                history.update({f"train_{k}": v for k, v in train_log.items()})
                history.update({f"val_{k}": v for k, v in val_log.items()})
                if self.use_wandb and self.wandb_run is not None:
                    wandb.log({**{f"train/{k}": v for k, v in train_log.items()}, **{f"val/{k}": v for k, v in val_log.items()}, "epoch": epoch})
                progress.console.log(
                    f"Epoch {epoch} | train_loss={train_log['loss']:.4f} val_loss={val_log['loss']:.4f} "
                    f"val_top1={val_log['top1']:.3f} val_top3={val_log['top3']:.3f}"
                )
                if val_log["top1"] > self.best_top1:
                    self.best_top1 = val_log["top1"]
                    self._save_checkpoint(epoch)
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()
        return history

    def _run_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        top1 = 0.0
        top3 = 0.0
        for batch in self.dataloaders.train_loader:
            total_steps += 1
            batch = batch.to_device(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(batch)
            losses = self.model.loss(
                outputs,
                batch,
                lambda_value=self.config.lambda_value,
                lambda_aux=self.config.lambda_aux,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += float(losses["total"].detach().cpu().item())
            logs = basic_metrics(outputs.policy_log_probs.detach(), batch.label_actions.detach())
            top1 += logs["top1"]
            top3 += logs["top3"]
        num_batches = max(1, total_steps)
        return {
            "loss": total_loss / num_batches,
            "top1": top1 / num_batches,
            "top3": top3 / num_batches,
        }

    @torch.no_grad()
    def _run_validation(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        top1 = 0.0
        top3 = 0.0
        for batch in self.dataloaders.val_loader:
            total_steps += 1
            batch = batch.to_device(self.device)
            outputs = self.model(batch)
            losses = self.model.loss(
                outputs,
                batch,
                lambda_value=self.config.lambda_value,
                lambda_aux=self.config.lambda_aux,
            )
            total_loss += float(losses["total"].detach().cpu().item())
            logs = basic_metrics(outputs.policy_log_probs, batch.label_actions)
            top1 += logs["top1"]
            top3 += logs["top3"]
        num_batches = max(1, total_steps)
        return {
            "loss": total_loss / num_batches,
            "top1": top1 / num_batches,
            "top3": top3 / num_batches,
        }

    def _save_checkpoint(self, epoch: int) -> None:
        payload = {
            "model_state": self.model.state_dict(),
            "epoch": epoch,
            "best_top1": self.best_top1,
        }
        torch.save(payload, self.best_path)


__all__ = ["SupervisedTrainer", "TrainerConfig"]


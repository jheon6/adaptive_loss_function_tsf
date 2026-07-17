"""
Trainer — backbone-agnostic training loop.

Design decisions:
  - A single optimizer is shared between the backbone and the loss function.
    This ensures that the WeightGeneratorMLP (inside AdaptiveLossWeighting)
    is updated via backpropagation in every training step.
  - Early stopping is based on validation loss with configurable patience.
  - The loss function's forward signature is:
        loss_fn(pred, target, x_enc)
    where x_enc is the encoder input.  FixedWeightLoss ignores x_enc;
    AdaptiveLossWeighting uses it for feature extraction.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metrics import compute_metrics
from utils.logger import ExperimentLogger


class EarlyStopping:
    """
    Stops training when validation loss does not improve for `patience` epochs.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.patience == 0:
            return False  # disabled
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Trains a backbone model with any compatible loss function.

    Args:
        backbone      : nn.Module — the forecasting backbone (e.g. DLinear)
        loss_fn       : nn.Module — loss function (FixedWeightLoss or AdaptiveLossWeighting)
        config        : ExperimentConfig
        train_loader  : DataLoader for training split
        val_loader    : DataLoader for validation split
        test_loader   : DataLoader for test split
        logger        : ExperimentLogger

    Optimizer covers both backbone.parameters() and loss_fn.parameters()
    so that the WeightGeneratorMLP is trained end-to-end.
    """

    def __init__(
        self,
        backbone: nn.Module,
        loss_fn: nn.Module,
        config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        logger: ExperimentLogger,
    ):
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger

        self.device = torch.device(config.device)
        self.backbone.to(self.device)
        self.loss_fn.to(self.device)

        # Joint optimizer — backbone + weight generator trained together,
        # but the weight generator (when it has learnable params, e.g. inside
        # AdaptiveLossWeighting) gets its own LR. It's the more sensitive,
        # instability-prone part of the graph, so it defaults to a fraction
        # of the backbone's LR (weight_gen_lr_scale) rather than sharing it.
        backbone_params = list(backbone.parameters())
        loss_fn_params = list(loss_fn.parameters())
        lr_scale = getattr(config, "weight_gen_lr_scale", 1.0)

        param_groups = [{"params": backbone_params, "lr": config.learning_rate}]
        if loss_fn_params:
            param_groups.append({"params": loss_fn_params, "lr": config.learning_rate * lr_scale})

        self.optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=config.weight_decay,
        )

        self.early_stopping = EarlyStopping(patience=config.patience)

    # ------------------------------------------------------------------
    # Training / evaluation steps
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.backbone.train()
        self.loss_fn.train()
        total_loss = 0.0

        for x_enc, y in self.train_loader:
            x_enc = x_enc.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.backbone(x_enc)            # (B, pred_len, C)
            loss = self.loss_fn(pred, y, x_enc)    # scalar
            loss.backward()
            # Gradient clipping stabilizes training with the weight generator
            torch.nn.utils.clip_grad_norm_(
                list(self.backbone.parameters()) + list(self.loss_fn.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple:
        """
        Returns:
            adaptive_loss : the loss_fn's val loss (used for logging only)
            monitor_mse   : raw MSE in normalized space (used for checkpointing)
                            This is scale-consistent across all loss types.
        """
        self.backbone.eval()
        self.loss_fn.eval()
        total_adaptive_loss = 0.0
        total_mse = 0.0

        for x_enc, y in loader:
            x_enc = x_enc.to(self.device)
            y = y.to(self.device)
            pred = self.backbone(x_enc)
            total_adaptive_loss += self.loss_fn(pred, y, x_enc).item()
            # Raw MSE (normalized space) — scale-consistent monitoring signal.
            # point_forecast() corrects for loss functions (e.g. skew-normal
            # NLL) whose optimal `pred` is a distribution location, not its
            # mean — the raw location is the wrong thing to score with MSE.
            eval_pred = self.loss_fn.point_forecast(pred, x_enc) if hasattr(self.loss_fn, "point_forecast") else pred
            total_mse += ((eval_pred - y) ** 2).mean().item()

        n = len(loader)
        return total_adaptive_loss / n, total_mse / n

    @torch.no_grad()
    def _test(self, loader: DataLoader, dataset) -> dict:
        """
        Run inference on the test set and compute evaluation metrics.
        Predictions are inverse-transformed before computing metrics.
        """
        self.backbone.eval()
        all_preds, all_targets = [], []

        for x_enc, y in loader:
            x_enc = x_enc.to(self.device)
            pred = self.backbone(x_enc)
            if hasattr(self.loss_fn, "point_forecast"):
                pred = self.loss_fn.point_forecast(pred, x_enc)
            pred = pred.cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y.numpy())

        preds = np.concatenate(all_preds, axis=0)       # (N, T, C)
        targets = np.concatenate(all_targets, axis=0)

        # Inverse-transform if the dataset has a scaler
        if hasattr(dataset, "inverse_transform"):
            B, T, C = preds.shape
            preds_flat = preds.reshape(-1, C)
            targets_flat = targets.reshape(-1, C)
            preds_flat = dataset.inverse_transform(preds_flat)
            targets_flat = dataset.inverse_transform(targets_flat)
            preds = preds_flat.reshape(B, T, C)
            targets = targets_flat.reshape(B, T, C)

        return compute_metrics(preds, targets)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """
        Full training loop with early stopping and checkpointing.
        Loads the best checkpoint for final test evaluation.
        """
        self.logger.log(f"Training started — device: {self.device}")
        self.logger.log(
            f"  Backbone params : {sum(p.numel() for p in self.backbone.parameters()):,}"
        )
        self.logger.log(
            f"  Loss fn  params : {sum(p.numel() for p in self.loss_fn.parameters()):,}"
        )

        epoch_history = []

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()
            val_loss, val_mse = self._eval_epoch(self.val_loader)

            # val_mse: raw MSE in normalized space
            # → scale-consistent across all loss types → used for checkpoint & early stopping
            # val_loss: the loss_fn's own val loss (may be on a different scale for adaptive)
            # → logged for analysis only
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mse": val_mse,
            }

            # Log adaptive weights and running means if available
            if hasattr(self.loss_fn, "get_current_weights"):
                sample_x, _ = next(iter(self.val_loader))
                sample_x = sample_x.to(self.device)
                weight_dict = self.loss_fn.get_current_weights(sample_x)
                metrics.update({f"w_{k}": v for k, v in weight_dict.items()})
            if hasattr(self.loss_fn, "get_running_means"):
                mean_dict = self.loss_fn.get_running_means()
                metrics.update({f"ema_{k}": v for k, v in mean_dict.items()})

            epoch_history.append(metrics)
            self.logger.log_epoch(epoch, metrics)
            self.logger.save_checkpoint(self.backbone, self.loss_fn, epoch, val_mse)

            if self.early_stopping.step(val_mse):
                self.logger.log(f"Early stopping triggered at epoch {epoch}.")
                break

        # Final evaluation on test set using best checkpoint
        self.logger.load_best_checkpoint(self.backbone, self.loss_fn, self.device)
        test_dataset = self.test_loader.dataset
        test_metrics = self._test(self.test_loader, test_dataset)

        self.logger.log("\n" + "=" * 50)
        self.logger.log("Test Results:")
        for k, v in test_metrics.items():
            self.logger.log(f"  {k}: {v:.6f}")
        self.logger.log("=" * 50)

        return test_metrics, epoch_history

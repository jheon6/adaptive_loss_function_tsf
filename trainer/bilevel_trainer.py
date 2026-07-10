"""
BilevelTrainer — prototype: meta-learn the adaptive loss weighting (phi)
against validation loss instead of fitting it to the same train-batch loss
it modulates.

Motivation:
    AdaptiveLossWeighting's uncertainty-weighting formulation already removes
    the "free lunch" of gaming the train loss (see losses/adaptive_loss.py) —
    the MLE optimum log_var_i* = log(train_loss_i) is an honest per-(sample,
    channel) noise estimate, not an exploit. But phi is still shaped entirely
    by what happened on this training batch; nothing tells it whether the
    resulting weighting actually helps the backbone generalize.

Bilevel objective:
    phi*   = argmin_phi  L_val( theta*(phi) )
    theta*(phi) = argmin_theta L_train(theta, phi)

Full bilevel differentiates through the entire inner optimization trajectory
(unrolled SGD across many steps, `higher`-style). This prototype uses the
1-step approximation instead (Ren et al., 2018, "Learning to Reweight
Examples for Robust Deep Learning" — adapted here to a persistent weight
generator network instead of free per-example scalars): unroll a single
virtual SGD step of theta, evaluate val loss at that virtual point, and
backprop through that one step into phi. No `higher` dependency —
torch.func.functional_call + torch.autograd.grad(create_graph=True) is
enough for a single step.

Per training batch:
    1. pred_a       = backbone(x_tr)                    [current theta]
    2. loss_train_a = loss_fn(pred_a, y_tr, x_tr)        [current phi, loss_fn.eval()
                                                           so this measurement pass
                                                           doesn't perturb EMA buffers
                                                           or apply dropout]
    3. grads_theta  = d(loss_train_a)/d(theta), create_graph=True
                      → differentiable w.r.t. phi (loss_train_a depends on phi)
    4. theta'       = theta - inner_lr * grads_theta      [virtual step, never applied]
    5. pred_val     = functional_call(backbone, theta', x_val)
    6. loss_meta    = MSE(pred_val, y_val)                [raw MSE — the generalization
                                                           signal; deliberately NOT run
                                                           through loss_fn, since phi is
                                                           exactly what we're scoring]
    7. grads_phi    = d(loss_meta)/d(phi)
       phi          <- phi - meta_lr * grads_phi           [real update]
    8. pred_b       = backbone(x_tr)                       [current theta, fresh forward,
                                                            loss_fn.train() restored]
       loss_train_b = loss_fn(pred_b, y_tr, x_tr)           [updated phi]
       theta        <- theta - lr * d(loss_train_b)/d(theta) [real update]

Steps 1-6 never call .step() on theta or phi — they only measure a
meta-gradient via a virtual, discarded update. Step 8 is a normal
forward/backward/step using the phi that was just updated in step 7, and
is what actually moves theta and what updates the EMA/running-stat buffers.

Caveat: DLinear has no dropout/BatchNorm, so toggling loss_fn.eval() for
step 2 only silences its own dropout+EMA. If this trainer is later pointed
at a backbone with dropout/BatchNorm, step 2 would need the same eval()
treatment on the backbone for the measurement to stay side-effect-free.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader

from .trainer import EarlyStopping
from utils.metrics import compute_metrics


class BilevelTrainer:
    """
    Same external interface as Trainer (build with backbone/loss_fn/config/
    loaders/logger, call .train()), but the weight generator inside loss_fn
    is updated via the 1-step bilevel meta-gradient instead of jointly with
    the backbone on the same train-batch loss.
    """

    def __init__(
        self,
        backbone: nn.Module,
        loss_fn: nn.Module,
        config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        logger,
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

        self.inner_lr = getattr(config, "bilevel_inner_lr", None) or config.learning_rate
        meta_lr_scale = getattr(config, "weight_gen_lr_scale", 0.5)

        self.backbone_optimizer = torch.optim.Adam(
            self.backbone.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        loss_fn_params = list(self.loss_fn.parameters())
        assert loss_fn_params, (
            "BilevelTrainer requires loss_fn to have learnable parameters "
            "(e.g. loss_type='adaptive'); got none."
        )
        self.meta_optimizer = torch.optim.Adam(
            loss_fn_params,
            lr=config.learning_rate * meta_lr_scale,
            weight_decay=config.weight_decay,
        )

        self.early_stopping = EarlyStopping(patience=config.patience)

        # Cycling iterator so every train step gets a meta-batch, independent
        # of val_loader's own epoch boundary.
        self._val_iter = iter(self.val_loader)

    def _next_val_batch(self):
        try:
            x_val, y_val = next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(self.val_loader)
            x_val, y_val = next(self._val_iter)
        return x_val.to(self.device), y_val.to(self.device)

    # ------------------------------------------------------------------
    # Training / evaluation steps
    # ------------------------------------------------------------------

    def _train_epoch(self) -> tuple:
        self.backbone.train()
        self.loss_fn.train()
        total_train_loss = 0.0
        total_meta_loss = 0.0
        loss_fn_params = list(self.loss_fn.parameters())
        backbone_named_params = list(self.backbone.named_parameters())

        for x_tr, y_tr in self.train_loader:
            x_tr = x_tr.to(self.device)
            y_tr = y_tr.to(self.device)
            x_val, y_val = self._next_val_batch()

            # ---- Steps 1-4: virtual backbone update ----
            self.loss_fn.eval()  # measurement pass: no dropout, no EMA side effects
            pred_a = self.backbone(x_tr)
            loss_train_a = self.loss_fn(pred_a, y_tr, x_tr)

            grads_theta = torch.autograd.grad(
                loss_train_a,
                [p for _, p in backbone_named_params],
                create_graph=True,
            )
            fast_params = {
                name: p - self.inner_lr * g
                for (name, p), g in zip(backbone_named_params, grads_theta)
            }

            # ---- Steps 5-6: meta objective on the val batch ----
            pred_val = functional_call(self.backbone, fast_params, (x_val,))
            loss_meta = F.mse_loss(pred_val, y_val)

            # ---- Step 7: meta-gradient -> update phi only ----
            grads_phi = torch.autograd.grad(loss_meta, loss_fn_params)
            self.meta_optimizer.zero_grad()
            for p, g in zip(loss_fn_params, grads_phi):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(loss_fn_params, max_norm=1.0)
            self.meta_optimizer.step()

            # ---- Step 8: real backbone update with the just-updated phi ----
            self.loss_fn.train()
            self.backbone_optimizer.zero_grad()
            pred_b = self.backbone(x_tr)
            loss_train_b = self.loss_fn(pred_b, y_tr, x_tr)
            loss_train_b.backward()
            torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=1.0)
            self.backbone_optimizer.step()

            total_train_loss += loss_train_b.item()
            total_meta_loss += loss_meta.item()

        n = len(self.train_loader)
        return total_train_loss / n, total_meta_loss / n

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple:
        self.backbone.eval()
        self.loss_fn.eval()
        total_adaptive_loss = 0.0
        total_mse = 0.0

        for x_enc, y in loader:
            x_enc = x_enc.to(self.device)
            y = y.to(self.device)
            pred = self.backbone(x_enc)
            total_adaptive_loss += self.loss_fn(pred, y, x_enc).item()
            total_mse += ((pred - y) ** 2).mean().item()

        n = len(loader)
        return total_adaptive_loss / n, total_mse / n

    @torch.no_grad()
    def _test(self, loader: DataLoader, dataset) -> dict:
        self.backbone.eval()
        all_preds, all_targets = [], []

        for x_enc, y in loader:
            x_enc = x_enc.to(self.device)
            pred = self.backbone(x_enc).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y.numpy())

        import numpy as np
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)

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
        self.logger.log(f"[Bilevel] Training started — device: {self.device}")
        self.logger.log(f"  inner_lr : {self.inner_lr}")
        self.logger.log(
            f"  Backbone params : {sum(p.numel() for p in self.backbone.parameters()):,}"
        )
        self.logger.log(
            f"  Loss fn  params : {sum(p.numel() for p in self.loss_fn.parameters()):,}"
        )

        epoch_history = []

        for epoch in range(1, self.config.epochs + 1):
            train_loss, meta_loss = self._train_epoch()
            val_loss, val_mse = self._eval_epoch(self.val_loader)

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "meta_loss": meta_loss,
                "val_loss": val_loss,
                "val_mse": val_mse,
            }

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

        self.logger.load_best_checkpoint(self.backbone, self.loss_fn, self.device)
        test_dataset = self.test_loader.dataset
        test_metrics = self._test(self.test_loader, test_dataset)

        self.logger.log("\n" + "=" * 50)
        self.logger.log("Test Results:")
        for k, v in test_metrics.items():
            self.logger.log(f"  {k}: {v:.6f}")
        self.logger.log("=" * 50)

        return test_metrics, epoch_history

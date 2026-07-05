"""
ExperimentLogger — handles console logging, CSV metric recording, and checkpointing.

Directory layout under `save_dir/exp_id/`:
    checkpoints/
        best_model.pt         — backbone state dict at best val loss
        best_loss_fn.pt       — loss function state dict (WeightGenerator params)
        last_model.pt         — most recent checkpoint
    metrics.csv               — per-epoch metrics (train_loss, val_loss, test_*)
    config.yaml               — copy of the run configuration
"""

import os
import csv
import time
import logging
import yaml
import torch
from dataclasses import asdict


class ExperimentLogger:
    """
    Centralizes all I/O for a single experiment run.

    Args:
        config     : experiment config dataclass
        exp_id     : unique string identifier for this run
        save_dir   : root directory for all experiment outputs
    """

    def __init__(self, config, exp_id: str, save_dir: str = "outputs"):
        self.config = config
        self.exp_id = exp_id
        self.run_dir = os.path.join(save_dir, exp_id)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")

        os.makedirs(self.ckpt_dir, exist_ok=True)

        self._setup_console_logger()
        self._setup_csv()
        self._save_config()

        self.best_val_loss = float("inf")
        self.start_time = time.time()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_console_logger(self):
        log_path = os.path.join(self.run_dir, "run.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path),
            ],
        )
        self.logger = logging.getLogger(self.exp_id)

    def _setup_csv(self):
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)
        self._csv_header_written = False

    def _save_config(self):
        config_path = os.path.join(self.run_dir, "config.yaml")
        try:
            config_dict = asdict(self.config)
        except TypeError:
            config_dict = vars(self.config)
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, msg: str):
        """Print and save a plain-text log line."""
        self.logger.info(msg)

    def log_epoch(self, epoch: int, metrics: dict):
        """
        Log per-epoch metrics to console and CSV.

        Args:
            epoch   : current epoch index (1-based)
            metrics : dict of metric_name → float
        """
        elapsed = time.time() - self.start_time
        header = f"Epoch {epoch:>4d} | {elapsed:6.0f}s"
        metric_str = "  ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
        self.logger.info(f"{header} | {metric_str}")

        # Append to CSV
        row = {"epoch": epoch, **metrics}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

    def save_checkpoint(self, backbone, loss_fn, epoch: int, val_loss: float):
        """
        Save the latest checkpoint and update the best checkpoint
        if val_loss improves.
        """
        state = {
            "epoch": epoch,
            "backbone": backbone.state_dict(),
            "loss_fn": loss_fn.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(state, os.path.join(self.ckpt_dir, "last_checkpoint.pt"))

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(state, os.path.join(self.ckpt_dir, "best_checkpoint.pt"))
            self.logger.info(f"  [✓] New best val loss: {val_loss:.6f} — checkpoint saved.")

    def load_best_checkpoint(self, backbone, loss_fn, device: torch.device):
        """Load the best checkpoint into backbone and loss_fn in-place."""
        path = os.path.join(self.ckpt_dir, "best_checkpoint.pt")
        assert os.path.exists(path), f"No checkpoint found at {path}"
        state = torch.load(path, map_location=device)
        backbone.load_state_dict(state["backbone"])
        loss_fn.load_state_dict(state["loss_fn"])
        self.logger.info(f"Loaded best checkpoint (epoch {state['epoch']}, val_loss {state['val_loss']:.6f})")

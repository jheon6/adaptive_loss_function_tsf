"""
Single-experiment entry point.

Usage:
    # Single run  → result/result1/report.html
    python experiments/run_experiment.py --config configs/experiments/adaptive_etth1.yaml

    # Ablation group run (called by run_ablation_etth1.sh)
    # → result/ablation_etth1_20240704_1200/dlinear_etth1_adaptive/report.html
    python experiments/run_experiment.py --config configs/experiments/adaptive_etth1.yaml \
        --result_dir result/ablation_etth1_20240704_1200
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from configs.base_config import ExperimentConfig
from datasets.data_factory import build_dataloader
from models import build_model
from losses import build_loss
from trainer import Trainer
from utils.logger import ExperimentLogger
from utils.html_reporter import generate_report
from utils.summary_reporter import append_result
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single TSF experiment.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config file.")

    # Allow overriding any config field from the command line
    parser.add_argument("--exp_id",        type=str,   default=None)
    parser.add_argument("--pred_len",      type=int,   default=None)
    parser.add_argument("--seq_len",       type=int,   default=None)
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--loss_type",     type=str,   default=None)
    parser.add_argument("--device",        type=str,   default=None)
    parser.add_argument("--save_dir",      type=str,   default=None)
    parser.add_argument("--seed",          type=int,   default=None)

    # Ablation group: when set, all reports go into this directory
    # with exp_id as subfolder name instead of auto-incremented resultN
    parser.add_argument("--result_dir",    type=str,   default=None,
                        help="Override report output directory (used by run_ablation_etth1.sh).")

    return parser.parse_args()


def run(config: ExperimentConfig, result_dir: str = "result"):
    set_seed(config.seed)

    logger = ExperimentLogger(config, config.exp_id, config.save_dir)
    logger.log(f"Experiment: {config.exp_id}")
    logger.log(f"  loss_type  : {config.loss_type}")
    logger.log(f"  pred_len   : {config.pred_len}")
    logger.log(f"  seq_len    : {config.seq_len}")
    logger.log(f"  result_dir : {result_dir}")

    train_loader = build_dataloader(config, split="train")
    val_loader   = build_dataloader(config, split="val")
    test_loader  = build_dataloader(config, split="test")

    config.num_features = train_loader.dataset.num_features

    backbone = build_model(config)
    loss_fn  = build_loss(config)

    trainer = Trainer(
        backbone=backbone,
        loss_fn=loss_fn,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        logger=logger,
    )

    test_metrics, epoch_history = trainer.train()

    report_path = generate_report(
        config=config,
        epoch_history=epoch_history,
        test_metrics=test_metrics,
        save_dir=result_dir,
    )
    logger.log(f"\nHTML report saved  → {report_path}")

    append_result(config, test_metrics, report_path)
    logger.log(f"Summary updated    → result/summary.html")

    return test_metrics


if __name__ == "__main__":
    args = parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    for key in ["exp_id", "pred_len", "seq_len", "epochs",
                "batch_size", "learning_rate", "loss_type", "device", "save_dir", "seed"]:
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    result_dir = args.result_dir if args.result_dir else "result"
    run(config, result_dir=result_dir)

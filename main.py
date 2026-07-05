"""
Main entry point.

Examples:
    python main.py --config configs/experiments/adaptive_etth1.yaml
    python main.py --config configs/experiments/adaptive_etth1.yaml --pred_len 192
    bash scripts/run_ablation_etth1.sh
"""

from experiments.run_experiment import parse_args, run, ExperimentConfig


if __name__ == "__main__":
    args = parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    for key in ["exp_id", "pred_len", "seq_len", "epochs",
                "batch_size", "learning_rate", "loss_type", "device", "save_dir"]:
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)

    result_dir = args.result_dir if args.result_dir else "result"
    run(config, result_dir=result_dir)

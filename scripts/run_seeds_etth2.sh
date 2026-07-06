#!/usr/bin/env bash
# ============================================================
# Multi-seed ablation study: ETTh2, DLinear, pred_len = 96
#
# Runs all 5 loss-type configs across multiple seeds so that
# mean/std can be compared (a single seed isn't enough to tell
# a real improvement from run-to-run noise).
#
# Results go under a single timestamped folder:
#   result/seeds_etth2_YYYYMMDD_HHMM/
#     dlinear_etth2_mse_only_seed42/report.html
#     dlinear_etth2_mse_only_seed43/report.html
#     ...
#
# Each run also appends a row to result/results_log.json, so after
# this script finishes, aggregate with e.g.:
#   python - <<'PY'
#   import json, statistics
#   data = json.load(open("result/results_log.json"))
#   group = "seeds_etth2_20260706_1200"  # <- your timestamp
#   rows = [d for d in data if group in d["report"]]
#   ...
#   PY
# ============================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"

SEEDS=(42 43 44 45 46)

TIMESTAMP=$(date +%Y%m%d_%H%M)
RESULT_DIR="result/seeds_etth2_${TIMESTAMP}"
echo "Multi-seed output dir: $ROOT/$RESULT_DIR"
echo ""

CONFIGS=(
    "mse_only_etth2"
    "mse_mae_fixed_etth2"
    "mse_trend_fixed_etth2"
    "mse_mae_trend_fixed_etth2"
    "adaptive_etth2"
)

for seed in "${SEEDS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        exp_id="dlinear_etth2_${cfg%_etth2}_seed${seed}"
        echo "============================================"
        echo "Running: $cfg  (seed=$seed)"
        echo "============================================"
        $PYTHON "$ROOT/experiments/run_experiment.py" \
            --config "$ROOT/configs/experiments/${cfg}.yaml" \
            --seed "$seed" \
            --exp_id "$exp_id" \
            --result_dir "$RESULT_DIR"
        echo ""
    done
done

echo "============================================"
echo "All multi-seed experiments completed."
echo "Reports : $ROOT/$RESULT_DIR/"
echo "Group id for aggregation: seeds_etth2_${TIMESTAMP}"
echo "============================================"

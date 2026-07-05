#!/usr/bin/env bash
# ============================================================
# Full ablation study: ETTm2, DLinear, pred_len = 96
# All 5 experiments are saved under a single timestamped folder:
#   result/ablation_ettm2_YYYYMMDD_HHMM/
#     dlinear_ettm2_mse_only/report.html
#     dlinear_ettm2_mse_mae_fixed/report.html
#     dlinear_ettm2_mse_trend_fixed/report.html
#     dlinear_ettm2_mse_mae_trend_fixed/report.html
#     dlinear_ettm2_adaptive/report.html
#
# frequency loss is excluded from this ablation set for now
# (see losses/components.py FrequencyLoss — still available, just unused here).
#
# result/summary.html is updated after each experiment
# (baseline = mse_only, latest = last finished experiment).
# ============================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"

# One shared folder for the entire ablation run
TIMESTAMP=$(date +%Y%m%d_%H%M)
RESULT_DIR="result/ablation_ettm2_${TIMESTAMP}"
echo "Ablation output dir: $ROOT/$RESULT_DIR"
echo ""

CONFIGS=(
    "mse_only_ettm2"
    "mse_mae_fixed_ettm2"
    "mse_trend_fixed_ettm2"
    "mse_mae_trend_fixed_ettm2"
    "adaptive_ettm2"
)

for cfg in "${CONFIGS[@]}"; do
    echo "============================================"
    echo "Running: $cfg"
    echo "============================================"
    $PYTHON "$ROOT/experiments/run_experiment.py" \
        --config "$ROOT/configs/experiments/${cfg}.yaml" \
        --result_dir "$RESULT_DIR"
    echo ""
done

echo "============================================"
echo "All ablation experiments completed."
echo "Reports : $ROOT/$RESULT_DIR/"
echo "Summary : $ROOT/result/summary.html"
echo "============================================"

#!/usr/bin/env bash
# ============================================================
# Pilot: skew-normal NLL vs. mse_only / adaptive on ETTh2, DLinear, pred_len=96
#
# Quick single-seed signal check for the skew-normal NLL redesign
# (see losses/distributional_loss.py) before committing to a full
# multi-dataset / multi-seed sweep. All 3 experiments are saved under
# a single timestamped folder:
#   result/pilot_skew_nll_etth2_YYYYMMDD_HHMM/
#     dlinear_etth2_mse_only/report.html
#     dlinear_etth2_adaptive/report.html
#     dlinear_etth2_skew_nll/report.html
#
# Usage:
#   bash scripts/run_pilot_skew_nll_etth2.sh              # full run (50 epochs)
#   bash scripts/run_pilot_skew_nll_etth2.sh --epochs 2   # smoke test
# ============================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"

TIMESTAMP=$(date +%Y%m%d_%H%M)
RESULT_DIR="result/pilot_skew_nll_etth2_${TIMESTAMP}"
echo "Pilot output dir: $ROOT/$RESULT_DIR"
echo ""

CONFIGS=(
    "mse_only_etth2"
    "adaptive_etth2"
    "skew_nll_etth2"
)

for cfg in "${CONFIGS[@]}"; do
    echo "============================================"
    echo "Running: $cfg"
    echo "============================================"
    $PYTHON "$ROOT/experiments/run_experiment.py" \
        --config "$ROOT/configs/experiments/${cfg}.yaml" \
        --result_dir "$RESULT_DIR" \
        "$@"
    echo ""
done

echo "============================================"
echo "Pilot completed."
echo "Reports : $ROOT/$RESULT_DIR/"
echo "Summary : $ROOT/result/summary.html"
echo "============================================"

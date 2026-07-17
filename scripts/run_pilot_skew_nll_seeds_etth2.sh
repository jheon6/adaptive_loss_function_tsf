#!/usr/bin/env bash
# ============================================================
# Multi-seed noise check: adaptive vs. skew_normal_nll on ETTh2, DLinear
#
# The single-seed pilot (seed 42) showed skew_normal_nll close behind
# adaptive (MSE 14.68 vs 14.59, MAE 2.42 vs 2.36) — this checks whether
# that gap is a real effect or within seed-to-seed noise, same pattern
# as scripts/run_seeds_etth2.sh.
#
# Results go under:
#   result/pilot_skew_nll_seeds_etth2_YYYYMMDD_HHMM/
#     dlinear_etth2_adaptive_seed42/report.html
#     dlinear_etth2_skew_nll_seed42/report.html
#     ...
# ============================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"

SEEDS=(42 43 44 45 46)

TIMESTAMP=$(date +%Y%m%d_%H%M)
RESULT_DIR="result/pilot_skew_nll_seeds_etth2_${TIMESTAMP}"
echo "Multi-seed output dir: $ROOT/$RESULT_DIR"
echo ""

CONFIGS=(
    "adaptive_etth2"
    "skew_nll_etth2"
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
echo "Group id for aggregation: pilot_skew_nll_seeds_etth2_${TIMESTAMP}"
echo "============================================"

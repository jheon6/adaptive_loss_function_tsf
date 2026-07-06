"""
Compare ablation results while filtering out stale "adaptive" runs that were
trained with a loss configuration that no longer matches the current
AdaptiveLossWeighting module (losses/adaptive_loss.py).

Background:
    Early ETTh1 ablation runs used a 4-component adaptive loss
    (mse/mae/trend/frequency). The frequency component caused training to
    diverge in some runs (val loss rising monotonically from epoch 1,
    final test MSE ~14.8-15.0 vs. ~9.3-9.5 for every other run). Frequency
    was later dropped from AdaptiveLossWeighting, and every "adaptive" run
    since has been stable. Because results_log.json has no version tag,
    naively averaging "adaptive" across all logged runs mixes the two
    populations and makes adaptive look worse than it currently is.

    This script detects which "adaptive" runs are legacy (4-loss) by
    inspecting the embedded weight chart in each run's report.html for a
    non-zero "Frequency weight" series, and excludes them by default.

Usage:
    python scripts/compare_results.py
    python scripts/compare_results.py --include-legacy   # show the unfiltered picture too
    python scripts/compare_results.py --csv out.csv       # also write the clean per-run table
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "result" / "results_log.json"


def load_results(log_path: Path) -> list[dict]:
    with open(log_path, encoding="utf-8") as f:
        return json.load(f)


def is_legacy_adaptive(report_path: str) -> bool:
    """True if this adaptive run's report.html shows a non-zero Frequency
    weight series, i.e. it was trained before FrequencyLoss was dropped
    from AdaptiveLossWeighting."""
    path = ROOT / report_path
    if not path.exists():
        return False
    html = path.read_text(encoding="utf-8")
    match = re.search(r"label: 'Frequency weight',\s*data: \[([^\]]*)\]", html)
    if not match:
        return False
    values = [float(v) for v in match.group(1).split(",") if v.strip()]
    return any(v != 0.0 for v in values)


def annotate(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["legacy"] = False
    adaptive_mask = df["loss_type"] == "adaptive"
    df.loc[adaptive_mask, "legacy"] = df.loc[adaptive_mask, "report"].apply(is_legacy_adaptive)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["dataset", "loss_type"])
        .agg(n=("mse", "size"), mse_mean=("mse", "mean"), mse_std=("mse", "std"),
             mae_mean=("mae", "mean"), mae_std=("mae", "std"))
        .reset_index()
    )
    grouped["mse_std"] = grouped["mse_std"].fillna(0.0)
    grouped["mae_std"] = grouped["mae_std"].fillna(0.0)

    # delta vs. mse_only baseline, per dataset
    baseline = grouped[grouped["loss_type"] == "mse_only"].set_index("dataset")["mse_mean"]
    grouped["delta_vs_mse_only"] = grouped.apply(
        lambda r: r["mse_mean"] - baseline.get(r["dataset"], float("nan")), axis=1
    )
    return grouped.sort_values(["dataset", "loss_type"]).reset_index(drop=True)


def print_table(title: str, table: pd.DataFrame):
    print(f"\n=== {title} ===")
    display = table.copy()
    for col in ("mse_mean", "mse_std", "mae_mean", "mae_std", "delta_vs_mse_only"):
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    print(display.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-legacy", action="store_true",
                         help="Also print the unfiltered summary (legacy 4-loss adaptive runs included) for comparison.")
    parser.add_argument("--csv", type=Path, default=None,
                         help="Write the clean (legacy-excluded) per-run table to this CSV path.")
    args = parser.parse_args()

    results = load_results(LOG_PATH)
    df = annotate(results)

    legacy_rows = df[df["legacy"]]
    if not legacy_rows.empty:
        print("Excluded legacy (4-loss) adaptive runs:")
        for _, r in legacy_rows.iterrows():
            print(f"  {r['timestamp']}  {r['dataset']:10s}  mse={r['mse']:.4f}  report={r['report']}")
    else:
        print("No legacy adaptive runs found.")

    clean = df[~df["legacy"]]
    print_table("Clean comparison (legacy adaptive runs excluded)", summarize(clean))

    if args.include_legacy:
        print_table("Unfiltered comparison (legacy adaptive runs included)", summarize(df))

    if args.csv:
        clean.drop(columns=["legacy"]).to_csv(args.csv, index=False)
        print(f"\nWrote clean per-run table to {args.csv}")


if __name__ == "__main__":
    main()

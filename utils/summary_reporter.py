"""
Summary Reporter — accumulates results across experiments and generates
a comparison HTML report at result/summary.html.

Each experiment appends one record to result/results_log.json.
summary.html is regenerated from scratch every time a new result is added,
so it always reflects the full history.

Comparison baseline: the most recent "mse_only" entry in the log.
Delta columns show improvement over that baseline (negative = better for MSE/MAE/RMSE).
"""

import os
import json
from datetime import datetime
from dataclasses import asdict



RESULTS_LOG = "result/results_log.json"
SUMMARY_HTML = "result/summary.html"


# ───────────────���──────────────────────���───────────────────────────────────────
# Registry helpers
# ─────────────────��─────────────────────��──────────────────────────────────────

def _load_log() -> list:
    if not os.path.exists(RESULTS_LOG):
        return []
    with open(RESULTS_LOG, "r") as f:
        return json.load(f)


def _save_log(records: list):
    os.makedirs("result", exist_ok=True)
    with open(RESULTS_LOG, "w") as f:
        json.dump(records, f, indent=2)


def append_result(config, test_metrics: dict, report_path: str):
    """
    Append this experiment's result to results_log.json and
    regenerate summary.html.

    Args:
        config       : ExperimentConfig dataclass
        test_metrics : {"mse":..., "mae":..., "rmse":...}
        report_path  : relative path to this experiment's report.html
    """
    try:
        config_dict = asdict(config)
    except TypeError:
        config_dict = vars(config)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp_id":    config_dict.get("exp_id", "—"),
        "model":     config_dict.get("model_name", "—"),
        "dataset":   config_dict.get("data_name", "—"),
        "loss_type": config_dict.get("loss_type", "—"),
        "pred_len":  config_dict.get("pred_len", "—"),
        "seq_len":   config_dict.get("seq_len", "—"),
        "epochs":    config_dict.get("epochs", "—"),
        "lr":        config_dict.get("learning_rate", "—"),
        "mse":       test_metrics.get("mse", 0.0),
        "mae":       test_metrics.get("mae", 0.0),
        "rmse":      test_metrics.get("rmse", 0.0),
        "report":    report_path,
    }

    records = _load_log()
    records.append(record)
    _save_log(records)

    _generate_summary(records)


# ─────────────────���───────────────────────────���─────────────────────────────���──
# Summary HTML generation
# ──────────────────────────────────���────────────────────────��──────────────────

def _get_ablation_group(report_path: str) -> str:
    """
    Extract the ablation group folder name from a report path.
    e.g. "result/ablation_20240704_1430/exp/report.html" → "ablation_20240704_1430"
    Returns "" if not an ablation run (e.g. "result/result1/report.html").
    """
    parts = report_path.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("ablation_"):
            return part
    return ""


def _generate_summary(records: list):
    """
    Rebuild summary.html.

    - Ablation run  : show all experiments from the latest ablation group
    - Single run    : show mse_only baseline + latest experiment only
    """
    if not records:
        return

    # Determine display records based on latest experiment's origin
    latest = records[-1]
    latest_group = _get_ablation_group(latest.get("report", ""))

    if latest_group:
        # Ablation mode: collect all records from the same ablation group
        display = [r for r in records if _get_ablation_group(r.get("report", "")) == latest_group]
        mode_label = f"ablation group: <strong>{latest_group}</strong>"
    else:
        # Single run mode: baseline + latest
        baseline_record = next(
            (r for r in reversed(records) if r["loss_type"] == "mse_only"), None
        )
        display = []
        if baseline_record:
            display.append(baseline_record)
        if latest is not baseline_record:
            display.append(latest)
        if not display:
            display = [latest]
        mode_label = "baseline + latest"

    # Baseline for delta computation: mse_only within the display set
    baseline = next((r for r in display if r["loss_type"] == "mse_only"), None)

    rows_html  = _build_table_rows(display, baseline)
    chart_html = _build_bar_chart(display)
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment Summary</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#f8f9fb; --card:#fff; --border:#e2e8f0;
    --accent:#4f46e5; --text:#1e293b; --muted:#64748b;
    --green:#10b981; --red:#ef4444; --yellow:#f59e0b;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
          background:var(--bg); color:var(--text); font-size:14px; }}
  .page {{ max-width:1200px; margin:0 auto; padding:40px 24px; }}

  .header {{ background:var(--accent); color:#fff; border-radius:12px;
             padding:28px 32px; margin-bottom:28px; }}
  .header h1 {{ font-size:22px; font-weight:700; }}
  .header .sub {{ opacity:.8; font-size:13px; margin-top:4px; }}

  .section {{ background:var(--card); border:1px solid var(--border);
              border-radius:12px; padding:24px 28px; margin-bottom:24px; }}
  .section h2 {{ font-size:15px; font-weight:700; color:var(--accent);
                 margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:12px; margin-bottom:18px; }}

  /* Table */
  .tbl-wrap {{ overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#f1f5f9; text-align:left; padding:10px 12px;
        font-weight:600; color:var(--muted); border-bottom:2px solid var(--border); }}
  td {{ padding:10px 12px; border-bottom:1px solid var(--border); vertical-align:middle; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:#f8fafc; }}
  .row-best td {{ background:#f0fdf4 !important; }}

  /* Metric cell */
  .metric {{ font-weight:700; font-variant-numeric:tabular-nums; }}
  .best-val {{ color:var(--green); }}

  /* Delta badge */
  .delta {{ display:inline-block; font-size:11px; font-weight:700;
            border-radius:999px; padding:1px 8px; }}
  .delta.better {{ background:#dcfce7; color:#16a34a; }}
  .delta.worse  {{ background:#fee2e2; color:#dc2626; }}
  .delta.same   {{ background:#f1f5f9; color:var(--muted); }}

  /* Baseline tag */
  .tag {{ display:inline-block; font-size:10px; font-weight:700;
          border-radius:4px; padding:1px 6px; margin-left:6px; }}
  .tag.baseline {{ background:#e0e7ff; color:#4338ca; }}
  .tag.proposed {{ background:#dcfce7; color:#15803d; }}

  /* Report link */
  .link {{ color:var(--accent); text-decoration:none; font-size:12px; }}
  .link:hover {{ text-decoration:underline; }}

  /* Chart */
  .chart-container {{ position:relative; height:320px; }}

  /* Legend note */
  .note {{ font-size:12px; color:var(--muted); margin-top:12px; }}

  .footer {{ text-align:center; color:var(--muted); font-size:12px; margin-top:28px; }}
</style>
</head>
<body>
<div class="page">

  <div class="header">
    <h1>Experiment Summary</h1>
    <div class="sub">Last updated: {timestamp} &nbsp;·&nbsp; {len(records)} experiment(s) in log &nbsp;·&nbsp; showing {mode_label}</div>
  </div>

  <!-- Comparison Table -->
  <div class="section">
    <h2>Results Comparison</h2>
    <p class="subtitle">
      Δ columns show difference vs. <strong>mse_only</strong> baseline
      &nbsp;(green = improvement, red = degradation).
      {"No mse_only baseline found yet — run mse_only first to see deltas." if baseline is None else ""}
    </p>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Experiment ID</th>
            <th>Loss Type</th>
            <th>Pred Len</th>
            <th>MSE</th>
            <th>Δ MSE</th>
            <th>MAE</th>
            <th>Δ MAE</th>
            <th>RMSE</th>
            <th>Report</th>
            <th>Time</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    <p class="note">* Best values per column are highlighted in green.</p>
  </div>

  <!-- Bar Chart -->
  <div class="section">
    <h2>Metric Comparison Chart</h2>
    <p class="subtitle">MSE and MAE across all experiments</p>
    <div class="chart-container">
      <canvas id="barChart"></canvas>
    </div>
  </div>

  <div class="footer">
    Adaptive Loss Weighting Framework &nbsp;·&nbsp; {timestamp}
  </div>
</div>

<script>
{chart_html}
</script>
</body>
</html>"""

    with open(SUMMARY_HTML, "w", encoding="utf-8") as f:
        f.write(html)


def _build_table_rows(records: list, baseline: dict) -> str:
    if not records:
        return '<tr><td colspan="11" style="text-align:center;color:#94a3b8;">No experiments yet.</td></tr>'

    best_mse  = min(r["mse"]  for r in records)
    best_mae  = min(r["mae"]  for r in records)
    best_rmse = min(r["rmse"] for r in records)

    rows = []
    for i, r in enumerate(records, 1):
        is_baseline = r["loss_type"] == "mse_only"
        is_adaptive = r["loss_type"] == "adaptive"
        is_best_row = r["mse"] == best_mse

        # Tags
        tags = ""
        if is_baseline:
            tags += '<span class="tag baseline">baseline</span>'
        if is_adaptive:
            tags += '<span class="tag proposed">proposed</span>'

        # Metric cells
        mse_cls  = ' best-val' if r["mse"]  == best_mse  else ''
        mae_cls  = ' best-val' if r["mae"]  == best_mae  else ''
        rmse_cls = ' best-val' if r["rmse"] == best_rmse else ''

        # Delta vs baseline
        d_mse = _delta_badge(r["mse"],  baseline["mse"],  lower_better=True)  if baseline and not is_baseline else '<span class="delta same">—</span>'
        d_mae = _delta_badge(r["mae"],  baseline["mae"],  lower_better=True)  if baseline and not is_baseline else '<span class="delta same">—</span>'

        # Report link (relative path from result/)
        rel_report = os.path.relpath(r["report"], "result") if os.path.exists(r["report"]) else r["report"]
        link = f'<a class="link" href="{rel_report}" target="_blank">open ↗</a>'

        row_cls = ' class="row-best"' if is_best_row else ''
        rows.append(f"""<tr{row_cls}>
          <td>{i}</td>
          <td><strong>{r['exp_id']}</strong>{tags}</td>
          <td><code>{r['loss_type']}</code></td>
          <td>{r['pred_len']}</td>
          <td class="metric{mse_cls}">{r['mse']:.4f}</td>
          <td>{d_mse}</td>
          <td class="metric{mae_cls}">{r['mae']:.4f}</td>
          <td>{d_mae}</td>
          <td class="metric{rmse_cls}">{r['rmse']:.4f}</td>
          <td>{link}</td>
          <td style="color:var(--muted);font-size:12px;">{r['timestamp']}</td>
        </tr>""")

    return "\n".join(rows)


def _delta_badge(current: float, baseline: float, lower_better: bool = True) -> str:
    diff = current - baseline
    pct  = (diff / baseline * 100) if baseline != 0 else 0.0
    sign = "+" if diff > 0 else ""
    label = f"{sign}{pct:.2f}%"

    if abs(diff) < 1e-6:
        return f'<span class="delta same">{label}</span>'
    improved = (diff < 0) if lower_better else (diff > 0)
    cls = "better" if improved else "worse"
    return f'<span class="delta {cls}">{label}</span>'


def _build_bar_chart(records: list) -> str:
    if not records:
        return ""

    labels   = [r["exp_id"] for r in records]
    mse_vals = [round(r["mse"], 4) for r in records]
    mae_vals = [round(r["mae"], 4) for r in records]

    return f"""
const barCtx = document.getElementById('barChart').getContext('2d');
new Chart(barCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(labels)},
    datasets: [
      {{
        label: 'MSE',
        data: {json.dumps(mse_vals)},
        backgroundColor: 'rgba(79,70,229,0.75)',
        borderColor: '#4f46e5',
        borderWidth: 1,
        borderRadius: 4,
      }},
      {{
        label: 'MAE',
        data: {json.dumps(mae_vals)},
        backgroundColor: 'rgba(6,182,212,0.75)',
        borderColor: '#06b6d4',
        borderWidth: 1,
        borderRadius: 4,
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{
      x: {{ ticks: {{ maxRotation: 30 }} }},
      y: {{ title: {{ display: true, text: 'Value (lower is better)' }}, beginAtZero: true }}
    }}
  }}
}});"""



"""
HTML Report Generator for experiment results.

Generates a self-contained HTML file (report.html) under result/resultN/.
Auto-increments the result index by scanning existing subdirectories.

Report layout:
    1. Header       — experiment ID, timestamp, config details
    2. Training     — loss curve chart (train / val loss per epoch)
    3. Adaptive weights chart (only for adaptive loss type)
    4. Test Metrics — final MAE / MSE / RMSE / MAPE
"""

import os
import json
from datetime import datetime
from dataclasses import asdict


def _make_result_dir(base: str, exp_id: str, is_group: bool) -> str:
    """
    Determine and create the directory for this experiment's report.

    - Single run  (is_group=False): result/result1/, result/result2/, ...
    - Ablation    (is_group=True) : base/<exp_id>/   (exp_id as folder name)
    """
    os.makedirs(base, exist_ok=True)
    if is_group:
        path = os.path.join(base, exp_id)
    else:
        existing = [
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and d.startswith("result")
        ]
        indices = []
        for name in existing:
            try:
                indices.append(int(name.replace("result", "")))
            except ValueError:
                pass
        next_idx = max(indices, default=0) + 1
        path = os.path.join(base, f"result{next_idx}")
    os.makedirs(path, exist_ok=True)
    return path


def generate_report(
    config,
    epoch_history: list,
    test_metrics: dict,
    save_dir: str = "result",
) -> str:
    """
    Generate an HTML report for one experiment.

    Args:
        config        : ExperimentConfig dataclass
        epoch_history : [{"epoch":1, "train_loss":..., "val_loss":..., ...}, ...]
        test_metrics  : {"mse":..., "mae":..., "rmse":...}
        save_dir      : directory to save into.
                        "result"  → auto-increment resultN subfolder (single run)
                        anything else (e.g. "result/ablation_20240704") →
                        use exp_id as subfolder name (ablation group run)

    Returns:
        Absolute path to the generated report.html.
    """
    is_group = (save_dir != "result")
    try:
        exp_id = config.exp_id
    except AttributeError:
        exp_id = vars(config).get("exp_id", "exp")

    result_dir = _make_result_dir(save_dir, exp_id, is_group)
    report_path = os.path.join(result_dir, "report.html")

    try:
        config_dict = asdict(config)
    except TypeError:
        config_dict = vars(config)

    html = _build_html(config_dict, epoch_history, test_metrics)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path


# ──────────────────────────────────────────────────────────────────────────────
# HTML construction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_html(config_dict: dict, epoch_history: list, test_metrics: dict) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    is_adaptive = config_dict.get("loss_type") == "adaptive"

    epochs       = [e["epoch"]      for e in epoch_history]
    train_losses = [e["train_loss"] for e in epoch_history]
    val_losses   = [e["val_loss"]   for e in epoch_history]

    # Adaptive weight history (optional)
    weight_names = ["w_mse", "w_mae", "w_trend", "w_frequency"]
    weight_series = {}
    if is_adaptive and weight_names[0] in epoch_history[0]:
        for wn in weight_names:
            weight_series[wn] = [e.get(wn, 0.0) for e in epoch_history]

    config_rows = _config_table_rows(config_dict)
    test_cards  = _test_metric_cards(test_metrics)
    loss_chart  = _loss_chart_js(epochs, train_losses, val_losses)
    weight_chart = _weight_chart_js(epochs, weight_series) if weight_series else ""
    weight_section = f"""
        <div class="section">
            <h2>Adaptive Weight Evolution</h2>
            <p class="subtitle">Average loss weights predicted by WeightGeneratorMLP per epoch</p>
            <div class="chart-container">
                <canvas id="weightChart"></canvas>
            </div>
        </div>
    """ if weight_series else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment Report — {config_dict.get('exp_id', 'exp')}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:      #f8f9fb;
    --card:    #ffffff;
    --border:  #e2e8f0;
    --accent:  #4f46e5;
    --accent2: #06b6d4;
    --text:    #1e293b;
    --muted:   #64748b;
    --green:   #10b981;
    --orange:  #f59e0b;
    --red:     #ef4444;
    --purple:  #8b5cf6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.6;
  }}
  .page {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px; }}

  /* Header */
  .header {{ background: var(--accent); color: #fff; border-radius: 12px;
             padding: 32px 36px; margin-bottom: 32px; }}
  .header h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 6px; }}
  .header .meta {{ opacity: 0.85; font-size: 13px; }}
  .badge {{ display: inline-block; background: rgba(255,255,255,0.2);
            border-radius: 999px; padding: 2px 12px; font-size: 12px;
            font-weight: 600; margin-right: 8px; margin-top: 8px; }}

  /* Section */
  .section {{ background: var(--card); border: 1px solid var(--border);
              border-radius: 12px; padding: 28px 32px; margin-bottom: 24px; }}
  .section h2 {{ font-size: 16px; font-weight: 700; color: var(--accent);
                 margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 12px; margin-bottom: 20px; }}

  /* Config table */
  .config-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                  gap: 12px; }}
  .config-item {{ background: var(--bg); border-radius: 8px; padding: 12px 14px; }}
  .config-key {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
                 letter-spacing: 0.05em; margin-bottom: 2px; }}
  .config-val {{ font-weight: 600; font-size: 14px; color: var(--text); }}

  /* Metric cards */
  .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
  .metric-card {{ border-radius: 10px; padding: 20px 18px; text-align: center; color: #fff; }}
  .metric-card .label {{ font-size: 12px; font-weight: 600; opacity: 0.85;
                          text-transform: uppercase; letter-spacing: 0.06em; }}
  .metric-card .value {{ font-size: 28px; font-weight: 800; margin-top: 6px; }}
  .metric-card.mse   {{ background: linear-gradient(135deg, #4f46e5, #7c3aed); }}
  .metric-card.mae   {{ background: linear-gradient(135deg, #06b6d4, #0284c7); }}
  .metric-card.rmse  {{ background: linear-gradient(135deg, #10b981, #059669); }}
  .metric-card.mape  {{ background: linear-gradient(135deg, #f59e0b, #d97706); }}

  /* Chart */
  .chart-container {{ position: relative; height: 300px; }}

  /* Footer */
  .footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 32px; }}
</style>
</head>
<body>
<div class="page">

  <!-- Header -->
  <div class="header">
    <h1>Experiment Report</h1>
    <div class="meta">{timestamp}</div>
    <div style="margin-top:10px;">
      <span class="badge">{config_dict.get('exp_id','—')}</span>
      <span class="badge">{config_dict.get('model_name','—')}</span>
      <span class="badge">{config_dict.get('data_name','—')}</span>
      <span class="badge">loss: {config_dict.get('loss_type','—')}</span>
      <span class="badge">pred_len: {config_dict.get('pred_len','—')}</span>
    </div>
  </div>

  <!-- Experiment Configuration -->
  <div class="section">
    <h2>Experiment Configuration</h2>
    <p class="subtitle">Full hyperparameter settings for this run</p>
    <div class="config-grid">
      {config_rows}
    </div>
  </div>

  <!-- Test Metrics -->
  <div class="section">
    <h2>Test Results</h2>
    <p class="subtitle">Evaluated on the held-out test set using the best validation checkpoint</p>
    <div class="metric-grid">
      {test_cards}
    </div>
  </div>

  <!-- Training Loss Curve -->
  <div class="section">
    <h2>Training Curve</h2>
    <p class="subtitle">Train and validation loss per epoch</p>
    <div class="chart-container">
      <canvas id="lossChart"></canvas>
    </div>
  </div>

  <!-- Adaptive Weight Evolution (only for adaptive loss) -->
  {weight_section}

  <div class="footer">
    Generated by Adaptive Loss Weighting Framework &nbsp;·&nbsp; {timestamp}
  </div>
</div>

<script>
{loss_chart}
{weight_chart}
</script>
</body>
</html>"""


def _config_table_rows(config_dict: dict) -> str:
    # Fields to highlight (in order); skip internal/path fields
    primary_keys = [
        "exp_id", "model_name", "data_name", "loss_type",
        "seq_len", "pred_len", "num_features",
        "epochs", "batch_size", "learning_rate", "weight_decay", "patience",
        "moving_avg", "weight_gen_hidden_dim", "weight_gen_dropout",
        "weight_gen_max_log_var", "num_stat_features", "device",
    ]
    skip = {"save_dir", "data_path", "features", "target"}
    rows = []
    for key in primary_keys:
        if key in config_dict and key not in skip:
            rows.append(
                f'<div class="config-item">'
                f'<div class="config-key">{key.replace("_", " ")}</div>'
                f'<div class="config-val">{config_dict[key]}</div>'
                f'</div>'
            )
    return "\n".join(rows)


def _test_metric_cards(test_metrics: dict) -> str:
    order = [("mse", "MSE"), ("mae", "MAE"), ("rmse", "RMSE")]
    cards = []
    for key, label in order:
        val = test_metrics.get(key, 0.0)
        cards.append(
            f'<div class="metric-card {key}">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{val:.4f}</div>'
            f'</div>'
        )
    return "\n".join(cards)


def _loss_chart_js(epochs, train_losses, val_losses) -> str:
    return f"""
const lossCtx = document.getElementById('lossChart').getContext('2d');
new Chart(lossCtx, {{
  type: 'line',
  data: {{
    labels: {json.dumps(epochs)},
    datasets: [
      {{
        label: 'Train Loss',
        data: {json.dumps([round(v, 6) for v in train_losses])},
        borderColor: '#4f46e5', backgroundColor: 'rgba(79,70,229,0.08)',
        borderWidth: 2, pointRadius: 3, tension: 0.3, fill: true,
      }},
      {{
        label: 'Val Loss',
        data: {json.dumps([round(v, 6) for v in val_losses])},
        borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,0.08)',
        borderWidth: 2, pointRadius: 3, tension: 0.3, fill: true,
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Epoch' }} }},
      y: {{ title: {{ display: true, text: 'Loss' }} }}
    }}
  }}
}});"""


def _weight_chart_js(epochs, weight_series: dict) -> str:
    colors = {
        "w_mse":       ("#4f46e5", "rgba(79,70,229,0.08)"),
        "w_mae":       ("#06b6d4", "rgba(6,182,212,0.08)"),
        "w_trend":     ("#10b981", "rgba(16,185,129,0.08)"),
        "w_frequency": ("#f59e0b", "rgba(245,158,11,0.08)"),
    }
    labels = {
        "w_mse": "MSE weight", "w_mae": "MAE weight",
        "w_trend": "Trend weight", "w_frequency": "Frequency weight",
    }
    datasets = []
    for key, values in weight_series.items():
        bc, bg = colors.get(key, ("#999", "rgba(153,153,153,0.08)"))
        datasets.append(f"""{{
        label: '{labels.get(key, key)}',
        data: {json.dumps([round(v, 4) for v in values])},
        borderColor: '{bc}', backgroundColor: '{bg}',
        borderWidth: 2, pointRadius: 3, tension: 0.3, fill: false,
      }}""")

    return f"""
const weightCtx = document.getElementById('weightChart').getContext('2d');
new Chart(weightCtx, {{
  type: 'line',
  data: {{
    labels: {json.dumps(epochs)},
    datasets: [{", ".join(datasets)}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'top' }} }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Epoch' }} }},
      y: {{
        title: {{ display: true, text: 'Weight' }},
        min: 0, max: 1,
        ticks: {{ callback: v => v.toFixed(2) }}
      }}
    }}
  }}
}});"""

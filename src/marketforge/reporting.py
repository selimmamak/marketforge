"""Portable, dependency-free HTML research reports with inline vector charts."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MODEL_NAMES = {
    "zero": "Zero Return",
    "historical_mean": "Historical Mean",
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
    "svr": "Support Vector Regression",
    "mlp": "Multilayer Perceptron",
    "regime_mixture": "Regime Mixture",
    "xgboost": "XGBoost",
    "torch_attention": "Feature Attention Transformer",
    "torch_residual": "Gated Residual Network",
    "temporal_ensemble": "Temporal Ensemble",
}

METRIC_NAMES = {
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
    "spearman_ic": "Spearman IC",
    "n_observations": "Observations",
    "ci_low": "95% CI Lower Bound",
    "ci_high": "95% CI Upper Bound",
    "cost_bps": "Cost (Basis Points)",
    "n_bootstrap": "Bootstrap Samples",
    "sum_cost_as_period_equity_fraction": "Sum of Period Cost Fractions",
}


def _model_name(value: Any) -> str:
    name = str(value)
    return MODEL_NAMES.get(name, name.replace("_", " "))


def _metric_name(value: Any) -> str:
    name = str(value)
    return METRIC_NAMES.get(name, name.replace("_", " ").title())


def _safe(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _format(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:,.6g}" if np.isfinite(value) else "undefined"
    if value is None:
        return "undefined"
    return _safe(value)


def _metrics(values: Any) -> str:
    if not isinstance(values, dict):
        return f"<p>{_safe(values)}</p>"
    rows = []
    for key, value in values.items():
        if isinstance(value, dict):
            continue
        rows.append(f"<tr><th>{_safe(_metric_name(key))}</th><td>{_format(value)}</td></tr>")
    return "<table class='metrics'><tbody>" + "".join(rows) + "</tbody></table>"


def _equity_chart(daily: pd.DataFrame) -> str:
    names = [name for name in ["equity", "benchmark_equity"] if name in daily]
    if daily.empty or not names:
        return "<p>No one-session strategy simulation was produced for this horizon.</p>"
    values = daily[names].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        return "<p>Equity chart unavailable because values were nonfinite.</p>"
    positions = np.unique(np.linspace(0, len(daily) - 1, min(len(daily), 1200)).astype(int))
    values = values.iloc[positions]
    width, height, left, right, top, bottom = 940, 310, 68, 20, 22, 38
    plot_width, plot_height = width - left - right, height - top - bottom
    minimum, maximum = min(1.0, float(values.min().min())), max(1.0, float(values.max().max()))
    padding = max((maximum - minimum) * 0.12, 0.02)
    minimum, maximum = minimum - padding, maximum + padding

    def y_coordinate(value: float) -> float:
        return top + plot_height * (maximum - value) / (maximum - minimum)

    lines = []
    for tick in np.linspace(minimum, maximum, 5):
        y = y_coordinate(tick)
        lines.append(f"<line x1='{left}' x2='{width-right}' y1='{y:.2f}' y2='{y:.2f}' stroke='#273648'/><text x='{left-12}' y='{y+4:.2f}' text-anchor='end' fill='#9cacc0' font-size='12'>{tick:.3f}</text>")
    colors = ["#65ddbe", "#abafff"]
    for index, name in enumerate(names):
        all_values = np.r_[1.0, values[name].to_numpy()]
        points = " ".join(f"{left + position/(len(all_values)-1)*plot_width:.2f},{y_coordinate(value):.2f}" for position, value in enumerate(all_values))
        lines.append(f"<polyline points='{points}' fill='none' stroke='{colors[index]}' stroke-width='2.4'/>")
    if "date" in daily:
        for location, value, anchor in [(left, daily["date"].iloc[0], "start"), (width-right, daily["date"].iloc[-1], "end")]:
            label = str(value)[:10]
            lines.append(f"<text x='{location}' y='{height-8}' text-anchor='{anchor}' fill='#9cacc0' font-size='12'>{_safe(label)}</text>")
    svg = f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Cumulative net strategy and benchmark equity, starting from one unit'>" + "".join(lines) + "</svg>"
    legend = "<div class='legend'><span style='color:#65ddbe'>● Selected strategy</span><span style='color:#abafff'>● Equal-weight long benchmark</span></div>"
    return legend + svg


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No results available.</p>"
    display = frame.copy()
    if "model" in display:
        display["model"] = display["model"].map(_model_name)
    display = display.rename(columns=_metric_name)
    return "<div class='table-scroll'>" + display.to_html(index=False, border=0, classes="leaderboard", escape=True, float_format=lambda value: f"{value:.6g}") + "</div>"


def render_report(summary: dict[str, Any], leaderboard: pd.DataFrame, daily: pd.DataFrame, output_path: Path) -> None:
    """Write a self-contained report; all untrusted text and tables are escaped."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    winner = summary.get("winner", "No selected model")
    if isinstance(winner, dict):
        winner = winner.get("name", winner.get("model", str(winner)))
    winner = _model_name(winner)
    holdout = summary.get("holdout_metrics", {})
    development = summary.get("development_rmse", "See leaderboard")
    source = summary.get("data_source", summary.get("dataset", {}).get("source", "See run manifest"))
    errors = summary.get("model_errors", {})
    display_errors = (
        {_model_name(name): detail for name, detail in errors.items()}
        if isinstance(errors, dict) else errors
    )
    diagnostics = summary.get("holdout_leaderboard", [])
    holdout_table = _table(pd.DataFrame(diagnostics)) if isinstance(diagnostics, list) and diagnostics else ""
    raw = json.dumps(summary, indent=2, default=str, ensure_ascii=False)
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light"><title>MarketForge · Research report</title>
<style>
:root{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#eaf0f8;background:#0b121c;line-height:1.55}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1200px;margin:auto;padding:48px 30px 80px}}header{{border-bottom:1px solid #273648;padding-bottom:30px;margin-bottom:30px}}.eyebrow{{font-size:12px;letter-spacing:.19em;font-weight:700;color:#65ddbe;text-transform:uppercase}}h1{{font-size:clamp(30px,5vw,48px);letter-spacing:-.04em;line-height:1.15;margin:14px 0}}h2{{font-size:22px;letter-spacing:-.02em;margin:0 0 18px}}p{{color:#acbbcf;max-width:88ch}}.cards{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:28px 0}}.card,section{{background:#121d2b;border:1px solid #273648;border-radius:12px;padding:22px}}.label{{font-size:12px;color:#9cacc0;text-transform:uppercase;letter-spacing:.08em}}.value{{font-size:25px;font-weight:650;margin-top:9px;overflow-wrap:anywhere}}section{{margin-top:20px}}.columns{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}.columns section{{min-width:0}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:11px 12px;border-bottom:1px solid #273648;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead{{color:#a7b8cd;background:#182536}}.metrics th{{font-weight:400;color:#acbbcf;text-align:left;white-space:normal}}.metrics td{{font-variant-numeric:tabular-nums;white-space:normal;overflow-wrap:anywhere}}.table-scroll{{overflow-x:auto}}.legend{{display:flex;gap:24px;font-size:13px;margin-bottom:12px;flex-wrap:wrap}}svg{{width:100%;height:auto}}details{{margin-top:20px}}summary{{cursor:pointer;color:#65ddbe}}pre{{overflow:auto;color:#acbbcf;font-size:12px;padding:16px;background:#0b121c;border-radius:8px}}a{{color:#65ddbe}}li{{color:#acbbcf;margin:9px 0}}footer{{font-size:12px;color:#8c9bb1;margin-top:30px}}@media(max-width:760px){{main{{padding:28px 16px}}.cards,.columns{{grid-template-columns:1fr}}section,.card{{padding:16px}}}}@media print{{:root{{color:#172437;background:white}}.card,section{{background:white;border-color:#ccd3db;break-inside:avoid}}p,li,.metrics th{{color:#344458}}th,td{{border-color:#ddd}}thead{{background:#eee;color:#172437}}details{{display:none}}}}
</style></head><body><main>
<header><div class="eyebrow">MarketForge / experiment review</div><h1>Forecasts measured against time.</h1><p>Model selection uses development data only. The final chronological holdout measures the frozen selection, with execution delayed until the following open.</p></header>
<div class="cards"><div class="card"><div class="label">Selected on development</div><div class="value">{_safe(winner)}</div></div><div class="card"><div class="label">Development RMSE</div><div class="value">{_format(development)}</div></div><div class="card"><div class="label">Data source</div><div class="value">{_safe(source)}</div></div></div>
<section><h2>Development model comparison</h2><p>Lower RMSE is better. Scores describe this dataset and evaluation period; they do not establish a universally best forecasting model.</p>{_table(leaderboard)}</section>
<section><h2>Holdout strategy equity</h2><p>Equity starts at 1. Costs include entry, rebalancing, and final liquidation. This is a research simulation with idealized fills.</p>{_equity_chart(daily)}</section>
<div class="columns"><section><h2>Frozen model: holdout forecasts</h2>{_metrics(holdout)}</section><section><h2>Holdout strategy</h2>{_metrics(summary.get('backtest_metrics', {}))}</section></div>
<div class="columns"><section><h2>Uncertainty calibration</h2>{_metrics(summary.get('interval', {}))}<p>Calibration uses an earlier, separate chronological segment. Serial dependence and changing regimes can invalidate exchangeability; inspect realized holdout coverage.</p></section><section><h2>Paired block bootstrap</h2>{_metrics(summary.get('bootstrap', {}))}<p>RMSE difference is selected model minus reference. Negative values favor the selected model. Date clusters preserve the panel structure; contiguous blocks preserve some serial dependence. This interval does not adjust for searching many models.</p></section></div>
<section><h2>Evaluation assumptions</h2><ul><li>Features are available at the signal close. Targets run from the next open to a later open; training labels overlapping an evaluation boundary are removed.</li><li>Hyperparameters and the winning model are selected inside the research segment. Calibration and the final holdout do not tune the selection.</li><li>Simple-return forecast accuracy and trading profitability answer different questions. Volatility-scaled directional positions do not optimize portfolio covariance.</li><li>Adjusted historical prices are retrospective. A fixed present-day ticker list has survivorship and selection bias. Delisted securities and point-in-time corporate actions require a specialist data source.</li><li>Borrow costs, financing, cash yield, taxes, order-book liquidity, and nonlinear market impact are omitted. Shorting availability and idealized open fills are not guaranteed.</li></ul></section>
{('<section><h2>Other holdout results. Diagnostic only.</h2><p>These results do not alter the selected model. Repeatedly inspecting and changing the experiment turns this holdout into development data.</p>' + holdout_table + '</section>') if holdout_table else ''}
{('<section><h2>Model execution notes</h2><pre>' + _safe(json.dumps(display_errors, indent=2, default=str)) + '</pre></section>') if errors else ''}
<details><summary>Run manifest and complete metrics</summary><pre>{_safe(raw)}</pre></details>
<footer>MarketForge · Reproducible forecasting research · Local report; no external scripts, stylesheets, fonts, or tracking.</footer>
</main></body></html>"""
    destination.write_text(content, encoding="utf-8")

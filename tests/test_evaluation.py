"""Numerical invariants for loss diagnostics and execution accounting."""
import numpy as np
import pandas as pd
import pytest

from marketforge.evaluation import (
    backtest,
    block_bootstrap_rmse_difference,
    conformal_radius,
    regression_metrics,
)
from marketforge.reporting import render_report


def prediction_frame(signals=(1.0, 1.0, 1.0), returns=(0.0, 0.0, 0.0)):
    dates = pd.bdate_range("2024-01-02", periods=len(signals) + 2)
    return pd.DataFrame({
        "date": dates[:-2], "entry_date": dates[1:-1], "exit_date": dates[2:],
        "ticker": "A", "target": returns, "prediction": signals,
        "f_volatility_20": 0.10 / np.sqrt(252),
    })


def test_metrics_constant_forecast_and_invalid_values():
    metrics = regression_metrics([-0.1, 0.0, 0.1], [0.0, 0.0, 0.0])
    assert metrics["spearman_ic"] == 0
    assert metrics["direction_accuracy"] == pytest.approx(1 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.02 / 3))
    assert regression_metrics([1, 1], [1, 1])["r2"] is None
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([np.nan], [0])


def test_conformal_finite_sample_order_and_insufficient_sample():
    assert conformal_radius(np.arange(1.0, 10), alpha=0.1) == 9.0
    assert conformal_radius([-1, -2, 3, 4], alpha=0.4) == 3.0
    assert np.isinf(conformal_radius([1, 2], alpha=0.1))
    with pytest.raises(ValueError):
        conformal_radius([1, 2], alpha=1.0)


def test_bootstrap_paired_date_clusters_are_deterministic():
    dates = pd.bdate_range("2024-01-02", periods=20).repeat(2)
    actual = np.sin(np.arange(40)) * 0.01
    first = actual.copy()
    second = np.zeros(40)
    result = block_bootstrap_rmse_difference(actual, first, second, dates=dates, block_size=4, n_bootstrap=40)
    assert result == block_bootstrap_rmse_difference(actual, first, second, dates=dates, block_size=4, n_bootstrap=40)
    assert result["sampling_units"] == 20
    assert result["date_clustered"] is True
    assert result["ci_high"] < 0
    identical = block_bootstrap_rmse_difference(actual, second, second, dates=dates, n_bootstrap=40)
    assert identical["ci_low"] == identical["ci_high"] == 0


def test_bootstrap_long_block_does_not_collapse_small_sample_uncertainty():
    actual = np.linspace(-0.10, 0.20, 10)
    result = block_bootstrap_rmse_difference(actual, actual, np.zeros(10), block_size=20, n_bootstrap=100)
    assert result["requested_block_size"] == 20
    assert result["block_size"] == 5
    assert result["ci_low"] < result["ci_high"]


def test_zero_signals_stay_in_cash_and_benchmark_pays_costs():
    daily, metrics = backtest(prediction_frame((0, 0, 0), (0.02, -0.01, 0.03)), cost_bps=10)
    assert (daily["equity"] == 1).all()
    assert (daily["trading_cost"] == 0).all()
    assert daily["benchmark_cost"].sum() > 0
    assert metrics["total_return"] == 0


def test_one_period_cost_includes_entry_and_liquidation():
    daily, _ = backtest(prediction_frame((1,), (0.10,)), cost_bps=10)
    # Entry trades 1 dollar, terminal liquidation trades 1.10 dollars.
    assert daily["turnover"].iloc[0] == pytest.approx(2.1)
    assert daily["trading_cost"].iloc[0] == pytest.approx(0.0021)
    assert daily["net_return"].iloc[0] == pytest.approx(0.0979)


def test_flip_and_missing_asset_exit_are_charged():
    frame = prediction_frame((1, -1), (0, 0))
    daily, _ = backtest(frame, cost_bps=0)
    assert daily["turnover"].tolist() == pytest.approx([1, 3])
    frame = prediction_frame((1, 1), (0, 0))
    frame.loc[1, "ticker"] = "B"
    daily, _ = backtest(frame, cost_bps=0)
    assert daily["turnover"].tolist() == pytest.approx([1, 3])


def test_holdings_drift_with_returns_before_next_rebalance():
    frame = prediction_frame((1, 0), (0.10, 0))
    daily, _ = backtest(frame, cost_bps=10)
    assert daily["turnover"].iloc[1] == pytest.approx(1.1 / 1.099)


def test_backtest_numeric_text_is_normalized_and_nonfinite_is_rejected():
    frame = prediction_frame((1,), (0.10,))
    frame["target"] = frame["target"].astype(str)
    daily, _ = backtest(frame, cost_bps=0)
    assert daily["net_return"].iloc[0] == pytest.approx(0.10)
    frame["prediction"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        backtest(frame)


def test_bankruptcy_is_rejected_before_compounding_invalid_equity():
    with pytest.raises(ValueError, match="non-positive"):
        backtest(prediction_frame((-1,), (2.0,)), cost_bps=0)


def test_lookahead_and_overlapping_periods_are_rejected():
    frame = prediction_frame()
    frame.loc[0, "entry_date"] = frame.loc[0, "date"]
    with pytest.raises(ValueError, match="prevent lookahead"):
        backtest(frame)
    frame = prediction_frame()
    frame.loc[0, "exit_date"] = frame.loc[1, "exit_date"]
    with pytest.raises(ValueError, match="non-overlapping"):
        backtest(frame)


def test_gross_exposure_is_capped_and_initial_loss_enters_drawdown():
    frame = prediction_frame((1,), (-0.10,))
    second = frame.assign(ticker="B")
    daily, metrics = backtest(pd.concat([frame, second]), cost_bps=0, max_leverage=1)
    assert daily["gross_exposure"].iloc[0] == 1
    assert metrics["max_drawdown"] == pytest.approx(-0.10)


def test_report_is_self_contained_and_escapes_inputs(tmp_path):
    destination = tmp_path / "report.html"
    daily, metrics = backtest(prediction_frame(), cost_bps=0)
    render_report({"winner": "<script>bad()</script>", "backtest_metrics": metrics}, pd.DataFrame({"model": ["<b>ridge</b>"], "rmse": [0.01]}), daily, destination)
    contents = destination.read_text()
    assert "<script>bad()" not in contents
    assert "&lt;script&gt;" in contents
    assert "<svg" in contents
    assert '<script src=' not in contents
    assert "&lt;b&gt;ridge&lt;/b&gt;" in contents

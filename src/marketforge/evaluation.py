"""Forecast diagnostics, dependence-aware resampling, and open-to-open simulation."""
from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _vectors(*arrays: Sequence[float]) -> list[np.ndarray]:
    values = [np.asarray(array, dtype=float).reshape(-1) for array in arrays]
    if not values or not len(values[0]):
        raise ValueError("At least one observation is required.")
    if any(len(value) != len(values[0]) for value in values):
        raise ValueError("Inputs must contain the same number of observations.")
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("Inputs must contain finite numbers only.")
    return values


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, Any]:
    """Metrics on simple returns; flat predictions count as zero directions.

    Rank correlation is defined as zero when either vector is constant. R² is
    undefined (None) for a constant target, rather than silently claiming skill.
    """
    actual, predicted = _vectors(y_true, y_pred)
    error = predicted - actual
    variance = float(np.mean((actual - actual.mean()) ** 2))
    nonconstant = len(actual) > 1 and np.ptp(actual) > 0 and np.ptp(predicted) > 0
    return {
        "n_observations": int(len(actual)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_error": float(np.mean(error)),
        "direction_accuracy": float(np.mean(np.sign(actual) == np.sign(predicted))),
        "spearman_ic": float(spearmanr(actual, predicted).statistic) if nonconstant else 0.0,
        "r2": float(1 - np.mean(error ** 2) / variance) if variance > 0 else None,
    }


def block_bootstrap_rmse_difference(
    y_true: Sequence[float],
    pred_a: Sequence[float],
    pred_b: Sequence[float],
    block_size: int = 20,
    n_bootstrap: int = 500,
    seed: int = 42,
    dates: Optional[Sequence[Any]] = None,
) -> dict[str, Any]:
    """Paired circular moving-block bootstrap for RMSE(a) minus RMSE(b).

    With dates, panel losses are first averaged within each date. Resampling
    whole consecutive dates preserves cross-sectional clusters and some serial
    dependence. Dates have equal weight, even when their ticker counts differ.
    Intervals are descriptive, not multiple-model-selection-adjusted tests.
    """
    actual, first, second = _vectors(y_true, pred_a, pred_b)
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer.")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 20:
        raise ValueError("n_bootstrap must be an integer of at least 20.")
    losses = pd.DataFrame({"a": (actual - first) ** 2, "b": (actual - second) ** 2})
    if dates is not None:
        if len(dates) != len(actual):
            raise ValueError("dates must match the prediction count.")
        losses["date"] = pd.to_datetime(np.asarray(dates), errors="raise")
        if losses["date"].isna().any():
            raise ValueError("dates must not contain missing values.")
        losses = losses.groupby("date", sort=True)[["a", "b"]].mean()
    matrix = losses[["a", "b"]].to_numpy()
    size = len(matrix)
    if size < 2:
        raise ValueError("Bootstrap uncertainty requires at least two sampling units.")
    # A full-length circular block visits every unit exactly once and falsely
    # produces zero bootstrap variability. Require at least two effective blocks.
    width = min(block_size, max(1, size // 2))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap)
    offsets = np.arange(width)
    for iteration in range(n_bootstrap):
        starts = rng.integers(0, size, size=math.ceil(size / width))
        indices = ((starts[:, None] + offsets) % size).reshape(-1)[:size]
        rmses = np.sqrt(matrix[indices].mean(axis=0))
        deltas[iteration] = rmses[0] - rmses[1]
    low, high = np.quantile(deltas, [0.025, 0.975])
    point = np.sqrt(matrix.mean(axis=0))
    return {
        "mean_delta": float(point[0] - point[1]),
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": 0.95,
        "sampling_units": size,
        "block_size": width,
        "requested_block_size": block_size,
        "n_bootstrap": n_bootstrap,
        "date_clustered": dates is not None,
    }


def conformal_radius(residuals: Sequence[float], alpha: float = 0.1) -> float:
    """Return the split-conformal absolute-error order statistic.

    Return infinity when the calibration sample is too short for the requested
    finite-sample rank. Financial time-series dependence means nominal coverage
    should be treated as a target and checked empirically, not as a guarantee.
    """
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    (values,) = _vectors(residuals)
    rank = math.ceil((len(values) + 1) * (1 - alpha))
    if rank > len(values):
        return float("inf")
    return float(np.partition(np.abs(values), rank - 1)[rank - 1])


def _portfolio_step(
    desired: pd.Series, returns: pd.Series, previous: pd.Series, cost_rate: float, liquidate: bool
) -> tuple[float, float, float, float, pd.Series]:
    assets = desired.index.union(previous.index)
    turnover = float((desired.reindex(assets, fill_value=0) - previous.reindex(assets, fill_value=0)).abs().sum())
    gross = float((desired * returns).sum())
    cost = turnover * cost_rate
    terminal_dollars = desired * (1 + returns)
    if liquidate:
        exit_turnover = float(terminal_dollars.abs().sum())
        turnover += exit_turnover
        cost += exit_turnover * cost_rate
    net = gross - cost
    if 1 + net <= 0:
        raise ValueError("Portfolio equity would become non-positive; reduce leverage or inspect returns.")
    drifted = terminal_dollars / (1 + net) if not liquidate else pd.Series(dtype=float)
    return gross, turnover, cost, net, drifted


def _performance(returns: pd.Series, equity: pd.Series) -> dict[str, Any]:
    n = len(returns)
    deviation = float(returns.std(ddof=1)) if n > 1 else 0.0
    annualized = float(np.expm1(np.log1p(returns).mean() * 252))
    full_equity = np.r_[1.0, equity.to_numpy(dtype=float)]
    drawdown = full_equity / np.maximum.accumulate(full_equity) - 1
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annualized_return": annualized,
        "annualized_volatility": deviation * math.sqrt(252),
        "sharpe": float(returns.mean() / deviation * math.sqrt(252)) if deviation > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "positive_period_fraction": float((returns > 0).mean()),
        "periods": n,
    }


def backtest(
    predictions: pd.DataFrame,
    cost_bps: float = 5.0,
    target_vol: float = 0.10,
    max_leverage: float = 1.0,
    signal_threshold: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate one-session open-to-open returns from prior-close forecasts.

    Inputs use target=open[t+2]/open[t+1]-1 and features available at close[t].
    Per-asset volatility scales signed forecasts; portfolio gross is capped.
    Costs are bps per dollar traded, including initial trades, drift-aware
    rebalancing, missing-asset exits, and final liquidation. The benchmark is
    an equal-weight long portfolio rebalanced each period with identical costs.

    Annualization assumes consecutive daily sessions. Borrow fees, cash interest,
    financing, execution limits, and market impact are outside this simulator.
    """
    required = {"date", "ticker", "target", "prediction", "f_volatility_20", "entry_date", "exit_date"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing backtest columns: {', '.join(sorted(missing))}")
    if predictions.empty:
        raise ValueError("Backtest predictions must not be empty.")
    for name, value, allow_zero in [
        ("cost_bps", cost_bps, True), ("target_vol", target_vol, False),
        ("max_leverage", max_leverage, False), ("signal_threshold", signal_threshold, True),
    ]:
        if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")
    frame = predictions.copy()
    for column in ["date", "entry_date", "exit_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="raise")
        if frame[column].isna().any():
            raise ValueError("Backtest dates must not be missing.")
    if frame["ticker"].isna().any() or frame.duplicated(["date", "ticker"]).any():
        raise ValueError("Each signal date and ticker must be unique and nonmissing.")
    for column in ["target", "prediction", "f_volatility_20"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"{column} must be finite.")
    if (frame["target"] < -1).any() or (frame["f_volatility_20"] < 0).any():
        raise ValueError("Returns must be >= -1 and volatility must be nonnegative.")
    if not ((frame["date"] < frame["entry_date"]) & (frame["entry_date"] < frame["exit_date"])).all():
        raise ValueError("Require signal date < entry date < exit date to prevent lookahead.")
    grouped = list(frame.sort_values(["date", "ticker"]).groupby("date", sort=True))
    previous = pd.Series(dtype=float)
    benchmark_previous = pd.Series(dtype=float)
    previous_exit = None
    rows = []
    for number, (signal_date, group) in enumerate(grouped):
        if group["entry_date"].nunique() != 1 or group["exit_date"].nunique() != 1:
            raise ValueError("All assets in a signal cohort must share entry and exit dates.")
        entry, exit_date = group["entry_date"].iloc[0], group["exit_date"].iloc[0]
        if previous_exit is not None and entry != previous_exit:
            raise ValueError("Backtest requires contiguous, non-overlapping one-session holding intervals.")
        previous_exit = exit_date
        group = group.set_index("ticker")
        direction = np.sign(group["prediction"]).where(group["prediction"].abs() > signal_threshold, 0.0)
        weights = direction * target_vol / math.sqrt(252) / group["f_volatility_20"].clip(lower=1e-6)
        gross_exposure = float(weights.abs().sum())
        weights /= max(1.0, gross_exposure / max_leverage)
        benchmark = pd.Series(1.0 / len(group), index=group.index)
        liquidate = number == len(grouped) - 1
        gross, turnover, cost, net, previous = _portfolio_step(weights, group["target"], previous, cost_bps / 10_000, liquidate)
        _, benchmark_turnover, benchmark_cost, benchmark_net, benchmark_previous = _portfolio_step(
            benchmark, group["target"], benchmark_previous, cost_bps / 10_000, liquidate
        )
        rows.append({
            "date": exit_date, "signal_date": signal_date, "entry_date": entry,
            "gross_return": gross, "turnover": turnover, "trading_cost": cost,
            "net_return": net, "gross_exposure": float(weights.abs().sum()),
            "benchmark_return": benchmark_net, "benchmark_turnover": benchmark_turnover,
            "benchmark_cost": benchmark_cost,
        })
    daily = pd.DataFrame(rows)
    daily["equity"] = (1 + daily["net_return"]).cumprod()
    daily["benchmark_equity"] = (1 + daily["benchmark_return"]).cumprod()
    metrics = _performance(daily["net_return"], daily["equity"])
    metrics.update({
        "mean_turnover": float(daily["turnover"].mean()),
        "sum_cost_as_period_equity_fraction": float(daily["trading_cost"].sum()),
        "mean_gross_exposure": float(daily["gross_exposure"].mean()),
        "cost_bps": float(cost_bps),
        "benchmark": _performance(daily["benchmark_return"], daily["benchmark_equity"]),
        "benchmark_name": "Equal-weight long, rebalanced each session",
    })
    return daily, metrics

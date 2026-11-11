"""Causal technical features and next-open executable forecasting labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketforge.data import validate_prices


def _asset_features(bars: pd.DataFrame, horizon: int) -> pd.DataFrame:
    bars = bars.sort_values("date").copy()
    close, high, low, volume = (bars[name] for name in ("close", "high", "low", "volume"))
    returns = close.pct_change(fill_method=None)
    log_returns = np.log(close).diff()
    for lag in range(6):
        bars[f"f_return_lag_{lag}"] = returns.shift(lag)
    for window in (5, 10, 20, 60):
        bars[f"f_momentum_{window}"] = close / close.shift(window) - 1
        bars[f"f_ma_distance_{window}"] = close / close.rolling(window).mean() - 1
        bars[f"f_volatility_{window}"] = returns.rolling(window).std(ddof=1)
    bars["f_volatility_ratio"] = bars["f_volatility_5"] / bars["f_volatility_60"].clip(lower=1e-12)
    bars["f_downside_volatility_20"] = np.sqrt(log_returns.clip(upper=0).pow(2).rolling(20).mean())
    bars["f_parkinson_volatility_20"] = np.sqrt(
        np.log(high / low).pow(2).rolling(20).mean() / (4 * np.log(2))
    )
    bars["f_drawdown_60"] = close / close.rolling(60).max() - 1
    bars["f_intraday_return"] = close / bars["open"] - 1
    bars["f_overnight_gap"] = bars["open"] / close.shift(1) - 1
    bars["f_range_fraction"] = (high - low) / close
    bars["f_close_location"] = ((close - low) / (high - low).replace(0, np.nan)).fillna(0.5)
    changes = close.diff()
    gains = changes.clip(lower=0).rolling(14).mean()
    losses = (-changes.clip(upper=0)).rolling(14).mean()
    bars["f_rsi_14"] = (gains / (gains + losses).replace(0, np.nan)).where(
        (gains + losses) != 0, 0.5
    )
    log_volume = np.log1p(volume)
    bars["f_log_volume_change"] = log_volume.diff()
    bars["f_volume_zscore_20"] = (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(
        20
    ).std(ddof=1).clip(lower=1e-12)
    bars["f_volume_ratio_5_20"] = volume.rolling(5).mean() / volume.rolling(20).mean().clip(lower=1)
    bars["f_return_skew_20"] = returns.rolling(20).skew()
    day = bars["date"].dt.dayofweek
    bars["f_weekday_sin"] = np.sin(2 * np.pi * day / 5)
    bars["f_weekday_cos"] = np.cos(2 * np.pi * day / 5)
    # Information at close t produces an order at open t+1, exited at open t+1+h.
    bars["target"] = bars["open"].shift(-(horizon + 1)) / bars["open"].shift(-1) - 1
    bars["entry_date"] = bars["date"].shift(-1)
    bars["exit_date"] = bars["date"].shift(-(horizon + 1))
    bars["label_end"] = bars["exit_date"]
    return bars


def build_dataset(
    prices: pd.DataFrame,
    horizon: int = 1,
    include_unlabeled: bool = False,
) -> pd.DataFrame:
    """Build a pooled asset panel without estimating anything from future bars.

    The forecast origin is the close of ``date``. ``target`` is the simple return
    from the following open to the open ``horizon`` observed sessions later.
    All contemporaneous panel features use the supplied universe; historical
    membership must therefore be supplied by the researcher for real studies.
    The final horizon+1 rows per asset are unlabelled and omitted unless
    ``include_unlabeled=True`` is requested for inference. Such rows contain
    missing target and future date fields and must never be used for fitting.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    prices = validate_prices(prices)
    frame = pd.concat(
        [_asset_features(group, horizon) for _, group in prices.groupby("ticker", sort=True)],
        ignore_index=True,
    )
    by_date = frame.groupby("date", sort=False)
    frame["f_cross_section_momentum_rank"] = by_date["f_momentum_20"].rank(pct=True)
    frame["f_cross_section_volatility_rank"] = by_date["f_volatility_20"].rank(pct=True)
    frame["f_market_return"] = by_date["f_return_lag_0"].transform("mean")
    frame["f_relative_return"] = frame["f_return_lag_0"] - frame["f_market_return"]
    features = [name for name in frame.columns if name.startswith("f_")]
    metadata = ["date", "ticker", "target", "label_end", "entry_date", "exit_date"]
    frame = frame.loc[:, metadata + features]
    finite_columns = features if include_unlabeled else features + ["target"]
    required_dates = (
        ["date"] if include_unlabeled else ["date", "label_end", "entry_date", "exit_date"]
    )
    complete = np.isfinite(frame[finite_columns].to_numpy(dtype=float)).all(axis=1)
    complete &= frame[required_dates].notna().all(axis=1)
    frame = frame.loc[complete].sort_values(["date", "ticker"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No complete samples: provide more than 60+horizon+1 daily bars per asset")
    return frame

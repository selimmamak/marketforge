"""Daily price ingestion, validation and reproducible synthetic market data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _session_date(value) -> pd.Timestamp:
    """Keep the local calendar date of a midnight session label."""
    if isinstance(value, (int, float, np.number)):
        raise ValueError("Dates must be daily session labels, not numeric timestamps")
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Dates must be valid and non-missing") from exc
    if pd.isna(stamp):
        raise ValueError("Dates must be valid and non-missing")
    if stamp != stamp.normalize():
        raise ValueError("Dates must be midnight daily session labels, not intraday timestamps")
    # An exchange session is a date label. Converting local midnight to UTC
    # before removing the timezone can silently move it to a different date.
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical daily panel; reject ambiguous or invalid market bars.

    Prices must use one consistent adjustment convention for every OHLC field.
    This validator cannot infer whether corporate actions were handled correctly.
    """
    missing = sorted(set(PRICE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing price columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Price data is empty")
    result = frame.loc[:, PRICE_COLUMNS].copy()
    if result["ticker"].isna().any():
        raise ValueError("Ticker identifiers cannot be missing")
    result["ticker"] = result["ticker"].astype(str).str.strip()
    if result["ticker"].eq("").any():
        raise ValueError("Ticker identifiers cannot be empty")
    result["date"] = pd.to_datetime(result["date"].map(_session_date))
    if result.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker bars are not allowed")
    for name in PRICE_COLUMNS[2:]:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    if not np.isfinite(result[PRICE_COLUMNS[2:]].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must be finite and non-missing")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be strictly positive")
    if (result["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    if (
        (result["high"] < result[["open", "close", "low"]].max(axis=1))
        | (result["low"] > result[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("Each high/low must contain the open and close")
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)
    calendar = pd.DatetimeIndex(result["date"].unique()).sort_values()
    for ticker, bars in result.groupby("ticker", sort=False):
        expected = calendar[(calendar >= bars["date"].min()) & (calendar <= bars["date"].max())]
        missing_sessions = expected.difference(pd.DatetimeIndex(bars["date"]))
        if len(missing_sessions):
            first_missing = missing_sessions[0].date()
            raise ValueError(
                f"Ticker {ticker} is missing an internal panel session on {first_missing}; "
                "all assets must use a consistent trading calendar"
            )
    return result


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a long-form CSV with date,ticker,open,high,low,close,volume."""
    return validate_prices(pd.read_csv(path))


def generate_synthetic(
    n_days: int = 1200,
    tickers: Iterable[str] = ("SYN_A", "SYN_B", "SYN_C"),
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate correlated OHLCV bars with persistent volatility regimes.

    These invented assets exercise the pipeline; model performance on them is
    not evidence about predictability or profitability of actual equities.
    """
    tickers = list(tickers)
    if n_days < 3:
        raise ValueError("n_days must be at least 3")
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Provide at least one unique ticker")
    if any(not isinstance(ticker, str) or not ticker.strip() for ticker in tickers):
        raise ValueError("Tickers must be nonempty strings")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_days)
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        switch_probability = 0.035 if regime[t - 1] == 0 else 0.12
        regime[t] = 1 - regime[t - 1] if rng.random() < switch_probability else regime[t - 1]
    sigma = np.where(regime == 0, 0.006, 0.019)
    market = rng.normal(size=n_days) * sigma + np.where(regime == 0, 0.0003, -0.0006)
    market_gap = rng.normal(size=n_days) * sigma * 0.30
    frames = []
    for i, ticker in enumerate(tickers):
        beta = 0.7 + 0.25 * i
        idiosyncratic = rng.normal(size=n_days) * sigma * 0.65
        returns = np.zeros(n_days)
        for t in range(1, n_days):
            coefficient = 0.12 if regime[t] == 0 else -0.10
            returns[t] = beta * market[t] + coefficient * returns[t - 1] + idiosyncratic[t]
        close = (80.0 + 35.0 * i) * np.exp(np.cumsum(returns))
        previous_close = np.r_[close[0], close[:-1]]
        open_price = previous_close * np.exp(
            beta * market_gap + rng.normal(size=n_days) * sigma * 0.15
        )
        excursion = np.abs(rng.normal(size=n_days)) * sigma * 0.65 + 0.0005
        high = np.maximum(open_price, close) * np.exp(excursion)
        low = np.minimum(open_price, close) * np.exp(-excursion)
        volume = np.round(
            (1_000_000 + 200_000 * i)
            * np.exp(rng.normal(0, 0.25, n_days) + 0.4 * regime + 12 * np.abs(returns))
        ).astype(np.int64)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return validate_prices(pd.concat(frames, ignore_index=True))


def download_yahoo(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted daily bars (inclusive start, exclusive end).

    Yahoo is an optional convenience source, not a point-in-time institutional
    dataset. The caller must respect its terms and inspect adjustment quality.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Yahoo downloads require the data extra: pip install '.[data]'") from exc
    names = list(tickers)
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide at least one unique ticker")
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start_date) or pd.isna(end_date):
        raise ValueError("start and end must be valid dates")
    if start_date >= end_date:
        raise ValueError("start must precede end")
    frames = []
    for ticker in names:
        bars = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
            actions=False,
        )
        if bars.empty:
            raise ValueError(f"No daily prices returned for {ticker}")
        if isinstance(bars.columns, pd.MultiIndex):
            bars.columns = bars.columns.get_level_values(0)
        bars = bars.reset_index()
        bars.columns = [str(column).lower() for column in bars.columns]
        if "date" not in bars.columns and "datetime" in bars.columns:
            bars = bars.rename(columns={"datetime": "date"})
        bars["ticker"] = ticker
        frames.append(bars)
    return validate_prices(pd.concat(frames, ignore_index=True))

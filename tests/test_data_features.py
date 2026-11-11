import numpy as np
import pandas as pd
import pytest

from marketforge.data import generate_synthetic, load_csv, validate_prices
from marketforge.features import build_dataset


def test_synthetic_reproducible_valid_panel():
    first = generate_synthetic(n_days=120, seed=11)
    second = generate_synthetic(n_days=120, seed=11)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 360
    assert first.groupby("date").size().eq(3).all()
    pd.testing.assert_frame_equal(first, validate_prices(first))
    assert not first.equals(generate_synthetic(n_days=120, seed=12))


@pytest.mark.parametrize(
    "fault", ["duplicate", "negative_price", "bad_high", "missing", "date", "negative_volume"]
)
def test_invalid_prices_rejected(fault):
    frame = generate_synthetic(n_days=10)
    if fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif fault == "negative_price":
        frame.loc[0, "open"] = -1
    elif fault == "bad_high":
        frame.loc[0, "high"] = frame.loc[0, "low"] / 2
    elif fault == "missing":
        frame.loc[0, "close"] = np.nan
    elif fault == "date":
        frame.loc[0, "date"] = pd.NaT
    elif fault == "negative_volume":
        frame.loc[0, "volume"] = -1
    with pytest.raises(ValueError):
        validate_prices(frame)


def test_csv_roundtrip(tmp_path):
    prices = generate_synthetic(n_days=10)
    filename = tmp_path / "prices.csv"
    prices.to_csv(filename, index=False)
    pd.testing.assert_frame_equal(load_csv(filename), prices)


@pytest.mark.parametrize("date", ["2024-01-02 16:00:00", 1704153600, "not-a-date"])
def test_daily_input_rejects_ambiguous_or_intraday_timestamps(date):
    prices = generate_synthetic(n_days=5).iloc[[0]].copy()
    prices["date"] = [date]
    with pytest.raises(ValueError, match="Dates must"):
        validate_prices(prices)


def test_timezone_aware_session_dates_keep_the_exchange_calendar_day():
    prices = generate_synthetic(n_days=5)
    expected = prices.copy()
    prices["date"] = prices.date.dt.tz_localize("Asia/Tokyo")
    pd.testing.assert_frame_equal(validate_prices(prices), expected)


def test_internal_calendar_holes_are_rejected_before_labels_are_built():
    prices = generate_synthetic(n_days=120)
    missing_date = prices.date.drop_duplicates().iloc[80]
    prices = prices.loc[~((prices.ticker == "SYN_B") & (prices.date == missing_date))]
    with pytest.raises(ValueError, match="missing an internal panel session"):
        build_dataset(prices)


def test_differing_history_start_and_end_dates_are_allowed():
    prices = generate_synthetic(n_days=120)
    dates = prices.date.drop_duplicates()
    short = (prices.ticker == "SYN_B") & (
        (prices.date < dates.iloc[10]) | (prices.date > dates.iloc[100])
    )
    trimmed = prices.loc[~short].reset_index(drop=True)
    pd.testing.assert_frame_equal(validate_prices(trimmed), trimmed)


def test_optional_download_never_substitutes_synthetic_data(monkeypatch):
    import sys
    from types import SimpleNamespace

    from marketforge.data import download_yahoo

    monkeypatch.setitem(
        sys.modules, "yfinance", SimpleNamespace(download=lambda *args, **kwargs: pd.DataFrame())
    )
    with pytest.raises(ValueError, match="No daily prices returned"):
        download_yahoo(["AAPL"], "2024-01-01", "2024-02-01")
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ImportError, match="data extra"):
        download_yahoo(["AAPL"], "2024-01-01", "2024-02-01")


def test_download_rejects_missing_dates_before_network_access(monkeypatch):
    import sys
    from types import SimpleNamespace

    from marketforge.data import download_yahoo

    def forbidden_download(*args, **kwargs):
        raise AssertionError("The invalid request must fail before contacting a provider")

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=forbidden_download))
    with pytest.raises(ValueError, match="valid dates"):
        download_yahoo(["AAPL"], "NaT", "2024-02-01")


def test_features_are_invariant_to_future_prices():
    prices = generate_synthetic(n_days=250)
    baseline = build_dataset(prices, horizon=5)
    cutoff = prices["date"].drop_duplicates().iloc[180]
    changed = prices.copy()
    later = changed["date"] > cutoff
    changed.loc[later, ["open", "high", "low", "close"]] *= 7
    changed.loc[later, "volume"] *= 13
    altered = build_dataset(changed, horizon=5)
    columns = ["date", "ticker"] + [name for name in baseline if name.startswith("f_")]
    pd.testing.assert_frame_equal(
        baseline.loc[baseline.date <= cutoff, columns].reset_index(drop=True),
        altered.loc[altered.date <= cutoff, columns].reset_index(drop=True),
    )
    prefix = build_dataset(prices.loc[prices.date <= cutoff], horizon=5)
    before = baseline.merge(prefix[["date", "ticker"]], on=["date", "ticker"])
    pd.testing.assert_frame_equal(before[columns], prefix[columns])


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_targets_use_executable_next_open_returns(horizon):
    prices = generate_synthetic(n_days=200, tickers=("A",))
    dataset = build_dataset(prices, horizon=horizon)
    sample = dataset.iloc[25]
    position = prices.index[prices.date.eq(sample.date)][0]
    expected_entry = prices.iloc[position + 1]
    expected_exit = prices.iloc[position + horizon + 1]
    assert sample.entry_date == expected_entry.date
    assert sample.exit_date == expected_exit.date
    assert sample.label_end == expected_exit.date
    assert sample.target == pytest.approx(expected_exit.open / expected_entry.open - 1)
    feature_names = [name for name in dataset if name.startswith("f_")]
    assert np.isfinite(dataset[feature_names].to_numpy()).all()
    assert "f_volatility_20" in feature_names


def test_heterogeneous_history_does_not_change_prefix_rank():
    prices = generate_synthetic(n_days=150)
    cutoff = prices.date.drop_duplicates().iloc[120]
    prices = prices.loc[~((prices.ticker == "SYN_C") & (prices.date > cutoff))]
    full = build_dataset(prices)
    prefix = build_dataset(prices.loc[prices.date <= cutoff])
    keys = ["date", "ticker"]
    feature_names = [name for name in prefix if name.startswith("f_")]
    match = full.merge(prefix[keys], on=keys)
    pd.testing.assert_frame_equal(match[keys + feature_names], prefix[keys + feature_names])


def test_inference_retains_latest_rows_without_labels():
    prices = generate_synthetic(n_days=120)
    frame = build_dataset(prices, horizon=5, include_unlabeled=True)
    latest = frame.groupby("ticker", sort=False).tail(1)
    assert latest.date.eq(prices.date.max()).all()
    assert latest.target.isna().all()
    assert latest.label_end.isna().all()
    feature_names = [name for name in frame if name.startswith("f_")]
    assert np.isfinite(latest[feature_names].to_numpy()).all()
    labeled = build_dataset(prices, horizon=5)
    pd.testing.assert_frame_equal(
        frame.loc[frame.target.notna()].reset_index(drop=True),
        labeled,
    )


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_horizon_rejected(horizon):
    with pytest.raises(ValueError, match="horizon"):
        build_dataset(generate_synthetic(n_days=100), horizon=horizon)

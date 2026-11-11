import numpy as np
import pandas as pd
import pytest

from marketforge.data import generate_synthetic
from marketforge.features import build_dataset
from marketforge.splitting import holdout_split, walk_forward_splits


def panel(n_dates=100, horizon=5):
    dates = pd.bdate_range("2020-01-01", periods=n_dates + horizon + 1)
    return pd.DataFrame(
        {
            "date": np.repeat(dates[:n_dates], 3),
            "label_end": np.repeat(dates[horizon + 1 : n_dates + horizon + 1], 3),
            "ticker": ["A", "B", "C"] * n_dates,
        }
    )


def assert_purged_grouped(frame, train, test):
    training, testing = frame.iloc[train], frame.iloc[test]
    assert set(train).isdisjoint(test)
    assert training.date.max() < testing.date.min()
    assert training.label_end.max() < testing.date.min()
    for date in testing.date.unique():
        assert testing.loc[testing.date == date].shape[0] == frame.loc[frame.date == date].shape[0]


def test_expanding_grouped_purged_walk_forward():
    frame = panel()
    folds = walk_forward_splits(frame, n_splits=3, test_size=10, min_train_size=20)
    seen = set()
    previous_train = set()
    for train, test in folds:
        assert_purged_grouped(frame, train, test)
        assert previous_train.issubset(set(train))
        assert seen.isdisjoint(test)
        assert len(test) == 30
        seen.update(test)
        previous_train = set(train)
    assert frame.iloc[folds[-1][1]].date.max() == frame.date.max()


def test_holdout_purges_entire_multi_day_labels():
    frame = panel(horizon=10)
    development, holdout = holdout_split(frame, holdout_size=20, embargo=0)
    assert_purged_grouped(frame, development, holdout)
    assert frame.iloc[holdout].date.nunique() == 20
    # Origin 68 labels end at origin 79, while origin 69 would end on test date 80.
    assert frame.iloc[development].date.nunique() == 69


def test_embargo_excludes_requested_feature_date_gap():
    frame = panel(horizon=1)
    development, holdout = holdout_split(frame, holdout_size=20, embargo=10)
    assert_purged_grouped(frame, development, holdout)
    assert frame.iloc[development].date.nunique() == 70


def test_splits_return_positions_for_unsorted_nonstandard_index():
    frame = panel().sample(frac=1, random_state=2)
    frame.index = np.arange(len(frame)) + 9000
    for train, test in walk_forward_splits(frame, n_splits=2, test_size=15, min_train_size=20):
        assert_purged_grouped(frame, train, test)
        assert train.max() < len(frame)


def test_real_dataset_boundaries_and_nested_split_are_purged():
    dataset = build_dataset(generate_synthetic(n_days=700), horizon=5)
    development, holdout = holdout_split(dataset, holdout_size=63)
    assert_purged_grouped(dataset, development, holdout)
    dev = dataset.iloc[development].reset_index(drop=True)
    for train, test in walk_forward_splits(dev, n_splits=3, test_size=42, min_train_size=252):
        assert_purged_grouped(dev, train, test)
        inner = dev.iloc[train].reset_index(drop=True)
        for inner_train, inner_test in walk_forward_splits(
            inner, n_splits=2, test_size=21, min_train_size=100
        ):
            assert_purged_grouped(inner, inner_train, inner_test)


def test_small_sample_and_invalid_dates_raise_clear_errors():
    with pytest.raises(ValueError, match="Not enough dates"):
        walk_forward_splits(panel(n_dates=10), n_splits=2, test_size=10)
    with pytest.raises(ValueError, match="training dates"):
        walk_forward_splits(panel(), n_splits=2, test_size=10, min_train_size=90)
    with pytest.raises(ValueError, match="no development"):
        holdout_split(panel(), holdout_size=100)
    with pytest.raises(ValueError, match="embargo"):
        holdout_split(panel(), embargo=-1)
    invalid = panel()
    invalid.loc[0, "label_end"] = invalid.loc[0, "date"]
    with pytest.raises(ValueError, match="strictly follow"):
        holdout_split(invalid, holdout_size=20)

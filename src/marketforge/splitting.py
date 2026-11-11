"""Date-grouped chronological splits with explicit label-overlap purging."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate(frame: pd.DataFrame, embargo: int) -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    if not isinstance(embargo, int) or isinstance(embargo, bool) or embargo < 0:
        raise ValueError("embargo must be a nonnegative number of observed dates")
    if not {"date", "label_end"}.issubset(frame.columns):
        raise ValueError("Splitting requires date and label_end columns")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    ends = pd.to_datetime(frame["label_end"], errors="coerce")
    if dates.isna().any() or ends.isna().any() or frame.empty:
        raise ValueError("Splitting requires nonempty, non-missing dates and label ends")
    if (ends <= dates).any():
        raise ValueError("Each label_end must strictly follow its forecast date")
    unique = pd.DatetimeIndex(dates.unique()).sort_values()
    return dates, ends, unique


def _train_before(
    dates: pd.Series,
    ends: pd.Series,
    unique: pd.DatetimeIndex,
    test_start: int,
    embargo: int,
) -> np.ndarray:
    # Embargo removes feature dates immediately before the test block; label
    # purging independently removes any labels unresolved at the test origin.
    cutoff_index = test_start - embargo
    if cutoff_index <= 0:
        return np.array([], dtype=int)
    return np.flatnonzero(((dates < unique[cutoff_index]) & (ends < unique[test_start])).to_numpy())


def walk_forward_splits(
    frame: pd.DataFrame,
    n_splits: int = 3,
    test_size: int = 63,
    min_train_size: int = 252,
    embargo: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding training windows and disjoint final test blocks.

    Sizes count unique forecast dates, not rows. Every available ticker on a
    test date is assigned to the same fold. Training labels must end strictly
    before the first test forecast date. The ``embargo`` is an additional
    pre-test gap on feature dates; it is not a substitute for label purging.
    """
    for name, value in (
        ("n_splits", n_splits),
        ("test_size", test_size),
        ("min_train_size", min_train_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    dates, ends, unique = _validate(frame, embargo)
    first_start = len(unique) - n_splits * test_size
    if first_start <= 0:
        raise ValueError("Not enough dates for requested walk-forward test blocks")
    splits = []
    for start in range(first_start, len(unique), test_size):
        train = _train_before(dates, ends, unique, start, embargo)
        stop = start + test_size
        test = np.flatnonzero(dates.isin(unique[start:stop]).to_numpy())
        train_dates = dates.iloc[train].nunique()
        if train_dates < min_train_size:
            raise ValueError(
                f"Only {train_dates} training dates remain after purging; "
                f"need min_train_size={min_train_size}. Add data or reduce split sizes."
            )
        splits.append((train, test))
    return splits


def holdout_split(
    frame: pd.DataFrame,
    holdout_size: int = 126,
    embargo: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve the final dates and purge development labels at their boundary."""
    if not isinstance(holdout_size, int) or isinstance(holdout_size, bool) or holdout_size < 1:
        raise ValueError("holdout_size must be a positive integer")
    dates, ends, unique = _validate(frame, embargo)
    start = len(unique) - holdout_size
    if start <= 0:
        raise ValueError("holdout_size leaves no development dates")
    train = _train_before(dates, ends, unique, start, embargo)
    if len(train) == 0:
        raise ValueError("No development rows remain after holdout label purging")
    test = np.flatnonzero(dates.isin(unique[start:]).to_numpy())
    return train, test

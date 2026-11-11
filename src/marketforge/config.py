"""Strict configuration: a typo must never silently change an experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ExperimentConfig:
    seed: int = 42
    horizon: int = 1
    holdout_size: int = 126
    calibration_size: int = 63
    n_splits: int = 3
    test_size: int = 63
    min_train_size: int = 252
    inner_splits: int = 2
    inner_test_size: int = 42
    inner_min_train_size: int = 126
    embargo: int = 1
    cost_bps: float = 5.0
    target_vol: float = 0.10
    max_leverage: float = 1.0
    signal_threshold: float = 0.0
    interval_alpha: float = 0.10
    bootstrap_samples: int = 500
    bootstrap_block_size: int = 20
    include_neural: bool = False
    include_boosting: bool = False
    quick: bool = False
    models: Optional[list[str]] = None

    def __post_init__(self):
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32-1]")
        for key in ("include_neural", "include_boosting", "quick"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be a boolean")
        positive = (
            "horizon", "holdout_size", "calibration_size", "n_splits", "test_size",
            "min_train_size", "inner_splits", "inner_test_size", "inner_min_train_size",
            "bootstrap_samples", "bootstrap_block_size",
        )
        for key in positive:
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.embargo, int) or isinstance(self.embargo, bool) or self.embargo < 0:
            raise ValueError("embargo must be a nonnegative integer")
        if self.bootstrap_samples < 20:
            raise ValueError("bootstrap_samples must be at least 20")
        if self.holdout_size < 2:
            raise ValueError("holdout_size must be at least two dates for block bootstrap")
        for key in ("interval_alpha", "cost_bps", "signal_threshold", "target_vol", "max_leverage"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        if not 0 < self.interval_alpha < 1:
            raise ValueError("interval_alpha must lie between zero and one")
        if self.cost_bps < 0 or self.signal_threshold < 0:
            raise ValueError("cost_bps and signal_threshold must be nonnegative")
        if self.target_vol <= 0 or not 0 < self.max_leverage <= 1:
            raise ValueError("target_vol must be positive; max_leverage must be in (0, 1]")
        if self.models is not None and (
            not isinstance(self.models, list) or not self.models
            or not all(isinstance(n, str) and n for n in self.models)
        ):
            raise ValueError("models must be a nonempty list of registry names")

    def to_dict(self):
        return asdict(self)


def load_config(path: Optional[Path] = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping")
    unknown = set(raw) - {field.name for field in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    return ExperimentConfig(**raw)

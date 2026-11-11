"""Forecasting estimators with training-only preprocessing and explicit candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class RegimeMixtureRegressor(RegressorMixin, BaseEstimator):
    """Soft Gaussian gating of regularized local linear experts.

    Standardization, PCA, mixture estimation and experts are fitted exclusively
    on the supplied training rows. Regularized Gaussian components identify
    statistical feature regimes, not verified economic regimes. Experts use
    posterior sample weights and shrink toward a global ridge forecast according
    to their effective sample size. Unseen rows are never used to refit the gate.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        alpha: float = 10.0,
        gate_dimensions: int = 6,
        shrinkage: float = 64.0,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.alpha = alpha
        self.gate_dimensions = gate_dimensions
        self.shrinkage = shrinkage
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=float, y_numeric=True)
        if self.n_regimes < 1 or self.gate_dimensions < 1:
            raise ValueError("n_regimes and gate_dimensions must be positive")
        if self.alpha < 0 or self.shrinkage < 0:
            raise ValueError("alpha and shrinkage must be nonnegative")
        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(X)
        scaled = self.scaler_.transform(X)
        self.global_ = Ridge(alpha=self.alpha).fit(scaled, y)
        n_unique = len(np.unique(scaled, axis=0))
        n_components = min(self.n_regimes, n_unique, max(1, len(X) // 20))
        self.pca_ = None
        self.gate_ = None
        if n_components > 1:
            dimension = min(self.gate_dimensions, X.shape[1], len(X) - 1)
            self.pca_ = PCA(n_components=dimension, random_state=self.random_state)
            projected = self.pca_.fit_transform(scaled)
            self.gate_ = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                reg_covar=1e-4,
                n_init=3,
                max_iter=200,
                random_state=self.random_state,
            ).fit(projected)
            responsibilities = self.gate_.predict_proba(projected)
        else:
            responsibilities = np.ones((len(X), 1))
        self.experts_ = []
        self.expert_strength_ = []
        for weights in responsibilities.T:
            effective_n = weights.sum() ** 2 / max(np.square(weights).sum(), 1e-12)
            strength = effective_n / (effective_n + self.shrinkage)
            self.experts_.append(Ridge(alpha=self.alpha).fit(scaled, y, sample_weight=weights))
            self.expert_strength_.append(strength)
        self.expert_strength_ = np.asarray(self.expert_strength_)
        self.n_regimes_ = len(self.experts_)
        return self

    def predict(self, X):
        check_is_fitted(self, "experts_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count differs from training")
        scaled = self.scaler_.transform(X)
        probabilities = (
            self.gate_.predict_proba(self.pca_.transform(scaled))
            if self.gate_ is not None
            else np.ones((len(X), 1))
        )
        global_prediction = self.global_.predict(scaled)
        experts = np.column_stack([expert.predict(scaled) for expert in self.experts_])
        shrunk = (
            experts * self.expert_strength_
            + global_prediction[:, None] * (1 - self.expert_strength_)
        )
        return np.sum(probabilities * shrunk, axis=1)


def _scaled(estimator):
    # Target scaling gives neural, kernel and penalized models sensible units.
    # Both transformers are cloned and fitted inside every training fold.
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), estimator),
        transformer=StandardScaler(),
    )


def make_model(name: str, params: dict[str, Any] | None = None, seed: int = 42):
    """Construct a fresh estimator. Parameters address the underlying model."""
    parameters = dict(params or {})
    if name == "zero":
        return DummyRegressor(strategy="constant", constant=0.0, **parameters)
    if name == "historical_mean":
        return DummyRegressor(strategy="mean", **parameters)
    if name == "ridge":
        return _scaled(Ridge(**parameters))
    if name == "elastic_net":
        return _scaled(ElasticNet(random_state=seed, max_iter=5000, **parameters))
    if name in {"random_forest", "extra_trees"}:
        cls = RandomForestRegressor if name == "random_forest" else ExtraTreesRegressor
        defaults = {"n_estimators": 120, "min_samples_leaf": 10, "max_features": 0.8}
        defaults.update(parameters)
        return cls(random_state=seed, n_jobs=1, **defaults)
    if name == "hist_gradient_boosting":
        defaults = {"max_iter": 150, "max_leaf_nodes": 15, "l2_regularization": 1.0}
        defaults.update(parameters)
        # Random holdbacks would mix dates; use fixed training iterations instead.
        return HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **defaults)
    if name == "svr":
        defaults = {"C": 1.0, "epsilon": 0.1, "cache_size": 256}
        defaults.update(parameters)
        return _scaled(SVR(**defaults))
    if name == "mlp":
        defaults = {
            "hidden_layer_sizes": (32, 16),
            "max_iter": 200,
            "alpha": 0.1,
            "learning_rate_init": 0.001,
            "early_stopping": False,
            "batch_size": "auto",
        }
        defaults.update(parameters)
        return _scaled(MLPRegressor(random_state=seed, **defaults))
    if name == "regime_mixture":
        return RegimeMixtureRegressor(random_state=seed, **parameters)
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            # XGBoostError can be raised while importing a missing native OpenMP
            # runtime, before its exception class can itself be imported.
            raise ImportError(
                "XGBoost could not be loaded. Install 'marketforge[boosting]' and "
                "its native OpenMP runtime (libomp on macOS). "
                f"Original error: {exc}"
            ) from exc
        defaults = {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 10.0,
            "tree_method": "hist",
            "objective": "reg:squarederror",
        }
        defaults.update(parameters)
        return XGBRegressor(random_state=seed, n_jobs=1, **defaults)
    if name in {"torch_attention", "torch_residual"}:
        try:
            from .neural import TorchTabularRegressor
        except ImportError as exc:
            raise ImportError("PyTorch models require: pip install 'marketforge[neural]'") from exc
        return TorchTabularRegressor(
            architecture=name.removeprefix("torch_"), random_state=seed, **parameters
        )
    raise ValueError(f"Unknown model: {name!r}")


def model_candidates(
    seed: int = 42,
    include_neural: bool = False,
    include_boosting: bool = False,
    quick: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Finite search space; model selection is performed by the experiment runner.

    ``seed`` belongs to this public factory contract; initialization uses the
    seed passed separately to :func:`make_model`, not randomized search grids.
    """
    candidates = {
        "zero": [{}],
        "historical_mean": [{}],
        "ridge": [{"alpha": 10.0}, {"alpha": 100.0}, {"alpha": 1000.0}],
        "elastic_net": [{"alpha": 0.01, "l1_ratio": 0.1}, {"alpha": 0.1, "l1_ratio": 0.5}],
        "random_forest": [{"max_depth": 6}, {"max_depth": 12}],
        "extra_trees": [{"max_depth": 6}, {"max_depth": 12}],
        "hist_gradient_boosting": [{"max_leaf_nodes": 7}, {"max_leaf_nodes": 15}],
        "svr": [{"C": 0.1}, {"C": 1.0}],
        "mlp": [{"hidden_layer_sizes": (32, 16)}, {"hidden_layer_sizes": (64, 32), "alpha": 1.0}],
        "regime_mixture": [{"n_regimes": 2, "alpha": 10.0}, {"n_regimes": 3, "alpha": 100.0}],
    }
    if quick:
        candidates = {name: [dict(grid[0])] for name, grid in candidates.items()}
        for name in ("random_forest", "extra_trees"):
            candidates[name][0]["n_estimators"] = 40
        candidates["hist_gradient_boosting"][0]["max_iter"] = 60
        candidates["mlp"][0].update({"max_iter": 80, "hidden_layer_sizes": (24,)})
    if include_boosting:
        candidates["xgboost"] = (
            [{"n_estimators": 60, "max_depth": 3}] if quick
            else [{"max_depth": 3}, {"max_depth": 5, "reg_lambda": 30.0}]
        )
    if include_neural:
        candidates["torch_attention"] = (
            [{"epochs": 12, "width": 16}] if quick
            else [{"epochs": 12, "width": 16}, {"epochs": 20, "width": 16}]
        )
        candidates["torch_residual"] = (
            [{"epochs": 12, "width": 32}] if quick
            else [{"epochs": 20, "width": 32}, {"epochs": 30, "width": 64}]
        )
    return candidates

"""Nested, purged walk-forward experiments with a frozen final evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import ExperimentConfig
from .evaluation import (
    backtest,
    block_bootstrap_rmse_difference,
    conformal_radius,
    regression_metrics,
)
from .features import build_dataset
from .models import make_model, model_candidates
from .reporting import render_report
from .splitting import holdout_split, walk_forward_splits

LOGGER = logging.getLogger(__name__)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, value):
    path.write_text(
        json.dumps(value, default=_json_default, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _predict(model, x):
    with threadpool_limits(limits=1):
        prediction = np.asarray(model.predict(x), dtype=float).reshape(-1)
    if len(prediction) != len(x) or not np.isfinite(prediction).all():
        raise ValueError("Estimator returned invalid predictions")
    return prediction


def _fit(name, params, x, y, seed):
    model = make_model(name, params, seed)
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model.fit(x, y)
    messages = sorted({str(w.message) for w in captured})
    return model, messages


def _tune(name, candidates, train, feature_columns, config):
    # A single candidate needs no selection; outer testing still evaluates it.
    if len(candidates) == 1:
        return candidates[0], []
    splits = walk_forward_splits(
        train, n_splits=config.inner_splits, test_size=config.inner_test_size,
        min_train_size=config.inner_min_train_size, embargo=config.embargo,
    )
    x = train[feature_columns].to_numpy(dtype=float)
    y = train.target.to_numpy(dtype=float)
    trials = []
    for params in candidates:
        squared_error, count = 0.0, 0
        try:
            for fit_idx, valid_idx in splits:
                model, _ = _fit(name, params, x[fit_idx], y[fit_idx], config.seed)
                pred = _predict(model, x[valid_idx])
                squared_error += float(np.square(y[valid_idx] - pred).sum())
                count += len(valid_idx)
            trials.append({"params": params, "rmse": float(np.sqrt(squared_error / count))})
        except (ValueError, RuntimeError, ImportError, OSError, FloatingPointError) as exc:
            trials.append({"params": params, "error": str(exc)})
    successful = [trial for trial in trials if "rmse" in trial]
    if not successful:
        raise RuntimeError(f"Every tuning candidate failed for {name}: {trials}")
    return min(successful, key=lambda trial: trial["rmse"])["params"], trials


def _inverse_error_weights(predictions, y):
    mse = np.mean(np.square(predictions - y[:, None]), axis=0)
    raw = 1.0 / np.maximum(mse, 1e-12)
    return raw / raw.sum()


def temporal_ensemble(predictions, y, fold_ids, dates=None, label_ends=None):
    """Score each fold with weights learned only from completed earlier folds."""
    result = np.empty(len(y))
    weight_history = []
    for fold in np.unique(fold_ids):
        current = fold_ids == fold
        previous = fold_ids < fold
        if dates is not None and label_ends is not None:
            previous &= np.asarray(label_ends) < np.min(np.asarray(dates)[current])
        weights = (
            _inverse_error_weights(predictions[previous], y[previous])
            if previous.any() else np.ones(predictions.shape[1]) / predictions.shape[1]
        )
        result[current] = predictions[current] @ weights
        weight_history.append({"fold": int(fold), "weights": weights.tolist()})
    return result, _inverse_error_weights(predictions, y), weight_history


def run_experiment(prices, config: ExperimentConfig, output: Path, data_source="csv"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new run directory.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frame = build_dataset(prices, horizon=config.horizon)
    features = sorted(column for column in frame if column.startswith("f_"))
    dev_idx, holdout_idx = holdout_split(frame, config.holdout_size, config.embargo)
    development = frame.iloc[dev_idx].reset_index(drop=True)
    research_idx, calibration_idx = holdout_split(
        development, config.calibration_size, config.embargo
    )
    research = development.iloc[research_idx].reset_index(drop=True)
    calibration = development.iloc[calibration_idx].reset_index(drop=True)
    holdout = frame.iloc[holdout_idx].reset_index(drop=True)
    if not np.isfinite(conformal_radius(np.zeros(len(calibration)), config.interval_alpha)):
        raise ValueError("Not enough calibration rows for interval_alpha; increase calibration_size")
    splits = walk_forward_splits(
        research, config.n_splits, config.test_size, config.min_train_size, config.embargo
    )
    registry = model_candidates(
        config.seed, config.include_neural, config.include_boosting, config.quick
    )
    if config.models is not None:
        unknown = set(config.models) - registry.keys()
        if unknown:
            raise ValueError(f"Unknown or disabled model names: {sorted(unknown)}")
        registry = {name: registry[name] for name in dict.fromkeys(["zero", *config.models])}
    x = research[features].to_numpy(dtype=float)
    y = research.target.to_numpy(dtype=float)
    oof_index = np.concatenate([test for _, test in splits])
    fold_ids = np.concatenate([np.full(len(test), i) for i, (_, test) in enumerate(splits)])
    oof_y = y[oof_index]
    oof_predictions, fold_records, model_errors, captured_warnings = {}, [], {}, {}
    fold_boundaries = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        fold_boundaries.append({
            "fold": fold, "train_start": research.date.iloc[train_idx].min(),
            "train_end": research.date.iloc[train_idx].max(),
            "latest_train_label": research.label_end.iloc[train_idx].max(),
            "test_start": research.date.iloc[test_idx].min(),
            "test_end": research.date.iloc[test_idx].max(),
            "train_rows": len(train_idx), "test_rows": len(test_idx),
        })
    for name, candidates in registry.items():
        LOGGER.info("Evaluating %s (%d candidates)", name, len(candidates))
        local_predictions, local_records, messages = [], [], set()
        try:
            for fold, (train_idx, test_idx) in enumerate(splits):
                LOGGER.info("%s: outer fold %d/%d", name, fold + 1, len(splits))
                params, trials = _tune(
                    name, candidates, research.iloc[train_idx].reset_index(drop=True),
                    features, config,
                )
                model, fit_warnings = _fit(name, params, x[train_idx], y[train_idx], config.seed)
                prediction = _predict(model, x[test_idx])
                local_predictions.append(prediction)
                messages.update(fit_warnings)
                local_records.append({
                    "model": name, "fold": fold, "params": params,
                    "inner_trials": trials, **regression_metrics(y[test_idx], prediction),
                })
            oof_predictions[name] = np.concatenate(local_predictions)
            fold_records.extend(local_records)
            captured_warnings[name] = sorted(messages)
        except (ImportError, ValueError, RuntimeError, OSError, FloatingPointError) as exc:
            model_errors[name] = str(exc)
            LOGGER.warning("Excluded %s: %s", name, exc)
    if not oof_predictions:
        raise RuntimeError(f"No model completed the temporal benchmark: {model_errors}")

    names = list(oof_predictions)
    matrix = np.column_stack([oof_predictions[name] for name in names])
    ensemble, weights, weight_history = temporal_ensemble(
        matrix, oof_y, fold_ids, dates=research.date.to_numpy()[oof_index],
        label_ends=research.label_end.to_numpy()[oof_index],
    )
    if len(names) > 1:
        oof_predictions["temporal_ensemble"] = ensemble
    leaderboard = pd.DataFrame([
        {"model": name, **regression_metrics(oof_y, pred)}
        for name, pred in oof_predictions.items()
    ]).sort_values(["rmse", "model"]).reset_index(drop=True)
    winner = str(leaderboard.iloc[0]["model"])
    LOGGER.info("Frozen development winner: %s", winner)

    # Re-tune on research data only. Calibration and holdout never affect choices.
    models, selected_parameters, cal_predictions, test_predictions = {}, {}, {}, {}
    final_trials = {}
    for name in names:
        LOGGER.info("Final research-only fit: %s", name)
        params, trials = _tune(name, registry[name], research, features, config)
        model, fit_warnings = _fit(name, params, x, y, config.seed)
        models[name] = model
        selected_parameters[name] = params
        final_trials[name] = trials
        captured_warnings[name] = sorted(set(captured_warnings[name] + fit_warnings))
        cal_predictions[name] = _predict(model, calibration[features].to_numpy(dtype=float))
        test_predictions[name] = _predict(model, holdout[features].to_numpy(dtype=float))
    if len(names) > 1:
        cal_predictions["temporal_ensemble"] = np.column_stack(
            [cal_predictions[n] for n in names]
        ) @ weights
        test_predictions["temporal_ensemble"] = np.column_stack(
            [test_predictions[n] for n in names]
        ) @ weights
    radius = conformal_radius(
        calibration.target.to_numpy() - cal_predictions[winner], config.interval_alpha
    )
    selected_prediction = test_predictions[winner]
    holdout_metrics = regression_metrics(holdout.target.to_numpy(), selected_prediction)
    holdout_board = pd.DataFrame([
        {"model": name, **regression_metrics(holdout.target.to_numpy(), pred)}
        for name, pred in test_predictions.items()
    ])
    predictions = holdout[[
        "date", "ticker", "target", "label_end", "entry_date", "exit_date", "f_volatility_20"
    ]].copy()
    predictions["prediction"] = selected_prediction
    predictions["lower"] = selected_prediction - radius
    predictions["upper"] = selected_prediction + radius
    predictions["model"] = winner
    covered = (predictions.target >= predictions.lower) & (predictions.target <= predictions.upper)
    interval = {
        "nominal_coverage": 1 - config.interval_alpha,
        "holdout_coverage": float(covered.mean()), "radius": radius,
        "mean_width": 2 * radius, "calibration_rows": len(calibration),
        "interpretation": "Empirical coverage; serial dependence invalidates IID guarantees.",
    }
    daily, trading = pd.DataFrame(), {}
    if config.horizon == 1:
        daily, trading = backtest(
            predictions, config.cost_bps, config.target_vol, config.max_leverage,
            config.signal_threshold,
        )
    else:
        trading = {"omitted": "Trading simulator supports one-session, non-overlapping labels only."}
    bootstrap = block_bootstrap_rmse_difference(
        holdout.target.to_numpy(), selected_prediction, np.zeros(len(holdout)),
        block_size=config.bootstrap_block_size, n_bootstrap=config.bootstrap_samples,
        seed=config.seed, dates=holdout.date.to_numpy(),
    )
    periods = {
        key: {"start": part.date.min(), "end": part.date.max(), "rows": len(part)}
        for key, part in (("research", research), ("calibration", calibration), ("holdout", holdout))
    }
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "yfinance"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    fingerprint = hashlib.sha256(
        prices.sort_values(["date", "ticker"]).to_csv(index=False).encode()
    ).hexdigest()
    summary = {
        "winner": winner, "selection_metric": "pooled development out-of-fold RMSE",
        "development_rmse": float(leaderboard.iloc[0].rmse),
        "holdout_metrics": holdout_metrics, "backtest_metrics": trading,
        "holdout_leaderboard": holdout_board.to_dict(orient="records"),
        "bootstrap": bootstrap, "interval": interval,
        "config": config.to_dict(), "data_source": str(data_source),
        "dataset": {"price_rows": len(prices), "labeled_rows": len(frame),
                    "tickers": sorted(prices.ticker.unique().tolist()),
                    "features": len(features), "sha256": fingerprint},
        "timeline": periods, "fold_boundaries": fold_boundaries,
        "ensemble_weights": dict(zip(names, weights.tolist())),
        "ensemble_fold_weights": weight_history,
        "selected_parameters": selected_parameters,
        "model_errors": model_errors, "fit_warnings": captured_warnings,
        "versions": versions, "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    holdout_board.to_csv(output / "holdout_metrics.csv", index=False)
    predictions.to_csv(output / "holdout_predictions.csv", index=False)
    all_predictions = holdout[["date", "ticker", "target"]].copy()
    for name, prediction in test_predictions.items():
        all_predictions[name] = prediction
    all_predictions.to_csv(output / "all_holdout_predictions.csv", index=False)
    oof = research.iloc[oof_index][["date", "ticker", "target"]].reset_index(drop=True)
    oof["fold"] = fold_ids
    for name, prediction in oof_predictions.items():
        oof[name] = prediction
    oof.to_csv(output / "development_predictions.csv", index=False)
    if not daily.empty:
        daily.to_csv(output / "daily_returns.csv", index=False)
    write_json(output / "summary.json", summary)
    write_json(output / "tuning.json", {"outer_folds": fold_records, "final_trials": final_trials})
    artifact = {
        "format_version": 1, "winner": winner, "models": models,
        "features": features, "ensemble_names": names, "ensemble_weights": weights,
        "interval_radius": radius, "horizon": config.horizon,
        "training_end": research.date.max(), "versions": versions,
        "tickers": sorted(prices.ticker.unique().tolist()),
    }
    joblib.dump(artifact, output / "model.joblib")
    render_report(summary, leaderboard, daily, output / "report.html")
    return summary


def forecast_latest(prices, artifact_path: Path):
    """Load a trusted local artifact and forecast the latest shared market close."""
    artifact = joblib.load(artifact_path)
    if artifact.get("format_version") != 1:
        raise ValueError("Unsupported model artifact version")
    actual = sorted(prices.ticker.unique().tolist())
    if actual != artifact["tickers"]:
        raise ValueError("Forecast ticker universe must match training; ranks depend on the universe")
    frame = build_dataset(prices, artifact["horizon"], include_unlabeled=True)
    latest_date = frame.date.max()
    if latest_date != pd.to_datetime(prices.date).max():
        raise ValueError("Latest price date has incomplete features; refusing a stale forecast")
    if latest_date <= pd.Timestamp(artifact["training_end"]):
        raise ValueError("Latest forecast date must be later than the model training period")
    latest = frame.loc[frame.date == latest_date].copy()
    if sorted(latest.ticker.unique().tolist()) != actual:
        raise ValueError("The latest feature date is missing one or more trained tickers")
    x = latest[artifact["features"]].to_numpy(dtype=float)
    winner = artifact["winner"]
    if winner == "temporal_ensemble":
        matrix = np.column_stack([
            _predict(artifact["models"][name], x) for name in artifact["ensemble_names"]
        ])
        prediction = matrix @ artifact["ensemble_weights"]
    else:
        prediction = _predict(artifact["models"][winner], x)
    result = latest[["date", "ticker"]].copy()
    result["predicted_return"] = prediction
    result["lower"] = prediction - artifact["interval_radius"]
    result["upper"] = prediction + artifact["interval_radius"]
    result["horizon_sessions"] = artifact["horizon"]
    result["model"] = winner
    result["training_end"] = artifact["training_end"]
    return result.reset_index(drop=True)

"""End-to-end tests of the information boundaries, artifact and ensemble."""

import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

import marketforge.experiment as experiment
from marketforge.config import ExperimentConfig
from marketforge.data import generate_synthetic
from marketforge.features import build_dataset
from marketforge.splitting import holdout_split


@pytest.fixture
def completed_run(tmp_path):
    prices = generate_synthetic(n_days=260, tickers=("SYN_A", "SYN_B"), seed=17)
    config = ExperimentConfig(
        models=["zero", "ridge"],
        holdout_size=30,
        calibration_size=20,
        n_splits=2,
        test_size=20,
        min_train_size=40,
        inner_splits=2,
        inner_test_size=10,
        inner_min_train_size=20,
        bootstrap_samples=20,
        bootstrap_block_size=5,
    )
    output = tmp_path / "original"
    summary = experiment.run_experiment(prices, config, output, data_source="synthetic")
    return prices, config, output, summary


def test_small_nested_experiment_writes_complete_artifacts(completed_run):
    prices, config, output, summary = completed_run
    expected = {
        "leaderboard.csv",
        "holdout_metrics.csv",
        "holdout_predictions.csv",
        "all_holdout_predictions.csv",
        "development_predictions.csv",
        "daily_returns.csv",
        "summary.json",
        "tuning.json",
        "model.joblib",
        "report.html",
    }
    assert expected.issubset({path.name for path in output.iterdir()})
    assert summary["model_errors"] == {}
    leaderboard = pd.read_csv(output / "leaderboard.csv")
    assert set(leaderboard.model) == {"zero", "ridge", "temporal_ensemble"}
    assert summary["winner"] == leaderboard.iloc[0].model
    holdout = pd.read_csv(output / "holdout_predictions.csv")
    assert len(holdout) == config.holdout_size * prices.ticker.nunique()
    assert np.isfinite(holdout[["prediction", "lower", "upper"]].to_numpy()).all()
    assert (holdout.lower <= holdout.prediction).all()
    assert (holdout.prediction <= holdout.upper).all()
    for boundary in summary["fold_boundaries"]:
        assert boundary["latest_train_label"] < boundary["test_start"]
    serialized = json.loads((output / "summary.json").read_text())
    assert serialized["winner"] == summary["winner"]
    tuning = json.loads((output / "tuning.json").read_text())
    ridge_trials = [fold for fold in tuning["outer_folds"] if fold["model"] == "ridge"]
    assert len(ridge_trials) == config.n_splits
    assert all(len(fold["inner_trials"]) == 3 for fold in ridge_trials)


def test_holdout_targets_cannot_change_selection(completed_run, tmp_path, monkeypatch):
    prices, config, original, summary = completed_run
    frame = build_dataset(prices, config.horizon)
    _, holdout = holdout_split(frame, config.holdout_size, config.embargo)
    altered = frame.copy()
    altered.loc[holdout, "target"] += 0.007
    monkeypatch.setattr(experiment, "build_dataset", lambda *args, **kwargs: altered.copy())
    destination = tmp_path / "altered_holdout"
    rerun = experiment.run_experiment(prices, config, destination, data_source="synthetic")
    assert rerun["winner"] == summary["winner"]
    assert rerun["selected_parameters"] == summary["selected_parameters"]
    assert rerun["ensemble_weights"] == summary["ensemble_weights"]
    assert rerun["interval"]["radius"] == summary["interval"]["radius"]
    pd.testing.assert_frame_equal(
        pd.read_csv(original / "leaderboard.csv"),
        pd.read_csv(destination / "leaderboard.csv"),
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(original / "all_holdout_predictions.csv").drop(columns="target"),
        pd.read_csv(destination / "all_holdout_predictions.csv").drop(columns="target"),
    )
    assert rerun["holdout_metrics"]["rmse"] != summary["holdout_metrics"]["rmse"]


def test_calibration_changes_radius_without_refitting_model(completed_run, tmp_path, monkeypatch):
    prices, config, original, summary = completed_run
    frame = build_dataset(prices, config.horizon)
    development_idx, _ = holdout_split(frame, config.holdout_size, config.embargo)
    development = frame.iloc[development_idx].reset_index(drop=True)
    research_idx, calibration_idx = holdout_split(
        development,
        config.calibration_size,
        config.embargo,
    )
    research = development.iloc[research_idx].reset_index(drop=True)
    altered = frame.copy()
    altered.loc[development_idx[calibration_idx], "target"] += 0.2
    monkeypatch.setattr(experiment, "build_dataset", lambda *args, **kwargs: altered.copy())
    fits = []
    original_fit = experiment._fit

    def recording_fit(name, params, x, y, seed):
        fits.append((name, x.copy(), y.copy()))
        return original_fit(name, params, x, y, seed)

    monkeypatch.setattr(experiment, "_fit", recording_fit)
    destination = tmp_path / "altered_calibration"
    rerun = experiment.run_experiment(prices, config, destination, data_source="synthetic")
    assert rerun["winner"] == summary["winner"]
    assert rerun["selected_parameters"] == summary["selected_parameters"]
    assert rerun["ensemble_weights"] == summary["ensemble_weights"]
    assert rerun["interval"]["radius"] > summary["interval"]["radius"]
    pd.testing.assert_frame_equal(
        pd.read_csv(original / "all_holdout_predictions.csv"),
        pd.read_csv(destination / "all_holdout_predictions.csv"),
    )
    feature_names = sorted(name for name in research if name.startswith("f_"))
    for model in ("zero", "ridge"):
        _, final_x, final_y = [fit for fit in fits if fit[0] == model][-1]
        np.testing.assert_array_equal(final_x, research[feature_names].to_numpy())
        np.testing.assert_array_equal(final_y, research.target.to_numpy())
    assert all(len(y) <= len(research) for _, _, y in fits)


def test_current_fold_targets_do_not_affect_ensemble_weights():
    matrix = np.array([[0.01, 0.02], [0.04, -0.01], [0.02, 0.03], [-0.01, 0.01]])
    targets = np.array([0.01, 0.02, 0.04, -0.02])
    folds = np.array([0, 0, 1, 1])
    first, _, history = experiment.temporal_ensemble(matrix, targets, folds)
    changed = targets.copy()
    changed[folds == 1] = [0.7, -0.4]
    second, _, changed_history = experiment.temporal_ensemble(matrix, changed, folds)
    np.testing.assert_array_equal(first, second)
    assert history == changed_history
    np.testing.assert_array_equal(history[0]["weights"], [0.5, 0.5])


def test_ensemble_excludes_previous_fold_labels_not_yet_resolved():
    matrix = np.array([[0.01, 0.02], [0.04, -0.01], [0.02, 0.03], [-0.01, 0.01]])
    targets = np.array([0.01, 0.02, 0.04, -0.02])
    folds = np.array([0, 0, 1, 1])
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    ends = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"])
    first, _, history = experiment.temporal_ensemble(
        matrix,
        targets,
        folds,
        dates=dates,
        label_ends=ends,
    )
    changed = targets.copy()
    changed[1] = 0.75  # Not yet known before the second fold starts.
    second, _, changed_history = experiment.temporal_ensemble(
        matrix,
        changed,
        folds,
        dates=dates,
        label_ends=ends,
    )
    np.testing.assert_array_equal(first, second)
    assert history == changed_history
    assert history[1]["weights"][0] > 0.99


def test_forecast_roundtrip_matches_saved_estimator(completed_run):
    prices, _, output, summary = completed_run
    forecast = experiment.forecast_latest(prices, output / "model.joblib")
    assert len(forecast) == prices.ticker.nunique()
    assert forecast.date.eq(prices.date.max()).all()
    assert forecast.model.eq(summary["winner"]).all()
    assert np.isfinite(forecast[["predicted_return", "lower", "upper"]].to_numpy()).all()
    artifact = joblib.load(output / "model.joblib")
    frame = build_dataset(prices, artifact["horizon"], include_unlabeled=True)
    latest = frame.loc[frame.date == frame.date.max()]
    x = latest[artifact["features"]].to_numpy()
    if artifact["winner"] == "temporal_ensemble":
        expected = (
            np.column_stack(
                [artifact["models"][name].predict(x) for name in artifact["ensemble_names"]]
            )
            @ artifact["ensemble_weights"]
        )
    else:
        expected = artifact["models"][artifact["winner"]].predict(x)
    np.testing.assert_allclose(forecast.predicted_return, expected)
    np.testing.assert_allclose(
        forecast.upper - forecast.lower,
        2 * artifact["interval_radius"],
    )
    with pytest.raises(ValueError, match="universe"):
        experiment.forecast_latest(prices.loc[prices.ticker == "SYN_A"], output / "model.joblib")


def test_output_directory_cannot_overwrite_previous_run(completed_run):
    prices, config, output, _ = completed_run
    with pytest.raises(ValueError, match="not empty"):
        experiment.run_experiment(prices, config, output)


def test_forecast_rejects_stale_feature_date(completed_run, monkeypatch):
    prices, _, output, _ = completed_run
    frame = build_dataset(prices, include_unlabeled=True)
    stale = frame.loc[frame.date < prices.date.max()].copy()
    monkeypatch.setattr(experiment, "build_dataset", lambda *args, **kwargs: stale)
    with pytest.raises(ValueError, match="stale forecast"):
        experiment.forecast_latest(prices, output / "model.joblib")


def _assert_identical_research_outputs(first, second):
    for name in (
        "leaderboard.csv",
        "development_predictions.csv",
        "holdout_predictions.csv",
        "all_holdout_predictions.csv",
        "holdout_metrics.csv",
        "daily_returns.csv",
    ):
        pd.testing.assert_frame_equal(pd.read_csv(first / name), pd.read_csv(second / name))
    assert json.loads((first / "tuning.json").read_text()) == json.loads(
        (second / "tuning.json").read_text()
    )


def test_interrupted_outer_fold_is_resumed_without_refitting(completed_run, tmp_path, monkeypatch):
    prices, config, clean, expected = completed_run
    output = tmp_path / "interrupted"
    original_save = experiment._save_checkpoint

    def interrupt_after_first_save(path, signature, identity, payload):
        original_save(path, signature, identity, payload)
        raise KeyboardInterrupt("Simulate a stopped process after a durable fold")

    monkeypatch.setattr(experiment, "_save_checkpoint", interrupt_after_first_save)
    with pytest.raises(KeyboardInterrupt):
        experiment.run_experiment(prices, config, output)
    assert (output / ".checkpoints" / "outer-zero-0.joblib").is_file()
    monkeypatch.setattr(experiment, "_save_checkpoint", original_save)
    computed = []
    original_outer = experiment._outer_payload

    def track_outer(name, candidates, research, features, configuration, fold, train, test):
        computed.append((name, fold))
        return original_outer(
            name, candidates, research, features, configuration, fold, train, test
        )

    monkeypatch.setattr(experiment, "_outer_payload", track_outer)
    actual = experiment.run_experiment(prices, config, output, resume=True)
    assert ("zero", 0) not in computed
    assert len(computed) == 3
    assert actual["winner"] == expected["winner"]
    assert actual["ensemble_weights"] == expected["ensemble_weights"]
    _assert_identical_research_outputs(clean, output)


def test_completed_resume_restores_artifacts_without_any_fit(completed_run, monkeypatch):
    prices, config, output, expected = completed_run
    expected_predictions = pd.read_csv(output / "holdout_predictions.csv")
    for name in ("model.joblib", "summary.json", "report.html"):
        (output / name).unlink()

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("A completed checkpoint must not be refitted")

    monkeypatch.setattr(experiment, "_fit", forbidden_fit)
    monkeypatch.setattr(experiment, "_tune", forbidden_fit)
    # Canonical sorting allows equivalent source rows in a different order.
    actual = experiment.run_experiment(prices.iloc[::-1], config, output, resume=True)
    assert actual["winner"] == expected["winner"]
    assert actual["holdout_metrics"] == expected["holdout_metrics"]
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "holdout_predictions.csv"), expected_predictions
    )
    assert all(
        (output / name).is_file() for name in ("model.joblib", "summary.json", "report.html")
    )


def test_interrupted_final_fit_restores_saved_estimator(completed_run, tmp_path, monkeypatch):
    prices, config, clean, _ = completed_run
    output = tmp_path / "interrupted_final"
    original_save = experiment._save_checkpoint

    def interrupt_after_final_save(path, signature, identity, payload):
        original_save(path, signature, identity, payload)
        if identity == {"stage": "final", "model": "zero"}:
            raise KeyboardInterrupt("Stop after the first final estimator is durable")

    monkeypatch.setattr(experiment, "_save_checkpoint", interrupt_after_final_save)
    with pytest.raises(KeyboardInterrupt):
        experiment.run_experiment(prices, config, output)
    monkeypatch.setattr(experiment, "_save_checkpoint", original_save)
    computed = []
    original_final = experiment._final_payload

    def track_final(name, *args, **kwargs):
        computed.append(name)
        return original_final(name, *args, **kwargs)

    monkeypatch.setattr(experiment, "_final_payload", track_final)
    experiment.run_experiment(prices, config, output, resume=True)
    assert computed == ["ridge"]
    _assert_identical_research_outputs(clean, output)


@pytest.mark.parametrize(
    "changed", ["data", "config", "features", "registry", "source", "versions"]
)
def test_resume_rejects_changed_experiment_identity(completed_run, monkeypatch, changed):
    prices, config, output, _ = completed_run
    manifest = output / ".checkpoints" / "manifest.json"
    original_manifest = manifest.read_bytes()
    if changed == "data":
        prices = prices.copy()
        prices["volume"] *= 2
    elif changed == "config":
        config = replace(config, seed=config.seed + 1)
    elif changed == "features":
        frame = build_dataset(prices, config.horizon)
        frame["f_new"] = 0.0
        monkeypatch.setattr(experiment, "build_dataset", lambda *args, **kwargs: frame)
    elif changed == "registry":
        original = experiment.model_candidates

        def changed_registry(*args, **kwargs):
            registry = original(*args, **kwargs)
            registry["ridge"][0]["alpha"] = 123.0
            return registry

        monkeypatch.setattr(experiment, "model_candidates", changed_registry)
    elif changed == "source":
        monkeypatch.setattr(experiment, "_source_hashes", lambda: {"experiment": "changed"})
    elif changed == "versions":
        monkeypatch.setattr(experiment, "_versions", lambda: {"python": "changed"})
    with pytest.raises(ValueError, match="manifest mismatch"):
        experiment.run_experiment(prices, config, output, resume=True)
    assert manifest.read_bytes() == original_manifest


def test_resume_requires_manifest_and_preserves_unrelated_files(tmp_path):
    prices = generate_synthetic(n_days=100)
    output = tmp_path / "unrelated"
    output.mkdir()
    sentinel = output / "important.txt"
    sentinel.write_text("keep this file")
    with pytest.raises(ValueError, match="valid checkpoint manifest"):
        experiment.run_experiment(prices, ExperimentConfig(), output, resume=True)
    assert sentinel.read_text() == "keep this file"
    with pytest.raises(ValueError, match="not empty"):
        experiment.run_experiment(prices, ExperimentConfig(), output)


def test_damaged_checkpoint_cannot_silently_change_results(completed_run):
    prices, config, output, _ = completed_run
    (output / ".checkpoints" / "outer-zero-0.joblib").write_bytes(b"interrupted invalid bytes")
    before = (output / "summary.json").read_bytes()
    with pytest.raises(ValueError, match="Cannot load checkpoint"):
        experiment.run_experiment(prices, config, output, resume=True)
    assert (output / "summary.json").read_bytes() == before


def test_atomic_checkpoint_failure_preserves_previous_complete_record(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.joblib"
    identity = {"stage": "outer", "model": "zero", "fold": 0}
    payload = {"status": "failed", "error": "original diagnostic"}
    experiment._save_checkpoint(path, "signature", identity, payload)
    previous = path.read_bytes()

    def interrupted_dump(record, temporary, **kwargs):
        temporary.write_bytes(b"partial replacement")
        raise OSError("Simulated interrupted disk write")

    monkeypatch.setattr(experiment.joblib, "dump", interrupted_dump)
    with pytest.raises(OSError, match="interrupted disk"):
        experiment._save_checkpoint(path, "signature", identity, payload)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_optional_model_failures_are_checkpointed_and_not_retried(
    completed_run, tmp_path, monkeypatch
):
    prices, config, _, _ = completed_run
    output = tmp_path / "optional_failure"
    original = experiment._fit

    def fail_ridge(name, *args, **kwargs):
        if name == "ridge":
            raise ImportError("Optional dependency unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(experiment, "_fit", fail_ridge)
    first = experiment.run_experiment(prices, config, output)
    assert set(first["model_errors"]) == {"ridge"}
    assert first["winner"] == "zero"

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Cached successful fits and failures must both be reused")

    monkeypatch.setattr(experiment, "_fit", forbidden_fit)
    second = experiment.run_experiment(prices, config, output, resume=True)
    assert second["model_errors"] == first["model_errors"]
    assert second["winner"] == first["winner"]


def test_parallel_jobs_match_serial_outputs(completed_run, tmp_path):
    prices, config, serial, expected = completed_run
    parallel = tmp_path / "parallel"
    actual = experiment.run_experiment(prices, config, parallel, workers=2)
    assert actual["winner"] == expected["winner"]
    assert actual["ensemble_weights"] == expected["ensemble_weights"]
    _assert_identical_research_outputs(serial, parallel)
    # Worker count is execution policy, not a change to experiment identity.
    resumed = experiment.run_experiment(prices, config, parallel, resume=True, workers=1)
    assert resumed["holdout_metrics"] == actual["holdout_metrics"]


def test_cli_exposes_resume_and_worker_count():
    from marketforge.cli import _parser

    args = _parser().parse_args(
        [
            "run",
            "--data",
            "prices.csv",
            "--output",
            "results/study",
            "--resume",
            "--workers",
            "3",
        ]
    )
    assert args.resume is True
    assert args.workers == 3

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from marketforge.models import RegimeMixtureRegressor, make_model, model_candidates


def _sample(seed=12, n=160):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = 0.01 * X[:, 0] - 0.005 * X[:, 1] + rng.normal(scale=0.002, size=n)
    return X, y


def test_baselines_are_training_only_and_zero_is_exact():
    X, y = _sample()
    query = X[:7] * 100
    assert np.array_equal(make_model("zero").fit(X, y).predict(query), np.zeros(7))
    assert np.allclose(make_model("historical_mean").fit(X, y).predict(query), y.mean())


@pytest.mark.parametrize("name", list(model_candidates(quick=True)))
def test_every_core_estimator_produces_finite_out_of_sample_forecasts(name):
    X, y = _sample()
    parameters = model_candidates(quick=True)[name][0]
    model = clone(make_model(name, parameters, seed=19)).fit(X[:120], y[:120])
    prediction = model.predict(X[120:])
    assert prediction.shape == (40,)
    assert np.isfinite(prediction).all()


def test_preprocessing_never_refits_on_prediction_rows():
    X, y = _sample()
    model = make_model("ridge", {"alpha": 1.0}).fit(X, y)
    feature_mean = model.regressor_.steps[0][1].mean_.copy()
    target_mean = model.transformer_.mean_.copy()
    prediction = model.predict(X[:3])
    model.predict(X * 1000 + 9999)
    np.testing.assert_array_equal(model.regressor_.steps[0][1].mean_, feature_mean)
    np.testing.assert_array_equal(model.transformer_.mean_, target_mean)
    np.testing.assert_array_equal(model.predict(X[:3]), prediction)


def test_regime_mixture_handles_constant_training_features_and_small_samples():
    model = RegimeMixtureRegressor(n_regimes=5).fit(np.ones((8, 4)), np.arange(8) / 100)
    assert model.n_regimes_ == 1
    assert model.gate_ is None
    np.testing.assert_allclose(model.predict(np.ones((3, 4))), 0.035)


def test_regime_mixture_learns_different_local_relationships():
    rng = np.random.default_rng(41)
    regimes = np.repeat([-1.0, 1.0], 200)
    X = np.column_stack([regimes * 4 + rng.normal(0, 0.2, 400), rng.normal(size=400)])
    y = regimes * X[:, 1] * 0.02
    model = RegimeMixtureRegressor(n_regimes=2, alpha=0.1, shrinkage=1.0).fit(X, y)
    assert model.n_regimes_ == 2
    assert np.mean((model.predict(X) - y) ** 2) < 0.00001
    state = model.gate_.means_.copy()
    model.predict(X + 100)
    np.testing.assert_array_equal(model.gate_.means_, state)


def test_regime_mixture_rejects_bad_inputs_and_requires_fit():
    with pytest.raises(NotFittedError):
        RegimeMixtureRegressor().predict([[1, 2]])
    X, y = _sample()
    with pytest.raises(ValueError, match="positive"):
        RegimeMixtureRegressor(n_regimes=0).fit(X, y)
    with pytest.raises(ValueError, match="feature count"):
        RegimeMixtureRegressor().fit(X, y).predict(np.ones((3, 2)))
    with pytest.raises(ValueError):
        make_model("ridge").fit(np.full((10, 2), np.nan), np.arange(10))


def test_candidate_grids_are_independent_and_optional_models_are_explicit():
    grids = model_candidates(quick=True)
    grids["ridge"][0]["alpha"] = -1
    assert model_candidates(quick=True)["ridge"][0]["alpha"] == 10
    assert "torch_attention" not in model_candidates()
    assert "xgboost" in model_candidates(include_boosting=True)
    assert "torch_attention" in model_candidates(include_neural=True)
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("imaginary")


@pytest.mark.parametrize("architecture", ["attention", "residual"])
def test_optional_neural_models_are_reproducible_and_serializable(architecture, tmp_path):
    torch = pytest.importorskip("torch")
    import joblib

    from marketforge.neural import TorchTabularRegressor

    torch.set_num_threads(1)
    X, y = _sample(n=40)
    kwargs = dict(architecture=architecture, width=8, epochs=2, batch_size=16, random_state=3)
    first = TorchTabularRegressor(**kwargs).fit(X[:30], y[:30])
    second = TorchTabularRegressor(**kwargs).fit(X[:30], y[:30])
    np.testing.assert_allclose(first.predict(X[30:]), second.predict(X[30:]), rtol=0, atol=1e-10)
    destination = tmp_path / "neural.joblib"
    joblib.dump(first, destination)
    np.testing.assert_array_equal(joblib.load(destination).predict(X[30:]), first.predict(X[30:]))
    assert len(first.loss_history_) == 2

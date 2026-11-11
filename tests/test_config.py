"""Reject configuration values that would silently change experimental meaning."""
import math

import pytest

from marketforge.config import ExperimentConfig, load_config


@pytest.mark.parametrize("field,value", [
    ("seed", True), ("seed", -1), ("seed", "42"),
    ("horizon", True), ("horizon", "1"), ("n_splits", 0),
    ("embargo", False), ("embargo", -1),
    ("include_neural", "false"), ("include_boosting", 1), ("quick", "true"),
    ("cost_bps", "5"), ("cost_bps", True), ("cost_bps", -1),
    ("target_vol", math.inf), ("target_vol", 0),
    ("signal_threshold", math.nan), ("signal_threshold", -0.1),
    ("max_leverage", 0), ("max_leverage", 1.1),
    ("interval_alpha", 0), ("interval_alpha", 1),
    ("bootstrap_samples", 19), ("bootstrap_block_size", 0),
    ("models", "ridge"), ("models", []), ("models", ["ridge", 1]),
])
def test_invalid_configuration_fails_at_construction(field, value):
    with pytest.raises(ValueError):
        ExperimentConfig(**{field: value})


@pytest.mark.parametrize("text", ["[]", "false", "0", "ridge", "- ridge\n- zero"])
def test_yaml_requires_a_mapping_even_when_falsey(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_unknown_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("holdout_szie: 126\n")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(path)


def test_round_trip_and_empty_yaml_defaults(tmp_path):
    config = ExperimentConfig(models=["zero", "ridge"], quick=True, bootstrap_samples=20)
    assert ExperimentConfig(**config.to_dict()) == config
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert load_config(path) == ExperimentConfig()

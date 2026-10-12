"""Foundation configuration tests."""

from pathlib import Path

import pytest

from river_water_quality.utils.config import load_config, load_yaml, validate_config


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / "configs/base.yaml")


def test_config_loads(config):
    assert config["paths"]["interim_data"] == "data/interim"
    assert config["target"]["column"] is None


@pytest.mark.parametrize("seed", [-1, 2**32, True, "42"])
def test_invalid_seed(config, seed):
    config["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        validate_config(config)


@pytest.mark.parametrize("section", ["paths", "target", "experiment", "project"])
def test_missing_section(config, section):
    del config[section]
    with pytest.raises(ValueError, match=section):
        validate_config(config)


def test_invalid_path(config):
    config["paths"]["raw_data"] = " "
    with pytest.raises(ValueError, match="paths.raw_data"):
        validate_config(config)


def test_target_requires_definition(config):
    config["target"]["column"] = "wqi"
    with pytest.raises(ValueError, match="definition_source"):
        validate_config(config)
    config["target"]["definition_source"] = "docs/target_definition.md"
    validate_config(config)


def test_invalid_experiment_flag(config):
    config["experiment"]["deterministic"] = "true"
    with pytest.raises(ValueError, match="deterministic"):
        validate_config(config)


def test_invalid_target_task(config):
    config["target"]["task"] = "invented"
    with pytest.raises(ValueError, match="target.task"):
        validate_config(config)


def test_yaml_root_validation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- item\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(path)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        load_config(path)

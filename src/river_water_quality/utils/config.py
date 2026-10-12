"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def validate_config(config: dict[str, Any]) -> None:
    """Validate foundation settings without inventing scientific targets."""
    seed = config.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    for section in ("project", "paths", "experiment", "target"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a mapping")
    for key in ("raw_data", "interim_data", "processed_data", "external_data", "outputs", "checkpoints"):
        value = config["paths"].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a non-empty path string")
    for section, key in (("project", "name"), ("project", "version"), ("experiment", "name")):
        value = config[section].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{section}.{key} must be a non-empty string")
    if type(config["experiment"].get("deterministic")) is not bool:
        raise ValueError("experiment.deterministic must be a boolean")
    target = config["target"]
    if target.get("task") not in ("regression", "classification"):
        raise ValueError("target.task must be regression or classification")
    for key in ("column", "definition_source"):
        value = target.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"target.{key} must be null or a non-empty string")
    if target.get("column") is not None and target.get("definition_source") is None:
        raise ValueError("target.definition_source is required when target.column is configured")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an experiment configuration; retain load_yaml for generic YAML."""
    config = load_yaml(path)
    validate_config(config)
    return config


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the root of {config_path}")
    return data

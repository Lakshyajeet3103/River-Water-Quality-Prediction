from pathlib import Path

from river_water_quality import __version__
from river_water_quality.utils.config import load_yaml


def test_package_version():
    assert __version__ == "0.1.0"


def test_base_config_loads():
    config_path = Path(__file__).parents[1] / "configs" / "base.yaml"
    config = load_yaml(config_path)
    assert config["seed"] == 42
    assert config["project"]["name"] == "river-water-quality-prediction"

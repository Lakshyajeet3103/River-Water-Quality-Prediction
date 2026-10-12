"""Check installed dependencies and foundation configuration, without downloading data."""

from importlib.metadata import version
from pathlib import Path
import sys

from river_water_quality.utils.config import load_config


def main() -> None:
    """Report the active interpreter, package versions, and configuration status."""
    print(f"Python: {sys.version.split()[0]}")
    for package in ("river-water-quality-prediction", "numpy", "pandas", "pyyaml", "pytest"):
        print(f"{package}: {version(package)}")
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")
    print("Foundation configuration valid.")
    print(f"Configured target: {config['target']['column'] or 'not yet selected'}")


if __name__ == "__main__":
    main()

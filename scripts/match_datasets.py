"""Match local datasets and write a new, auditable output directory."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import hashlib
import json

from river_water_quality.data.ingestion import load_dataset
from river_water_quality.data.matching import MatchingSettings, match_observations
from river_water_quality.utils.config import load_config


def main() -> None:
    """Require explicit inputs/settings; refuse to overwrite an output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_csv")
    parser.add_argument("ground_manifest")
    parser.add_argument("satellite_csv")
    parser.add_argument("satellite_manifest")
    parser.add_argument("--ground-kind", choices=["cpcb", "usgs"], default="cpcb")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        settings = MatchingSettings.from_config(config)
        ground = load_dataset(args.ground_csv, args.ground_manifest, kind=args.ground_kind)
        satellite = load_dataset(args.satellite_csv, args.satellite_manifest, kind="sentinel2")
        result = match_observations(ground.data, satellite.data, settings)
        report = {
            "summary": result.summary, "configuration": config, "matching": asdict(settings),
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "sources": {kind: {"provenance": dataset.provenance,
                                "source_sha256": dataset.source_sha256,
                                "manifest_sha256": dataset.manifest_sha256}
                        for kind, dataset in ((args.ground_kind, ground), ("sentinel2", satellite))},
            "versions": {name: version(name) for name in ("river-water-quality-prediction", "pandas", "numpy", "pyyaml", "tzdata")},
            "limitations": "Date-level matching only; raw inputs/quality thresholds and shared-scene leakage require review.",
        }
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        for name in ("matches", "ground", "satellite", "exclusions", "candidates"):
            getattr(result, name).to_csv(output / f"{name}.csv", index=False)
        (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Matching failed: {exc}\n")
    print(json.dumps(result.summary, indent=2))
    print(f"Outputs written to {output}")


if __name__ == "__main__":
    main()

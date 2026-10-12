"""Run Phase 3 regression baselines on an existing local matching run."""

import argparse
import csv
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform

import pandas as pd

from river_water_quality.data.schemas import GROUND_NUMERIC, SATELLITE_NUMERIC
from river_water_quality.utils.config import load_config


def read_table(path: Path, numeric: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read known matching outputs without losing IDs or hiding duplicate headers."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if not header or len(header) != len(set(header)):
            raise ValueError(f"{path.name} must have unique, non-empty headers")
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Invalid record width in {path.name}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, skip_blank_lines=False)
    for column in numeric:
        if column not in frame.columns:
            raise ValueError(f"Missing {column} in {path.name}")
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric values in {path.name}:{column}") from exc
    return frame


def main() -> None:
    """Require an explicit target and data purpose; refuse existing output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matching_run", help="Phase 2 directory containing run.json and matched CSVs")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if config["target"]["task"] != "regression":
            raise ValueError("Phase 3 currently supports regression only; no class thresholds are assumed")
        if not config["target"].get("column") or not config["target"].get("definition_source"):
            raise ValueError("Set a justified target.column and target.definition_source before training")
        options = config.get("baseline")
        if not isinstance(options, dict) or options.get("data_purpose") not in ("research", "software_test"):
            raise ValueError("baseline.data_purpose must explicitly be research or software_test")
        if not isinstance(options.get("target_unit"), str) or not options["target_unit"].strip():
            raise ValueError("baseline.target_unit must be explicitly configured")
        try:
            import joblib
            from river_water_quality.training.baselines import train_baselines
        except ImportError as exc:
            raise ValueError("Install Phase 3 dependencies with python -m pip install -r requirements-ml.txt") from exc
        from river_water_quality.training.dataset import build_baseline_dataset
        from river_water_quality.data.matching import MatchingSettings

        directory = Path(args.matching_run)
        output = Path(args.output)
        if output.exists():
            raise ValueError("Output directory already exists; choose a new run path")
        input_paths = [directory / name for name in ("run.json", "matches.csv", "ground.csv", "satellite.csv")]
        matching_report = json.loads(input_paths[0].read_text(encoding="utf-8"))
        if not isinstance(matching_report, dict) or not isinstance(matching_report.get("matching"), dict) or not isinstance(matching_report.get("sources"), dict) or not matching_report["sources"]:
            raise ValueError("Matching run must contain settings and source provenance")
        providers = set(matching_report["sources"])
        if providers not in ({"cpcb", "sentinel2"}, {"usgs", "sentinel2"}):
            raise ValueError("Matching provenance must contain one ground provider and sentinel2")
        for provider, source in matching_report["sources"].items():
            if not isinstance(source, dict) or not isinstance(source.get("provenance"), dict):
                raise ValueError(f"Missing source provenance for {provider}")
            if source["provenance"].get("provider") != provider:
                raise ValueError(f"Source provenance provider mismatch for {provider}")
            for field in ("dataset_id", "source", "access_date", "license", "timestamp_convention"):
                value = source["provenance"].get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Missing provenance {field} for {provider}")
            if source["provenance"].get("coordinate_reference_system") != "EPSG:4326" or not isinstance(source["provenance"].get("units"), dict) or not source["provenance"]["units"]:
                raise ValueError(f"Invalid coordinate/unit provenance for {provider}")
            for field in ("source_sha256", "manifest_sha256"):
                value = source.get(field)
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"Missing or invalid {field} for {provider}")
        settings = MatchingSettings(**matching_report["matching"])
        ground_table = read_table(input_paths[2], GROUND_NUMERIC)
        if not ground_table.quality_flag.isin(settings.accepted_ground_flags).all():
            raise ValueError("Ground quality flags disagree with the matching configuration")
        data = build_baseline_dataset(
            read_table(input_paths[1]), ground_table,
            read_table(input_paths[3], SATELLITE_NUMERIC),
            target_parameter=config["target"]["column"], target_unit=options["target_unit"],
            ground_units=options.get("ground_units"), satellite_units=options.get("satellite_units"),
            timezone=settings.timezone,
        )
        ground_provider = next(provider for provider in providers if provider != "sentinel2")
        source_ground_units = matching_report["sources"][ground_provider]["provenance"]["units"]
        for feature, unit in {config["target"]["column"]: options["target_unit"], **options["ground_units"]}.items():
            if source_ground_units.get(feature) != unit:
                raise ValueError(f"Configured ground unit disagrees with source provenance for {feature}")
        source_satellite_units = matching_report["sources"]["sentinel2"]["provenance"]["units"]
        for feature, unit in options["satellite_units"].items():
            if source_satellite_units.get(feature) != unit:
                raise ValueError(f"Configured satellite unit disagrees with source provenance for {feature}")
        result = train_baselines(data, options.get("split"), seed=config["seed"],
            ridge_alpha=options.get("ridge_alpha", 1.0), forest_trees=options.get("forest_trees", 100),
            forest_min_leaf=options.get("forest_min_leaf", 2))
        report = {
            "data_purpose": options["data_purpose"],
            "evidence_label": "SOFTWARE TEST ONLY — NOT SCIENTIFIC EVIDENCE" if options["data_purpose"] == "software_test" else "USER-DECLARED RESEARCH INPUTS — SCIENTIFIC SUITABILITY REQUIRES REVIEW",
            "configuration": config,
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "matching_run": matching_report,
            "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths},
            "features": {"ground": list(data.ground.columns), "satellite": list(data.satellite.columns)},
            "cohort_events": len(data.metadata), "cohort_excluded_events": len(data.exclusions),
            "split_counts": result.assignments.split.value_counts().to_dict(),
            "selection": "Lowest validation MAE per modality; fixed parameters, no validation refit; test evaluated only for selected models and mean reference",
            "versions": {"python": platform.python_version(), **{name: version(name) for name in ("river-water-quality-prediction", "numpy", "pandas", "scikit-learn", "scipy", "joblib", "pyyaml", "tzdata")}},
            "limitations": "Same-event ground predictors estimate a measured target; availability, target-derived aliases, nearby stations, source quality, and generalization need review.",
        }
        output.mkdir(parents=True, exist_ok=False)
        data.exclusions.to_csv(output / "cohort_exclusions.csv", index=False)
        result.assignments.join(data.metadata).to_csv(output / "split_assignments.csv")
        result.metrics.assign(data_purpose=report["data_purpose"], evidence_label=report["evidence_label"]).to_csv(output / "metrics.csv", index=False)
        result.predictions.assign(data_purpose=report["data_purpose"], evidence_label=report["evidence_label"]).to_csv(output / "test_predictions.csv", index=False)
        joblib.dump({"models": result.models, "features": report["features"],
                     "target": config["target"], "target_unit": options["target_unit"],
                     "data_purpose": report["data_purpose"], "evidence_label": report["evidence_label"]}, output / "models.joblib")
        (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(1, f"Baseline run failed: {exc}\n")
    print(report["evidence_label"])
    print(f"Baseline artifacts written to {output}; inspect metrics.csv and exclusions.")


if __name__ == "__main__":
    main()

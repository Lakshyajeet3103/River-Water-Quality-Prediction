"""Run a small, explicitly configured Earth Engine extraction pilot."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import pandas as pd

from river_water_quality.data.ingestion import load_dataset
from river_water_quality.data.schemas import SATELLITE_COLUMNS, validate_satellite
from river_water_quality.data.sentinel import COLLECTION, extract_event, validate_extraction_settings
from river_water_quality.utils.config import load_yaml


def main() -> None:
    """Reuse existing Google credentials; save results/audits without overwriting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_csv")
    parser.add_argument("ground_manifest")
    parser.add_argument("--config", default="configs/extraction_pilot.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        settings = load_yaml(args.config)
        validate_extraction_settings(settings)
        ground = load_dataset(args.ground_csv, args.ground_manifest, kind="cpcb")
        events = ground.data[["observation_id", "station_id", "sample_date", "latitude", "longitude"]].drop_duplicates()
        if len(events) > settings["max_events"]:
            raise ValueError("Input exceeds max_events; supply an explicit pilot subset")
        out = Path(args.output)
        if out.exists():
            raise ValueError("Output already exists; choose a new run path")
        import ee
        ee.Initialize(project=settings["project_id"])
        features, audits = [], []
        for index, event in enumerate(events.to_dict("records"), 1):
            print(f"Extracting pilot event {index}/{len(events)}: {event['station_id']} {event['sample_date']}", flush=True)
            try:
                rows, log = extract_event(ee, event, settings)
                features.extend(rows)
                audits.extend(log)
            except Exception as exc:
                audits.append({"observation_id": event["observation_id"], "station_id": event["station_id"],
                    "sample_date": event["sample_date"], "status": "query_error", "error_type": type(exc).__name__})
        frame = pd.DataFrame(features) if features else pd.DataFrame(columns=SATELLITE_COLUMNS)
        if not frame.empty:
            # Shared scenes are emitted once; audit retains individual event links.
            frame = frame.drop_duplicates(["station_id", "scene_id", "band"])
            validate_satellite(frame)
        now = datetime.now(timezone.utc)
        manifest = {"dataset_id": "sentinel2_up_three_event_pilot", "provider": "sentinel2", "source": COLLECTION,
            "access_date": now.date().isoformat(), "license": "Copernicus Sentinel Data Terms and Conditions; see Earth Engine catalog",
            "coordinate_reference_system": "EPSG:4326", "timestamp_convention": "Acquisition timestamps UTC; pilot ground dates provisionally Asia/Kolkata",
            "units": {band: "reflectance" for band in settings["bands"]}, "column_mapping": {},
            "settings": settings, "processing": "DN * 0.0001; common B5 20m grid with default nearest resampling; SCL water=6, cloud/shadow=3/8/9/10; unweighted counts and means include pixel centres within ROI; common-valid water pixels",
            "source_catalog": "https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED",
            "limitations": "Provisional pilot settings and ground timezone/CRS; SCL water classification requires validation; not research evidence"}
        summary = {"attempted_events": len(events), "selected_events": sum(row["status"] == "selected" for row in audits),
            "spectral_rows": len(frame), "status_counts": pd.Series([row["status"] for row in audits]).value_counts().to_dict(),
            "ground_source_sha256": ground.source_sha256, "ground_manifest_sha256": ground.manifest_sha256,
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(), "settings": settings,
            "run_utc": now.isoformat(), "earthengine_version": ee.__version__}
        out.mkdir(parents=True, exist_ok=False)
        frame.to_csv(out / "features.csv", index=False)
        pd.DataFrame(audits).to_csv(out / "audit.csv", index=False)
        (out / "features.provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (out / "run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as exc:
        parser.exit(1, f"Pilot failed ({type(exc).__name__}). Check input/configuration and Earth Engine project access.\n")
    print(json.dumps(summary, indent=2))
    if summary["status_counts"].get("query_error"):
        parser.exit(1, "One or more queries failed; inspect audit.csv. Missing features were not fabricated.\n")


if __name__ == "__main__":
    main()

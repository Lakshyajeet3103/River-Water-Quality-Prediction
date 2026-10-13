"""Prepare a bounded USGS river case study from downloaded WQP exports."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
from pyproj import Transformer
import yaml

from river_water_quality.data.wqp import screen_results, pair_samples
from river_water_quality.data.schemas import validate_ground
from river_water_quality.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw/us_search_001")
    parser.add_argument("--stations", nargs="+", default=["USGS-02198920", "USGS-02226010", "USGS-02212940"])
    parser.add_argument("--output", default="data/interim/us_demo_input_001")
    args = parser.parse_args()
    source, out = Path(args.source), Path(args.output)
    if out.exists():
        raise ValueError("Output exists; choose a new directory")
    bod_raw = pd.read_csv(source / "usgs_bod_2017_2023.csv", dtype=str, keep_default_na=False)
    fecal_raw = pd.read_csv(source / "usgs_fecal_2017_2023.csv", dtype=str, keep_default_na=False)
    stations = pd.read_csv(source / "usgs_bod_stations.csv", dtype=str, keep_default_na=False)
    bod, bod_audit = screen_results(bod_raw, "bod")
    fecal, fecal_audit = screen_results(fecal_raw, "fecal_coliform")
    pairs, pairing_audit = pair_samples(bod, fecal)
    chosen = pairs[pairs.MonitoringLocationIdentifier.isin(args.stations) & pairs.ActivityStartDate.ge("2019-01-01")].copy()
    if chosen.empty or len(chosen) > 12:
        raise ValueError("Need 1-12 explicitly bounded station events from 2019 onward")
    records, registry = [], []
    for row in chosen.to_dict("records"):
        station = stations[stations.MonitoringLocationIdentifier.eq(row["MonitoringLocationIdentifier"])]
        if len(station) != 1:
            raise ValueError("Missing or ambiguous station metadata")
        site = station.iloc[0]
        if site.MonitoringLocationTypeName not in ("Stream", "River/Stream", "Stream: Tidal stream"):
            raise ValueError("Case study requires river/stream stations")
        if site.HorizontalCoordinateReferenceSystemDatumName != "NAD83":
            raise ValueError("Unreviewed station coordinate datum")
        transformer = Transformer.from_crs("EPSG:4269", "EPSG:4326", always_xy=True, allow_ballpark=False)
        lon, lat = transformer.transform(float(site.LongitudeMeasure), float(site.LatitudeMeasure))
        operation = transformer.get_last_used_operation()
        registry.append({"station_id": site.MonitoringLocationIdentifier, "station_name": site.MonitoringLocationName,
            "station_type": site.MonitoringLocationTypeName,
            "source_latitude": site.LatitudeMeasure, "source_longitude": site.LongitudeMeasure,
            "source_datum": "NAD83", "latitude": lat, "longitude": lon, "output_crs": "EPSG:4326",
            "transform_description": operation.description, "transform_accuracy_m": operation.accuracy,
            "source_horizontal_accuracy": site["HorizontalAccuracyMeasure/MeasureValue"],
            "source_horizontal_accuracy_unit": site["HorizontalAccuracyMeasure/MeasureUnitCode"]})
        for parameter, suffix, unit in (("bod", "bod", "mg/L"), ("fecal_coliform", "fecal", "MPN/100mL")):
            records.append({"observation_id": row["observation_id"], "station_id": site.MonitoringLocationIdentifier,
                "station_name": site.MonitoringLocationName, "sample_date": row["ActivityStartDate"],
                "latitude": lat, "longitude": lon, "parameter": parameter,
                "value": float(row[f"ResultMeasureValue_{suffix}"]), "unit": unit,
                "quality_flag": "accepted_uncensored", "source_activity_id": row[f"ActivityIdentifier_{suffix}"],
                "source_parameter_code": row[f"USGSPCode_{suffix}"], "ground_offset_minutes": row["ground_offset_minutes"],
                "source_sample_time": row[f"ActivityStartTime/Time_{suffix}"],
                "source_time_zone": row[f"ActivityStartTime/TimeZoneCode_{suffix}"],
                "source_method": row[f"ResultAnalyticalMethod/MethodName_{suffix}"],
                "source_unit": row[f"ResultMeasure/MeasureUnitCode_{suffix}"],
                "source_file": f"usgs_{'bod' if suffix == 'bod' else 'fecal'}_2017_2023.csv",
                "source_record": row[f"source_record_{suffix}"]})
    ground = pd.DataFrame(records)
    validate_ground(ground)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.glob("*.csv")}
    provenance = {"dataset_id": "usgs_three_river_near_time_demo", "provider": "usgs",
        "source": "USGS NWIS records downloaded via https://www.waterqualitydata.us; original queries in .download.json sidecars",
        "access_date": datetime.now(timezone.utc).date().isoformat(),
        "license": "USGS-origin government data; see https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits",
        "coordinate_reference_system": "EPSG:4326", "coordinate_processing": "pyproj EPSG:4269 to EPSG:4326; original coordinates and operation retained in stations.csv",
        "timestamp_convention": "WQP calendar dates; EST/EDT source clocks retained; image matching America/New_York",
        "units": {"bod": "mg/L", "fecal_coliform": "MPN/100mL"}, "source_sha256": hashes,
        "quality": "Accepted actual routine water results; qualifiers, nondetects, non-MPN fecal units and non-00310 BOD excluded",
        "pairing": "Separate BOD/fecal samples, same organization/station/date/time zone, unique nearest clock time within five minutes",
        "limitations": "US five-day BOD and presumptive MPN fecal method known; CPCB laboratory methods unverified; exploratory method comparability only. Savannah station is classified as tidal stream."}
    extraction = load_yaml("configs/extraction_pilot.yaml")
    extraction.update(timezone="America/New_York", timezone_status="WQP local dates with source EST/EDT clocks preserved", max_events=len(chosen))
    matching = load_yaml("configs/pilot_matching.yaml")
    matching["matching"].update(timezone="America/New_York", accepted_ground_flags=["accepted_uncensored"])
    out.mkdir(parents=True, exist_ok=False)
    ground.to_csv(out / "ground.csv", index=False)
    pd.DataFrame(registry).drop_duplicates().to_csv(out / "stations.csv", index=False)
    pairs.to_csv(out / "compatible_pairs.csv", index=False)
    chosen.to_csv(out / "selected_pairs.csv", index=False)
    pd.concat([bod_audit, fecal_audit], ignore_index=True).to_csv(out / "result_exclusions.csv", index=False)
    pairing_audit.to_csv(out / "pairing_exclusions.csv", index=False)
    (out / "ground.provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for name, config in (("extraction.yaml", extraction), ("matching.yaml", matching)):
        (out / name).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(ground[ground.parameter.eq("bod")][["station_name", "sample_date", "value", "ground_offset_minutes"]].to_string(index=False))
    print(f"Compatible near-time US pairs: {len(pairs)}; selected: {len(chosen)}")


if __name__ == "__main__":
    main()

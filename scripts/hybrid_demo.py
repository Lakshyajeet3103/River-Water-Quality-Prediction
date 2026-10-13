"""Prepare a small station/date subset and export real satellite-ground pairs."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from river_water_quality.data.ingestion import load_dataset
from river_water_quality.data.matching import MatchingSettings, match_observations
from river_water_quality.data.hybrid import choose_events, paired_table
from river_water_quality.utils.config import load_yaml


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def prepare(args):
    out = Path(args.output)
    if out.exists():
        raise ValueError("Output exists; choose a new directory")
    source = load_dataset(args.ground, args.manifest, kind="cpcb")
    selected, selection = choose_events(source.data, args.stations, args.year, args.months)
    settings = load_yaml("configs/extraction_pilot.yaml")
    settings["max_events"] = int(selected.observation_id.nunique())
    out.mkdir(parents=True, exist_ok=False)
    selected.to_csv(out / "ground.csv", index=False)
    selection.to_csv(out / "selection.csv", index=False)
    provenance = {**source.provenance, "dataset_id": "small_station_hybridization_demo",
                  "parent_source_sha256": source.source_sha256, "parent_manifest_sha256": source.manifest_sha256,
                  "selection": "For each requested station/month, earliest BOD sampling date nearest day 15; retain all measured parameters of that event"}
    save_json(out / "ground.provenance.json", provenance)
    save_json(out / "selection.json", {"year": args.year, "months": args.months, "station_ids": args.stations,
              "rule": "Nearest day 15 within each requested month; date and event ID break ties; BOD magnitude never used",
              "parent_source_sha256": source.source_sha256, "parent_manifest_sha256": source.manifest_sha256})
    (out / "extraction.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    print(selection.to_string(index=False))
    print(f"Prepared {settings['max_events']} events in {out}")


def export(args):
    out, prepared, extracted = Path(args.output), Path(args.prepared), Path(args.extracted)
    if out.exists():
        raise ValueError("Output exists; choose a new directory")
    ground = load_dataset(prepared / "ground.csv", prepared / "ground.provenance.json", kind="cpcb")
    satellite = load_dataset(extracted / "features.csv", extracted / "features.provenance.json", kind="sentinel2")
    settings = MatchingSettings.from_config(load_yaml("configs/pilot_matching.yaml"))
    result = match_observations(ground.data, satellite.data, settings)
    paired = paired_table(result)
    audit = pd.read_csv(extracted / "audit.csv")
    extraction_report = json.loads((extracted / "run.json").read_text())
    if extraction_report["ground_source_sha256"] != ground.source_sha256:
        raise ValueError("Extraction ground hash differs from prepared input")
    if extraction_report["ground_manifest_sha256"] != ground.manifest_sha256:
        raise ValueError("Extraction ground manifest differs from prepared input")
    if set(audit.observation_id) != set(ground.data.observation_id):
        raise ValueError("Extraction audit must cover exactly the planned events")
    events = ground.data[["observation_id", "station_id", "station_name", "sample_date", "latitude", "longitude"]].drop_duplicates()
    events["matched"] = events.observation_id.isin(result.matches.observation_id)
    events["extraction_outcome"] = events.observation_id.map(audit.groupby("observation_id", sort=False).status.last())
    events["candidate_rejections"] = events.observation_id.map(
        audit.groupby("observation_id", sort=False).status.apply(lambda statuses: ";".join(sorted(set(statuses) - {"selected", "no_usable_scene"}))))
    counts = events.groupby(["station_id", "station_name"]).agg(attempted=("observation_id", "size"), matched=("matched", "sum")).reset_index()
    counts["unmatched"] = counts.attempted - counts.matched
    report = {"scope": "Small-station real-data satellite-ground hybridization feasibility demonstration",
              "summary": result.summary, "by_station": counts.to_dict("records"), "matching": asdict(settings),
              "extraction": extraction_report, "paired_rows": len(paired),
              "units": {"ground": ground.provenance["units"], "satellite": satellite.provenance["units"]},
              "source_hashes": {"ground": ground.source_sha256, "satellite": satellite.source_sha256,
                                "ground_manifest": ground.manifest_sha256, "satellite_manifest": satellite.manifest_sha256,
                                "extraction_audit": hashlib.sha256((extracted / "audit.csv").read_bytes()).hexdigest()},
              "limitations": ["Convenience subset selected for feasibility; coverage is not representative of all stations.",
                              "Supplied station coordinates are not independently verified sampling coordinates.",
                              "Ground source URL/license/QC/CRS/time convention remain provisional; see source manifests.",
                              "Retrospective date-level matching permits images before or after sampling; not a forecast.",
                              "ROI centre distance is zero by construction; pixels occupy a 60m buffer, not an exact sampling point.",
                              "Visual review of the selected locations shows crossings near the ROIs; SCL water pixels may include bridge/bank mixtures at 20m resolution.",
                              "This dataset demonstrates hybridization, not dual-branch fusion, prediction accuracy or cross-region generalization."]}
    out.mkdir(parents=True, exist_ok=False)
    for name in ("matches", "ground", "satellite", "exclusions", "candidates"):
        getattr(result, name).to_csv(out / f"{name}.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    events.to_csv(out / "event_outcomes.csv", index=False)
    counts.to_csv(out / "station_summary.csv", index=False)
    save_json(out / "run.json", report)
    lines = ["# Satellite-ground matching demonstration", "",
             f"{len(events)} planned events; {len(paired)} matched events across {len(counts)} stations.", "",
             "| Station | Attempted | Matched | Unmatched |", "|---|---:|---:|---:|"]
    lines.extend(f"| {row.station_name} | {row.attempted} | {row.matched} | {row.unmatched} |" for row in counts.itertuples())
    lines.extend(["", "Open paired.csv: one row per matched event, with CPCB measurements and Sentinel-2 spectral bands.",
                  "Ground BOD is mg/L; fecal coliform is MPN/100mL; satellite bands are scaled surface reflectance.",
                  "event_outcomes.csv covers every attempted event; detailed image rejections are in the extraction audit.csv.",
                  "", "## Limits", "", *[f"- {item}" for item in report["limitations"]]])
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(counts.to_string(index=False))
    print(f"Saved {len(paired)} paired events to {out / 'paired.csv'}")
    if events.extraction_outcome.eq("query_error").any():
        raise ValueError("Partial dataset saved: query errors remain; inspect event_outcomes.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--ground", default="data/interim/cpcb_preparation_001/ground.csv")
    p.add_argument("--manifest", default="data/interim/cpcb_preparation_001/ground.provenance.json")
    p.add_argument("--stations", nargs="+", required=True)
    p.add_argument("--year", type=int, default=2021)
    p.add_argument("--months", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11])
    p.add_argument("--output", required=True)
    p = sub.add_parser("export")
    p.add_argument("--prepared", required=True)
    p.add_argument("--extracted", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        (prepare if args.command == "prepare" else export)(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Hybrid demo failed: {exc}\n")


if __name__ == "__main__":
    main()

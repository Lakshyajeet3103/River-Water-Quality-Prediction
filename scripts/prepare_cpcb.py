"""Prepare user-supplied CPCB exports and a bounded, explicitly provisional pilot."""

import argparse
from datetime import date
from pathlib import Path
import json

from river_water_quality.data.cpcb import PARAMETERS, prepare_cpcb


def main() -> None:
    """Write a new preparation run with audits, provenance, registry, and pilot events."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = prepare_cpcb([Path(p) for p in args.input])
        out = Path(args.output)
        if out.exists():
            raise ValueError("Output already exists; choose a new run directory")
        bod = result.ground.query("parameter == 'bod'").copy()
        pilot = []
        # Pilot names are provisional river hints, not verified station-to-river labels.
        for name in ("GANGA AT VARANASI U/S (ASSIGHAT) U.P", "GANGA AT TRIGHAT (GHAZIPUR) U.P", "GANGA U/S ANOOPSHAHAR"):
            candidates = bod[(bod.station_name == name) & bod.sample_date.str.startswith("2021-")].copy()
            if candidates.empty:
                continue
            candidates["distance_to_pilot_date"] = candidates.sample_date.map(lambda x: abs((date.fromisoformat(x) - date(2021, 3, 15)).days))
            chosen = candidates.sort_values(["distance_to_pilot_date", "sample_date", "observation_id"]).iloc[0]
            pilot.append(chosen.observation_id)
        if not pilot:
            raise ValueError("No selected 2021 pilot events found")
        pilot_ground = result.ground[result.ground.observation_id.isin(pilot)].copy()
        registry = result.ground[["station_id", "station_name", "latitude", "longitude", "river_label"]].drop_duplicates()
        summary = {"source_files": result.sources, "numeric_measurements": len(result.ground),
            "events": int(result.ground.observation_id.nunique()), "stations": int(registry.station_id.nunique()),
            "parameter_counts": result.ground.parameter.value_counts().to_dict(),
            "excluded_cells_or_records": len(result.exclusions), "exclusion_reasons": result.exclusions.reason.value_counts().to_dict(),
            "pilot_events": len(pilot), "limitations": ["Original source URL/license/download date unresolved", "Coordinates treated as WGS84 pending verification", "Day-first timestamp parsing; source timezone and 08:30 meaning unverified", "River membership inferred from names for pilot only", "Numeric does not imply source quality accepted"]}
        provenance = {"dataset_id": "user_supplied_up_biological_cpcb_prepared", "provider": "cpcb",
            "source": "User-supplied local CPCB-labelled CSVs; original URL unresolved; see preparation.json source hashes",
            "access_date": "2026-10-13", "license": "Unverified; must confirm original source terms before research publication",
            "coordinate_reference_system": "EPSG:4326", "coordinate_assumption": "Header coordinates assumed WGS84; unverified",
            "timestamp_convention": "DD-MM-YYYY HH:MM parsed as displayed calendar date; timezone/08:30 meaning unverified",
            "units": dict(PARAMETERS.values()), "column_mapping": {}, "quality_flags": {"source_not_reported": "No source QC field; numeric parsing only, not quality acceptance"},
            "identifier_rule": "Event: raw-file SHA256 prefix + CSV record; station: name and coordinates SHA256 prefix",
            "inspection_date_note": "access_date is local inspection date, not original download date"}
        out.mkdir(parents=True, exist_ok=False)
        result.ground.to_csv(out / "ground.csv", index=False)
        result.exclusions.to_csv(out / "exclusions.csv", index=False)
        registry.to_csv(out / "stations.csv", index=False)
        pilot_ground.to_csv(out / "pilot_ground.csv", index=False)
        for name in ("ground.provenance.json", "pilot_ground.provenance.json"):
            (out / name).write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        (out / "preparation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

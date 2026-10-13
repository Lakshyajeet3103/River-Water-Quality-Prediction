"""Resume bounded extraction at existing stations until 100 total pairs are saved."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import ee
import pandas as pd

from river_water_quality.data.ingestion import load_dataset
from river_water_quality.data.hybrid import paired_table
from river_water_quality.data.matching import MatchingSettings, match_observations
from river_water_quality.data.schemas import validate_satellite
from river_water_quality.data.sentinel import extract_event, validate_extraction_settings
from river_water_quality.utils.config import load_yaml

OUT = Path("outputs/matches_expansion_100_20261014")
TARGET = 86  # 14 original + 50 previous additions + 36 requested additions.


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    source = load_dataset("data/interim/cpcb_preparation_001/ground.csv",
                          "data/interim/cpcb_preparation_001/ground.provenance.json", kind="cpcb")
    original_path = Path("outputs/hybrid_demo_001/paired.csv")
    original = pd.read_csv(original_path)
    settings = load_yaml("configs/extraction_pilot.yaml")
    settings["max_events"] = 200
    validate_extraction_settings(settings)
    matching = MatchingSettings.from_config(load_yaml("configs/pilot_matching.yaml"))
    ground = source.data
    values = ground.pivot(index="observation_id", columns="parameter", values="value")
    eligible = values.dropna(subset=["bod", "fecal_coliform"]).index
    events = ground[ground.observation_id.isin(eligible)
                    & ground.station_id.isin(original.station_id.unique())][
        ["observation_id", "station_id", "station_name", "sample_date", "latitude", "longitude"]].drop_duplicates()
    events = events[~events.observation_id.isin(original.observation_id)].copy()
    # Spread attempts across stations, starting with the latest dates; no target-value ranking.
    events = events.sort_values(["station_id", "sample_date", "observation_id"], ascending=[True, False, True])
    events["station_order"] = events.groupby("station_id").cumcount()
    events = events.sort_values(["station_order", "station_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUT / "planned_events.csv", index=False)
    state_path = OUT / "checkpoint.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"attempts": [], "audits": [], "features": [], "pairs": []}
    signature = {"source_sha256": source.source_sha256, "manifest_sha256": source.manifest_sha256,
                 "original_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(), "settings": settings}
    if "signature" in state and state["signature"] != signature:
        raise ValueError("Resume inputs/settings changed")
    state["signature"] = signature
    ee.Initialize(project=settings["project_id"])
    completed = {a["observation_id"] for a in state["attempts"] if a["status"] != "query_error"}
    for event in events.to_dict("records"):
        if len(state["pairs"]) >= TARGET:
            break
        if event["observation_id"] in completed:
            continue
        print(f"Attempt {len(state['attempts']) + 1}: {event['station_name']} {event['sample_date']} ({len(state['pairs'])}/{TARGET} new matches)", flush=True)
        rows, logs = [], []
        try:
            rows, logs = extract_event(ee, event, settings)
            if rows:
                spectral = pd.DataFrame(rows)
                validate_satellite(spectral)
                event_ground = ground[ground.observation_id.eq(event["observation_id"])].copy()
                result = match_observations(event_ground, spectral, matching)
                paired = paired_table(result)
                if len(paired) != 1:
                    raise ValueError("Selected extraction did not produce exactly one matched pair")
                state["features"].extend(rows)
                state["pairs"].extend(paired.to_dict("records"))
            status = logs[-1]["status"] if logs else "no_audit"
        except Exception as exc:
            status = "query_error"
            logs.append({"observation_id": event["observation_id"], "status": status,
                         "error_type": type(exc).__name__, "message": str(exc)[:300]})
        state["audits"].extend(logs)
        state["attempts"].append({**event, "status": status})
        write_json(state_path, state)
        print(f"  {status}; now {len(state['pairs'])}/{TARGET} new matches", flush=True)
        save_outputs(state, original, source)
    save_outputs(state, original, source)
    if len(state["pairs"]) != TARGET:
        raise RuntimeError(f"Only {len(state['pairs'])} new matches; eligible events exhausted")


def save_outputs(state, original, source):
    new = pd.DataFrame(state["pairs"])
    pd.DataFrame(state["attempts"]).to_csv(OUT / "event_outcomes.csv", index=False)
    pd.DataFrame(state["audits"]).to_csv(OUT / "extraction_audit.csv", index=False)
    if new.empty:
        return
    if new.observation_id.duplicated().any() or new.observation_id.isin(original.observation_id).any():
        raise ValueError("Duplicate/new-original overlap")
    combined = pd.concat([original, new], ignore_index=True).sort_values(["station_id", "sample_date"])
    new.to_csv(OUT / "new_paired.csv", index=False)
    combined.to_csv(OUT / "paired.csv", index=False)
    source.data[source.data.observation_id.isin(new.observation_id)].to_csv(OUT / "new_ground.csv", index=False)
    write_json(OUT / "new_ground.provenance.json", source.provenance)
    pd.DataFrame(state["features"]).drop_duplicates(["station_id", "scene_id", "band"]).to_csv(OUT / "new_satellite_features.csv", index=False)
    report = {"target_additional_matches": TARGET, "new_matches": len(new), "original_matches": len(original),
              "combined_matches": len(combined), "attempts": len(state["attempts"]),
              "status_counts": pd.Series([x["status"] for x in state["attempts"]]).value_counts().to_dict(),
              "by_station": new.groupby(["station_id", "station_name"]).size().reset_index(name="new_matches").to_dict("records"),
              "date_range": [new.sample_date.min(), new.sample_date.max()], "inputs": state["signature"],
              "updated_utc": datetime.now(timezone.utc).isoformat(), "earthengine_version": ee.__version__,
              "selection": f"Complete BOD/fecal events at original three stations; newest-first round robin; stop after {TARGET} additional successes; no BOD-value filtering",
              "limitations": ["Existing source provenance/timezone/coordinate assumptions remain provisional.",
                              "Retrospective +/-7 day matching is estimation, not forecasting.",
                              "Shared satellite scenes must remain in one fold during future training.",
                              "SCL water classification and bank/bridge contamination remain pilot limitations."]}
    write_json(OUT / "run.json", report)


if __name__ == "__main__":
    main()

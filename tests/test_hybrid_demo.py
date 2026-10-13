"""Synthetic checks for date selection and event-level hybridization."""

import pandas as pd
import pytest

from river_water_quality.data.hybrid import choose_events, paired_table
from river_water_quality.data.matching import MatchingSettings, match_observations


@pytest.fixture
def ground():
    rows = []
    for index, day in enumerate(("2021-03-14", "2021-03-16", "2021-05-15")):
        for parameter, value, unit in (("bod", 2.0 + index, "mg/L"), ("fecal_coliform", 100 + index, "MPN/100mL")):
            rows.append({"observation_id": f"SYNTHETIC_{index}", "station_id": "001", "station_name": "SYNTHETIC_STATION",
                         "sample_date": day, "latitude": 25.0, "longitude": 83.0, "parameter": parameter,
                         "value": value, "unit": unit, "quality_flag": "source_not_reported"})
    return pd.DataFrame(rows)


def test_selection_uses_dates_not_target_magnitude(ground):
    first, log = choose_events(ground, ["001"], 2021, [3, 5, 7])
    assert set(first.observation_id) == {"SYNTHETIC_0", "SYNTHETIC_2"}
    assert len(first) == 4  # Both measured parameters, not just the BOD filter.
    assert log.status.tolist() == ["selected", "selected", "no_ground_observation"]
    changed = ground.copy()
    changed.loc[changed.parameter == "bod", "value"] = 9999
    other, _ = choose_events(changed, ["001"], 2021, [3, 5, 7])
    assert other.observation_id.tolist() == first.observation_id.tolist()


@pytest.mark.parametrize("stations,months", [(["bad"], [3]), (["001", "001"], [3]),
                                           (["001"], [0]), (["001"], [3, 3])])
def test_invalid_selection_is_rejected(ground, stations, months):
    with pytest.raises(ValueError):
        choose_events(ground, stations, 2021, months)


def test_wide_pairs_preserve_measurements_without_cartesian_duplication(ground):
    selected, _ = choose_events(ground, ["001"], 2021, [3, 5])
    satellite = pd.DataFrame([{"scene_id": f"SYNTHETIC_SCENE_{index}", "station_id": "001",
        "acquired_at": f"{day}T05:00:00+00:00", "latitude": 25.0, "longitude": 83.0,
        "band": band, "value": value, "unit": "reflectance", "product_id": "SYNTHETIC_PRODUCT",
        "cloud_fraction": 0.0, "water_fraction": 0.8, "pixel_count": 8}
        for index, day in enumerate(("2021-03-14", "2021-05-15"))
        for band, value in (("B2", 0.1 + index), ("B3", 0.2 + index))])
    settings = MatchingSettings("Asia/Kolkata", 60, 0.2, 0.2, ("source_not_reported",), ("B2", "B3"), "retrospective")
    result = match_observations(selected, satellite, settings)
    pairs = paired_table(result)
    assert len(pairs) == 2
    assert pairs.observation_id.is_unique
    assert pairs.ground_bod.tolist() == [2.0, 4.0]
    assert pairs.ground_fecal_coliform.tolist() == [100, 102]
    assert pairs.satellite_B2.tolist() == [0.1, 1.1]
    assert pairs.pixel_count.tolist() == [8, 8]


def test_cli_prepare_export_and_hash_guard(ground, tmp_path):
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    import sys
    from test_ingestion import files
    raw = tmp_path / "raw"
    raw.mkdir()
    csv, manifest = files(raw, ground)
    root = Path(__file__).parents[1]
    prepared, extracted, output = (tmp_path / name for name in ("prepared", "extracted", "output"))
    command = [sys.executable, "scripts/hybrid_demo.py", "prepare", "--ground", str(csv), "--manifest", str(manifest),
               "--stations", "001", "--months", "3", "5", "--output", str(prepared)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    extracted.mkdir()
    features = pd.DataFrame([{"scene_id": "SYNTHETIC_SCENE", "station_id": "001", "acquired_at": "2021-03-14T05:00:00Z",
        "latitude": 25.0, "longitude": 83.0, "band": band, "value": 0.1, "unit": "reflectance",
        "product_id": "SYNTHETIC_PRODUCT", "cloud_fraction": 0.0, "water_fraction": 0.8, "pixel_count": 8}
        for band in ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")])
    feature_csv, feature_manifest = files(extracted, features, kind="sentinel2")
    feature_csv.rename(extracted / "features.csv")
    feature_manifest.rename(extracted / "features.provenance.json")
    pd.DataFrame({"observation_id": ["SYNTHETIC_0", "SYNTHETIC_2"], "status": ["selected", "no_usable_scene"]}).to_csv(extracted / "audit.csv", index=False)
    report = {"ground_source_sha256": hashlib.sha256((prepared / "ground.csv").read_bytes()).hexdigest(),
              "ground_manifest_sha256": hashlib.sha256((prepared / "ground.provenance.json").read_bytes()).hexdigest()}
    (extracted / "run.json").write_text(json.dumps(report))
    command = [sys.executable, "scripts/hybrid_demo.py", "export", "--prepared", str(prepared), "--extracted", str(extracted), "--output", str(output)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    paired = pd.read_csv(output / "paired.csv", dtype={"station_id": str})
    assert paired.station_id.tolist() == ["001"]
    assert paired.ground_bod.tolist() == [2.0]
    assert len(pd.read_csv(output / "event_outcomes.csv")) == 2
    report["ground_source_sha256"] = "wrong"
    (extracted / "run.json").write_text(json.dumps(report))
    command[-1] = str(tmp_path / "bad_export")
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode != 0 and "hash differs" in result.stderr
    assert not (tmp_path / "bad_export").exists()

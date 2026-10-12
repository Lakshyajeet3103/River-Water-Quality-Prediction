"""Synthetic software tests, never water-quality experimental evidence."""

from dataclasses import asdict, replace
from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import pytest
import yaml

from test_ingestion import ground, satellite, files  # shared synthetic fixtures
from river_water_quality.data.matching import MatchingSettings, distance_m, match_observations
from river_water_quality.utils.config import load_config


@pytest.fixture
def settings():
    return MatchingSettings(
        timezone="UTC", max_distance_m=60, max_cloud_fraction=0.2,
        min_water_fraction=0.8, accepted_ground_flags=("SYNTHETIC_TEST_ONLY",),
        required_bands=("TEST_BAND_A", "TEST_BAND_B"), direction="retrospective",
    )


def another_scene(satellite, name, timestamp, **values):
    data = satellite.copy(deep=True)
    data["scene_id"] = name
    data["acquired_at"] = timestamp
    for key, value in values.items():
        data[key] = value
    return data


def test_same_date_preserves_relational_rows_and_inputs(ground, satellite, settings):
    before_ground, before_satellite = ground.copy(deep=True), satellite.copy(deep=True)
    result = match_observations(ground, satellite, settings)
    assert len(result.matches) == 1
    assert len(result.ground) == len(result.satellite) == 2
    assert result.matches.temporal_offset_days.tolist() == [0]
    assert result.matches.search_tier.tolist() == ["primary"]
    assert result.exclusions.empty
    assert result.summary["matched_ground_rows"] + result.summary["excluded_ground_rows"] == len(ground)
    json.dumps(result.summary)
    pd.testing.assert_frame_equal(before_ground, ground)
    pd.testing.assert_frame_equal(before_satellite, satellite)


@pytest.mark.parametrize("timestamp,offset,tier", [
    ("2019-12-29T12:00:00Z", -3, "primary"),
    ("2020-01-04T12:00:00Z", 3, "primary"),
    ("2019-12-25T12:00:00Z", -7, "fallback"),
    ("2020-01-08T12:00:00Z", 7, "fallback"),
])
def test_inclusive_windows(ground, satellite, settings, timestamp, offset, tier):
    satellite["acquired_at"] = timestamp
    result = match_observations(ground, satellite, settings)
    assert result.matches.temporal_offset_days.tolist() == [offset]
    assert result.matches.search_tier.tolist() == [tier]


@pytest.mark.parametrize("field,value,status", [
    ("acquired_at", "2020-01-09T00:00:00Z", "outside_temporal_window"),
    ("latitude", 10.01, "outside_spatial_tolerance"),
    ("cloud_fraction", 0.21, "cloud_fraction"),
    ("water_fraction", 0.79, "water_fraction"),
])
def test_exclusion_reasons(ground, satellite, settings, field, value, status):
    satellite[field] = value
    result = match_observations(ground, satellite, settings)
    assert result.matches.empty and result.ground.empty and result.satellite.empty
    assert result.exclusions.reason.tolist() == ["no_usable_scene"] * 2
    assert result.candidates.status.tolist() == [status]
    assert result.summary["exclusion_rate_rows"] == 1
    assert "observation_id" in result.satellite.columns


def test_station_ids_do_not_cross_match(ground, satellite, settings):
    satellite["station_id"] = "OTHER_TEST_STATION"
    result = match_observations(ground, satellite, settings)
    assert result.candidates.empty
    assert result.exclusions.reason.tolist() == ["no_station_candidate"] * 2


def test_required_bands(ground, satellite, settings):
    result = match_observations(ground, satellite.iloc[:1], settings)
    assert result.candidates.status.tolist() == ["missing_required_bands"]


def test_quality_filters_run_before_selection(ground, satellite, settings):
    satellite["cloud_fraction"] = 0.9
    fallback = another_scene(satellite, "TEST_FALLBACK", "2020-01-06T10:00:00Z", cloud_fraction=0.1)
    result = match_observations(ground, pd.concat([satellite, fallback]), settings)
    assert result.matches.scene_id.tolist() == ["TEST_FALLBACK"]
    assert result.matches.search_tier.tolist() == ["fallback"]


def test_nearest_eligible_scene_and_deterministic_ties(ground, satellite, settings):
    second = another_scene(satellite, "A_TEST_SCENE", "2020-01-01T10:00:00Z")
    combined = pd.concat([satellite, second], ignore_index=True)
    first = match_observations(ground, combined, settings)
    shuffled = match_observations(ground, combined.sample(frac=1, random_state=3), settings)
    assert first.matches.scene_id.tolist() == ["A_TEST_SCENE"]
    pd.testing.assert_frame_equal(first.matches, shuffled.matches)
    assert sorted(first.candidates.status) == ["eligible_not_selected", "selected"]


def test_signed_prior_offset_and_same_day_rejection(ground, satellite, settings):
    previous = another_scene(satellite, "TEST_PREVIOUS", "2019-12-31T10:00:00Z")
    future = another_scene(satellite, "TEST_FUTURE", "2020-01-02T10:00:00Z")
    result = match_observations(ground, pd.concat([satellite, previous, future]), replace(settings, direction="prior_dates_only"))
    assert result.matches.temporal_offset_days.tolist() == [-1]
    assert result.candidates.status.tolist().count("not_prior_date") == 2


def test_timezone_calendar_boundary(ground, satellite, settings):
    satellite["acquired_at"] = "2019-12-31T20:00:00Z"
    result = match_observations(ground, satellite, replace(settings, timezone="Asia/Kolkata"))
    assert result.matches.satellite_local_date.tolist() == ["2020-01-01"]
    assert result.matches.temporal_offset_days.tolist() == [0]


def test_partial_ground_quality_exclusion(ground, satellite, settings):
    ground.loc[0, "quality_flag"] = "TEST_REJECTED"
    result = match_observations(ground, satellite, settings)
    assert len(result.ground) == 1
    assert result.exclusions.reason.tolist() == ["ground_quality_flag"]
    assert result.summary["exclusion_rate_rows"] == 0.5


def test_all_ground_flags_rejected(ground, satellite, settings):
    ground["quality_flag"] = "TEST_REJECTED"
    result = match_observations(ground, satellite, settings)
    assert result.matches.empty
    assert result.exclusions.reason.tolist() == ["ground_quality_flag"] * 2


def test_reused_scene_keeps_event_links(ground, satellite, settings):
    replicate = ground.copy()
    replicate["observation_id"] = "TEST_REPLICATE"
    result = match_observations(pd.concat([ground, replicate]), satellite, settings)
    assert len(result.matches) == 2
    assert len(result.satellite) == 4
    assert result.satellite.observation_id.nunique() == 2


@pytest.mark.parametrize("field,value", [
    ("timezone", None), ("timezone", "INVALID/TEST_ZONE"),
    ("max_distance_m", -1), ("max_distance_m", float("inf")),
    ("max_cloud_fraction", None), ("max_cloud_fraction", 1.1),
    ("min_water_fraction", True), ("accepted_ground_flags", []),
    ("required_bands", ["TEST_BAND", "TEST_BAND"]),
    ("direction", "future_only"), ("primary_window_days", -1),
    ("primary_window_days", True), ("max_window_days", 2),
])
def test_invalid_settings(settings, field, value):
    with pytest.raises(ValueError, match="matching"):
        replace(settings, **{field: value})


def test_default_matching_settings_require_study_choices():
    config = load_config(Path(__file__).parents[1] / "configs/base.yaml")
    with pytest.raises(ValueError, match="timezone"):
        MatchingSettings.from_config(config)


def test_distance_identity_and_dateline():
    assert distance_m(10, 20, 10, 20) == 0
    assert 200 < distance_m(0, 179.999, 0, -179.999) < 230


def test_matching_cli_outputs_and_refuses_overwrite(tmp_path, ground, satellite, settings):
    ground_dir, sat_dir = tmp_path / "ground", tmp_path / "satellite"
    ground_dir.mkdir()
    sat_dir.mkdir()
    ground_paths = files(ground_dir, ground)
    satellite_paths = files(sat_dir, satellite, "sentinel2")
    config = load_config(Path(__file__).parents[1] / "configs/base.yaml")
    config["matching"] = asdict(settings)
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "outputs"
    command = [sys.executable, str(Path(__file__).parents[1] / "scripts/match_datasets.py"),
               *map(str, ground_paths), *map(str, satellite_paths),
               "--config", str(config_path), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert {p.name for p in output.iterdir()} == {"matches.csv", "ground.csv", "satellite.csv", "exclusions.csv", "candidates.csv", "run.json"}
    report = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert report["summary"]["matched_events"] == 1
    assert report["sources"]["cpcb"]["source_sha256"]
    previous = (output / "run.json").read_bytes()
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode == 1
    assert (output / "run.json").read_bytes() == previous

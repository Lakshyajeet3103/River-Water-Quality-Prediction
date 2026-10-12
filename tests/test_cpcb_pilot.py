"""Synthetic adapter/quality tests; no fabricated provider observations."""

from pathlib import Path

import pandas as pd
import pytest

from river_water_quality.data.cpcb import PARAMETERS, prepare_cpcb
from river_water_quality.data.sentinel import candidate_reason, validate_extraction_settings
from river_water_quality.utils.config import load_yaml


def source(tmp_path, **changes):
    row = {"Station": "SYNTHETIC_TEST_STATION", "Latitude": "10", "Longitude": "20",
        "Data Acquisition Time": "15-03-2021 08:30", **{column: "" for column in PARAMETERS}}
    row["Biochemical Oxygen Demand (mg/L)"] = "2.5"
    row.update(changes)
    path = tmp_path / "synthetic.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def test_numeric_preservation_and_missing_audit(tmp_path):
    path = source(tmp_path)
    before = path.read_bytes()
    result = prepare_cpcb([path])
    assert result.ground.value.tolist() == [2.5]
    assert result.ground.raw_measurement.tolist() == ["2.5"]
    assert result.ground.sample_date.tolist() == ["2021-03-15"]
    assert result.ground.quality_flag.tolist() == ["source_not_reported"]
    assert len(result.exclusions) == 3
    assert path.read_bytes() == before
    assert result.sources[0]["sha256"]


def test_text_is_excluded_not_converted(tmp_path):
    path = source(tmp_path, **{"Fecal Coliform (MPN/100mL)": "Clear"})
    result = prepare_cpcb([path])
    assert result.exclusions.query("reason == 'non_numeric_or_nonfinite'").raw_value.tolist() == ["Clear"]


def test_same_station_stable_across_files(tmp_path):
    first = source(tmp_path)
    second = tmp_path / "second.csv"
    frame = pd.read_csv(first, dtype=str, keep_default_na=False)
    frame["Data Acquisition Time"] = "16-03-2021 08:30"
    frame.to_csv(second, index=False)
    result = prepare_cpcb([first, second])
    assert result.ground.station_id.nunique() == 1
    assert result.ground.observation_id.nunique() == 2


def test_no_numeric_observations_does_not_manufacture_data(tmp_path):
    path = source(tmp_path, **{"Biochemical Oxygen Demand (mg/L)": "Clear"})
    with pytest.raises(ValueError, match="No numeric"):
        prepare_cpcb([path])


@pytest.fixture
def settings():
    return load_yaml(Path(__file__).parents[1] / "configs/extraction_pilot.yaml")


def test_pilot_settings_valid(settings):
    validate_extraction_settings(settings)


@pytest.mark.parametrize("field,value", [("scale_m", 10), ("max_events", 0), ("max_cloud_fraction", 1.1), ("max_window_days", 1)])
def test_invalid_pilot_choices(settings, field, value):
    settings[field] = value
    with pytest.raises(ValueError):
        validate_extraction_settings(settings)


@pytest.mark.parametrize("change,reason", [
    ({"covered": 10, "cloud": 0, "water": 8, "usable": 8}, "insufficient_scene_coverage"),
    ({"cloud": 30, "water": 60}, "cloud_fraction"),
    ({"water": 1, "usable": 1}, "water_fraction"),
    ({"usable": 0}, "insufficient_valid_water_pixels"),
])
def test_quality_exclusions(settings, change, reason):
    stats = {"total": 100, "covered": 100, "cloud": 10, "water": 80, "usable": 10}
    stats.update(change)
    assert candidate_reason(stats, {b: 0.1 for b in settings["bands"]}, settings["bands"], settings) == reason


def test_missing_band_not_substituted(settings):
    stats = {"total": 100, "covered": 100, "cloud": 10, "water": 80, "usable": 10}
    assert candidate_reason(stats, {}, settings["bands"], settings) == "missing_spectral_value"
    assert candidate_reason(stats, {b: 0.1 for b in settings["bands"]}, settings["bands"], settings) is None


def test_fractional_counts_cannot_be_reported_as_integer_pixels(settings):
    stats = {"total": 100, "covered": 100, "cloud": 10, "water": 80, "usable": 10.2}
    assert candidate_reason(stats, {b: 0.1 for b in settings["bands"]}, settings["bands"], settings) == "invalid_pixel_counts"


@pytest.mark.parametrize("field,value", [("covered", None), ("cloud", "0"), ("cloud", True),
    ("total", float("nan")), ("water", float("inf")), ("usable", -1),
    ("covered", 101), ("usable", 81), ("cloud", 21), ("water", 101)])
def test_invalid_counts_fail_closed_before_fractions(settings, field, value):
    stats = {"total": 100, "covered": 100, "cloud": 10, "water": 80, "usable": 10}
    stats[field] = value
    assert candidate_reason(stats, {}, settings["bands"], settings) == "invalid_pixel_counts"


@pytest.mark.parametrize("total", [0, 100])
def test_zero_coverage_does_not_divide_by_zero(settings, total):
    settings["min_coverage_fraction"] = 0
    stats = {"total": total, "covered": 0, "cloud": 0, "water": 0, "usable": 0}
    assert candidate_reason(stats, {}, settings["bands"], settings) == "insufficient_scene_coverage"


@pytest.mark.parametrize("value", [None, "0.1", True, float("nan"), float("inf")])
def test_invalid_spectra_are_rejected_without_type_errors(settings, value):
    stats = {"total": 100, "covered": 100, "cloud": 10, "water": 80, "usable": 10}
    means = {b: 0.1 for b in settings["bands"]}
    means[settings["bands"][0]] = value
    assert candidate_reason(stats, means, settings["bands"], settings) == "missing_spectral_value"

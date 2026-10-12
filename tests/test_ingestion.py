"""Synthetic software fixtures only; these are not provider observations."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from river_water_quality.data.ingestion import load_dataset
from river_water_quality.data.schemas import validate_ground, validate_satellite


@pytest.fixture
def ground():
    return pd.DataFrame({
        "observation_id": ["TEST_SAMPLE_001", "TEST_SAMPLE_001"],
        "station_id": ["001", "001"], "sample_date": ["2020-01-01"] * 2,
        "latitude": [10.0] * 2, "longitude": [20.0] * 2,
        "parameter": ["TEST_PARAMETER_A", "TEST_PARAMETER_B"],
        "value": [1.0, 2.0], "unit": ["TEST_UNIT"] * 2,
        "quality_flag": ["SYNTHETIC_TEST_ONLY"] * 2,
    })


@pytest.fixture
def satellite():
    return pd.DataFrame({
        "scene_id": ["TEST_SCENE"] * 2, "station_id": ["001"] * 2,
        "acquired_at": ["2020-01-01T10:00:00Z"] * 2,
        "latitude": [10.0] * 2, "longitude": [20.0] * 2,
        "band": ["TEST_BAND_A", "TEST_BAND_B"], "value": [0.1, 0.2],
        "unit": ["TEST_UNIT"] * 2, "product_id": ["TEST_PRODUCT"] * 2,
        "cloud_fraction": [0.1] * 2, "water_fraction": [0.8] * 2,
        "pixel_count": [4] * 2,
    })


def files(tmp_path, data, kind="cpcb", **overrides):
    feature = "band" if kind == "sentinel2" else "parameter"
    manifest = {
        "dataset_id": "SYNTHETIC_SOFTWARE_TEST_ONLY", "provider": kind,
        "source": "Synthetic unit-test fixture; not scientific evidence",
        "access_date": "2020-01-01", "license": "Software test fixture only",
        "coordinate_reference_system": "EPSG:4326",
        "timestamp_convention": "Synthetic date/UTC convention for software testing",
        "units": dict(zip(data[feature], data["unit"])),
    }
    manifest.update(overrides)
    csv_path = tmp_path / "synthetic.csv"
    metadata_path = tmp_path / "synthetic.provenance.json"
    data.to_csv(csv_path, index=False)
    metadata_path.write_text(json.dumps(manifest), encoding="utf-8")
    return csv_path, metadata_path


@pytest.mark.parametrize("kind", ["cpcb", "usgs"])
def test_ground_adapter_preserves_ids_flags_and_source(tmp_path, ground, kind):
    paths = files(tmp_path, ground, kind)
    original = paths[0].read_bytes()
    result = load_dataset(*paths, kind=kind)
    assert result.data["station_id"].tolist() == ["001", "001"]
    assert result.data["quality_flag"].tolist() == ground["quality_flag"].tolist()
    assert result.source_sha256 == hashlib.sha256(original).hexdigest()
    assert result.manifest_sha256 == hashlib.sha256(paths[1].read_bytes()).hexdigest()
    assert paths[0].read_bytes() == original
    assert len(result.data) == 2


def test_satellite_adapter(tmp_path, satellite):
    result = load_dataset(*files(tmp_path, satellite, "sentinel2"), kind="sentinel2")
    assert result.data["value"].tolist() == [0.1, 0.2]


def test_explicit_column_mapping(tmp_path, ground):
    paths = files(tmp_path, ground, column_mapping={"Site": "station_id"})
    ground.rename(columns={"station_id": "Site"}).to_csv(paths[0], index=False)
    assert load_dataset(*paths, kind="cpcb").data.station_id.tolist() == ["001", "001"]


@pytest.mark.parametrize("field,value,message", [
    ("source", "", "source"), ("access_date", "2020-02-30", "access_date"),
    ("provider", "usgs", "provider"),
    ("coordinate_reference_system", "unknown", "EPSG:4326"),
    ("units", {}, "units"), ("column_mapping", {"absent": "station_id"}, "not found"),
    ("column_mapping", {"station_id": "observation_id"}, "duplicate column"),
])
def test_invalid_provenance(tmp_path, ground, field, value, message):
    with pytest.raises(ValueError, match=message):
        load_dataset(*files(tmp_path, ground, **{field: value}), kind="cpcb")


def test_unit_mismatch(tmp_path, ground):
    paths = files(tmp_path, ground, units={"TEST_PARAMETER_A": "WRONG_UNIT", "TEST_PARAMETER_B": "TEST_UNIT"})
    with pytest.raises(ValueError, match="does not match provenance"):
        load_dataset(*paths, kind="cpcb")


def test_censored_value_is_not_fabricated(tmp_path, ground):
    ground["value"] = ["<1", "2"]
    with pytest.raises(ValueError, match="Invalid numeric values in value"):
        load_dataset(*files(tmp_path, ground), kind="cpcb")


@pytest.mark.parametrize("column,value,message", [
    ("latitude", 91, "latitude"), ("longitude", -181, "longitude"),
    ("value", float("inf"), "finite"), ("value", float("nan"), "Missing required"),
    ("sample_date", "2020-02-30", "sample_date"),
    ("sample_date", "01/02/2020", "sample_date"),
    ("sample_date", "2020-01-01T00:00:00Z", "sample_date"),
    ("quality_flag", "", "Missing required"),
])
def test_invalid_ground(ground, column, value, message):
    ground[column] = value
    with pytest.raises(ValueError, match=message):
        validate_ground(ground)


def test_duplicate_parameter_rejected_but_real_replicate_id_allowed(ground):
    data = pd.concat([ground, ground.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate observations"):
        validate_ground(data)
    data.loc[2, "observation_id"] = "TEST_SAMPLE_002"
    validate_ground(data)


def test_conflicting_ground_metadata(ground):
    ground.loc[1, "station_id"] = "OTHER_TEST_STATION"
    with pytest.raises(ValueError, match="Conflicting"):
        validate_ground(ground)


@pytest.mark.parametrize("column,value,message", [
    ("acquired_at", "2020-01-01", "UTC offset"),
    ("acquired_at", "2020-01-01T10:00:00", "UTC offset"),
    ("cloud_fraction", 1.1, "cloud_fraction"),
    ("water_fraction", -0.1, "water_fraction"),
    ("pixel_count", 0, "positive integers"),
    ("pixel_count", 1.5, "positive integers"),
])
def test_invalid_satellite(satellite, column, value, message):
    satellite[column] = value
    with pytest.raises(ValueError, match=message):
        validate_satellite(satellite)


def test_conflicting_satellite_metadata(satellite):
    satellite.loc[1, "product_id"] = "OTHER_TEST_PRODUCT"
    with pytest.raises(ValueError, match="Conflicting"):
        validate_satellite(satellite)


def test_satellite_duplicate_band(satellite):
    satellite.loc[1, "band"] = satellite.loc[0, "band"]
    with pytest.raises(ValueError, match="Duplicate observations"):
        validate_satellite(satellite)


def test_missing_column(tmp_path, ground):
    paths = files(tmp_path, ground)
    ground.drop(columns="latitude").to_csv(paths[0], index=False)
    with pytest.raises(ValueError, match="Missing required columns.*latitude"):
        load_dataset(*paths, kind="cpcb")


@pytest.mark.parametrize("problem", ["duplicate_header", "short_row", "blank_row"])
def test_malformed_csv_not_silently_changed(tmp_path, ground, problem):
    paths = files(tmp_path, ground)
    content = paths[0].read_text(encoding="utf-8")
    if problem == "duplicate_header":
        content = content.replace("observation_id,station_id", "station_id,station_id", 1)
    elif problem == "short_row":
        content += "INCOMPLETE_RECORD\n"
    else:
        content += "\n"
    paths[0].write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="unique column|fields"):
        load_dataset(*paths, kind="cpcb")


def test_extra_columns_preserved(tmp_path, ground):
    ground["source_notes"] = "SOURCE_TEST_NOTE"
    result = load_dataset(*files(tmp_path, ground), kind="cpcb")
    assert result.data.source_notes.tolist() == ground.source_notes.tolist()


@pytest.mark.parametrize("valid", [True, False])
def test_validation_cli(tmp_path, ground, valid):
    paths = files(tmp_path, ground)
    if not valid:
        ground["value"] = "<1"
        ground.to_csv(paths[0], index=False)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts/validate_dataset.py"),
         "cpcb", str(paths[0]), str(paths[1])],
        capture_output=True, text=True, check=False,
    )
    if valid:
        assert result.returncode == 0
        assert "Validated 2 rows" in result.stdout
        assert "Source SHA256:" in result.stdout
    else:
        assert result.returncode == 1
        assert "Invalid numeric values in value" in result.stderr
        assert "Traceback" not in result.stderr

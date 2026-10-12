"""Synthetic software checks; fitted values and metrics are not research evidence."""

from dataclasses import asdict, replace
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("sklearn", reason="Install requirements-ml.txt for Phase 3 tests")

from river_water_quality.data.matching import MatchingSettings, match_observations
from river_water_quality.training.dataset import build_baseline_dataset
from river_water_quality.training.splits import assign_splits
from river_water_quality.training.baselines import train_baselines, regression_metrics
from river_water_quality.utils.config import load_config
from test_ingestion import files


@pytest.fixture
def matched():
    ground, satellite = [], []
    for i in range(9):
        event, station = f"TEST_EVENT_{i}", f"00{i // 3}"
        sample_date = f"2020-01-{i + 1:02}"
        for parameter, value in (("TEST_TARGET", 2.0 * i + 1), ("TEST_INPUT", float(i))):
            ground.append({"observation_id": event, "station_id": station,
                "sample_date": sample_date, "latitude": 10.0, "longitude": 20.0,
                "parameter": parameter, "value": value, "unit": "TEST_UNIT",
                "quality_flag": "SYNTHETIC_TEST_ONLY"})
        satellite.append({"station_id": station, "scene_id": f"TEST_SCENE_{i}",
            "acquired_at": f"{sample_date}T10:00:00Z", "latitude": 10.0, "longitude": 20.0,
            "band": "TEST_BAND", "value": i / 10.0, "unit": "TEST_UNIT",
            "product_id": "TEST_PRODUCT", "cloud_fraction": 0.1,
            "water_fraction": 0.9, "pixel_count": 4})
    settings = MatchingSettings(timezone="UTC", max_distance_m=60, max_cloud_fraction=0.2,
        min_water_fraction=0.8, accepted_ground_flags=("SYNTHETIC_TEST_ONLY",),
        required_bands=("TEST_BAND",), direction="retrospective")
    return match_observations(pd.DataFrame(ground), pd.DataFrame(satellite), settings), settings


def build(result, **overrides):
    options = {"target_parameter": "TEST_TARGET", "target_unit": "TEST_UNIT",
        "ground_units": {"TEST_INPUT": "TEST_UNIT"},
        "satellite_units": {"TEST_BAND": "TEST_UNIT"}, "timezone": "UTC"}
    options.update(overrides)
    return build_baseline_dataset(result.matches, result.ground, result.satellite, **options)


@pytest.fixture
def dataset(matched):
    return build(matched[0])


@pytest.fixture
def station_split():
    return {"mode": "station_holdout", "train_stations": ["000"],
        "validation_stations": ["001"], "test_stations": ["002"]}


@pytest.fixture
def temporal_split():
    return {"mode": "chronological", "train_end_date": "2020-01-03",
            "validation_end_date": "2020-01-06", "gap_days": 0}


def test_dataset_views_target_and_metadata_separation(matched):
    result = matched[0]
    before = result.ground.copy(deep=True)
    data = build(result)
    assert len(data.target) == 9
    assert data.ground.columns.tolist() == ["ground::TEST_INPUT"]
    assert data.satellite.columns.tolist() == ["satellite::TEST_BAND"]
    assert data.metadata.station_id.iloc[0] == "000"
    assert data.exclusions.empty
    pd.testing.assert_frame_equal(result.ground, before)


@pytest.mark.parametrize("options,message", [
    ({"ground_units": {"TEST_TARGET": "TEST_UNIT"}}, "Target leakage"),
    ({"target_unit": "OTHER_UNIT"}, "Ground unit mismatch"),
    ({"satellite_units": {"TEST_BAND": "OTHER_UNIT"}}, "Satellite unit mismatch"),
    ({"target_parameter": ""}, "target_parameter"),
    ({"ground_units": {}}, "ground_units"),
])
def test_invalid_feature_choices(matched, options, message):
    with pytest.raises(ValueError, match=message):
        build(matched[0], **options)


@pytest.mark.parametrize("parameter,reason", [("TEST_TARGET", "missing_target"), ("TEST_INPUT", "missing_ground_predictor")])
def test_incomplete_events_are_logged(matched, parameter, reason):
    result = matched[0]
    ground = result.ground[~((result.ground.observation_id == "TEST_EVENT_0") & (result.ground.parameter == parameter))]
    data = build(replace(result, ground=ground))
    assert len(data.target) == 8
    assert data.exclusions.to_dict("records") == [{"observation_id": "TEST_EVENT_0", "reason": reason}]
    assert data.ground.index.equals(data.satellite.index)


def test_mismatched_link_ids_rejected(matched):
    result = matched[0]
    with pytest.raises(ValueError, match="Event IDs"):
        build(replace(result, matches=result.matches.iloc[1:]))


def test_timestamp_local_date_mismatch(matched):
    result = matched[0]
    matches = result.matches.copy()
    matches.loc[0, "satellite_local_date"] = "2020-01-02"
    with pytest.raises(ValueError, match="local date"):
        build(replace(result, matches=matches))


def test_shared_complete_cohort_excludes_missing_band(matched):
    result = matched[0]
    satellite = result.satellite.copy()
    satellite.loc[satellite.observation_id == "TEST_EVENT_0", "band"] = "OTHER_TEST_BAND"
    data = build(replace(result, satellite=satellite))
    assert len(data.ground) == len(data.satellite) == 8
    assert data.exclusions.reason.tolist() == ["missing_satellite_predictor"]


@pytest.mark.parametrize("mode", ["station", "time"])
def test_three_folds_and_no_row_randomization(dataset, station_split, temporal_split, mode):
    splits = assign_splits(dataset.metadata, station_split if mode == "station" else temporal_split)
    assert splits.split.tolist() == ["train"] * 3 + ["validation"] * 3 + ["test"] * 3


def test_disjoint_station_lists(dataset, station_split):
    station_split["test_stations"] = ["000"]
    with pytest.raises(ValueError, match="disjoint"):
        assign_splits(dataset.metadata, station_split)


def test_shared_scene_leakage_blocked(dataset, station_split):
    metadata = dataset.metadata.copy()
    metadata.loc["TEST_EVENT_3", "scene_id"] = "TEST_SCENE_0"
    with pytest.raises(ValueError, match="Shared satellite scene"):
        assign_splits(metadata, station_split)


@pytest.mark.parametrize("alias", [
    "20210316T045659_20210316T050603_T44RPN",
    "COPERNICUS/S2_SR_HARMONIZED/20210316T045659_20210317T050603_T44RQP",
])
def test_overlapping_tile_and_processing_alias_leakage_blocked(dataset, station_split, alias):
    metadata = dataset.metadata.copy()
    metadata.loc["TEST_EVENT_0", "scene_id"] = "20210316T045659_20210316T050603_T44RQP"
    metadata.loc["TEST_EVENT_3", "scene_id"] = alias
    with pytest.raises(ValueError, match="Shared Sentinel-2 acquisition"):
        assign_splits(metadata, station_split)


def test_overlapping_tiles_allowed_within_one_fold(dataset, station_split):
    metadata = dataset.metadata.copy()
    metadata.loc["TEST_EVENT_0", "scene_id"] = "20210316T045659_20210316T050603_T44RQP"
    metadata.loc["TEST_EVENT_1", "scene_id"] = "20210316T045659_20210316T050603_T44RPN"
    assert assign_splits(metadata, station_split).loc["TEST_EVENT_1", "split"] == "train"


def test_distinct_sentinel_acquisitions_allowed_across_folds(dataset, station_split):
    metadata = dataset.metadata.copy()
    metadata.loc["TEST_EVENT_0", "scene_id"] = "20210316T045659_20210316T050603_T44RQP"
    metadata.loc["TEST_EVENT_3", "scene_id"] = "20210321T045701_20210321T050108_T44RQP"
    assert assign_splits(metadata, station_split).loc["TEST_EVENT_3", "split"] == "validation"


def test_boundary_crossing_scene_excluded(dataset, temporal_split):
    metadata = dataset.metadata.copy()
    metadata.loc["TEST_EVENT_2", "satellite_local_date"] = "2020-01-04"
    assignments = assign_splits(metadata, temporal_split)
    assert assignments.loc["TEST_EVENT_2", "reason"] == "temporal_boundary_or_gap"


def test_time_gap_excluded(dataset, temporal_split):
    temporal_split["gap_days"] = 1
    assignments = assign_splits(dataset.metadata, temporal_split)
    assert assignments.loc["TEST_EVENT_3", "split"] == "excluded"
    assert assignments.loc["TEST_EVENT_6", "split"] == "excluded"


def test_unassigned_station_is_audited(dataset, station_split):
    metadata = dataset.metadata.copy()
    metadata.loc["TEST_EVENT_2", "station_id"] = "UNKNOWN_TEST_STATION"
    assignments = assign_splits(metadata, station_split)
    assert assignments.loc["TEST_EVENT_2", "reason"] == "unassigned_station"


def test_too_small_fold_rejected(dataset, station_split):
    with pytest.raises(ValueError, match="at least two"):
        assign_splits(dataset.metadata.iloc[:7], station_split)


def test_unknown_split_mode_rejected(dataset):
    with pytest.raises(ValueError, match="split.mode"):
        assign_splits(dataset.metadata, {"mode": "random_rows"})


def train(data, split):
    return train_baselines(data, split, seed=42, ridge_alpha=0.01, forest_trees=10, forest_min_leaf=1)


def test_train_only_scaling_mean_reference_and_selected_test_models(dataset, station_split):
    result = train(dataset, station_split)
    assert result.metrics.shape[0] == 8
    assert len(result.predictions) == 9
    scaler = result.models["ground"].named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, [1.0])
    assert result.models["mean_reference"].constant_.item() == 3.0
    assert not np.isclose(scaler.mean_[0], dataset.ground.iloc[:, 0].mean())
    assert result.predictions.modality.unique().tolist() == ["reference", "ground", "satellite"]


def test_test_labels_do_not_affect_model_selection(dataset, station_split):
    original = train(dataset, station_split)
    changed_y = dataset.target.copy()
    changed_y.iloc[6:] = [1000.0, -1000.0, 5000.0]
    changed = train(replace(dataset, target=changed_y), station_split)
    pd.testing.assert_series_equal(original.predictions.predicted, changed.predictions.predicted)
    before = original.metrics.query("split == 'validation'").reset_index(drop=True)
    after = changed.metrics.query("split == 'validation'").reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_forest_and_pipeline_reproducibility(dataset, station_split):
    first = train(dataset, station_split)
    second = train(dataset, station_split)
    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)


def test_selected_forest_and_constant_target_metrics(dataset, station_split):
    constant = replace(dataset, target=pd.Series(2.0, index=dataset.target.index, name=dataset.target.name))
    result = train(constant, station_split)
    assert result.models["ground"].random_state == 42
    assert result.metrics.r2.isna().all()
    np.testing.assert_allclose(result.predictions.predicted, 2.0)


def test_undefined_r2_is_explicit():
    metrics = regression_metrics(pd.Series([2.0, 2.0]), np.array([1.0, 3.0]))
    assert metrics == {"mae": 1.0, "rmse": 1.0, "r2": None}


@pytest.mark.parametrize("setting,value", [("ridge_alpha", 0), ("forest_trees", 0), ("forest_min_leaf", True), ("seed", -1)])
def test_invalid_model_settings(dataset, station_split, setting, value):
    options = {"seed": 42, setting: value}
    with pytest.raises(ValueError):
        train_baselines(dataset, station_split, **options)


def test_training_cli_label_artifacts_reload_and_no_overwrite(tmp_path, matched, station_split):
    import joblib

    result, matching_settings = matched
    directory = tmp_path / "matching"
    ground_dir, satellite_dir = tmp_path / "raw_ground", tmp_path / "raw_satellite"
    ground_dir.mkdir()
    satellite_dir.mkdir()
    ground_paths = files(ground_dir, result.ground)
    satellite_paths = files(satellite_dir, result.satellite.drop(columns="observation_id"), "sentinel2")
    config = load_config(Path(__file__).parents[1] / "configs/base.yaml")
    config["target"] = {"task": "regression", "column": "TEST_TARGET", "definition_source": "Synthetic software fixture only, not a scientific target"}
    config["matching"] = asdict(matching_settings)
    config["baseline"].update({"data_purpose": "software_test", "target_unit": "TEST_UNIT",
        "ground_units": {"TEST_INPUT": "TEST_UNIT"}, "satellite_units": {"TEST_BAND": "TEST_UNIT"},
        "ridge_alpha": 0.01, "forest_trees": 10, "forest_min_leaf": 1, "split": station_split})
    config_path = tmp_path / "test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    matching_command = [sys.executable, str(Path(__file__).parents[1] / "scripts/match_datasets.py"),
        *map(str, ground_paths), *map(str, satellite_paths),
        "--config", str(config_path), "--output", str(directory)]
    matched_run = subprocess.run(matching_command, capture_output=True, text=True, check=False)
    assert matched_run.returncode == 0, matched_run.stderr
    output = tmp_path / "baseline"
    command = [sys.executable, str(Path(__file__).parents[1] / "scripts/train_baselines.py"),
        str(directory), "--config", str(config_path), "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "SOFTWARE TEST ONLY" in completed.stdout
    report = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert report["data_purpose"] == "software_test"
    assert len(report["input_sha256"]) == 4
    assert report["split_counts"] == {"train": 3, "validation": 3, "test": 3}
    artifact = joblib.load(output / "models.joblib")
    data = build(result)
    values = artifact["models"]["ground"].predict(data.ground.iloc[6:])
    recorded = pd.read_csv(output / "test_predictions.csv").query("modality == 'ground'").predicted
    np.testing.assert_allclose(values, recorded)
    assert artifact["data_purpose"] == "software_test"
    assert pd.read_csv(output / "metrics.csv").evidence_label.str.contains("SOFTWARE TEST ONLY").all()
    previous = (output / "run.json").read_bytes()
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode == 1 and "already exists" in again.stderr
    assert (output / "run.json").read_bytes() == previous


def test_training_cli_refuses_unset_target(tmp_path):
    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts/train_baselines.py"),
        str(tmp_path / "not_real_data"), "--config", str(root / "configs/base.yaml"),
        "--output", str(tmp_path / "not_created")], capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "justified target" in result.stderr
    assert not (tmp_path / "not_created").exists()

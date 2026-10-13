import pandas as pd
import pytest

from river_water_quality.training.fusion import FEATURES, prepare_data


def make_inputs(tmp_path):
    rows = []
    for i, split in enumerate(("train", "train", "train", "validation", "test")):
        row = {"observation_id": f"id-{i}", "ground_bod": 1.0 + i,
               "ground_fecal_coliform": 10.0 + i}
        row.update({name: 0.1 + i / 100 for name in FEATURES if name.startswith("satellite_")})
        rows.append(row)
    paired = tmp_path / "paired.csv"
    splits = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(paired, index=False)
    pd.DataFrame({
        "observation_id": [f"id-{i}" for i in range(5)],
        "split": ("train", "train", "train", "validation", "test"),
    }).to_csv(splits, index=False)
    return paired, splits


def test_prepare_data_preserves_saved_splits(tmp_path):
    paired, splits = make_inputs(tmp_path)
    data = prepare_data(paired, splits)
    assert data["split"].value_counts().to_dict() == {
        "train": 3, "validation": 1, "test": 1
    }


def test_prepare_data_rejects_missing_feature_values(tmp_path):
    paired, splits = make_inputs(tmp_path)
    data = pd.read_csv(paired)
    data.loc[0, "satellite_B2"] = float("nan")
    data.to_csv(paired, index=False)
    with pytest.raises(ValueError, match="finite and non-missing"):
        prepare_data(paired, splits)


def test_prepare_data_rejects_negative_fecal_coliform(tmp_path):
    paired, splits = make_inputs(tmp_path)
    data = pd.read_csv(paired)
    data.loc[0, "ground_fecal_coliform"] = -1
    data.to_csv(paired, index=False)
    with pytest.raises(ValueError, match="non-negative"):
        prepare_data(paired, splits)

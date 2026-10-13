"""Synthetic tests of fusion computation, leakage prevention and persistence."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
from river_water_quality.training.dataset import BaselineDataset
from river_water_quality.training.fusion import DualBranchFusion, GROUND, SATELLITE, BANDS, predict_fusion, train_fusion

SPLIT = {"mode": "chronological", "train_end_date": "2021-03-31", "validation_end_date": "2021-09-30", "gap_days": 7}


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    rows = [{"observation_id": f"SYNTHETIC_{month}_{station}", "station_id": f"00{station}",
             "sample_date": f"2021-{month:02d}-15", "satellite_local_date": f"2021-{month:02d}-15",
             "scene_id": f"SYNTHETIC_SCENE_{month}_{station}"}
            for month in (1, 3, 5, 7, 9, 11) for station in range(3)]
    meta = pd.DataFrame(rows).set_index("observation_id")
    return BaselineDataset(meta, pd.Series(rng.uniform(1, 4, len(meta)), index=meta.index, name="bod"),
        pd.DataFrame(rng.uniform(100, 1000, (len(meta), 1)), index=meta.index, columns=GROUND),
        pd.DataFrame(rng.uniform(0.01, 0.2, (len(meta), 10)), index=meta.index, columns=SATELLITE),
        pd.DataFrame(columns=["observation_id", "reason"]))


def test_both_branches_affect_output_and_receive_gradients():
    net = DualBranchFusion()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.fill_(0.1)
    s, g = torch.zeros(2, 10), torch.zeros(2, 1)
    base = net(s, g)
    assert not torch.allclose(base, net(s + 1, g))
    assert not torch.allclose(base, net(s, g + 1))
    net(s + 1, g + 1).sum().backward()
    for branch in (net.satellite_branch, net.ground_branch, net.fusion_head):
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
    assert sum(p.numel() for p in net.parameters()) == 81


def test_test_targets_cannot_change_scaling_weights_or_checkpoint_selection(data):
    before, metrics, _, _, _ = train_fusion(data, SPLIT, epochs=20)
    target = data.target.copy()
    target.loc[data.metadata.sample_date > "2021-09-30"] = 999
    after, other_metrics, *_ = train_fusion(replace(data, target=target), SPLIT, epochs=20)
    assert before["best_epoch"] == after["best_epoch"]
    assert before["satellite_scaler"] == after["satellite_scaler"]
    for key in before["state_dict"]:
        assert torch.equal(before["state_dict"][key], after["state_dict"][key])
    pd.testing.assert_frame_equal(metrics.query("split == 'validation'"), other_metrics.query("split == 'validation'"))
    np.testing.assert_allclose(before["satellite_scaler"]["mean"], data.satellite.iloc[:6].mean())
    np.testing.assert_allclose(before["ground_scaler"]["mean"], np.log1p(data.ground.iloc[:6]).mean())


def test_checkpoint_round_trip_and_input_column_order(data, tmp_path):
    checkpoint, *_ = train_fusion(data, SPLIT, epochs=10)
    first = predict_fusion(checkpoint, data.satellite, data.ground)
    torch.save(checkpoint, tmp_path / "model.pt")
    loaded = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=True)
    np.testing.assert_array_equal(first, predict_fusion(loaded, data.satellite, data.ground))
    assert (first >= 0).all()
    with pytest.raises(ValueError, match="ordered"):
        predict_fusion(loaded, data.satellite.iloc[:, ::-1], data.ground)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1])
def test_invalid_ground_input_rejected(data, bad):
    ground = data.ground.copy()
    ground.iloc[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        train_fusion(replace(data, ground=ground), SPLIT, epochs=1)


def test_target_cannot_be_added_as_predictor(data):
    ground = data.ground.assign(**{"ground::bod": data.target})
    with pytest.raises(ValueError, match="BOD is never an input"):
        train_fusion(replace(data, ground=ground), SPLIT, epochs=1)


def test_shared_scene_across_splits_rejected(data):
    metadata = data.metadata.copy()
    metadata.loc[metadata.index[-1], "scene_id"] = metadata.scene_id.iloc[0]
    with pytest.raises(ValueError, match="scene crosses splits"):
        train_fusion(replace(data, metadata=metadata), SPLIT, epochs=1)


def test_cli_train_predict_without_target_and_refuse_overwrite(data, tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    source, out = tmp_path / "input", tmp_path / "trained"
    source.mkdir()
    matches = data.metadata.assign(acquired_at=data.metadata.sample_date + "T05:00:00Z")
    matches.to_csv(source / "matches.csv")
    ground, satellite = [], []
    for event, row in matches.iterrows():
        common = {"observation_id": event, "station_id": row.station_id, "latitude": 25.0, "longitude": 83.0}
        for name, value, unit in (("bod", data.target[event], "mg/L"), ("fecal_coliform", data.ground.loc[event].iloc[0], "MPN/100mL")):
            ground.append({**common, "sample_date": row.sample_date, "parameter": name, "value": value,
                           "unit": unit, "quality_flag": "SYNTHETIC"})
        for band in BANDS:
            satellite.append({**common, "scene_id": row.scene_id, "acquired_at": row.acquired_at,
                "band": band, "value": data.satellite.loc[event, f"satellite::{band}"], "unit": "reflectance",
                "product_id": "SYNTHETIC", "cloud_fraction": 0.0, "water_fraction": 1.0, "pixel_count": 10})
    pd.DataFrame(ground).to_csv(source / "ground.csv", index=False)
    pd.DataFrame(satellite).to_csv(source / "satellite.csv", index=False)
    (source / "run.json").write_text(json.dumps({"purpose": "synthetic software test"}))
    root = Path(__file__).parents[1]
    result = subprocess.run([sys.executable, "scripts/fusion.py", "train", "--data", str(source), "--output", str(out), "--epochs", "10"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ground_bod" not in pd.read_csv(out / "example_inputs.csv")
    prediction_path = tmp_path / "prediction.csv"
    cmd = [sys.executable, "scripts/fusion.py", "predict", "--model", str(out / "model.pt"),
           "--input", str(out / "example_inputs.csv"), "--output", str(prediction_path)]
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = pd.read_csv(prediction_path)
    expected = pd.read_csv(out / "test_predictions.csv")
    np.testing.assert_allclose(actual.predicted_bod_mg_l, expected.fusion_bod_mg_l, rtol=1e-6)
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    assert result.returncode != 0 and "exists" in result.stderr

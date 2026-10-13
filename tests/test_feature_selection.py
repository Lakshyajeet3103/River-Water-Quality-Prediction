"""Synthetic checks for selection isolation and genuinely excluded inputs."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
from test_fusion import data, SPLIT
from river_water_quality.training.feature_selection import rank_features, select_features
from river_water_quality.training.fusion import SATELLITE, predict_fusion, train_fusion
from river_water_quality.training.splits import assign_splits


def test_rankings_depend_only_on_training_rows(data):
    assignments = assign_splits(data.metadata, SPLIT)
    before = rank_features(data, assignments)
    changed_target, changed_features = data.target.copy(), data.satellite.copy()
    held_out = assignments.split.ne("train")
    changed_target.loc[held_out] = 999
    changed_features.loc[held_out] = 888
    after = rank_features(replace(data, target=changed_target, satellite=changed_features), assignments)
    pd.testing.assert_frame_equal(before, after)
    assert "bod" not in before.feature.tolist()


def test_excluded_features_and_ground_cannot_affect_prediction(data):
    checkpoint, *_ = train_fusion(data, SPLIT, epochs=10, selected_bands=SATELLITE[:3], use_ground=False)
    before = predict_fusion(checkpoint, data.satellite, data.ground)
    spectral, ground = data.satellite.copy(), data.ground.copy()
    spectral.loc[:, SATELLITE[3:]] = 999
    ground.iloc[:, 0] = 888
    np.testing.assert_array_equal(before, predict_fusion(checkpoint, spectral, ground))


def test_candidate_search_does_not_score_test(data):
    _, metrics, predictions, _, _ = train_fusion(data, SPLIT, epochs=10, evaluate_test=False)
    assert metrics.split.tolist() == ["validation"]
    assert predictions.empty


def test_test_targets_do_not_affect_selection_or_weights(data):
    before, ranking, candidates, *_ = select_features(data, SPLIT, epochs=10)
    target = data.target.copy()
    target.loc[data.metadata.sample_date > "2021-09-30"] = 999
    after, other_ranking, other_candidates, *_ = select_features(replace(data, target=target), SPLIT, epochs=10)
    assert before["selection_candidate"] == after["selection_candidate"]
    pd.testing.assert_frame_equal(ranking, other_ranking)
    pd.testing.assert_frame_equal(candidates, other_candidates)
    for key in before["state_dict"]:
        assert torch.equal(before["state_dict"][key], after["state_dict"][key])
    assert len(candidates) == 6 and candidates.selected.sum() == 1


def test_legacy_checkpoint_keeps_all_inputs(data):
    checkpoint, *_ = train_fusion(data, SPLIT, epochs=10)
    before = predict_fusion(checkpoint, data.satellite, data.ground)
    checkpoint.pop("selected_bands")
    checkpoint.pop("use_ground")
    np.testing.assert_array_equal(before, predict_fusion(checkpoint, data.satellite, data.ground))


@pytest.mark.parametrize("selected", [[], ["ground::bod"], [SATELLITE[0], SATELLITE[0]]])
def test_invalid_masks_rejected(data, selected):
    with pytest.raises(ValueError, match="selected_bands"):
        train_fusion(data, SPLIT, epochs=1, selected_bands=selected)


def test_prediction_cli_needs_only_selected_inputs(data, tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    checkpoint, *_ = train_fusion(data, SPLIT, epochs=10, selected_bands=SATELLITE[:3], use_ground=False)
    torch.save(checkpoint, tmp_path / "model.pt")
    inputs = data.satellite[SATELLITE[:3]].rename(columns=lambda c: c.replace("::", "_"))
    inputs.to_csv(tmp_path / "inputs.csv")
    result = subprocess.run([sys.executable, "scripts/fusion.py", "predict", "--model", str(tmp_path / "model.pt"),
        "--input", str(tmp_path / "inputs.csv"), "--output", str(tmp_path / "predictions.csv")],
        cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(pd.read_csv(tmp_path / "predictions.csv").predicted_bod_mg_l,
        predict_fusion(checkpoint, data.satellite, data.ground), rtol=1e-6)

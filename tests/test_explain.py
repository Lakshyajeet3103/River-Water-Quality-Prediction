"""Synthetic integrated-gradients completeness and frozen-checkpoint checks."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")
from test_fusion import data, SPLIT
from river_water_quality.training.fusion import train_fusion, predict_fusion
from river_water_quality.training.explain import explain_fusion


def test_attributions_add_to_prediction_without_changing_weights(data):
    checkpoint, *_ = train_fusion(data, SPLIT, epochs=10)
    weights = {key: value.clone() for key, value in checkpoint["state_dict"].items()}
    explanation = explain_fusion(checkpoint, data.satellite.iloc[:3], data.ground.iloc[:3])
    expected = predict_fusion(checkpoint, data.satellite.iloc[:3], data.ground.iloc[:3])
    np.testing.assert_allclose(explanation.predicted_bod_mg_l, expected, rtol=1e-6)
    assert explanation.integration_residual_mg_l.abs().max() < 0.001
    for key in weights:
        assert torch.equal(weights[key], checkpoint["state_dict"][key])


def test_disabled_ground_has_zero_attribution(data):
    checkpoint, *_ = train_fusion(data, SPLIT, epochs=10, use_ground=False)
    explanation = explain_fusion(checkpoint, data.satellite.iloc[:2], data.ground.iloc[:2])
    assert explanation.ground_contribution_mg_l.eq(0).all()

"""Validate mechanisms, not just output shapes."""
import numpy as np
import pandas as pd
import torch

from river_water_quality.training.experimental_network import (
    ExperimentalNetwork, Switches, ReverseGradient, nonnegative_penalty, fit_network, predict_network)
from river_water_quality.training.sklearn_demonstrations import ALL, TARGET


def test_gradient_reversal_changes_feature_gradient_sign():
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    (ReverseGradient.apply(x, .2)*torch.tensor([4.0, 5.0])).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([-.8, -1.0]))


def test_penalty_pushes_negative_prediction_upward():
    raw = torch.tensor([-2., 3.], requires_grad=True)
    penalty = nonnegative_penalty(raw)
    penalty.backward()
    assert penalty.item() == 2
    assert raw.grad[0] < 0 and raw.grad[1] == 0


def test_gate_changes_with_gap_and_off_is_fixed():
    torch.manual_seed(42)
    x = torch.zeros(2, 13)
    gaps = torch.tensor([0., 7.])
    enabled = ExperimentalNetwork(Switches(gate=True))
    disabled = ExperimentalNetwork(Switches())
    _, weight, _ = enabled(x, gaps)
    assert weight[0] > weight[1]
    torch.testing.assert_close(disabled(x, gaps)[1], torch.tensor([.5, .5]))


def test_off_station_head_and_target_free_inference():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.uniform(.01, .1, (9, 13)), columns=ALL)
    data['ground_fecal_coliform'] *= 10000
    data[TARGET] = 2.
    data['station_id'] = ['s1', 's2', 's3']*3
    data['temporal_offset_days'] = np.arange(9)%7
    checkpoint, history = fit_network(data, Switches(), epochs=3)
    assert not history.constraint_loss_applied.any()
    assert not history.station_cross_entropy.any()
    assert all(v == 0 for k, v in checkpoint['weight_changes'].items() if k.startswith(('station_head', 'gate_layer')))
    expected = predict_network(checkpoint, data)['predicted']
    np.testing.assert_array_equal(expected, predict_network(checkpoint, data[[*ALL, 'temporal_offset_days']])['predicted'])
    checkpoint, _ = fit_network(data, Switches(True, True, True), epochs=3)
    assert any(v > 0 for k, v in checkpoint['weight_changes'].items() if k.startswith('station_head'))

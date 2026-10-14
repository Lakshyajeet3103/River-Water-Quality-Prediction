"""Smoke tests for optional college-demo advanced models."""
from pathlib import Path
import importlib.util

import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("matplotlib") is None:
    pytest.skip("Install the optional advanced-demo dependencies to run these tests", allow_module_level=True)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "demo_advanced_features.py"
spec = importlib.util.spec_from_file_location("demo_advanced_features", SCRIPT)
demo = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(demo)


def test_synthetic_data_shapes_and_reproducibility():
    source_a, target_a = demo.make_demo_data(n_source=12, n_target=8, seed=7)
    source_b, target_b = demo.make_demo_data(n_source=12, n_target=8, seed=7)
    assert source_a[0].shape == (12, 10)
    assert source_a[1].shape == (12, 3)
    assert source_a[2].shape == (12, 2)
    assert target_a[3].shape == (8,)
    assert (source_a[0] == source_b[0]).all()
    assert (target_a[3] == target_b[3]).all()


def test_dann_model_output_shapes_and_gradient_reversal():
    import torch
    model = demo.DannModel(input_dim=13)
    x = torch.randn(5, 13, requires_grad=True)
    prediction, domain_logits = model(x, grl_strength=0.5)
    assert prediction.shape == (5,)
    assert domain_logits.shape == (5, 2)
    domain_logits.sum().backward()
    assert x.grad is not None


def test_temporal_gate_outputs_valid_gate_values():
    import torch
    model = demo.TemporalGateModel()
    prediction, gate = model(torch.randn(7, 10), torch.randn(7, 3), torch.randn(7, 2))
    assert prediction.shape == (7,)
    assert gate.shape == (7,)
    assert torch.all(gate >= 0) and torch.all(gate <= 1)

import numpy as np
import pytest
import torch

from river_water_quality.training.oxygen_pinn import (
    CS, KD, KR, OxygenPINN, analytical, initial_demand, residuals, predict)


def test_initial_conditions_and_bod5_conversion():
    torch.manual_seed(42)
    model = OxygenPINN()
    start = model(torch.zeros(2), torch.tensor([2., 3.]), torch.tensor([7., 8.]))
    torch.testing.assert_close(start, torch.tensor([[2., 7.], [3., 8.]]))
    l0 = initial_demand([1., 4.])
    remaining, _ = analytical(5, l0, [7., 8.])
    np.testing.assert_allclose(l0-remaining, [1., 4.])


def test_analytical_solution_satisfies_both_ode_residuals():
    class Exact(torch.nn.Module):
        def forward(self, t, l, o):
            return torch.stack([l*torch.exp(-KD*t),
                CS+(o-CS)*torch.exp(-KR*t)-KD*l/(KR-KD)*(torch.exp(-KD*t)-torch.exp(-KR*t))], dim=1)
    t = torch.tensor([0., 1., 3., 5.], requires_grad=True)
    _, rl, ro = residuals(Exact(), t, torch.ones(4)*3, torch.ones(4)*8)
    torch.testing.assert_close(rl, torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(ro, torch.zeros(4), atol=1e-6, rtol=0)


def test_ode_loss_can_train_neural_parameters():
    model = OxygenPINN()
    t = torch.tensor([1., 2.], requires_grad=True)
    _, rl, ro = residuals(model, t, torch.tensor([3., 5.]), torch.tensor([7., 8.]))
    (rl.square().mean()+ro.square().mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_invalid_scenario_inputs_rejected():
    with pytest.raises(ValueError):
        initial_demand(-1)
    with pytest.raises(ValueError):
        predict({}, [6], [2], [8])
    with pytest.raises(ValueError, match='exhaustion'):
        predict({}, [5], [100], [1])

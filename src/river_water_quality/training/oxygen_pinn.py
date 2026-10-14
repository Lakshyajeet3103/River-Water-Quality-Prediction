"""Small conditional oxygen-balance PINN for explicitly assumed scenarios.

L'=-kd*L; O'=kr*(Cs-O)-kd*L. Not a calibrated river forecast.
"""
import numpy as np
import torch
from torch import nn

KD = 0.23  # Assumed natural-exponential first-order decay rate, day^-1.
KR = 0.50  # Assumed replenishment/reaeration rate, day^-1.
CS = 8.26  # Assumed freshwater oxygen saturation near 25 C, sea level, mg/L.
HORIZON = 5.0


def initial_demand(bod5):
    bod5 = np.asarray(bod5, dtype=float)
    if not np.isfinite(bod5).all() or (bod5 < 0).any():
        raise ValueError('BOD scenario input must be finite and nonnegative')
    return bod5 / (1 - np.exp(-KD * HORIZON))


def analytical(time, l0, o0):
    t = np.asarray(time, dtype=float)
    remaining = np.asarray(l0) * np.exp(-KD*t)
    oxygen = CS + (np.asarray(o0)-CS)*np.exp(-KR*t) - KD*np.asarray(l0)/(KR-KD)*(np.exp(-KD*t)-np.exp(-KR*t))
    return remaining, oxygen


class OxygenPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 24), nn.Tanh(), nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 2))

    def forward(self, time, l0, o0):
        x = torch.stack([time/HORIZON, l0/10, o0/10], dim=1)
        changes = self.net(x)
        # Hard initial conditions; network learns changes from the starting state.
        return torch.stack([l0 + time*changes[:, 0], o0 + time*changes[:, 1]], dim=1)


def residuals(model, time, l0, o0):
    state = model(time, l0, o0)
    dl = torch.autograd.grad(state[:, 0].sum(), time, create_graph=True)[0]
    do = torch.autograd.grad(state[:, 1].sum(), time, create_graph=True)[0]
    return state, dl + KD*state[:, 0], do - KR*(CS-state[:, 1]) + KD*state[:, 0]


def fit(bod, oxygen, epochs=1500):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    l0 = initial_demand(bod)
    o0 = np.asarray(oxygen, dtype=float)
    if l0.shape != o0.shape or l0.ndim != 1 or len(l0) == 0 or not np.isfinite(o0).all() or (o0 < 0).any():
        raise ValueError('Invalid starting oxygen/demand pairs')
    end_l, end_o = analytical(HORIZON, l0, o0)
    if (end_o < 0).any():
        raise ValueError('Scenario includes oxygen exhaustion; simple kinetics unsupported')
    times = np.tile(np.linspace(0, HORIZON, 11), len(l0))
    lt = torch.tensor(np.repeat(l0, 11), dtype=torch.float32)
    ot = torch.tensor(np.repeat(o0, 11), dtype=torch.float32)
    model = OxygenPINN()
    opt = torch.optim.Adam(model.parameters(), lr=.004)
    history = []
    for epoch in range(1, epochs+1):
        opt.zero_grad()
        t = torch.tensor(times, dtype=torch.float32, requires_grad=True)
        state, rl, ro = residuals(model, t, lt, ot)
        physical = rl.square().mean() + ro.square().mean()
        nonnegative = torch.relu(-state).square().mean()
        loss = physical + nonnegative
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite PINN loss')
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 25 == 0:
            history.append({'epoch': epoch, 'total_loss': float(loss.detach()),
                'ode_loss': float(physical.detach()), 'negative_state_penalty': float(nonnegative.detach())})
    checkpoint = {'state_dict': model.state_dict(), 'epochs': epochs, 'seed': 42,
        'kd_per_day': KD, 'kr_per_day': KR, 'oxygen_saturation_mg_l': CS,
        'horizon_days': HORIZON, 'training_initial_conditions': len(l0),
        'training_l0_range': [float(l0.min()), float(l0.max())],
        'training_do_range': [float(o0.min()), float(o0.max())],
        'training_method': 'ODE residual + nonnegative penalty; hard initial conditions; no trajectory labels'}
    return checkpoint, history


def predict(checkpoint, time, bod, oxygen):
    t, l0, o0 = np.broadcast_arrays(np.asarray(time, dtype=float), initial_demand(bod), np.asarray(oxygen, dtype=float))
    if not np.isfinite(t).all() or not np.isfinite(o0).all() or (t < 0).any() or (t > HORIZON).any() or (o0 < 0).any():
        raise ValueError('Scenario requires finite nonnegative inputs and time within 0-5 days')
    if (analytical(t, l0, o0)[1] < 0).any():
        raise ValueError('Oxygen-exhaustion scenario unsupported')
    model = OxygenPINN()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        result = model(*[torch.tensor(a.ravel(), dtype=torch.float32) for a in [t, l0, o0]])
    # Preserve raw values for auditing; do not hide violations by clipping.
    return result.numpy()

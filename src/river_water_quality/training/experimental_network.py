"""Tiny two-branch experiments: temporal gate, nonnegative penalty and GRL."""
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from river_water_quality.training.sklearn_demonstrations import ALL, values, check_separation
from river_water_quality.training.sklearn_fusion import TARGET


@dataclass(frozen=True)
class Switches:
    gate: bool = False
    constraint: bool = False
    station_adversarial: bool = False


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, strength):
        ctx.strength = strength
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def nonnegative_penalty(raw_prediction):
    return torch.relu(-raw_prediction).square().mean()


class ExperimentalNetwork(nn.Module):
    """Date gap and numeric readings are inputs; station identity is loss-only."""
    def __init__(self, switches):
        super().__init__()
        self.switches = switches
        self.satellite_branch = nn.Sequential(nn.Linear(10, 4), nn.Tanh())
        self.ground_branch = nn.Sequential(nn.Linear(3, 4), nn.Tanh())
        self.bod_head = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 1))
        self.gate_layer = nn.Linear(1, 1)
        # A transparent starting preference: older imagery initially gets less weight.
        nn.init.constant_(self.gate_layer.weight, -1.0)
        nn.init.zeros_(self.gate_layer.bias)
        self.station_head = nn.Linear(8, 3)

    def forward(self, x, absolute_gap, reversal_strength=0.0):
        s, g = self.satellite_branch(x[:, :10]), self.ground_branch(x[:, 10:])
        if self.switches.gate:
            weight = torch.sigmoid(self.gate_layer(absolute_gap.reshape(-1, 1) / 7))
        else:
            weight = torch.full((len(x), 1), .5, dtype=x.dtype, device=x.device)
        representation = torch.cat((weight*s, (1-weight)*g), dim=1)
        raw = self.bod_head(representation).squeeze(1)
        station = self.station_head(ReverseGradient.apply(representation, reversal_strength)) if self.switches.station_adversarial else None
        return raw, weight.squeeze(1), station


def setup(seed=42):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def tensors(frame, scaler):
    x = (values(frame, ALL)-np.array(scaler['mean'])) / np.array(scaler['scale'])
    gap = frame.temporal_offset_days.abs().to_numpy(dtype=float)
    if not np.isfinite(gap).all() or (gap > 7).any():
        raise ValueError('Date gaps must be finite and within the extraction window')
    return torch.tensor(x, dtype=torch.float32), torch.tensor(gap, dtype=torch.float32)


def fit_network(frame, switches, epochs=400, schedule_epochs=400):
    """Fit on supplied rows only; no test/validation data enter the optimizer."""
    setup()
    features = values(frame, ALL)
    scale = features.std(axis=0)
    scale[scale == 0] = 1
    scaler = {'mean': features.mean(axis=0).tolist(), 'scale': scale.tolist()}
    x, gap = tensors(frame, scaler)
    target = torch.tensor(frame[TARGET].to_numpy(dtype=float), dtype=torch.float32)
    if not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError('Invalid training BOD')
    stations = sorted(frame.station_id.unique())
    if len(stations) != 3:
        raise ValueError('This bounded station experiment requires the three training stations')
    station_labels = torch.tensor([stations.index(s) for s in frame.station_id], dtype=torch.long)
    counts = torch.bincount(station_labels, minlength=3).float()
    class_weights = len(frame)/(3*counts)
    model = ExperimentalNetwork(switches)
    initial = deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.001)
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        strength = .2 * (2/(1+np.exp(-10*epoch/schedule_epochs))-1)
        raw, gate, logits = model(x, gap, strength)
        mse = (raw-target).square().mean()
        physical = nonnegative_penalty(raw)
        domain = nn.functional.cross_entropy(logits, station_labels, weight=class_weights) if logits is not None else raw.new_tensor(0.)
        loss = mse + (physical if switches.constraint else 0.) + .1*domain
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        history.append({'epoch': epoch, 'total_loss': float(loss.detach()), 'bod_mse': float(mse.detach()),
                        'negative_bod_penalty': float(physical.detach()),
                        'constraint_loss_applied': float(physical.detach()) if switches.constraint else 0.,
                        'station_cross_entropy': float(domain.detach()),
                        'station_accuracy': float((logits.argmax(1)==station_labels).float().mean().detach()) if logits is not None else None,
                        'reversal_strength': float(strength) if switches.station_adversarial else 0.,
                        'mean_satellite_weight': float(gate.mean().detach()),
                        'negative_raw_train_predictions': int((raw < 0).sum().detach())})
    changes = {name: float((model.state_dict()[name]-before).abs().max()) for name, before in initial.items()}
    checkpoint = {'format_version': 1, 'switches': asdict(switches), 'features': ALL, 'target': TARGET,
                  'station_classes': stations, 'scaler': scaler, 'state_dict': model.state_dict(),
                  'epochs': epochs, 'seed': 42, 'settings': {'lr': .01, 'weight_decay': .001,
                  'constraint_weight': 1.0, 'station_loss_weight': .1, 'max_reversal_strength': .2},
                  'weight_changes': changes}
    return checkpoint, pd.DataFrame(history)


def predict_network(checkpoint, frame):
    if checkpoint['format_version'] != 1 or checkpoint['features'] != ALL:
        raise ValueError('Unexpected experimental checkpoint')
    model = ExperimentalNetwork(Switches(**checkpoint['switches']))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    x, gap = tensors(frame, checkpoint['scaler'])
    with torch.no_grad():
        raw, weight, logits = model(x, gap)
    return {'raw': raw.numpy(), 'predicted': raw.clamp(min=0).numpy(), 'satellite_weight': weight.numpy(),
            'ground_weight': 1-weight.numpy(), 'station_class': logits.argmax(1).numpy() if logits is not None else None}


def select_epochs(train, switches, max_epochs=400):
    """Use an inner chronological split, then refit the selected duration on all 80."""
    train = train.sort_values(['sample_date', 'station_id', 'observation_id']).reset_index(drop=True)
    for cut in range(60, 39, -1):
        try:
            check_separation(train.iloc[:cut], train.iloc[cut:])
            break
        except ValueError:
            continue
    else:
        raise ValueError('No valid inner split')
    # Same optimizer implementation, with one checkpoint per epoch for validation.
    setup()
    inner, validation = train.iloc[:cut], train.iloc[cut:]
    features = values(inner, ALL)
    scale = features.std(axis=0); scale[scale == 0] = 1
    scaler = {'mean': features.mean(axis=0).tolist(), 'scale': scale.tolist()}
    x, gap = tensors(inner, scaler)
    xv, gv = tensors(validation, scaler)
    y = torch.tensor(inner[TARGET].to_numpy(dtype=float), dtype=torch.float32)
    classes = sorted(inner.station_id.unique())
    if len(classes) != 3:
        raise ValueError('Inner training lacks one of the three stations')
    labels = torch.tensor([classes.index(s) for s in inner.station_id], dtype=torch.long)
    class_weights = len(inner)/(3*torch.bincount(labels, minlength=3).float())
    model = ExperimentalNetwork(switches)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.001)
    history, best_mae, best_epoch = [], float('inf'), 1
    for epoch in range(1, max_epochs+1):
        optimizer.zero_grad()
        strength = .2*(2/(1+np.exp(-10*epoch/max_epochs))-1)
        raw, _, logits = model(x, gap, strength)
        physical = nonnegative_penalty(raw)
        domain = nn.functional.cross_entropy(logits, labels, weight=class_weights) if logits is not None else raw.new_tensor(0.)
        loss = (raw-y).square().mean() + (physical if switches.constraint else 0.) + .1*domain
        loss.backward(); optimizer.step()
        with torch.no_grad():
            predicted = model(xv, gv)[0].clamp(min=0).numpy()
        mae = float(np.abs(predicted-validation[TARGET].to_numpy()).mean())
        history.append({'epoch': epoch, 'selection_validation_mae': mae})
        if mae < best_mae:
            best_mae, best_epoch = mae, epoch
    split = train[['observation_id']].copy()
    split['split'] = ['selection_train']*cut + ['selection_validation']*(len(train)-cut)
    return best_epoch, best_mae, pd.DataFrame(history), split

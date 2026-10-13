"""Small two-branch spectral/laboratory network for BOD estimation."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from river_water_quality.training.baselines import regression_metrics
from river_water_quality.training.splits import assign_splits
from river_water_quality.training.dataset import build_baseline_dataset

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
SATELLITE = [f"satellite::{band}" for band in BANDS]
GROUND = ["ground::fecal_coliform"]


def load_paired_data(source, timezone="Asia/Kolkata"):
    """Shared validated inputs and fingerprints for fusion and feature selection."""
    source = Path(source)
    frames, hashes = {}, {}
    for name in ("matches", "ground", "satellite"):
        path = source / f"{name}.csv"
        frames[name] = pd.read_csv(path, dtype={"observation_id": str, "station_id": str})
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = source / "run.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = build_baseline_dataset(frames["matches"], frames["ground"], frames["satellite"],
        target_parameter="bod", target_unit="mg/L", ground_units={"fecal_coliform": "MPN/100mL"},
        satellite_units={band: "reflectance" for band in BANDS}, timezone=timezone)
    return data, hashes, provenance


class DualBranchFusion(nn.Module):
    """Separate trainable encoders, concatenation, then a positive BOD output."""

    def __init__(self):
        super().__init__()
        self.satellite_branch = nn.Sequential(nn.Linear(10, 4), nn.Tanh())
        self.ground_branch = nn.Sequential(nn.Linear(1, 2), nn.Tanh())
        self.fusion_head = nn.Sequential(nn.Linear(6, 4), nn.Tanh(), nn.Linear(4, 1), nn.Softplus())

    def forward(self, satellite, ground):
        joined = torch.cat((self.satellite_branch(satellite), self.ground_branch(ground)), dim=1)
        return self.fusion_head(joined).squeeze(1)


def raw_features(satellite, ground):
    if list(satellite.columns) != SATELLITE or list(ground.columns) != GROUND:
        raise ValueError("Fusion requires exactly the ten ordered bands and fecal coliform; BOD is never an input")
    if satellite.empty or not satellite.index.equals(ground.index):
        raise ValueError("Both branches need nonempty, aligned event rows")
    spectral = satellite.to_numpy(dtype=np.float64)
    laboratory = ground.to_numpy(dtype=np.float64)
    if not np.isfinite(spectral).all() or not np.isfinite(laboratory).all() or (laboratory < 0).any():
        raise ValueError("Inputs must be finite; fecal coliform must be nonnegative")
    return spectral, np.log1p(laboratory)


def fit_scaler(values):
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale == 0] = 1
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def tensor(values, scaler):
    transformed = (values - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    if not np.isfinite(transformed).all() or np.abs(transformed).max() > np.finfo(np.float32).max:
        raise ValueError("Input magnitude exceeds supported floating-point range")
    return torch.tensor(transformed, dtype=torch.float32)


def selection_mask(selected_bands=None, use_ground=True):
    """Zero excluded standardized inputs; legacy checkpoints use every input."""
    selected = SATELLITE if selected_bands is None else selected_bands
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or not set(selected).issubset(SATELLITE):
        raise ValueError("selected_bands must be a nonempty unique subset of satellite features")
    if type(use_ground) is not bool:
        raise ValueError("use_ground must be boolean")
    return torch.tensor([float(name in selected) for name in SATELLITE]), float(use_ground)


def predict_fusion(checkpoint, satellite, ground):
    if checkpoint.get("format_version") != 1 or checkpoint.get("satellite_features") != SATELLITE or checkpoint.get("ground_features") != GROUND:
        raise ValueError("Unsupported fusion checkpoint or feature order")
    spectral, laboratory = raw_features(satellite, ground)
    model = DualBranchFusion()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    smask, gmask = selection_mask(checkpoint.get("selected_bands"), checkpoint.get("use_ground", True))
    with torch.no_grad():
        values = model(tensor(spectral, checkpoint["satellite_scaler"]) * smask,
                       tensor(laboratory, checkpoint["ground_scaler"]) * gmask).numpy()
    if not np.isfinite(values).all():
        raise ValueError("Fusion produced nonfinite predictions")
    return values


def train_fusion(data, split_settings, *, epochs=500, patience=60, seed=42,
                 selected_bands=None, use_ground=True, evaluate_test=True):
    if type(epochs) is not int or not 1 <= epochs <= 5000 or type(patience) is not int or patience < 1:
        raise ValueError("epochs must be 1-5000; patience must be positive")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid random seed")
    if data.target.name != "bod" or not data.metadata.index.equals(data.target.index) or not data.metadata.index.equals(data.satellite.index):
        raise ValueError("Fusion requires aligned BOD targets and metadata")
    if not np.isfinite(data.target).all() or data.target.lt(0).any():
        raise ValueError("BOD targets must be finite and nonnegative")
    spectral, laboratory = raw_features(data.satellite, data.ground)
    smask, gmask = selection_mask(selected_bands, use_ground)
    assignments = assign_splits(data.metadata, split_settings)
    masks = {fold: assignments.split.eq(fold).to_numpy() for fold in ("train", "validation", "test")}
    train, validation, test = (masks[name] for name in ("train", "validation", "test"))
    scalers = [fit_scaler(values[train]) for values in (spectral, laboratory)]
    s, g = tensor(spectral, scalers[0]), tensor(laboratory, scalers[1])
    s, g = s * smask, g * gmask
    y = torch.tensor(data.target.to_numpy(dtype=np.float32))
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = DualBranchFusion()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    best_score, best_epoch, best_state, stale, history = float("inf"), 0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(s[train], g[train]), y[train])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(torch.mean(torch.abs(model(s[validation], g[validation]) - y[validation])))
        history.append({"epoch": epoch, "train_mse": float(loss.detach()), "validation_mae": score})
        if score < best_score:
            best_score, best_epoch, best_state, stale = score, epoch, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    checkpoint = {"format_version": 1, "state_dict": best_state, "target": "bod", "unit": "mg/L",
                  "satellite_features": SATELLITE, "ground_features": GROUND,
                  "satellite_scaler": scalers[0], "ground_scaler": scalers[1], "ground_transform": "log1p",
                  "best_epoch": best_epoch, "seed": seed, "split_settings": split_settings,
                  "architecture": "satellite 10->4 tanh; ground 1->2 tanh; concatenate 6->4 tanh->1 softplus",
                  "parameter_count": sum(p.numel() for p in model.parameters())}
    checkpoint["selected_bands"] = list(SATELLITE if selected_bands is None else selected_bands)
    checkpoint["use_ground"] = use_ground
    if not evaluate_test:
        # Search candidates never generate test predictions or test metrics.
        metrics = pd.DataFrame([{"model": "fusion", "split": "validation", "n": int(validation.sum()), "mae": best_score}])
        return checkpoint, metrics, data.metadata.iloc[:0].copy(), assignments, pd.DataFrame(history)
    fusion_values = predict_fusion(checkpoint, data.satellite, data.ground)
    baseline_values = {"training_mean": np.full(len(y), float(data.target.iloc[np.flatnonzero(train)].mean()))}
    for name, features in (("ground_ridge", laboratory), ("satellite_ridge", spectral)):
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver="svd"))
        estimator.fit(features[train], data.target.to_numpy()[train])
        baseline_values[name] = estimator.predict(features)
    metrics, predictions = [], data.metadata.loc[test].copy()
    predictions["actual_bod_mg_l"] = data.target.loc[test]
    for name, values in {"fusion": fusion_values, **baseline_values}.items():
        for fold in ("validation", "test"):
            mask = masks[fold]
            metrics.append({"model": name, "split": fold, "n": int(mask.sum()),
                            **regression_metrics(data.target.loc[mask], values[mask])})
        predictions[f"{name}_bod_mg_l"] = values[test]
    return checkpoint, pd.DataFrame(metrics), predictions, assignments, pd.DataFrame(history)

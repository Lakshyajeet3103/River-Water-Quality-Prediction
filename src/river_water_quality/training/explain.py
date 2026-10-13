"""Integrated-gradients attribution relative to training-mean transformed inputs."""

import pandas as pd
import torch

from river_water_quality.training.fusion import DualBranchFusion, SATELLITE, GROUND, raw_features, tensor, selection_mask


def explain_fusion(checkpoint, satellite, ground, steps=512):
    """Additive model attribution, not causality or a water-safety judgement."""
    if type(steps) is not int or not 16 <= steps <= 4096:
        raise ValueError("Integration steps must be 16-4096")
    spectral, laboratory = raw_features(satellite, ground)
    smask, gmask = selection_mask(checkpoint.get("selected_bands"), checkpoint.get("use_ground", True))
    s = tensor(spectral, checkpoint["satellite_scaler"]) * smask
    g = tensor(laboratory, checkpoint["ground_scaler"]) * gmask
    model = DualBranchFusion()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        base = float(model(torch.zeros(1, 10), torch.zeros(1, 1))[0])
        predicted = model(s, g).numpy()
    fractions = ((torch.arange(steps, dtype=torch.float32) + 0.5) / steps).reshape(-1, 1)
    rows = []
    for index, event in enumerate(satellite.index):
        path_s = (fractions * s[index]).detach().requires_grad_(True)
        path_g = (fractions * g[index]).detach().requires_grad_(True)
        outputs = model(path_s, path_g)
        gs, gg = torch.autograd.grad(outputs.sum(), (path_s, path_g))
        attr_s = (s[index] * gs.mean(dim=0)).detach().numpy()
        attr_g = (g[index] * gg.mean(dim=0)).detach().numpy()
        s_total, g_total = float(attr_s.sum()), float(attr_g.sum())
        rows.append({"observation_id": event, "predicted_bod_mg_l": float(predicted[index]),
            "reference_bod_mg_l": base, "satellite_contribution_mg_l": s_total,
            "ground_contribution_mg_l": g_total,
            "integration_residual_mg_l": float(predicted[index]) - base - s_total - g_total,
            "larger_absolute_contribution": "satellite" if abs(s_total) > abs(g_total) else "ground",
            **{f"attribution_{name.replace('::', '_')}": float(value)
               for name, value in zip(SATELLITE + GROUND, list(attr_s) + list(attr_g))}})
    return pd.DataFrame(rows).set_index("observation_id")

"""Exact two-group Shapley attribution relative to a training background."""
import numpy as np


def explain_groups(predictor, example, background, satellite, ground):
    if set(satellite) & set(ground) or not satellite or not ground or background.empty:
        raise ValueError('Require disjoint nonempty satellite/ground groups and background')
    features = satellite + ground
    base = background[features].copy()
    satellite_only = base.copy()
    ground_only = base.copy()
    satellite_only.loc[:, satellite] = example[satellite].to_numpy()
    ground_only.loc[:, ground] = example[ground].to_numpy()
    v0 = float(np.mean(predictor(base)))
    vs = float(np.mean(predictor(satellite_only)))
    vg = float(np.mean(predictor(ground_only)))
    full = base.iloc[:1].copy()
    full.loc[:, features] = example[features].to_numpy()
    prediction = float(predictor(full)[0])
    sat = .5 * ((vs - v0) + (prediction - vg))
    grd = .5 * ((vg - v0) + (prediction - vs))
    if not np.isfinite([v0, prediction, sat, grd]).all():
        raise ValueError('Nonfinite explanation')
    return {'predicted_bod_mg_l': prediction, 'background_bod_mg_l': v0,
            'satellite_contribution_mg_l': sat, 'ground_contribution_mg_l': grd,
            'dominant_signal': 'satellite' if abs(sat) > abs(grd) else
                               'ground' if abs(grd) > abs(sat) else 'equal',
            'reconstruction_error_mg_l': prediction - v0 - sat - grd}

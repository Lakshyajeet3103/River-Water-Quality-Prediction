"""Fixed satellite ratios and a small regularisation search; no ground predictors."""
import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from river_water_quality.training.sklearn_fusion import SATELLITE_FEATURES, TARGET, regression_metrics
from river_water_quality.training.sklearn_demonstrations import check_separation

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
RATIOS = ['ratio_B3_B4', 'ratio_B8_B4', 'ratio_B5_B4', 'contrast_B3_B8', 'contrast_B4_B3']


def features(frame, mode):
    if mode not in ('bands', 'bands_ratios'):
        raise ValueError('Unknown input mode')
    data = frame[SATELLITE_FEATURES].copy()
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite satellite bands')
    if mode == 'bands':
        return data
    definitions = {'ratio_B3_B4': ('B3', 'B4', False), 'ratio_B8_B4': ('B8', 'B4', False),
                   'ratio_B5_B4': ('B5', 'B4', False), 'contrast_B3_B8': ('B3', 'B8', True),
                   'contrast_B4_B3': ('B4', 'B3', True)}
    for name, (left, right, normalized) in definitions.items():
        a, b = data['satellite_' + left], data['satellite_' + right]
        numerator, denominator = (a - b, a + b) if normalized else (a, b)
        if denominator.abs().lt(1e-6).any():
            raise ValueError(f'Near-zero denominator: {name}; no silent filling/clipping')
        data[name] = numerator / denominator
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite ratio features')
    return data


def fit(frame, mode, alpha):
    x = features(frame, mode)
    y = frame[[TARGET]].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid BOD training labels')
    if alpha not in ALPHAS:
        raise ValueError('Alpha outside fixed experiment grid')
    xs, ys = StandardScaler().fit(x.to_numpy()), StandardScaler().fit(y)
    estimator = MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='lbfgs',
                             alpha=alpha, max_iter=3000, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        estimator.fit(xs.transform(x.to_numpy()), ys.transform(y).ravel())
    return {'model': estimator, 'x_scaler': xs, 'y_scaler': ys, 'features': x.columns.tolist(),
            'input_mode': mode, 'alpha': alpha, 'target': TARGET, 'seed': 42,
            'warnings': [str(item.message) for item in caught]}


def predict(model, frame):
    x = features(frame, model['input_mode'])
    if x.columns.tolist() != model['features']:
        raise ValueError('Feature schema differs from saved model')
    return np.maximum(0, model['y_scaler'].inverse_transform(model['model'].predict(
        model['x_scaler'].transform(x.to_numpy())).reshape(-1, 1)).ravel())


def select(fitting, validation):
    """Both arguments must be internal training subsets; no outer test received."""
    if fitting.empty or validation.empty or not set(validation.station_id).issubset(set(fitting.station_id)):
        raise ValueError('Need nonempty familiar-station fitting/validation')
    check_separation(fitting, validation)
    rows = []
    for mode in ['bands', 'bands_ratios']:
        for alpha in ALPHAS:
            model = fit(fitting, mode, alpha)
            metrics = regression_metrics(validation[TARGET], predict(model, validation))
            rows.append({'input_mode': mode, 'alpha': alpha, 'input_count': len(model['features']),
                         'validation_mae': metrics['mae'], 'validation_rmse': metrics['rmse'],
                         'training_iterations': int(model['model'].n_iter_), 'warnings': '; '.join(model['warnings'])})
    scores = pd.DataFrame(rows)
    order = ['validation_mae', 'input_count', 'alpha', 'input_mode']
    overall = scores.sort_values(order).iloc[0]
    raw = scores.query("input_mode == 'bands'").sort_values(order).iloc[0]
    scores['selected_pipeline'] = scores.input_mode.eq(overall.input_mode) & scores.alpha.eq(overall.alpha)
    scores['selected_raw_alpha'] = scores.input_mode.eq('bands') & scores.alpha.eq(raw.alpha)
    return scores, {'input_mode': overall.input_mode, 'alpha': float(overall.alpha)}, float(raw.alpha)

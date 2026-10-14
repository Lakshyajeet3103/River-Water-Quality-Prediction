"""Small comparable MLP experiments and training-only feature selection."""
import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from river_water_quality.training.sklearn_fusion import SATELLITE_FEATURES, TARGET, regression_metrics
from river_water_quality.training.splits import acquisition_group

GROUND = ['ground_fecal_coliform', 'ground_ph', 'ground_dissolved_oxygen']
ALL = SATELLITE_FEATURES + GROUND


def values(frame, features):
    x = frame[features].to_numpy(dtype=float).copy()
    if not np.isfinite(x).all():
        raise ValueError('Missing or non-finite predictors')
    if 'ground_fecal_coliform' in features:
        i = features.index('ground_fecal_coliform')
        if (x[:, i] < 0).any():
            raise ValueError('Negative fecal counts')
        x[:, i] = np.log1p(x[:, i])
    return x


def fit(frame, features):
    y = frame[[TARGET]].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid BOD training labels')
    xs = StandardScaler().fit(values(frame, features))
    ys = StandardScaler().fit(y)
    estimator = MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='lbfgs',
                             alpha=1.0, max_iter=3000, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        estimator.fit(xs.transform(values(frame, features)), ys.transform(y).ravel())
    return {'model': estimator, 'features': list(features), 'target': TARGET, 'seed': 42,
            'x_scaler': xs, 'y_scaler': ys, 'log1p_fecal': 'ground_fecal_coliform' in features,
            'warnings': [str(w.message) for w in caught]}


def predict(model, frame):
    x = model['x_scaler'].transform(values(frame, model['features']))
    return np.maximum(0, model['y_scaler'].inverse_transform(model['model'].predict(x).reshape(-1, 1)).ravel())


def check_separation(left, right):
    if set(left.observation_id) & set(right.observation_id):
        raise ValueError('Event crosses folds')
    if set(left.scene_id.map(acquisition_group)) & set(right.scene_id.map(acquisition_group)):
        raise ValueError('Acquisition crosses folds')
    if left[['sample_date', 'satellite_local_date']].max().max() >= right[['sample_date', 'satellite_local_date']].min().min():
        raise ValueError('Date intervals overlap folds')


def feature_search(train):
    """Select only inside the outer training set; never receives outer test data."""
    train = train.sort_values(['sample_date', 'station_id', 'observation_id']).reset_index(drop=True)
    for cutoff in range(min(60, len(train)-2), 39, -1):
        inner, validation = train.iloc[:cutoff], train.iloc[cutoff:]
        try:
            check_separation(inner, validation)
            break
        except ValueError:
            continue
    else:
        raise ValueError('No valid inner chronological selection split')
    x, y = values(inner, ALL), inner[TARGET].to_numpy(dtype=float)
    ranking = []
    for i, name in enumerate(ALL):
        xc, yc = x[:, i]-x[:, i].mean(), y-y.mean()
        denominator = np.linalg.norm(xc)*np.linalg.norm(yc)
        r = float(xc @ yc/denominator) if denominator else 0.0
        ranking.append({'feature': name, 'modality': 'satellite' if name in SATELLITE_FEATURES else 'ground',
                        'correlation': r, 'score': abs(r), 'ranking_train_rows': len(inner)})
    ranking = pd.DataFrame(ranking).sort_values(['score', 'feature'], ascending=[False, True])
    bands = ranking.query("modality == 'satellite'").feature.tolist()
    ground = ranking.query("modality == 'ground'").feature.tolist()
    candidates = []
    for count in (3, 5, 10):
        for ground_count in (0, 1, 2, 3):
            features = [f for f in ALL if f in bands[:count] + ground[:ground_count]]
            model = fit(inner, features)
            score = regression_metrics(validation[TARGET], predict(model, validation))
            candidates.append({'candidate': f'top{count}_bands_top{ground_count}_ground',
                               'features': ';'.join(features), 'input_count': len(features),
                               'validation_mae': score['mae'], 'validation_rmse': score['rmse'],
                               'warnings': ';'.join(model['warnings'])})
    table = pd.DataFrame(candidates)
    choice = table.sort_values(['validation_mae', 'input_count', 'candidate']).iloc[0]
    selected = choice.features.split(';')
    ranking['selected'] = ranking.feature.isin(selected)
    table['selected'] = table.candidate.eq(choice.candidate)
    inner_splits = train[['observation_id']].copy()
    inner_splits['split'] = ['selection_train']*len(inner) + ['selection_validation']*len(validation)
    return selected, ranking.reset_index(drop=True), table, inner_splits

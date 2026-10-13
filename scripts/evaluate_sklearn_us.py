"""Evaluate the frozen expanded India scikit-learn model on four real US pairs."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from river_water_quality.training.sklearn_fusion import FEATURES, TARGET, regression_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    model_path = Path('outputs/fusion_sklearn_expanded_20261014/fusion_model.joblib')
    source = Path('outputs/crossregion_demo_001/us_paired.csv')
    out = Path('outputs/crossregion_sklearn_expanded_20261014')
    if out.exists():
        raise ValueError('Output exists; refusing to overwrite')
    before = sha(model_path)
    model = joblib.load(model_path)
    us = pd.read_csv(source)
    if len(us) != 4 or not us.observation_id.is_unique or not us.station_id.str.startswith('USGS-').all():
        raise ValueError('Expected four unique real USGS observations')
    if model['features'] != FEATURES or model['target'] != TARGET:
        raise ValueError('Unexpected saved model inputs/target')
    if not np.isfinite(us[FEATURES + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError('Missing or non-finite values')
    if (us[['ground_fecal_coliform', TARGET]] < 0).any().any():
        raise ValueError('Negative ground measurement')
    x = us[FEATURES].to_numpy(dtype=float).copy()
    x[:, -1] = np.log1p(x[:, -1])
    prediction = np.maximum(0, model['y_scaler'].inverse_transform(
        model['model'].predict(model['x_scaler'].transform(x)).reshape(-1, 1)).ravel())
    baseline = float(model['y_scaler'].mean_[0])
    rows = us[['observation_id', 'station_id', 'station_name', 'sample_date', 'scene_id']].copy()
    rows['actual_bod_mg_l'] = us[TARGET]
    rows['predicted_bod_mg_l'] = prediction
    rows['absolute_error_mg_l'] = np.abs(prediction - us[TARGET].to_numpy())
    rows['india_training_mean_bod_mg_l'] = baseline
    metrics = pd.DataFrame([
        {'model': 'frozen_india_sklearn_mlp', **regression_metrics(us[TARGET], prediction)},
        {'model': 'india_training_mean', **regression_metrics(us[TARGET], np.full(len(us), baseline))}])
    if sha(model_path) != before:
        raise ValueError('Model changed during evaluation')
    out.mkdir(parents=True)
    rows.to_csv(out / 'predictions.csv', index=False)
    metrics.to_csv(out / 'metrics.csv', index=False)
    report = {'model_sha256': before, 'us_paired_sha256': sha(source),
              'model_unchanged': True, 'us_data_used_for_training': False,
              'features': FEATURES, 'target': TARGET, 'metrics': metrics.to_dict('records'),
              'limitations': ['Only four previously examined US observations; exploratory geographic transfer.',
                              'Ground fecal-coliform measurement is required for prediction.',
                              'Existing sample-method, five-minute pairing, coordinate and water-pixel assumptions remain.',
                              'US sites are not assumed clean; retrospective estimation is not forecasting.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    saved = pd.read_csv(out / 'predictions.csv')
    np.testing.assert_allclose(saved.predicted_bod_mg_l, prediction, rtol=1e-12, atol=1e-12)
    print(rows[['station_name', 'sample_date', 'actual_bod_mg_l', 'predicted_bod_mg_l', 'absolute_error_mg_l']].to_string(index=False))
    print(metrics.to_string(index=False))
    print(f'Saved: {out}; frozen model verified unchanged')


if __name__ == '__main__':
    main()

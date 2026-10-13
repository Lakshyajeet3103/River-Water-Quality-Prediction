"""Train a separate satellite-only version of the friend's scikit-learn MLP."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from river_water_quality.training.sklearn_fusion import SATELLITE_FEATURES, TARGET, regression_metrics
from river_water_quality.training.splits import acquisition_group


def predict(model, frame):
    # Only satellite columns enter prediction; all ground values are ignored.
    x = frame[model['features']].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Satellite inputs must be finite')
    return np.maximum(0, model['y_scaler'].inverse_transform(
        model['model'].predict(model['x_scaler'].transform(x)).reshape(-1, 1)).ravel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', default='outputs/matches_expansion_100_20261014/paired.csv')
    parser.add_argument('--output', default='outputs/satellite_only_80_20_20261014')
    args = parser.parse_args()
    source, out = Path(args.paired), Path(args.output)
    if out.exists():
        raise ValueError('Output exists; refusing to overwrite')
    data = pd.read_csv(source).sort_values(['sample_date', 'station_id', 'observation_id']).reset_index(drop=True)
    if len(data) != 100 or not data.observation_id.is_unique:
        raise ValueError('Expected exactly 100 unique matched events')
    if not np.isfinite(data[SATELLITE_FEATURES + [TARGET]].to_numpy(dtype=float)).all() or data[TARGET].lt(0).any():
        raise ValueError('Invalid satellite inputs or BOD targets')
    train, test = data.iloc[:80].copy(), data.iloc[80:].copy()
    if set(train.scene_id.map(acquisition_group)) & set(test.scene_id.map(acquisition_group)):
        raise ValueError('Satellite acquisition crosses folds; cannot use this 80/20 boundary')
    train_last = train[['sample_date', 'satellite_local_date']].max().max()
    test_first = test[['sample_date', 'satellite_local_date']].min().min()
    if train_last >= test_first:
        raise ValueError('Sampling/acquisition date intervals overlap between folds')
    x_scaler = StandardScaler().fit(train[SATELLITE_FEATURES].to_numpy(dtype=float))
    y_scaler = StandardScaler().fit(train[[TARGET]].to_numpy(dtype=float))
    estimator = MLPRegressor(hidden_layer_sizes=(8,), activation='relu', solver='lbfgs',
                             alpha=1.0, max_iter=3000, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        estimator.fit(x_scaler.transform(train[SATELLITE_FEATURES].to_numpy(dtype=float)),
                      y_scaler.transform(train[[TARGET]].to_numpy(dtype=float)).ravel())
    model = {'model': estimator, 'x_scaler': x_scaler, 'y_scaler': y_scaler,
             'features': SATELLITE_FEATURES, 'target': TARGET, 'seed': 42, 'satellite_only': True}
    predicted = predict(model, test)
    metrics = pd.DataFrame([
        {'model': 'satellite_only_mlp', 'split': 'train', **regression_metrics(train[TARGET], predict(model, train))},
        {'model': 'satellite_only_mlp', 'split': 'test', **regression_metrics(test[TARGET], predicted)},
        {'model': 'training_mean', 'split': 'test', **regression_metrics(test[TARGET], np.full(20, float(y_scaler.mean_[0])))}])
    out.mkdir(parents=True)
    joblib.dump(model, out / 'model.joblib')
    restored = joblib.load(out / 'model.joblib')
    np.testing.assert_allclose(predict(restored, test[SATELLITE_FEATURES]), predicted, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x_scaler.mean_, train[SATELLITE_FEATURES].to_numpy().mean(axis=0))
    assignments = data[['observation_id', 'station_id', 'sample_date', 'scene_id']].copy()
    assignments['split'] = ['train'] * 80 + ['test'] * 20
    assignments.to_csv(out / 'split_assignments.csv', index=False)
    rows = test[['observation_id', 'station_id', 'station_name', 'sample_date', 'scene_id']].copy()
    rows['actual_bod_mg_l'] = test[TARGET]
    rows['predicted_bod_mg_l'] = predicted
    rows['absolute_error_mg_l'] = np.abs(predicted - test[TARGET].to_numpy())
    rows.to_csv(out / 'test_predictions.csv', index=False)
    test[['observation_id', *SATELLITE_FEATURES]].to_csv(out / 'example_inputs.csv', index=False)
    metrics.to_csv(out / 'metrics.csv', index=False)
    report = {'model': 'Satellite-only scikit-learn MLPRegressor', 'train_rows': 80, 'test_rows': 20,
              'features': SATELLITE_FEATURES, 'target': TARGET, 'target_unit': 'mg/L', 'seed': 42,
              'hidden_layer_sizes': [8], 'activation': 'relu', 'solver': 'lbfgs', 'alpha': 1.0,
              'max_iter': 3000, 'iterations': int(estimator.n_iter_), 'sklearn_version': sklearn.__version__,
              'train_only_scaling': True, 'train_interval_end': train_last, 'test_interval_start': test_first,
              'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256((out / 'model.joblib').read_bytes()).hexdigest(),
              'checkpoint_reload_verified': True, 'prediction_without_ground_columns_verified': True,
              'warnings': [str(w.message) for w in caught], 'metrics': metrics.to_dict('records'),
              'limitations': ['Known-station retrospective estimation, not future forecasting or water safety.',
                              'This test contains previously examined observations; exploratory, not a new untouched final test.',
                              'Old combined-model scores use different training/test rows and are not a controlled modality comparison.',
                              'Existing CPCB provenance and satellite pixel-quality assumptions remain.']}
    (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(metrics.to_string(index=False))
    print(f'Saved satellite-only model: {out}')


if __name__ == '__main__':
    main()

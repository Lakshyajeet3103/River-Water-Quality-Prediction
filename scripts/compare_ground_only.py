"""Add a ground-only MLP to the unchanged saved 80/20 modality comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('outputs/matplotlib_cache').resolve()))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from river_water_quality.training.sklearn_demonstrations import (
    ALL, GROUND, SATELLITE_FEATURES, TARGET, fit, predict, check_separation)
from river_water_quality.training.sklearn_fusion import regression_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_cohort(data, splits):
    if not data.observation_id.is_unique or not splits.observation_id.is_unique:
        raise ValueError('Duplicate event IDs')
    if set(data.observation_id) != set(splits.observation_id):
        raise ValueError('Saved split does not cover exactly the dataset')
    if set(splits.split) != {'train', 'test'}:
        raise ValueError('Unexpected split labels')
    data = data.merge(splits[['observation_id', 'split']], on='observation_id', validate='one_to_one')
    data = data.sort_values(['sample_date', 'station_id', 'observation_id']).reset_index(drop=True)
    train, test = data.query("split == 'train'").copy(), data.query("split == 'test'").copy()
    if len(train) != 80 or len(test) != 20:
        raise ValueError('Expected the unchanged 80/20 split')
    check_separation(train, test)
    return train, test


def verify_model_contract(model, features):
    estimator = model['model']
    expected = {'hidden_layer_sizes': (8,), 'activation': 'relu', 'solver': 'lbfgs',
                'alpha': 1.0, 'max_iter': 3000, 'random_state': 42}
    if model['features'] != features or any(estimator.get_params()[k] != v for k, v in expected.items()):
        raise ValueError('Saved model does not have the expected inputs/settings')
    if model['target'] != TARGET:
        raise ValueError('Unexpected model target')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/ground_only_comparison_20261014'))
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise ValueError('Output exists; choose a fresh directory')
    data_path = Path('outputs/india_100_chemical_dataset_20261014/paired.csv')
    split_path = Path('outputs/satellite_only_80_20_20261014/split_assignments.csv')
    old = Path('outputs/straightforward_demos_20261014')
    files = [data_path, split_path, old / 'satellite_only.joblib', old / 'combined_all_chemicals.joblib',
             old / 'test_predictions.csv', Path('outputs/satellite_only_80_20_20261014/model.joblib'),
             Path('outputs/fusion_sklearn_expanded_20261014/fusion_model.joblib')]
    hashes = {str(p): sha(p) for p in files}
    manifest = json.loads(Path('demo_bundle/manifest.json').read_text(encoding='utf-8'))
    expected = {entry['path']: entry['sha256'] for entry in manifest['files']}
    if any(expected[p.as_posix()] != hashes[str(p)] for p in files):
        raise ValueError('Saved source differs from preserved bundle')
    splits = pd.read_csv(split_path)
    train, test = saved_cohort(pd.read_csv(data_path), splits)
    ground_model = fit(train, GROUND)
    verify_model_contract(ground_model, GROUND)
    # Scaling has learned only the 80 training observations, including log1p fecal.
    expected_ground = train[GROUND].to_numpy(dtype=float).copy()
    expected_ground[:, 0] = np.log1p(expected_ground[:, 0])
    np.testing.assert_allclose(ground_model['x_scaler'].mean_, expected_ground.mean(axis=0))
    np.testing.assert_allclose(ground_model['y_scaler'].mean_, train[TARGET].mean())
    models = {'satellite_only': joblib.load(old / 'satellite_only.joblib'),
              'ground_only': ground_model,
              'combined_all_chemicals': joblib.load(old / 'combined_all_chemicals.joblib')}
    verify_model_contract(models['satellite_only'], SATELLITE_FEATURES)
    verify_model_contract(models['combined_all_chemicals'], ALL)
    previous = pd.read_csv(old / 'test_predictions.csv').set_index('observation_id')
    if set(previous.index) != set(test.observation_id):
        raise ValueError('Old comparison contains different test events')
    previous = previous.loc[test.observation_id]
    np.testing.assert_allclose(previous.actual_bod_mg_l, test[TARGET], rtol=0, atol=1e-12)
    rows = test[['observation_id', 'station_id', 'station_name', 'sample_date', 'scene_id']].copy()
    rows['actual_bod_mg_l'] = test[TARGET]
    metric_rows = []
    for name, model in models.items():
        prediction = predict(model, test[model['features']])
        if name != 'ground_only':
            np.testing.assert_allclose(prediction, previous[name + '_bod_mg_l'], rtol=0, atol=1e-7)
        rows[name + '_bod_mg_l'] = prediction
        rows[name + '_absolute_error_mg_l'] = np.abs(prediction - test[TARGET].to_numpy())
        metric_rows.append({'model': name, 'input_count': len(model['features']), 'split': 'test',
                            **regression_metrics(test[TARGET], prediction)})
    mean = np.full(20, float(train[TARGET].mean()))
    rows['training_mean_bod_mg_l'] = mean
    metric_rows.append({'model': 'training_mean', 'input_count': 0, 'split': 'test',
                        **regression_metrics(test[TARGET], mean)})
    metrics = pd.DataFrame(metric_rows)
    station_rows = []
    for station, group in rows.groupby('station_name', sort=True):
        for name in models:
            station_rows.append({'station_name': station, 'model': name,
                **regression_metrics(group.actual_bod_mg_l, group[name + '_bod_mg_l'])})
    out.mkdir(parents=True)
    joblib.dump(ground_model, out / 'ground_only.joblib')
    np.testing.assert_allclose(predict(joblib.load(out / 'ground_only.joblib'), test[GROUND]),
                               rows.ground_only_bod_mg_l, rtol=0, atol=1e-12)
    rows.to_csv(out / 'test_predictions.csv', index=False)
    metrics.to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(station_rows).to_csv(out / 'station_metrics.csv', index=False)
    splits.to_csv(out / 'split_assignments.csv', index=False)
    test[['observation_id', *GROUND]].to_csv(out / 'ground_only_example_inputs.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ['Satellite only', 'Ground only', 'Combined', 'Training mean']
    colors = ['#4389b1', '#c58c36', '#2a927c', '#a3a3a3']
    for ax, metric in zip(axes, ['mae', 'rmse']):
        ax.bar(labels, metrics[metric], color=colors)
        ax.set_ylabel(f'{metric.upper()} (BOD mg/L; lower is better)')
        ax.tick_params(axis='x', labelrotation=20)
        for i, value in enumerate(metrics[metric]):
            ax.text(i, value + .015, f'{value:.3f}', ha='center')
        ax.set_ylim(0, float(metrics[metric].max()) * 1.2)
    fig.suptitle('Same 80 training / 20 test events, same MLP settings')
    fig.tight_layout()
    fig.savefig(out / 'modality_comparison.png', dpi=160)
    plt.close(fig)
    model_metrics = metrics.set_index('model')
    ground_mae = float(model_metrics.loc['ground_only', 'mae'])
    combined_mae = float(model_metrics.loc['combined_all_chemicals', 'mae'])
    report = {'input_sha256': hashes, 'train_rows': 80, 'test_rows': 20,
              'test_ids': test.observation_id.tolist(), 'features_by_model': {n: m['features'] for n, m in models.items()},
              'settings': {'hidden_layer_sizes': [8], 'activation': 'relu', 'solver': 'lbfgs',
                           'alpha': 1.0, 'max_iter': 3000, 'seed': 42},
              'ground_model_training_iterations': int(ground_model['model'].n_iter_),
              'ground_model_warnings': ground_model['warnings'],
              'ground_model_sha256': sha(out / 'ground_only.joblib'),
              'combined_minus_ground_mae_mg_l': combined_mae - ground_mae,
              'combined_mae_improvement_percent': 100 * (ground_mae - combined_mae) / ground_mae,
              'cases_combined_lower_error': int((rows.combined_all_chemicals_absolute_error_mg_l < rows.ground_only_absolute_error_mg_l).sum()),
              'cases_ground_lower_error': int((rows.ground_only_absolute_error_mg_l < rows.combined_all_chemicals_absolute_error_mg_l).sum()),
              'limitations': ['Same known stations and previously examined test cases; exploratory comparison.',
                              'Same model settings with different input counts do not mean equal parameter counts.',
                              'Ground-only inputs are fecal coliform, pH and dissolved oxygen; BOD is never an input.',
                              'No hyperparameter tuning, test-based selection, or replacement of the main saved model.',
                              'This evaluates BOD, not independent WQI prediction or unseen stations.']}
    if any(sha(p) != hashes[str(p)] for p in files):
        raise ValueError('Preserved source changed')
    report['output_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(metrics.to_string(index=False))
    print(json.dumps({key: report[key] for key in ['combined_minus_ground_mae_mg_l',
          'combined_mae_improvement_percent', 'cases_combined_lower_error', 'cases_ground_lower_error']}))


if __name__ == '__main__':
    main()

"""Satellite-only band selection on the original familiar-station 80/20 split."""
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

from river_water_quality.training.satellite_selection import candidates, select_chronological_bands
from river_water_quality.training.sklearn_demonstrations import fit, predict, check_separation
from river_water_quality.training.sklearn_fusion import TARGET, regression_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/satellite_band_selection_seen_20261014'))
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise ValueError('Output exists; choose a fresh directory')
    data_path = Path('outputs/india_100_chemical_dataset_20261014/paired.csv')
    split_path = Path('outputs/satellite_only_80_20_20261014/split_assignments.csv')
    selection_path = Path('outputs/straightforward_demos_20261014/selection_split_assignments.csv')
    model_path = Path('outputs/satellite_only_80_20_20261014/model.joblib')
    prediction_path = Path('outputs/satellite_only_80_20_20261014/test_predictions.csv')
    protected = [data_path, split_path, selection_path, model_path, prediction_path]
    hashes = {str(p): sha(p) for p in protected}
    expected = {e['path']: e['sha256'] for e in json.loads(Path('demo_bundle/manifest.json').read_text(encoding='utf-8'))['files']}
    if any(hashes[str(p)] != expected[p.as_posix()] for p in protected):
        raise ValueError('Saved source differs from preserved bundle')
    data, splits = pd.read_csv(data_path), pd.read_csv(split_path)
    if not data.observation_id.is_unique or not splits.observation_id.is_unique or set(data.observation_id) != set(splits.observation_id):
        raise ValueError('Invalid preserved cohort/split')
    data = data.merge(splits[['observation_id', 'split']], on='observation_id', validate='one_to_one')
    data = data.sort_values(['sample_date', 'station_id', 'observation_id']).reset_index(drop=True)
    train, test = data.query("split == 'train'").drop(columns='split'), data.query("split == 'test'").drop(columns='split')
    if len(train) != 80 or len(test) != 20 or not set(test.station_id).issubset(set(train.station_id)):
        raise ValueError('Expected unchanged familiar-station 80/20 split')
    check_separation(train, test)
    internal = pd.read_csv(selection_path)
    features, choice, scores = select_chronological_bands(train, internal)
    # Choice is now fixed, before any outer test metric is calculated.
    model = fit(train, features)
    selected_prediction = predict(model, test[features])
    original = joblib.load(model_path)
    original_prediction = predict(original, test[original['features']])
    previous = pd.read_csv(prediction_path).set_index('observation_id')
    if set(previous.index) != set(test.observation_id):
        raise ValueError('Baseline test IDs differ')
    previous = previous.loc[test.observation_id]
    np.testing.assert_allclose(previous.actual_bod_mg_l, test[TARGET], rtol=0, atol=1e-12)
    np.testing.assert_allclose(previous.predicted_bod_mg_l, original_prediction, rtol=0, atol=1e-7)
    rows = test[['observation_id', 'station_id', 'station_name', 'sample_date', 'scene_id']].copy()
    rows['actual_bod_mg_l'] = test[TARGET]
    rows['selected_bod_mg_l'] = selected_prediction
    rows['all_bands_bod_mg_l'] = original_prediction
    rows['training_mean_bod_mg_l'] = train[TARGET].mean()
    metrics = pd.DataFrame([{'model': name, 'input_count': len(features) if name == 'selected' else 10 if name == 'all_bands' else 0,
                            **regression_metrics(rows.actual_bod_mg_l, rows[name + '_bod_mg_l'])}
                           for name in ['selected', 'all_bands', 'training_mean']])
    station_metrics = pd.DataFrame([{'station_name': station, 'model': name,
                                   **regression_metrics(group.actual_bod_mg_l, group[name + '_bod_mg_l'])}
                                  for station, group in rows.groupby('station_name')
                                  for name in ['selected', 'all_bands', 'training_mean']])
    out.mkdir(parents=True)
    joblib.dump(model, out / 'selected_model.joblib')
    np.testing.assert_allclose(predict(joblib.load(out / 'selected_model.joblib'), test[features]), selected_prediction, rtol=0, atol=1e-12)
    scores.to_csv(out / 'candidate_validation_scores.csv', index=False)
    internal.to_csv(out / 'selection_split_assignments.csv', index=False)
    splits.to_csv(out / 'split_assignments.csv', index=False)
    rows.to_csv(out / 'test_predictions.csv', index=False)
    metrics.to_csv(out / 'metrics.csv', index=False)
    station_metrics.to_csv(out / 'station_metrics.csv', index=False)
    test[['observation_id', *features]].to_csv(out / 'example_inputs.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = ['Selected bands', 'All ten bands', 'Training mean']
    for ax, metric in zip(axes, ['mae', 'rmse']):
        ax.bar(labels, metrics[metric], color=['#2a927c', '#4389b1', '#999999'])
        ax.set_ylabel(f'BOD {metric.upper()} (mg/L; lower is better)')
        ax.set_ylim(0, metrics[metric].max() * 1.2)
        ax.tick_params(axis='x', labelrotation=15)
        for i, value in enumerate(metrics[metric]):
            ax.text(i, value + .015, f'{value:.3f}', ha='center')
    fig.suptitle('Same familiar stations: 80 training / 20 later test observations')
    fig.tight_layout()
    fig.savefig(out / 'seen_station_comparison.png', dpi=160)
    plt.close(fig)
    if any(sha(p) != hashes[str(p)] for p in protected):
        raise ValueError('Original saved model/data changed')
    report = {'input_sha256': hashes, 'train_rows': 80, 'test_rows': 20,
              'selection_train_rows': 60, 'selection_validation_rows': 20,
              'selected_candidate': choice, 'selected_features': features,
              'selected_model_sha256': sha(out / 'selected_model.joblib'),
              'training_iterations': int(model['model'].n_iter_), 'training_warnings': model['warnings'],
              'settings': {'hidden_layer_sizes': [8], 'activation': 'relu', 'solver': 'lbfgs', 'alpha': 1, 'max_iter': 3000, 'seed': 42},
              'candidate_subsets': candidates(),
              'limitations': ['Band choice uses the internal 60/20 chronological split, not outer test answers.',
                              'Same known stations, original 100 observations; not the expanded seven-station unseen test.',
                              'The outer 20 examples were previously examined; exploratory internal comparison.',
                              'No chemical inputs, new indices, hyperparameter tuning, category/WQI or main-model replacement.',
                              'Input counts vary with selection, so parameter counts differ despite fixed hidden layer/settings.']}
    report['output_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Selected using inner validation only: {choice}: {features}')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()

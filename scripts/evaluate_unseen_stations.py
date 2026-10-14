"""Leave each station out; purge shared sensing acquisitions from training."""
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

from river_water_quality.training.sklearn_demonstrations import ALL, GROUND, SATELLITE_FEATURES, TARGET, fit, predict
from river_water_quality.training.sklearn_fusion import regression_metrics
from river_water_quality.training.splits import acquisition_group

LABELS = {'cpcb-0ecd9698ec1a2eb1': 'Anupshahar', 'cpcb-1c6a3feabae131a4': 'Varanasi downstream',
          'cpcb-baa51b840df3bdc2': 'Bijnor', 'cpcb-802a0bf7ab0a50d0': 'Garhmukteshwar',
          'cpcb-65a5002389d94ee5': 'Sangam', 'cpcb-d9f0323fba4e4a7c': 'Narora',
          'cpcb-06c5a8086252cc40': 'Mirzapur upstream', 'cpcb-a78396bed449fa69': 'Dalmau',
          'cpcb-d80ef48e84ee36a6': 'Varanasi upstream'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def station_fold(data, held_out):
    if not data.observation_id.is_unique:
        raise ValueError('Duplicate events')
    test = data[data.station_id.eq(held_out)].copy()
    candidates = data[~data.station_id.eq(held_out)].copy()
    if test.empty or candidates.station_id.nunique() < 2:
        raise ValueError('Need test station and at least two other training stations')
    shared = candidates.scene_id.map(acquisition_group).isin(set(test.scene_id.map(acquisition_group)))
    train, purged = candidates[~shared].copy(), candidates[shared].copy()
    if len(train) < 10 or train.station_id.nunique() < 2:
        raise ValueError('Insufficient training data after shared-acquisition purge')
    if set(train.observation_id) & set(test.observation_id) or held_out in set(train.station_id):
        raise ValueError('Test identity leaked into training')
    if set(train.scene_id.map(acquisition_group)) & set(test.scene_id.map(acquisition_group)):
        raise ValueError('Shared acquisition leaked')
    return train, test, purged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, default=Path('outputs/india_100_chemical_dataset_20261014/paired.csv'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; choose a fresh directory')
    data = pd.read_csv(args.paired).sort_values(['station_id', 'sample_date', 'observation_id']).reset_index(drop=True)
    if data.station_id.nunique() < 3 or not np.isfinite(data[ALL + [TARGET]]).all().all():
        raise ValueError('Need three stations and complete finite inputs/targets')
    if data[TARGET].lt(0).any():
        raise ValueError('Negative BOD')
    data_hash = sha(args.paired)
    provenance = args.paired.with_name('provenance.json')
    run = args.paired.with_name('run.json')
    expected = json.loads((provenance if provenance.exists() else run).read_text(encoding='utf-8'))
    if expected.get('output_sha256', expected.get('paired_sha256')) != data_hash:
        raise ValueError('Dataset provenance hash mismatch')
    preserved = [Path('outputs/satellite_only_80_20_20261014/model.joblib'),
                 Path('outputs/fusion_sklearn_expanded_20261014/fusion_model.joblib')]
    old_hashes = {str(p): sha(p) for p in preserved}
    args.output.mkdir(parents=True)
    all_predictions, fold_metrics, assignments, diagnostics = [], [], [], []
    features = {'satellite_only': SATELLITE_FEATURES, 'ground_only': GROUND, 'combined': ALL}
    for station in sorted(data.station_id.unique()):
        train, test, purged = station_fold(data, station)
        station_name = test.station_name.iloc[0]
        for label, frame in [('train', train), ('test', test), ('purged_shared_acquisition', purged)]:
            table = frame[['observation_id', 'station_id', 'sample_date', 'scene_id']].copy()
            table['held_out_station_id'], table['assignment'] = station, label
            assignments.append(table)
        prediction_rows = test[['observation_id', 'station_id', 'station_name', 'sample_date', 'scene_id']].copy()
        prediction_rows['held_out_station_id'] = station
        prediction_rows['actual_bod_mg_l'] = test[TARGET]
        for name, inputs in features.items():
            model = fit(train, inputs)
            prediction = predict(model, test[inputs])
            path = args.output / f'{station}_{name}.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(predict(joblib.load(path), test[inputs]), prediction, rtol=0, atol=1e-12)
            prediction_rows[name + '_bod_mg_l'] = prediction
            fold_metrics.append({'held_out_station_id': station, 'station_name': station_name,
                                 'model': name, 'training_rows': len(train), 'purged_rows': len(purged),
                                 **regression_metrics(test[TARGET], prediction)})
            diagnostics.append({'held_out_station_id': station, 'model': name,
                                'iterations': int(model['model'].n_iter_), 'warnings': model['warnings'],
                                'model_sha256': sha(path)})
        mean_prediction = np.full(len(test), train[TARGET].mean())
        prediction_rows['training_mean_bod_mg_l'] = mean_prediction
        fold_metrics.append({'held_out_station_id': station, 'station_name': station_name,
                             'model': 'training_mean', 'training_rows': len(train), 'purged_rows': len(purged),
                             **regression_metrics(test[TARGET], mean_prediction)})
        all_predictions.append(prediction_rows)
        print(f'{station_name}: train {len(train)}, test {len(test)}, purged {len(purged)}', flush=True)
    predictions, metrics = pd.concat(all_predictions, ignore_index=True), pd.DataFrame(fold_metrics)
    if len(predictions) != len(data) or not predictions.observation_id.is_unique:
        raise ValueError('Each observation must be tested exactly once at an unseen station')
    pooled = pd.DataFrame([{'model': name, **regression_metrics(predictions.actual_bod_mg_l, predictions[name + '_bod_mg_l']),
                           'station_macro_mae': float(metrics.query('model == @name').mae.mean())}
                          for name in [*features, 'training_mean']])
    predictions.to_csv(args.output / 'test_predictions.csv', index=False)
    metrics.to_csv(args.output / 'station_metrics.csv', index=False)
    pooled.to_csv(args.output / 'pooled_metrics.csv', index=False)
    pd.concat(assignments, ignore_index=True).to_csv(args.output / 'fold_assignments.csv', index=False)
    # Expanded-data runs also test frozen existing models only on genuinely new stations.
    frozen_metrics = None
    new_path = args.paired.with_name('new_paired.csv')
    if new_path.exists():
        new = pd.read_csv(new_path)
        original = pd.read_csv('outputs/india_100_chemical_dataset_20261014/paired.csv')
        if set(new.station_id) & set(original.station_id) or set(new.observation_id) & set(original.observation_id):
            raise ValueError('Frozen-model external cases must be genuinely new stations/events')
        frozen_paths = {'satellite_only': Path('outputs/satellite_only_80_20_20261014/model.joblib'),
                        'ground_only': Path('outputs/ground_only_comparison_20261014/ground_only.joblib'),
                        'combined': Path('outputs/straightforward_demos_20261014/combined_all_chemicals.joblib')}
        frozen_rows = new[['observation_id', 'station_id', 'station_name', 'sample_date', TARGET]].copy()
        frozen_metrics = []
        manifest = json.loads(Path('demo_bundle/manifest.json').read_text(encoding='utf-8'))
        expected_hashes = {entry['path']: entry['sha256'] for entry in manifest['files']}
        for name, path in frozen_paths.items():
            if sha(path) != expected_hashes[path.as_posix()]:
                raise ValueError('Frozen model differs from preserved bundle')
            model = joblib.load(path)
            prediction = predict(model, new[model['features']])
            frozen_rows[name + '_bod_mg_l'] = prediction
            frozen_metrics.append({'model': name, **regression_metrics(new[TARGET], prediction),
                                   'frozen_model_sha256': sha(path)})
        mean = np.full(len(new), float(joblib.load(frozen_paths['satellite_only'])['y_scaler'].mean_[0]))
        frozen_rows['training_mean_bod_mg_l'] = mean
        frozen_metrics.append({'model': 'training_mean', **regression_metrics(new[TARGET], mean),
                               'frozen_model_sha256': None})
        frozen_rows.to_csv(args.output / 'frozen_models_new_station_predictions.csv', index=False)
        pd.DataFrame(frozen_metrics).to_csv(args.output / 'frozen_models_new_station_metrics.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, color in [('satellite_only', '#4389b1'), ('ground_only', '#c58c36'), ('combined', '#2a927c'), ('training_mean', '#999999')]:
        group = metrics.query('model == @name')
        ax.plot(range(len(group)), group.mae, 'o-', label=name.replace('_', ' '), color=color)
    combined_metrics = metrics.query("model == 'combined'")
    labels = [LABELS.get(row.held_out_station_id, row.station_name) for row in combined_metrics.itertuples()]
    ax.set_xticks(range(data.station_id.nunique()), labels,
                  rotation=20, ha='right', fontsize=8)
    ax.set(ylabel='BOD MAE (mg/L; lower is better)', title='Each test station is absent from its training fold')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output / 'station_holdout.png', dpi=160)
    plt.close(fig)
    if sha(args.paired) != data_hash or any(sha(p) != old_hashes[str(p)] for p in preserved):
        raise ValueError('Saved source/model changed')
    report = {'paired_sha256': data_hash, 'paired_path': str(args.paired), 'rows': len(data),
              'stations': int(data.station_id.nunique()), 'features': features,
              'preserved_model_hashes': old_hashes, 'fold_diagnostics': diagnostics,
              'settings': {'hidden_layer_sizes': [8], 'activation': 'relu', 'solver': 'lbfgs', 'alpha': 1, 'max_iter': 3000, 'seed': 42},
              'frozen_models_new_station_metrics': frozen_metrics,
              'purge': 'Remove every training event sharing a sensing-acquisition group with any held-out event.',
              'limitations': ['Retrospective station holdout; training dates can overlap test dates. Not future forecasting.',
                              'Different folds train different models; main model is preserved.',
                              'Shared-date/source biases and provisional coordinate/laboratory QC remain.',
                              'No station-specific tuning, model selection or WQI/category changes.',
                              'This is within-Ganga station transfer, not foreign-river validation.']}
    report['output_sha256'] = {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(pooled.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

"""Evaluate frozen ratio, selected-feature and five network models at new stations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/unseen_saved_mpl_cache'))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from river_water_quality.training.experimental_network import predict_network
from river_water_quality.training.satellite_refinements import predict as ratio_predict
from river_water_quality.training.sklearn_demonstrations import predict as selected_predict
from river_water_quality.training.sklearn_fusion import regression_metrics
from river_water_quality.training.splits import acquisition_group

VARIANTS = ['baseline', 'gate_only', 'constraint_only', 'station_adversarial_only', 'all_components']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort_audit(train, new):
    if not train.observation_id.is_unique or not new.observation_id.is_unique:
        raise ValueError('Duplicate event IDs')
    if set(train.observation_id) & set(new.observation_id):
        raise ValueError('Training/test events overlap')
    if set(train.station_id) & set(new.station_id):
        raise ValueError('Test station present in training')
    if new.station_id.isna().any() or new.scene_id.isna().any():
        raise ValueError('Missing test station/acquisition')
    result = new[['observation_id', 'station_id', 'station_name', 'sample_date', 'scene_id']].copy()
    acquisitions = set(train.scene_id.map(acquisition_group))
    result['shared_training_acquisition'] = new.scene_id.map(acquisition_group).isin(acquisitions)
    result['eligible_strict_test'] = ~result.shared_training_acquisition
    return result


def score(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (actual <= 0).any():
        raise ValueError('MAPE requires finite positive BOD references')
    return {**regression_metrics(actual, predicted),
            'mape_percent': float(np.mean(np.abs(predicted-actual)/actual) * 100)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/saved_models_unseen_results')
    args = parser.parse_args()
    paired_path = ROOT / 'outputs/india_100_chemical_dataset_20261014/paired.csv'
    split_path = ROOT / 'outputs/satellite_only_80_20_20261014/split_assignments.csv'
    new_path = ROOT / 'outputs/new_station_matches_v2_20261014/new_paired.csv'
    spec_path = ROOT / 'configs/working_satellite_model.json'
    demo_path = ROOT / 'outputs/straightforward_demos_20261014/run.json'
    network_path = ROOT / 'outputs/experimental_network_20261014/run.json'
    paired, splits, new = pd.read_csv(paired_path), pd.read_csv(split_path), pd.read_csv(new_path)
    if set(paired.observation_id) != set(splits.observation_id) or not splits.observation_id.is_unique:
        raise ValueError('Original dataset/split mismatch')
    train = paired.set_index('observation_id').loc[splits.loc[splits.split.eq('train'), 'observation_id']].reset_index()
    seen = paired.set_index('observation_id').loc[splits.loc[splits.split.eq('test'), 'observation_id']].reset_index()
    if len(train) != 80 or len(seen) != 20 or len(new) != 30 or new.station_id.nunique() != 4:
        raise ValueError('Unexpected saved evaluation cohorts')
    audit = cohort_audit(train, new)
    strict = new.loc[audit.eligible_strict_test].copy()
    spec, demos, networks = [json.loads(p.read_text()) for p in [spec_path, demo_path, network_path]]
    if networks['input_sha256'] != digest(paired_path) or networks['split_sha256'] != digest(split_path):
        raise ValueError('Experimental checkpoint training sources differ')
    definitions = [
        ('satellite_ratios', ROOT / spec['model_path'], spec['model_sha256'], 'ratio'),
        ('selected_four_inputs', ROOT / 'outputs/straightforward_demos_20261014/selected_features.joblib',
         demos['model_hashes']['selected_features'], 'selected')]
    for name in VARIANTS:
        definitions.append(('network_' + name, ROOT / f'outputs/experimental_network_20261014/{name}.pt',
                            networks['variants'][name]['model_sha256'], 'network'))
    predictors, hashes = {}, {}
    torch.set_num_threads(1)
    for name, path, expected, kind in definitions:
        if digest(path) != expected:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
        hashes[str(path.relative_to(ROOT))] = expected
        if kind == 'network':
            model = torch.load(path, map_location='cpu', weights_only=True)
            predictors[name] = lambda frame, m=model: predict_network(m, frame)['predicted']
        else:
            model = joblib.load(path)
            fn = ratio_predict if kind == 'ratio' else selected_predict
            predictors[name] = lambda frame, m=model, f=fn: f(m, frame)
    mean = float(train.ground_bod.mean())
    predictors['training_mean'] = lambda frame: np.full(len(frame), mean)
    metric_rows, station_rows, prediction_rows = [], [], []
    for cohort, frame in [('familiar_20', seen), ('new_stations_all_30', new), ('new_stations_strict_26', strict)]:
        rows = frame[['observation_id', 'station_id', 'station_name', 'sample_date']].copy()
        rows.insert(0, 'cohort', cohort)
        rows['actual_bod_mg_l'] = frame.ground_bod.to_numpy()
        for name, predictor in predictors.items():
            # Remove target and station identity from inference, ensuring no answer/ID leakage.
            inputs = frame.drop(columns=['ground_bod', 'station_id'])
            values = predictor(inputs)
            rows[name + '_bod_mg_l'] = values
            metric_rows.append({'cohort': cohort, 'model': name, **score(frame.ground_bod, values)})
            for station_id in frame.station_id.unique():
                mask = frame.station_id.eq(station_id).to_numpy()
                station_rows.append({'cohort': cohort, 'model': name, 'station_id': station_id,
                    'station_name': frame.loc[mask, 'station_name'].iloc[0],
                    **score(frame.loc[mask, 'ground_bod'], values[mask])})
        prediction_rows.append(rows)
    if any(digest(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError('Saved models changed during evaluation')
    metrics = pd.DataFrame(metric_rows)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(station_rows).to_csv(out / 'station_metrics.csv', index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out / 'predictions.csv', index=False)
    audit.to_csv(out / 'cohort_audit.csv', index=False)
    new.to_csv(out / 'new_station_inputs.csv', index=False)
    shown = metrics[metrics.cohort.isin(['familiar_20', 'new_stations_strict_26'])]
    fig, ax = plt.subplots(figsize=(11, 6))
    shown.pivot(index='model', columns='cohort', values='mae').plot.barh(ax=ax)
    ax.set_xlabel('BOD mean absolute error (mg/L; lower is better)')
    ax.set_title('Frozen models: familiar stations versus four unseen stations')
    fig.tight_layout()
    fig.savefig(out / 'seen_vs_unseen.png', dpi=140)
    plt.close(fig)
    report = {'training_rows': 80, 'familiar_test_rows': 20, 'new_station_rows': 30,
        'strict_new_station_rows': len(strict), 'excluded_shared_acquisition_rows': int(audit.shared_training_acquisition.sum()),
        'strict_station_counts': strict.groupby('station_name').size().to_dict(),
        'model_sha256': hashes, 'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
            [paired_path, split_path, new_path, spec_path, demo_path, network_path]},
        'retraining': False, 'test_based_tuning': False,
        'mape_definition': '100 * mean(abs(predicted BOD - measured BOD) / measured BOD)',
        'limitations': ['Station transfer within Uttar Pradesh/Ganga, not cross-country validation.',
            'Dates may overlap training period: retrospective estimation, not forecasting.',
            'Four of 30 new cases share a training acquisition; primary report excludes them.',
            'New stations have a narrow BOD range; compare the training-mean reference.',
            'Existing site-selection, laboratory, cloud/water-pixel and source assumptions remain.',
            'No threshold, feature or checkpoint choice is changed using these new results.']}
    report['output_sha256'] = {p.name: digest(p) for p in out.iterdir() if p.name != 'provenance.json'}
    (out / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()

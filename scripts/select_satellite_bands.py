"""Nested station-validation band selection with fixed existing MLP settings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/matplotlib_cache'))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.evaluate_unseen_stations import LABELS, station_fold
from river_water_quality.training.satellite_selection import candidates, select_bands
from river_water_quality.training.sklearn_demonstrations import fit, predict
from river_water_quality.training.sklearn_fusion import TARGET, regression_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/satellite_band_selection_20261014'))
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise ValueError('Output exists; choose a fresh directory')
    paired = Path('outputs/new_station_matches_v2_20261014/paired.csv')
    baseline_folder = Path('outputs/unseen_stations_expanded_20261014')
    baseline_path = baseline_folder / 'test_predictions.csv'
    protected = [paired, baseline_path, Path('outputs/satellite_only_80_20_20261014/model.joblib')]
    hashes = {str(p): sha(p) for p in protected}
    manifest = json.loads(Path('demo_bundle/manifest.json').read_text(encoding='utf-8'))
    expected = {entry['path']: entry['sha256'] for entry in manifest['files']}
    if any(sha(p) != expected[p.as_posix()] for p in protected):
        raise ValueError('Frozen input differs from saved bundle')
    data = pd.read_csv(paired).sort_values(['station_id', 'sample_date', 'observation_id']).reset_index(drop=True)
    baseline = pd.read_csv(baseline_path).set_index('observation_id')
    if len(data) != 130 or data.station_id.nunique() != 7 or set(data.observation_id) != set(baseline.index):
        raise ValueError('Expected the same 130-event/seven-station evaluation cohort')
    out.mkdir(parents=True)
    predictions, metrics, selected, scores_all, details_all, inner_all, outer_all = [], [], [], [], [], [], []
    for station in sorted(data.station_id.unique()):
        train, test, purged = station_fold(data, station)
        print(f'{LABELS.get(station, station)}: selecting bands using {len(train)} training rows / six stations only...', flush=True)
        bands, choice, scores, details, inner_assignments = select_bands(train)
        model = fit(train, bands)
        path = out / f'{station}_selected.joblib'
        joblib.dump(model, path)
        prediction = predict(model, test[bands])
        np.testing.assert_allclose(predict(joblib.load(path), test[bands]), prediction, rtol=0, atol=1e-12)
        reference = baseline.loc[test.observation_id]
        np.testing.assert_allclose(reference.actual_bod_mg_l, test[TARGET], rtol=0, atol=1e-12)
        base_model_path = baseline_folder / f'{station}_satellite_only.joblib'
        if sha(base_model_path) != expected[base_model_path.as_posix()]:
            raise ValueError('Baseline fold checkpoint changed')
        base_model = joblib.load(base_model_path)
        np.testing.assert_allclose(predict(base_model, test[base_model['features']]), reference.satellite_only_bod_mg_l, rtol=0, atol=1e-7)
        np.testing.assert_allclose(base_model['y_scaler'].mean_, train[TARGET].mean(), rtol=0, atol=1e-12)
        rows = test[['observation_id', 'station_id', 'station_name', 'sample_date', 'scene_id']].copy()
        rows['actual_bod_mg_l'], rows['selected_bod_mg_l'] = test[TARGET], prediction
        rows['all_bands_bod_mg_l'] = reference.satellite_only_bod_mg_l.to_numpy()
        rows['training_mean_bod_mg_l'] = train[TARGET].mean()
        rows['selected_candidate'] = choice
        predictions.append(rows)
        for name in ['selected', 'all_bands', 'training_mean']:
            metrics.append({'station_id': station, 'station_name': test.station_name.iloc[0], 'model': name,
                            'training_rows': len(train), 'purged_rows': len(purged),
                            **regression_metrics(test[TARGET], rows[name + '_bod_mg_l'])})
        selected.append({'held_out_station_id': station, 'candidate': choice, 'features': bands,
                         'inner_validation_macro_mae': float(scores.loc[scores.selected, 'macro_validation_mae'].iloc[0]),
                         'training_iterations': int(model['model'].n_iter_), 'warnings': model['warnings'],
                         'model_sha256': sha(path)})
        for table, collection in [(scores, scores_all), (details, details_all), (inner_assignments, inner_all)]:
            table['outer_test_station_id'] = station
            collection.append(table)
        for label, frame in [('train', train), ('test', test), ('purged_shared_acquisition', purged)]:
            table = frame[['observation_id', 'station_id', 'scene_id']].copy()
            table['outer_test_station_id'], table['assignment'] = station, label
            outer_all.append(table)
        pd.DataFrame(selected).to_json(out / 'selected_subsets.json', orient='records', indent=2)
        print(f'  Chosen {choice}: {len(bands)} bands; held-out predictions saved.', flush=True)
    rows, station_metrics = pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)
    if len(rows) != 130 or not rows.observation_id.is_unique:
        raise ValueError('Every event must be predicted once at an unseen station')
    pooled = pd.DataFrame([{'model': name, **regression_metrics(rows.actual_bod_mg_l, rows[name + '_bod_mg_l']),
                           'station_macro_mae': float(station_metrics.query('model == @name').mae.mean())}
                          for name in ['selected', 'all_bands', 'training_mean']])
    rows.to_csv(out / 'test_predictions.csv', index=False)
    station_metrics.to_csv(out / 'station_metrics.csv', index=False)
    pooled.to_csv(out / 'pooled_metrics.csv', index=False)
    for collection, name in [(scores_all, 'candidate_scores'), (details_all, 'inner_candidate_metrics'),
                             (inner_all, 'inner_assignments'), (outer_all, 'outer_assignments')]:
        pd.concat(collection, ignore_index=True).to_csv(out / f'{name}.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = ['Training-selected bands', 'All ten bands', 'Training mean']
    for ax, metric in zip(axes, ['mae', 'rmse']):
        ax.bar(names, pooled[metric], color=['#2a927c', '#4389b1', '#999999'])
        ax.set_ylabel(f'BOD {metric.upper()} (mg/L; lower is better)')
        ax.tick_params(axis='x', labelrotation=15)
        ax.set_ylim(0, pooled[metric].max() * 1.2)
        for index, value in enumerate(pooled[metric]):
            ax.text(index, value + .02, f'{value:.3f}', ha='center')
    fig.suptitle('Seven unseen stations: band choice uses internal training stations only')
    fig.tight_layout()
    fig.savefig(out / 'band_selection_comparison.png', dpi=160)
    plt.close(fig)
    if any(sha(p) != hashes[str(p)] for p in protected):
        raise ValueError('Saved dataset/model changed')
    report = {'input_sha256': hashes, 'rows': 130, 'outer_stations': 7, 'candidate_subsets': candidates(),
              'selection': 'Lowest mean inner-station validation MAE; fewer inputs and name break ties.',
              'settings': {'hidden_layer_sizes': [8], 'activation': 'relu', 'solver': 'lbfgs', 'alpha': 1, 'max_iter': 3000, 'seed': 42},
              'selected_by_outer_fold': selected,
              'limitations': ['Only satellite bands are inputs; no station chemical values or target BOD in inference.',
                              'Nested validation prevents outer test answers entering band selection; examples were examined in earlier experiments.',
                              'Shared sensing groups purged at outer and inner levels; retrospective dates can overlap.',
                              'No hyperparameter search, new indices, date filtering or main-model replacement.',
                              'These evaluate a selection procedure; the seven checkpoints differ by training/selected subset.',
                              'Small uneven station groups; no reliable foreign-river or future generalisation claim.']}
    report['output_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(pooled.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

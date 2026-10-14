"""Evaluate frozen BOD-to-WQI substitution on the saved chronological 20 cases.

Both reference and estimated WQI are assumption-based scenarios. No retraining.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs' / 'wqi_matplotlib_cache'))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.calculate_wqi import calculate_rows
from river_water_quality.training.sklearn_demonstrations import predict
from river_water_quality.wqi import METHOD_SOURCE, OXYGEN_SOURCE

CLASSES = ['Very Poor', 'Poor', 'Good', 'Good-Excellent']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_test_inputs(paired, assignments, saved_predictions):
    if not all(f.observation_id.is_unique for f in (paired, assignments, saved_predictions)):
        raise ValueError('Duplicate observation identifiers')
    ids = assignments.loc[assignments.split.eq('test'), 'observation_id'].tolist()
    if len(ids) != 20 or set(saved_predictions.observation_id) != set(ids):
        raise ValueError('Expected exactly the same saved 20 test observations')
    if len(assignments) != 100 or assignments.split.eq('train').sum() != 80:
        raise ValueError('Expected preserved 80/20 assignments')
    result = assignments.loc[assignments.split.eq('test'), ['observation_id']].merge(
        paired, on='observation_id', how='left', validate='one_to_one', indicator=True)
    if not result._merge.eq('both').all():
        raise ValueError('Missing test chemical observation')
    saved = saved_predictions.set_index('observation_id').loc[ids]
    for name in ['station_id', 'sample_date', 'scene_id']:
        if result[name].tolist() != saved[name].tolist():
            raise ValueError(f'Test metadata differs: {name}')
    np.testing.assert_allclose(result.ground_bod, saved.actual_bod_mg_l, rtol=0, atol=1e-12)
    result['predicted_bod_mg_l'] = saved.predicted_bod_mg_l.to_numpy()
    return result.drop(columns='_merge')


def score_pairs(frame, temperatures, pressure, salinity, mean_bod):
    estimated = calculate_rows(frame, temperatures, pressure, salinity)
    reference = calculate_rows(frame.assign(predicted_bod_mg_l=frame.ground_bod), temperatures, pressure, salinity)
    baseline = calculate_rows(frame.assign(predicted_bod_mg_l=mean_bod), temperatures, pressure, salinity)
    keys = ['observation_id', 'assumed_temperature_c']
    fields = ['wqi', 'category', 'status', 'message', 'contribution_bod', 'sub_index_bod']
    result = estimated.rename(columns={f: 'estimated_' + f for f in fields})
    for name, table in [('reference', reference), ('training_mean_bod', baseline)]:
        result = result.merge(table[keys + fields].rename(columns={f: name + '_' + f for f in fields}),
                              on=keys, how='left', validate='one_to_one')
    truth = frame.set_index('observation_id').ground_bod
    result['measured_bod_mg_l'] = result.observation_id.map(truth)
    result['comparable'] = result.estimated_wqi.notna() & result.reference_wqi.notna()
    result['absolute_wqi_error'] = (result.estimated_wqi - result.reference_wqi).abs()
    result['signed_wqi_error'] = result.estimated_wqi - result.reference_wqi
    result['category_agrees'] = pd.Series(pd.NA, index=result.index, dtype='boolean')
    mask = result.comparable & result.estimated_category.notna() & result.reference_category.notna()
    result.loc[mask, 'category_agrees'] = result.loc[mask, 'estimated_category'].eq(result.loc[mask, 'reference_category'])
    return result


def summaries(result):
    metrics, confusion = [], []
    for t, group in result.groupby('assumed_temperature_c', sort=True):
        valid = group.loc[group.comparable]
        common = {'assumed_temperature_c': float(t), 'attempted': len(group), 'scored': len(valid),
                  'excluded': int((~group.comparable).sum()),
                  'formula_warning_rows': int((group.estimated_status.eq('formula_warning')
                                                | group.reference_status.eq('formula_warning')).sum())}
        for name, col in [('saved_satellite_only_bod_model', 'estimated'), ('training_mean_bod', 'training_mean_bod')]:
            scored = valid.loc[valid[col + '_wqi'].notna()]
            error = scored[col + '_wqi'] - scored.reference_wqi
            label_counts = scored.reference_category.value_counts().to_dict()
            metrics.append({**common, 'model': name, 'metric_rows': len(scored),
                'wqi_mae': float(error.abs().mean()) if len(scored) else None,
                'wqi_rmse': float(np.sqrt(np.mean(error**2))) if len(scored) else None,
                'mean_signed_wqi_error': float(error.mean()) if len(scored) else None,
                'category_agreement_count': int(scored[col + '_category'].eq(scored.reference_category).sum()),
                'category_agreement_fraction': float(scored[col + '_category'].eq(scored.reference_category).mean()) if len(scored) else None,
                'reference_category_counts': json.dumps(label_counts, sort_keys=True),
                'reference_majority_fraction': max(label_counts.values()) / len(scored) if len(scored) else None})
        for actual in CLASSES:
            for predicted in CLASSES:
                confusion.append({'assumed_temperature_c': t, 'reference_category': actual,
                    'estimated_category': predicted,
                    'count': int((valid.reference_category.eq(actual) & valid.estimated_category.eq(predicted)).sum())})
    return pd.DataFrame(metrics), pd.DataFrame(confusion)


def charts(result, metrics, output):
    temperatures = sorted(result.assumed_temperature_c.unique())
    fig, axes = plt.subplots(1, len(temperatures), figsize=(4.3 * len(temperatures), 4), squeeze=False)
    for ax, t in zip(axes[0], temperatures):
        group = result.loc[result.assumed_temperature_c.eq(t) & result.comparable]
        ax.scatter(group.reference_wqi, group.estimated_wqi, color='#197a8a', alpha=0.8)
        ax.plot([35, 85], [35, 85], '--', color='gray', lw=1)
        ax.set(xlim=(35, 85), ylim=(35, 85), xlabel='Reference scenario WQI', ylabel='Estimated scenario WQI',
               title=f'Assumed {t:g} C: {len(group)}/20 scored')
        ax.grid(alpha=0.2)
    fig.suptitle('Same measured pH, oxygen and fecal inputs; only BOD is substituted')
    fig.tight_layout()
    fig.savefig(output / 'wqi_comparison.png', dpi=160)
    plt.close(fig)
    main = metrics.query("model == 'saved_satellite_only_bod_model'")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = [f'{t:g} C' for t in main.assumed_temperature_c]
    axes[0].bar(labels, main.wqi_mae, color='#197a8a')
    axes[0].set(ylabel='Mean absolute WQI difference', title='Score differences')
    axes[1].bar(labels, main.category_agreement_fraction * 100, color='#e49b39')
    axes[1].set(ylim=(0, 110), ylabel='Category agreement (%)', title='Agreement within each scenario')
    for i, row in enumerate(main.itertuples()):
        axes[1].text(i, row.category_agreement_fraction * 100 + 2,
                     f'{row.category_agreement_count}/{row.metric_rows}', ha='center')
    fig.suptitle('Assumed pressure 1 atm and freshwater; missing temperatures limit interpretation')
    fig.tight_layout()
    fig.savefig(output / 'wqi_metrics.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists; choose a fresh directory')
    folder = Path('outputs/satellite_only_80_20_20261014')
    paired_path = Path('outputs/india_100_chemical_dataset_20261014/paired.csv')
    paths = [paired_path, folder / 'model.joblib', folder / 'split_assignments.csv', folder / 'test_predictions.csv']
    hashes = {str(p): digest(p) for p in paths}
    expected = json.loads(paired_path.with_name('provenance.json').read_text(encoding='utf-8'))
    if hashes[str(paired_path)] != expected['output_sha256']:
        raise ValueError('Saved paired dataset changed')
    for path, value in expected['preserved_model_hashes'].items():
        if digest(path) != value:
            raise ValueError('Saved model changed')
    # Verify all frozen input files against the committed bundle manifest.
    bundle = json.loads(Path('demo_bundle/manifest.json').read_text(encoding='utf-8'))
    entries = {entry['path']: entry['sha256'] for entry in bundle['files']}
    if any(entries[p.as_posix()] != hashes[str(p)] for p in paths):
        raise ValueError('Frozen source differs from bundle')
    paired, splits, predictions = [pd.read_csv(p) for p in [paired_path, folder / 'split_assignments.csv', folder / 'test_predictions.csv']]
    frame = prepare_test_inputs(paired, splits, predictions)
    model = joblib.load(folder / 'model.joblib')
    np.testing.assert_allclose(predict(model, frame[model['features']]), frame.predicted_bod_mg_l, rtol=0, atol=1e-7)
    mean_bod = float(model['y_scaler'].mean_[0])
    result = score_pairs(frame, [20, 25, 30], 1, 0, mean_bod)
    metrics, confusion = summaries(result)
    # In the cohort's BOD range, the formula has a known 1.33 WQI-point/mg/L slope.
    valid = result.loc[result.comparable & result.predicted_bod_mg_l.le(10) & result.measured_bod_mg_l.le(10)]
    np.testing.assert_allclose(valid.signed_wqi_error,
        -1.33 * (valid.predicted_bod_mg_l - valid.measured_bod_mg_l), rtol=0, atol=1e-10)
    args.output.mkdir(parents=True)
    frame[['observation_id', 'station_id', 'station_name', 'sample_date', 'scene_id',
           'ground_bod', 'predicted_bod_mg_l', 'ground_ph', 'ground_dissolved_oxygen',
           'ground_fecal_coliform']].to_csv(args.output / 'test_inputs.csv', index=False)
    result.to_csv(args.output / 'wqi_predictions.csv', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    confusion.to_csv(args.output / 'category_confusion.csv', index=False)
    stability = result.groupby('observation_id').agg(scored_scenarios=('comparable', 'sum'),
        reference_category_count=('reference_category', 'nunique'), estimated_category_count=('estimated_category', 'nunique'))
    stability.to_csv(args.output / 'temperature_category_stability.csv')
    charts(result, metrics, args.output)
    report = {'model': 'Preserved satellite-only 80/20 BOD model; unchanged', 'input_sha256': hashes,
              'test_ids': frame.observation_id.tolist(), 'train_rows': 80, 'test_rows': 20,
              'training_mean_bod_mg_l': mean_bod, 'temperatures_c': [20, 25, 30],
              'pressure_atm': 1, 'salinity_ppt': 0, 'method_source': METHOD_SOURCE, 'oxygen_source': OXYGEN_SOURCE,
              'common_supported_test_count': int(stability.scored_scenarios.eq(3).sum()),
              'reference_category_changes': int(stability.reference_category_count.gt(1).sum()),
              'estimated_category_changes': int(stability.estimated_category_count.gt(1).sum()),
              'limitations': ['Both reference and estimated WQI use assumed, not measured, water conditions.',
                              'Both share pH, DO and fecal inputs; agreement only tests BOD substitution.',
                              'A WQI category is not a drinking-water safety decision.',
                              'Temperature-specific unsupported rows remain explicit; no capping or imputation.',
                              'The same 20 examples were previously examined; this is exploratory, not new external validation.',
                              'The baseline substitutes training-mean BOD into shared measured parameters; it is not a fitted ground-only model.']}
    if any(digest(p) != hashes[str(p)] for p in paths):
        raise ValueError('Frozen source changed during evaluation')
    report['output_sha256'] = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / 'provenance.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(metrics.drop(columns=['reference_category_counts']).to_string(index=False))
    print(json.dumps({k: report[k] for k in ['common_supported_test_count', 'reference_category_changes', 'estimated_category_changes']}))


if __name__ == '__main__':
    main()

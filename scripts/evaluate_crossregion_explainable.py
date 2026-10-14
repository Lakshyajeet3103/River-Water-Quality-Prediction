"""Frozen India-trained BOD models on India test rows and four real US matches."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/crossregion_mpl_cache'))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from river_water_quality.training.satellite_refinements import predict as satellite_predict
from river_water_quality.training.sklearn_demonstrations import predict as combined_predict
from river_water_quality.training.sklearn_fusion import SATELLITE_FEATURES, regression_metrics
from river_water_quality.training.modality_explanation import explain_groups


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/crossregion_explainable_results')
    args = parser.parse_args()
    spec_path = ROOT / 'configs/working_satellite_model.json'
    spec = json.loads(spec_path.read_text())
    satellite_path = ROOT / spec['model_path']
    combined_path = ROOT / 'outputs/straightforward_demos_20261014/satellite_plus_fecal.joblib'
    combined_run_path = ROOT / 'outputs/straightforward_demos_20261014/run.json'
    combined_run = json.loads(combined_run_path.read_text())
    if digest(satellite_path) != spec['model_sha256'] or digest(combined_path) != combined_run['model_hashes']['satellite_plus_fecal']:
        raise ValueError('Saved model hash mismatch')
    satellite, combined = joblib.load(satellite_path), joblib.load(combined_path)
    ground = ['ground_fecal_coliform']
    if combined['features'] != SATELLITE_FEATURES + ground:
        raise ValueError('Expected compatible satellite-plus-fecal schema')
    paired_path = ROOT / 'outputs/india_100_chemical_dataset_20261014/paired.csv'
    split_path = ROOT / 'outputs/satellite_only_80_20_20261014/split_assignments.csv'
    us_path = ROOT / 'demo_bundle/files/outputs/crossregion_demo_001/us_paired.csv'
    paired, split, us = pd.read_csv(paired_path), pd.read_csv(split_path), pd.read_csv(us_path)
    if len(us) != 4 or not us.observation_id.is_unique or not paired.observation_id.is_unique:
        raise ValueError('Expected unique original India observations and four US cases')
    train = paired.set_index('observation_id').loc[split.loc[split.split.eq('train'), 'observation_id']].reset_index()
    india = paired.set_index('observation_id').loc[split.loc[split.split.eq('test'), 'observation_id']].reset_index()
    if len(train) != 80 or len(india) != 20 or set(us.observation_id) & set(paired.observation_id):
        raise ValueError('Unexpected split or shared India/US event IDs')
    background = train.sample(n=20, random_state=42)
    metric_rows, prediction_rows, explanations = [], [], []
    baseline = float(train.ground_bod.mean())
    for region, frame in [('India', india), ('US', us)]:
        rows = frame[['observation_id', 'station_id', 'station_name', 'sample_date']].copy()
        rows.insert(0, 'region', region)
        rows['actual_bod_mg_l'] = frame.ground_bod.to_numpy()
        for name, values in [('selected_satellite', satellite_predict(satellite, frame)),
                             ('satellite_plus_fecal', combined_predict(combined, frame)),
                             ('India_training_mean', np.full(len(frame), baseline))]:
            rows[name + '_bod_mg_l'] = values
            metric_rows.append({'region': region, 'model': name, **regression_metrics(frame.ground_bod, values)})
        prediction_rows.append(rows)
        # All four US cases and one India test example per station.
        examples = frame if region == 'US' else frame.groupby('station_id').head(1)
        for _, example in examples.iterrows():
            explanation = explain_groups(lambda f: combined_predict(combined, f), example,
                                         background, SATELLITE_FEATURES, ground)
            if abs(explanation['reconstruction_error_mg_l']) > 1e-8:
                raise ValueError('Attributions fail to reconstruct prediction')
            explanations.append({'region': region, 'observation_id': example.observation_id,
                                 'station_id': example.station_id, **explanation})
    metrics, explanation = pd.DataFrame(metric_rows), pd.DataFrame(explanations)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / 'metrics.csv', index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out / 'test_predictions.csv', index=False)
    explanation.to_csv(out / 'modality_explanations.csv', index=False)
    background[['observation_id', *SATELLITE_FEATURES, *ground]].to_csv(out / 'training_background.csv', index=False)
    us.to_csv(out / 'us_inputs.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pivot = metrics.pivot(index='model', columns='region', values='mae')
    pivot.plot.barh(ax=axes[0])
    axes[0].set_xlabel('BOD MAE (mg/L; lower is better)')
    axes[0].set_title('Frozen India-trained models: India n=20, US n=4')
    foreign = explanation[explanation.region.eq('US')].reset_index(drop=True)
    y = np.arange(len(foreign))
    axes[1].barh(y - .18, foreign.satellite_contribution_mg_l, height=.35, label='Satellite')
    axes[1].barh(y + .18, foreign.ground_contribution_mg_l, height=.35, label='Ground: fecal')
    axes[1].set_yticks(y, [f'US case {i+1}: {r.station_id}' for i, r in enumerate(foreign.itertuples())], fontsize=8)
    axes[1].axvline(0, color='black', linewidth=.7)
    axes[1].set_xlabel('Group Shapley contribution to BOD (mg/L)')
    axes[1].set_title('Combined model: US signal contributions')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / 'crossregion_explanation.png', dpi=130)
    plt.close(fig)
    report = {'model_hashes': {'selected_satellite': digest(satellite_path), 'satellite_plus_fecal': digest(combined_path)},
        'data_hashes': {str(p.relative_to(ROOT)): digest(p) for p in [paired_path, split_path, us_path, spec_path, combined_run_path]},
        'training_country': 'India', 'training_count': 80, 'foreign_training_or_tuning': False,
        'explanation': {'method': 'Exact two-player grouped Shapley', 'players': ['all ten satellite bands', 'fecal coliform'],
            'background': 'Seed-42 sample of 20 India training events', 'explained_cases': len(explanation),
            'max_reconstruction_error_mg_l': float(explanation.reconstruction_error_mg_l.abs().max())},
        'limitations': ['Four US cases at three stations; not broad evidence of robustness.',
            'Foreign rivers are not assumed clean; measured BOD spans 2 to 9 mg/L.',
            'No foreign pH/DO, so all-chemistry models and station-average categories are not evaluated here.',
            'Main satellite model has no ground input; group explanation belongs to the separate compatible combined model.',
            'Shapley attributions depend on background and are not causal or learned gate weights.',
            'Laboratory methods, time/location matching and water-pixel assumptions may differ by country.',
            'India 20-case scores are exploratory and have been repeatedly inspected.']}
    (out / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(metrics.to_string(index=False))
    print(f'Saved report: {out}')


if __name__ == '__main__':
    main()

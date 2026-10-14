"""Ratios, regularisation and tighter dates on the original seen-station split."""
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

from river_water_quality.training.satellite_refinements import ALPHAS, RATIOS, fit, predict, select
from river_water_quality.training.sklearn_demonstrations import check_separation, predict as original_predict
from river_water_quality.training.sklearn_fusion import TARGET, regression_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/satellite_seen_refinements_20261014'))
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise ValueError('Output exists; choose a fresh directory')
    paired = Path('outputs/india_100_chemical_dataset_20261014/paired.csv')
    split = Path('outputs/satellite_only_80_20_20261014/split_assignments.csv')
    internal_path = Path('outputs/straightforward_demos_20261014/selection_split_assignments.csv')
    old_model = Path('outputs/satellite_only_80_20_20261014/model.joblib')
    old_predictions = Path('outputs/satellite_only_80_20_20261014/test_predictions.csv')
    protected = [paired, split, internal_path, old_model, old_predictions]
    hashes = {str(p): sha(p) for p in protected}
    expected = {e['path']: e['sha256'] for e in json.loads(Path('demo_bundle/manifest.json').read_text(encoding='utf-8'))['files']}
    if any(hashes[str(p)] != expected[p.as_posix()] for p in protected):
        raise ValueError('Protected input differs from saved bundle')
    data, assignments = pd.read_csv(paired), pd.read_csv(split)
    if not data.observation_id.is_unique or not assignments.observation_id.is_unique or set(data.observation_id) != set(assignments.observation_id):
        raise ValueError('Invalid original cohort')
    data = data.merge(assignments[['observation_id', 'split']], on='observation_id', validate='one_to_one')
    data = data.sort_values(['sample_date', 'station_id', 'observation_id']).reset_index(drop=True)
    train, test = data.query("split == 'train'").drop(columns='split'), data.query("split == 'test'").drop(columns='split')
    if len(train) != 80 or len(test) != 20:
        raise ValueError('Expected saved 80/20 split')
    check_separation(train, test)
    internal = pd.read_csv(internal_path)
    if set(internal.observation_id) != set(train.observation_id) or not internal.observation_id.is_unique:
        raise ValueError('Internal assignments must contain only the outer training IDs')
    inner = train.merge(internal, on='observation_id', validate='one_to_one')
    fitting, validation = inner.query("split == 'selection_train'"), inner.query("split == 'selection_validation'")
    if len(fitting) != 60 or len(validation) != 20:
        raise ValueError('Expected saved internal 60/20 split')
    tight_train = train[train.temporal_offset_days.abs().le(3)]
    tight_test = test[test.temporal_offset_days.abs().le(3)]
    tight_fit = fitting[fitting.temporal_offset_days.abs().le(3)]
    tight_val = validation[validation.temporal_offset_days.abs().le(3)]
    check_separation(tight_train, tight_test)
    scores, choice, raw_alpha = select(fitting, validation)
    tight_scores, tight_choice, _ = select(tight_fit, tight_val)
    # All choices are fixed using internal validation, before outer test metrics.
    specs = {'ratios_only_alpha1': ('bands_ratios', 1.0, train),
             'regularisation_only': ('bands', raw_alpha, train),
             'selected_pipeline': (choice['input_mode'], choice['alpha'], train),
             'three_day_bands_alpha1': ('bands', 1.0, tight_train),
             'three_day_selected_pipeline': (tight_choice['input_mode'], tight_choice['alpha'], tight_train)}
    models = {name: fit(frame, mode, alpha) for name, (mode, alpha, frame) in specs.items()}
    original = joblib.load(old_model)
    reference = original_predict(original, test[original['features']])
    previous = pd.read_csv(old_predictions).set_index('observation_id').loc[test.observation_id]
    np.testing.assert_allclose(previous.predicted_bod_mg_l, reference, rtol=0, atol=1e-7)
    np.testing.assert_allclose(previous.actual_bod_mg_l, test[TARGET], rtol=0, atol=1e-12)
    # Prove the unchanged training implementation reproduces the original model.
    raw_reference = fit(train, 'bands', 1.0)
    np.testing.assert_allclose(predict(raw_reference, test), reference, rtol=0, atol=1e-7)
    out.mkdir(parents=True)
    rows = test[['observation_id', 'station_id', 'station_name', 'sample_date', 'temporal_offset_days']].copy()
    rows['actual_bod_mg_l'], rows['original_bod_mg_l'] = test[TARGET], reference
    rows['eligible_three_day'] = rows.temporal_offset_days.abs().le(3)
    for name, model in models.items():
        joblib.dump(model, out / f'{name}.joblib')
        prediction = predict(model, test)
        np.testing.assert_allclose(predict(joblib.load(out / f'{name}.joblib'), test), prediction, rtol=0, atol=1e-12)
        rows[name + '_bod_mg_l'] = prediction
    rows['training_mean_80_bod_mg_l'] = train[TARGET].mean()
    rows['training_mean_three_day_bod_mg_l'] = tight_train[TARGET].mean()
    full_models = ['original', 'ratios_only_alpha1', 'regularisation_only', 'selected_pipeline', 'training_mean_80']
    tight_models = [*full_models, 'three_day_bands_alpha1', 'three_day_selected_pipeline', 'training_mean_three_day']
    records = []
    for cohort, frame, names in [('original_20', rows, full_models),
                                 ('identical_three_day_test_cases', rows[rows.eligible_three_day], tight_models)]:
        for name in names:
            records.append({'test_cohort': cohort, 'model': name,
                            'training_rows': len(tight_train) if name.startswith('three_day') or name == 'training_mean_three_day' else 80,
                            **regression_metrics(frame.actual_bod_mg_l, frame[name + '_bod_mg_l'])})
    metrics = pd.DataFrame(records)
    rows.to_csv(out / 'test_predictions.csv', index=False)
    metrics.to_csv(out / 'metrics.csv', index=False)
    scores.to_csv(out / 'candidate_validation_scores.csv', index=False)
    tight_scores.to_csv(out / 'three_day_candidate_validation_scores.csv', index=False)
    eligibility = data[['observation_id', 'station_id', 'sample_date', 'split', 'temporal_offset_days']].copy()
    eligibility['eligible_three_day'] = eligibility.temporal_offset_days.abs().le(3)
    eligibility.to_csv(out / 'date_eligibility.csv', index=False)
    internal.to_csv(out / 'selection_split_assignments.csv', index=False)
    for cohort, filename in [('original_20', 'full_test_comparison.png'),
                              ('identical_three_day_test_cases', 'three_day_comparison.png')]:
        group = metrics[metrics.test_cohort.eq(cohort)]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(group.model.str.replace('_', ' '), group.mae, color='#4389b1')
        ax.invert_yaxis()
        ax.set(xlabel='BOD MAE (mg/L; lower is better)', title=f"{cohort.replace('_', ' ')}: n={group.n.iloc[0]}")
        for i, value in enumerate(group.mae):
            ax.text(value + .01, i, f'{value:.3f}', va='center')
        ax.set_xlim(0, group.mae.max() * 1.16)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)
    if any(sha(p) != hashes[str(p)] for p in protected):
        raise ValueError('Saved model/data changed')
    report = {'input_sha256': hashes, 'alpha_grid': ALPHAS, 'ratio_features': RATIOS,
              'selected_pipeline': choice, 'selected_raw_alpha': raw_alpha, 'three_day_selected_pipeline': tight_choice,
              'coverage': {'training_all': 80, 'test_all': 20, 'training_three_day': len(tight_train),
                           'test_three_day': len(tight_test), 'inner_fit_all': 60, 'inner_validation_all': 20,
                           'inner_fit_three_day': len(tight_fit), 'inner_validation_three_day': len(tight_val)},
              'model_details': {name: {'features': m['features'], 'alpha': m['alpha'], 'input_mode': m['input_mode'],
                                      'iterations': int(m['model'].n_iter_), 'warnings': m['warnings']} for name, m in models.items()},
              'settings_fixed': {'hidden_layer_sizes': [8], 'activation': 'relu', 'solver': 'lbfgs', 'max_iter': 3000, 'seed': 42},
              'limitations': ['All choices use internal validation, not outer test metrics; these test cases were examined earlier.',
                              'Date-only effect compares 80-row and 57-row training on identical eligible test cases.',
                              'Three-day filtering removes fallback matches; no new imagery or shifted sample dates.',
                              'Ratios add numeric inputs, so input-layer parameter count changes; hidden architecture remains the same.',
                              'No main-model promotion or WQI/category changes; same known three stations only.',
                              'Small validation/test groups; tuning performance is exploratory, not external validation.']}
    report['output_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['selected_pipeline', 'selected_raw_alpha', 'three_day_selected_pipeline', 'coverage']}, indent=2))
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    main()

"""Run fair MLP comparisons, feature selection, and three actual SHAP examples."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
os.environ.setdefault('MPLCONFIGDIR', str(Path('outputs/matplotlib_cache').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn

from river_water_quality.training.sklearn_demonstrations import (
    ALL, GROUND, SATELLITE_FEATURES, TARGET, fit, predict, feature_search, check_separation)
from river_water_quality.training.sklearn_fusion import regression_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/straightforward_demos_20261014')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise ValueError('Output exists; choose a fresh directory')
    data_path = Path('outputs/india_100_chemical_dataset_20261014/paired.csv')
    data = pd.read_csv(data_path)
    provenance = json.loads(data_path.with_name('provenance.json').read_text())
    if sha(data_path) != provenance['output_sha256']:
        raise ValueError('Prepared dataset hash mismatch')
    split_path = Path('outputs/satellite_only_80_20_20261014/split_assignments.csv')
    splits = pd.read_csv(split_path)[['observation_id', 'split']]
    if set(data.observation_id) != set(splits.observation_id) or not splits.observation_id.is_unique:
        raise ValueError('Saved split does not cover exactly the dataset')
    data = data.merge(splits, on='observation_id', validate='one_to_one').sort_values(['sample_date', 'station_id', 'observation_id'])
    train, test = data.query("split == 'train'"), data.query("split == 'test'")
    if len(train) != 80 or len(test) != 20:
        raise ValueError('Expected the established 80/20 split')
    check_separation(train, test)
    preserved = [Path('outputs/satellite_only_80_20_20261014/model.joblib'),
                 Path('outputs/fusion_sklearn_expanded_20261014/fusion_model.joblib')]
    old_hashes = {str(p): sha(p) for p in preserved}
    print('Selecting features inside the 80 training rows only...', flush=True)
    selected, ranking, candidates, inner_splits = feature_search(train)
    feature_sets = {'satellite_only': SATELLITE_FEATURES,
                    'satellite_plus_fecal': SATELLITE_FEATURES + GROUND[:1],
                    'combined_all_chemicals': ALL, 'selected_features': selected}
    models = {name: fit(train, features) for name, features in feature_sets.items()}
    # Prove the satellite reference reproduces the already saved model.
    np.testing.assert_allclose(predict(models['satellite_only'], test),
                               predict(joblib.load(preserved[0]), test), rtol=1e-7, atol=1e-7)
    rows = test[['observation_id', 'station_id', 'station_name', 'sample_date']].copy()
    rows['actual_bod_mg_l'] = test[TARGET]
    metric_rows = []
    for name, model in models.items():
        prediction = predict(model, test)
        rows[f'{name}_bod_mg_l'] = prediction
        metric_rows.append({'model': name, 'split': 'test', 'input_count': len(model['features']),
                            **regression_metrics(test[TARGET], prediction)})
    baseline = np.full(len(test), train[TARGET].mean())
    rows['training_mean_bod_mg_l'] = baseline
    metric_rows.append({'model': 'training_mean', 'split': 'test', 'input_count': 0,
                        **regression_metrics(test[TARGET], baseline)})
    metrics = pd.DataFrame(metric_rows)
    out.mkdir(parents=True)
    for name, model in models.items():
        joblib.dump(model, out / f'{name}.joblib')
        np.testing.assert_allclose(predict(joblib.load(out / f'{name}.joblib'), test), predict(model, test))
    splits.to_csv(out / 'split_assignments.csv', index=False)
    inner_splits.to_csv(out / 'selection_split_assignments.csv', index=False)
    ranking.to_csv(out / 'feature_ranking.csv', index=False)
    candidates.to_csv(out / 'feature_candidates.csv', index=False)
    rows.to_csv(out / 'test_predictions.csv', index=False)
    metrics.to_csv(out / 'metrics.csv', index=False)
    print(metrics.to_string(index=False), flush=True)
    print('Computing Kernel SHAP for one fixed test example per station...', flush=True)
    model = models['combined_all_chemicals']
    background = train.sample(n=20, random_state=42)
    examples = test.groupby('station_id', sort=True).head(1)
    background[['observation_id', *ALL]].to_csv(out / 'shap_background.csv', index=False)
    examples[['observation_id', *ALL]].to_csv(out / 'shap_inputs.csv', index=False)
    np.random.seed(42)
    explainer = shap.KernelExplainer(lambda x: predict(model, pd.DataFrame(x, columns=ALL)),
                                    background[ALL].to_numpy(), feature_names=ALL, link='identity')
    contributions = np.asarray(explainer.shap_values(examples[ALL].to_numpy(), nsamples=512, l1_reg=0, silent=True))
    expected = float(explainer.expected_value)
    actual_predictions = predict(model, examples)
    residual = actual_predictions - (expected + contributions.sum(axis=1))
    if not np.isfinite(contributions).all() or np.abs(residual).max() > 1e-6:
        raise ValueError('SHAP contributions do not reconstruct model predictions')
    explanation = examples[['observation_id', 'station_id', 'station_name', 'sample_date']].reset_index(drop=True)
    explanation['predicted_bod_mg_l'] = actual_predictions
    explanation['reference_bod_mg_l'] = expected
    explanation['satellite_contribution_mg_l'] = contributions[:, :10].sum(axis=1)
    explanation['ground_contribution_mg_l'] = contributions[:, 10:].sum(axis=1)
    explanation['reconstruction_residual_mg_l'] = residual
    for i, feature in enumerate(ALL):
        explanation[f'shap_{feature}_mg_l'] = contributions[:, i]
    explanation.to_csv(out / 'shap_explanations.csv', index=False)
    plots(out, ranking, metrics, explanation, contributions)
    if {str(p): sha(p) for p in preserved} != old_hashes:
        raise ValueError('An existing model changed')
    report = {'data_sha256': sha(data_path), 'saved_split_sha256': sha(split_path),
              'preserved_model_hashes': old_hashes, 'train_rows': 80, 'test_rows': 20,
              'model_settings': {'hidden_layer_sizes': [8], 'activation': 'relu', 'solver': 'lbfgs',
                                 'alpha': 1.0, 'max_iter': 3000, 'seed': 42},
              'features_by_model': feature_sets, 'selected_candidate': candidates.query('selected').candidate.iloc[0],
              'selection_split_rows': inner_splits.split.value_counts().to_dict(),
              'warnings_by_model': {name: model['warnings'] for name, model in models.items()},
              'metrics': metrics.to_dict('records'),
              'shap': {'method': 'Kernel SHAP', 'model': 'combined_all_chemicals', 'background_rows': 20,
                       'background': 'Seed-42 sample of training observations only', 'explained_test_rows': 3,
                       'nsamples': 512, 'l1_reg': 0, 'maximum_residual_mg_l': float(np.abs(residual).max()),
                       'reference': expected, 'seed': 42, 'version': shap.__version__,
                       'documentation': 'https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html'},
              'sklearn_version': sklearn.__version__, 'model_hashes': {name: sha(out / f'{name}.joblib') for name in models},
              'limitations': ['Same known stations and previously examined test events; exploratory comparison.',
                              'Feature choice uses only inner chronological training/validation, then refits on 80 rows.',
                              'No test-based tuning or switching the main saved model.',
                              'Kernel SHAP is approximate and reference-dependent; correlated bands make individual attributions difficult to interpret causally.',
                              'Ground inputs must actually be available at prediction time for combined models.',
                              'Source sampling-method, coordinate/time and water-pixel assumptions remain unchanged.']}
    (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved models, comparisons, feature selection and SHAP charts: {out}', flush=True)


def plots(out, ranking, metrics, explanation, contributions):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, column in zip(axes, ['mae', 'rmse']):
        ax.barh(metrics.model, metrics[column], color='#267caa')
        ax.set_xlabel(f'{column.upper()} (mg/L; lower is better)')
        ax.set_title('Same 20 test observations')
    fig.tight_layout()
    fig.savefig(out / 'model_comparison.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ranking.feature.iloc[::-1], ranking.score.iloc[::-1],
            color=['#b36b23' if f in GROUND else '#267caa' for f in ranking.feature.iloc[::-1]])
    ax.set_xlabel('Absolute correlation with BOD (inner training rows only)')
    fig.tight_layout()
    fig.savefig(out / 'feature_ranking.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    for i, ax in enumerate(axes):
        ax.barh(ALL, contributions[i], color=np.where(contributions[i] >= 0, '#b64c48', '#267caa'))
        ax.axvline(0, color='black', linewidth=.6)
        ax.set_title(f"{explanation.station_id.iloc[i]}\nBOD estimate {explanation.predicted_bod_mg_l.iloc[i]:.2f} mg/L")
        ax.set_xlabel('SHAP contribution (mg/L)')
    fig.suptitle('Combined model: contributions relative to training-background prediction')
    fig.tight_layout()
    fig.savefig(out / 'shap_examples.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()

"""Train the friend's unchanged scikit-learn model on expanded real pairs."""
from pathlib import Path
import hashlib
import json
import subprocess
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning

from river_water_quality.training.sklearn_fusion import train_fusion, FEATURES, TARGET
from river_water_quality.training.splits import assign_splits


def main():
    paired_path = Path('outputs/matches_expansion_20261014/paired.csv')
    out = Path('outputs/fusion_sklearn_expanded_20261014')
    if out.exists():
        raise ValueError('Output already exists; refusing to overwrite')
    paired = pd.read_csv(paired_path)
    settings = {'mode': 'chronological', 'train_end_date': '2021-05-31',
                'validation_end_date': '2021-09-30', 'gap_days': 7}
    splits = assign_splits(paired.set_index('observation_id'), settings)
    print(splits.split.value_counts().to_string(), flush=True)
    out.mkdir(parents=True)
    split_path = out / 'split_assignments.csv'
    splits.reset_index().to_csv(split_path, index=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        metrics = train_fusion(paired_path, split_path, out, seed=42)
    # Preserve exclusions/reasons as well as the retained assignment file.
    splits.reset_index().to_csv(out / 'all_split_assignments.csv', index=False)
    model = joblib.load(out / 'fusion_model.joblib')
    test = paired[paired.observation_id.isin(splits.index[splits.split.eq('test')])]
    x = test[FEATURES].to_numpy(dtype=float).copy()
    x[:, -1] = np.log1p(x[:, -1])
    predicted = np.maximum(0, model['y_scaler'].inverse_transform(
        model['model'].predict(model['x_scaler'].transform(x)).reshape(-1, 1)).ravel())
    saved = pd.read_csv(out / 'test_predictions.csv')
    np.testing.assert_allclose(predicted, saved.predicted_bod_mg_l, rtol=1e-12, atol=1e-12)
    report = json.loads((out / 'run.json').read_text())
    report.update({'split_settings': settings, 'excluded_events': int(splits.split.eq('excluded').sum()),
                   'paired_sha256': hashlib.sha256(paired_path.read_bytes()).hexdigest(),
                   'friend_source_commit': subprocess.check_output(['git','rev-parse','origin/main'], text=True).strip(),
                   'sklearn_version': sklearn.__version__, 'iterations': int(model['model'].n_iter_),
                   'warnings': [str(w.message) for w in caught], 'checkpoint_reload_verified': True})
    report['limitations'].extend([
        'Chronological holdout is October–December 2021 at known stations, not unseen-region testing.',
        'The earlier 3-event test overlaps this holdout and was previously examined; this is exploratory evaluation.',
        'Do not directly compare old 3-event metrics with new larger-test metrics as proof of improvement.',
        'Input provenance and water-pixel assumptions remain those of the extraction pilot.'])
    (out / 'run.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    example = test.iloc[:1][['observation_id', *FEATURES]].copy()
    example.to_csv(out / 'example_inputs.csv', index=False)
    model_row = metrics.query("model == 'sklearn_mlp_fusion' and split == 'test'").iloc[0]
    baseline_row = metrics.query("model == 'training_mean_baseline'").iloc[0]
    (out / 'SUMMARY.md').write_text(
        '# Expanded scikit-learn BOD demonstration\n\n'
        + f"Rows: {report['rows']}; excluded by boundary/gap: {report['excluded_events']}.\n\n"
        + f"Test MAE: {model_row.mae:.4f} mg/L; RMSE: {model_row.rmse:.4f} mg/L; R²: {model_row.r2:.4f}.\n\n"
        + f"Training-mean test MAE: {baseline_row.mae:.4f} mg/L.\n\n"
        + 'The friend\'s original model/settings are unchanged: 11 inputs, 8 ReLU hidden neurons, L-BFGS, alpha=1, seed=42. Scaling is fitted only on training rows. No hyperparameters were tuned on the test set.\n\n'
        + 'The saved model was reloaded and reproduced the saved test predictions. This small retrospective demonstration does not establish water safety or future/unseen-region performance. Earlier and expanded metrics use different test sets.\n', encoding='utf-8')
    print(metrics.to_string(index=False))
    print(f'Saved and verified: {out}')


if __name__ == '__main__':
    main()

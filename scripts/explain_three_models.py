"""Kernel SHAP for the exact three presentation checkpoints, without refitting."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/three_model_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from three_model_support import load, sha


def main():
    train, _, _, examples, models, source_hashes = load()
    background = train.sample(n=20, random_state=42)
    out = ROOT / 'docs/three_model_shap_results'
    out.mkdir(parents=True, exist_ok=True)
    background.to_csv(out / 'training_background.csv', index=False)
    examples.to_csv(out / 'explained_examples.csv', index=False)
    reports = {}
    for name, model in models.items():
        print('Explaining ' + name, flush=True)
        features, predictor = model['features'], model['predict']
        def wrapped(x):
            return np.asarray(predictor(pd.DataFrame(x, columns=features)), dtype=float)
        np.random.seed(42)
        explainer = shap.KernelExplainer(wrapped, background[features].to_numpy(), link='identity')
        values = np.asarray(explainer.shap_values(examples[features].to_numpy(),
                             nsamples=512, l1_reg=0, silent=True))
        reference = float(explainer.expected_value)
        predictions = wrapped(examples[features].to_numpy())
        residual = predictions - reference - values.sum(axis=1)
        if not np.isfinite(values).all() or np.max(np.abs(residual)) > 1e-5:
            raise ValueError('SHAP reconstruction failed')
        rows = examples[['observation_id', 'station_id', 'cohort']].copy()
        rows['predicted_bod_mg_l'] = predictions
        rows['reference_bod_mg_l'] = reference
        rows['reconstruction_error_mg_l'] = residual
        for i, feature in enumerate(features):
            rows['shap_' + feature] = values[:, i]
        for group, prefix in [('satellite', 'satellite_'), ('ground', 'ground_'), ('date_gap', 'temporal_')]:
            indices = [i for i, feature in enumerate(features) if feature.startswith(prefix)]
            rows[group + '_contribution_mg_l'] = values[:, indices].sum(axis=1)
        rows.to_csv(out / (name + '_explanations.csv'), index=False)
        importance = pd.DataFrame({'feature': features, 'mean_absolute_shap_mg_l': np.abs(values).mean(axis=0)})
        importance.sort_values('mean_absolute_shap_mg_l', ascending=False).to_csv(out / (name + '_ranking.csv'), index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        order = np.argsort(np.abs(values).mean(axis=0))
        axes[0].barh(np.asarray(features)[order], np.abs(values).mean(axis=0)[order], color='#267caa')
        axes[0].set_title('Mean absolute SHAP: seven demonstration cases')
        axes[0].set_xlabel('Contribution magnitude (mg/L)')
        axes[1].barh(features, values[0], color=np.where(values[0] >= 0, '#b64c48', '#267caa'))
        axes[1].axvline(0, color='black', linewidth=.6)
        axes[1].set_title(f'First example: BOD {predictions[0]:.2f} mg/L')
        axes[1].set_xlabel('Contribution relative to training reference (mg/L)')
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(out / (name + '_shap.png'), dpi=130)
        plt.close(fig)
        reports[name] = {'model_sha256': model['sha256'], 'input_features': features,
            'max_reconstruction_error': float(np.max(np.abs(residual))), 'reference': reference}
        if sha(model['path']) != model['sha256']:
            raise ValueError('Model changed')
    report = {'method': 'Kernel SHAP', 'shap_version': shap.__version__, 'background_rows': 20,
        'background_selection': 'Seed-42 sample of original 80 training rows only',
        'explained_rows_per_model': len(examples), 'example_selection': 'First saved test observation per station: three familiar, four unseen',
        'nsamples': 512, 'l1_reg': 0, 'models': reports, 'source_hashes': source_hashes,
        'limitations': ['Explains BOD, not downstream categories or the oxygen PINN.',
            'Ratio model is explained through raw bands; ratios are recomputed for every masked input.',
            'Date-gap contribution is separate from satellite and ground contributions.',
            'Seven-case ranking is local demonstration importance, not global feature selection.',
            'Correlated features and artificial masked combinations make SHAP reference-dependent, not causal.']}
    report['output_sha256'] = {p.name: sha(p) for p in out.iterdir() if p.name != 'provenance.json'}
    (out / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Saved SHAP for all three checkpoints:', out)


if __name__ == '__main__':
    main()

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from three_model_support import load, sha
from river_water_quality.training.oxygen_pinn import predict


def test_shap_background_is_training_only_and_explanations_reconstruct_exact_models():
    train, _, _, examples, models, _ = load()
    folder = ROOT / 'docs/three_model_shap_results'
    background = pd.read_csv(folder / 'training_background.csv')
    assert set(background.observation_id) <= set(train.observation_id)
    assert not set(background.observation_id) & set(examples.observation_id)
    for name, model in models.items():
        saved = pd.read_csv(folder / (name + '_explanations.csv'))
        actual = model['predict'](examples[model['features']])
        np.testing.assert_allclose(saved.predicted_bod_mg_l, actual, atol=1e-6, rtol=0)
        attribution = saved.filter(regex='^shap_').sum(axis=1)
        np.testing.assert_allclose(saved.reference_bod_mg_l+attribution, actual, atol=1e-5, rtol=0)
        if name == 'satellite_ratios':
            assert saved.ground_contribution_mg_l.eq(0).all()
    report = json.loads((folder / 'provenance.json').read_text())
    for name, digest in report['output_sha256'].items():
        assert sha(folder / name) == digest


def test_saved_pinn_reloads_and_equation_accuracy_is_bounded():
    folder = ROOT / 'docs/oxygen_pinn_results'
    checkpoint = torch.load(folder / 'oxygen_pinn.pt', map_location='cpu', weights_only=True)
    rows = pd.read_csv(folder / 'oxygen_scenarios.csv')
    result = predict(checkpoint, rows.time_days, rows.predicted_bod5_scenario_mg_l, rows.initial_measured_do_mg_l)
    np.testing.assert_allclose(result[:, 1], rows.pinn_do_mg_l, atol=3e-6, rtol=0)
    assert rows.absolute_oxygen_equation_error_mg_l.max() < .1
    assert (rows.pinn_do_mg_l >= 0).all()
    report = json.loads((folder / 'provenance.json').read_text())
    for name, digest in report['output_sha256'].items():
        assert sha(folder / name) == digest

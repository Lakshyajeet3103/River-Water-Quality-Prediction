import numpy as np
import pandas as pd
import pytest

from river_water_quality.training.satellite_refinements import ALPHAS, features, fit, predict, select
from river_water_quality.training.sklearn_fusion import SATELLITE_FEATURES


def raw(n=1):
    frame = pd.DataFrame({name: np.full(n, .2) for name in SATELLITE_FEATURES})
    frame['satellite_B3'], frame['satellite_B8'], frame['satellite_B5'] = .4, .8, .3
    return frame


def test_ratio_arithmetic_and_no_mutation_or_chemical_inputs():
    frame = raw().assign(ground_bod=999, ground_ph=999)
    original = frame.copy(deep=True)
    engineered = features(frame, 'bands_ratios')
    pd.testing.assert_frame_equal(frame, original)
    assert engineered.ratio_B3_B4.iloc[0] == pytest.approx(2)
    assert engineered.ratio_B8_B4.iloc[0] == pytest.approx(4)
    assert engineered.ratio_B5_B4.iloc[0] == pytest.approx(1.5)
    assert engineered.contrast_B3_B8.iloc[0] == pytest.approx(-1/3)
    assert engineered.contrast_B4_B3.iloc[0] == pytest.approx(-1/3)
    assert not any(name.startswith('ground_') for name in engineered)


def test_near_zero_denominator_is_not_filled():
    frame = raw()
    frame['satellite_B4'] = 0
    with pytest.raises(ValueError, match='Near-zero denominator'):
        features(frame, 'bands_ratios')


def test_prediction_is_target_free_and_scalers_are_training_only():
    frame = raw(30)
    frame['satellite_B4'] = np.linspace(.1, .3, 30)
    frame['ground_bod'] = np.linspace(1, 3, 30)
    model = fit(frame, 'bands_ratios', 1)
    x = frame.iloc[:2][SATELLITE_FEATURES].copy()
    np.testing.assert_array_equal(predict(model, x), predict(model, x.assign(ground_bod=-999, ground_ph=999)))
    np.testing.assert_allclose(model['x_scaler'].mean_, features(frame, 'bands_ratios').mean())
    assert model['model'].hidden_layer_sizes == (8,) and model['model'].alpha == 1


def test_grid_fitting_never_receives_validation_rows(monkeypatch):
    import river_water_quality.training.satellite_refinements as module
    frame = raw(30)
    frame['observation_id'] = [str(i) for i in range(30)]
    frame['station_id'] = [str(i % 3) for i in range(30)]
    frame['scene_id'] = [f'scene-{i}' for i in range(30)]
    frame['sample_date'] = pd.date_range('2021-01-01', periods=30).strftime('%Y-%m-%d')
    frame['satellite_local_date'] = frame.sample_date
    frame['ground_bod'] = 2.0
    calls = []

    class Estimator:
        n_iter_ = 1

    def fake_fit(data, mode, alpha):
        calls.append((set(data.observation_id), mode, alpha))
        return {'model': Estimator(), 'features': list(features(data, mode)), 'warnings': []}

    monkeypatch.setattr(module, 'fit', fake_fit)
    monkeypatch.setattr(module, 'predict', lambda model, data: np.full(len(data), 2.0))
    scores, choice, _ = select(frame.iloc[:20], frame.iloc[20:])
    assert len(calls) == 2 * len(ALPHAS)
    assert all(ids == set(frame.iloc[:20].observation_id) for ids, _, _ in calls)
    assert scores.selected_pipeline.sum() == 1 and choice['input_mode'] == 'bands'

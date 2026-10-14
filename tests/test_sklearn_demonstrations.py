"""Meaningful split, preprocessing and target-isolation safeguards."""
import numpy as np
import pandas as pd
import pytest

from river_water_quality.training.sklearn_demonstrations import ALL, TARGET, fit, predict, check_separation, feature_search


def fixture():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.uniform(.01, .1, (80, len(ALL))), columns=ALL)
    frame['ground_fecal_coliform'] *= 10000
    frame['ground_ph'] = 7 + frame.ground_ph
    frame['ground_dissolved_oxygen'] = 8 + frame.ground_dissolved_oxygen
    frame[TARGET] = 1 + 10*frame.satellite_B2
    frame['observation_id'] = [f'event-{i}' for i in range(80)]
    frame['scene_id'] = [f'scene-{i}' for i in range(80)]
    frame['station_id'] = 'synthetic-station'
    frame['sample_date'] = pd.date_range('2020-01-01', periods=80).strftime('%Y-%m-%d')
    frame['satellite_local_date'] = frame.sample_date
    return frame


def test_prediction_does_not_read_target_and_scaling_is_training_only():
    data = fixture()
    training, heldout = data.iloc[:60], data.iloc[60:]
    model = fit(training, ALL)
    original = predict(model, heldout)
    modified = heldout.copy()
    modified[TARGET] = 99999
    np.testing.assert_array_equal(original, predict(model, modified))
    np.testing.assert_array_equal(original, predict(model, heldout[ALL]))
    np.testing.assert_allclose(model['y_scaler'].mean_, [training[TARGET].mean()])


def test_aliases_of_same_satellite_acquisition_cannot_cross_folds():
    data = fixture()
    left, right = data.iloc[:60].copy(), data.iloc[60:].copy()
    left.loc[left.index[0], 'scene_id'] = '20200101T050000_20200101T060000_T44RKS'
    right.loc[right.index[0], 'scene_id'] = '20200101T050000_20200102T060000_T44RKT'
    with pytest.raises(ValueError, match='Acquisition crosses'):
        check_separation(left, right)


def test_ranking_never_uses_inner_validation_answers():
    data = fixture()
    _, ranking, _, _ = feature_search(data)
    modified = data.copy()
    modified.loc[60:, TARGET] = 999
    _, other, _, _ = feature_search(modified)
    pd.testing.assert_frame_equal(ranking.drop(columns='selected'), other.drop(columns='selected'))

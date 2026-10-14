import numpy as np
import pandas as pd
import pytest

from river_water_quality.bod_category import assess, category, classify, training_profiles


def fixture_frames():
    paired = pd.DataFrame({'observation_id': ['a', 'b', 'c'], 'station_id': ['s'] * 3,
        'sample_date': ['2020-01-01', '2020-02-01', '2020-03-01'],
        'ground_ph': [7, 8, 2], 'ground_dissolved_oxygen': [7, 8, 0],
        'ground_fecal_coliform': [100, 300, 90000]})
    split = pd.DataFrame({'observation_id': ['a', 'b', 'c'], 'split': ['train', 'train', 'test']})
    return paired, split


def test_profiles_exclude_test_values():
    paired, split = fixture_frames()
    before = training_profiles(paired, split)
    paired.loc[2, ['ground_ph', 'ground_dissolved_oxygen', 'ground_fecal_coliform']] = np.nan
    pd.testing.assert_frame_equal(before, training_profiles(paired, split))
    assert before.iloc[0].ground_ph == 7.5
    assert before.iloc[0].ground_fecal_coliform == 200
    assert before.iloc[0].training_count == 2


def test_category_boundaries_and_monotonicity():
    assert [category(x) for x in [0, 59.999, 60, 79.999, 80, 100]] == [
        'Bad', 'Bad', 'Good', 'Good', 'Best', 'Best']
    scores = [assess(bod, 7.5, 8, 200)['custom_score'] for bod in [0, 2, 4, 10, 20]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize('bod', [-1, float('nan'), float('inf')])
def test_invalid_bod_rejected(bod):
    with pytest.raises(ValueError):
        assess(bod, 7.5, 8, 200)


def test_inference_needs_no_actual_chemistry_and_rejects_unknown_station():
    paired, split = fixture_frames()
    profiles = training_profiles(paired, split)
    frame = pd.DataFrame({'station_id': ['s']})
    actual = classify(frame, [2], profiles)
    assert actual.iloc[0].bod_led_category in ['Best', 'Good', 'Bad']
    frame['ground_bod'] = -999
    frame['ground_ph'] = -999
    pd.testing.assert_frame_equal(actual, classify(frame, [2], profiles))
    with pytest.raises(ValueError, match='No training chemistry'):
        classify(pd.DataFrame({'station_id': ['unknown']}), [2], profiles)


def test_duplicates_rejected():
    paired, split = fixture_frames()
    paired.loc[1, 'observation_id'] = 'a'
    with pytest.raises(ValueError, match='Duplicate'):
        training_profiles(paired, split)

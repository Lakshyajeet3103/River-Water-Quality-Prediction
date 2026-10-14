import numpy as np
import pandas as pd
import pytest

from scripts.compare_ground_only import saved_cohort
from river_water_quality.training.sklearn_demonstrations import GROUND, fit, predict


def test_saved_split_rejects_shared_satellite_acquisition():
    data = pd.DataFrame({'observation_id': [str(i) for i in range(100)], 'station_id': 's',
        'sample_date': pd.date_range('2021-01-01', periods=100).strftime('%Y-%m-%d'),
        'satellite_local_date': pd.date_range('2021-01-01', periods=100).strftime('%Y-%m-%d'),
        'scene_id': [f'unique-{i}' for i in range(100)]})
    splits = data[['observation_id']].assign(split=['train'] * 80 + ['test'] * 20)
    train, test = saved_cohort(data.sample(frac=1, random_state=42), splits)
    assert len(train) == 80 and len(test) == 20
    data.loc[80, 'scene_id'] = data.loc[79, 'scene_id']
    with pytest.raises(ValueError, match='Acquisition crosses'):
        saved_cohort(data, splits)


def test_ground_only_prediction_needs_neither_bod_nor_satellite():
    rng = np.random.default_rng(42)
    train = pd.DataFrame({'ground_fecal_coliform': rng.uniform(100, 10000, 40),
                         'ground_ph': rng.uniform(7, 8, 40),
                         'ground_dissolved_oxygen': rng.uniform(6, 10, 40)})
    train['ground_bod'] = 0.5 + train.ground_fecal_coliform / 5000
    model = fit(train, GROUND)
    example = train.iloc[:3][GROUND].copy()
    original = predict(model, example)
    perturbed = example.assign(ground_bod=999, satellite_B2=999, satellite_B8=-999)
    np.testing.assert_array_equal(predict(model, perturbed), original)
    expected = train[GROUND].to_numpy().copy()
    expected[:, 0] = np.log1p(expected[:, 0])
    np.testing.assert_allclose(model['x_scaler'].mean_, expected.mean(axis=0))

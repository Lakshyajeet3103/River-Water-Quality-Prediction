import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_wqi import prepare_test_inputs, score_pairs, summaries


def cohort():
    paired = pd.DataFrame({'observation_id': [f'e{i}' for i in range(100)],
        'station_id': 's', 'sample_date': '2021-01-01', 'scene_id': 'scene',
        'ground_bod': np.linspace(1, 4, 100)})
    splits = paired[['observation_id']].assign(split=['train'] * 80 + ['test'] * 20)
    saved = paired.iloc[80:].rename(columns={'ground_bod': 'actual_bod_mg_l'}).copy()
    saved['predicted_bod_mg_l'] = 2.0
    return paired, splits, saved


def test_only_exact_test_ids_in_preserved_order_even_with_shuffled_inputs():
    paired, splits, saved = cohort()
    result = prepare_test_inputs(paired.sample(frac=1, random_state=2), splits,
                                 saved.sample(frac=1, random_state=3))
    assert result.observation_id.tolist() == splits.iloc[80:].observation_id.tolist()
    assert result.predicted_bod_mg_l.eq(2).all()


def test_train_prediction_cannot_replace_test_case():
    paired, splits, saved = cohort()
    saved.iloc[0, saved.columns.get_loc('observation_id')] = 'e0'
    with pytest.raises(ValueError, match='same saved 20'):
        prepare_test_inputs(paired, splits, saved)


def test_mismatched_truth_is_rejected():
    paired, splits, saved = cohort()
    saved.iloc[0, saved.columns.get_loc('actual_bod_mg_l')] += 1
    with pytest.raises(AssertionError):
        prepare_test_inputs(paired, splits, saved)


def test_invalid_saturation_not_in_agreement_denominator_and_shared_input_cancels():
    frame = pd.DataFrame({'observation_id': ['valid', 'unsupported'],
        'predicted_bod_mg_l': [2, 2], 'ground_bod': [3, 3], 'ground_ph': [7.8, 7.8],
        'ground_fecal_coliform': [100, 100], 'ground_dissolved_oxygen': [8, 20]})
    original = frame.copy(deep=True)
    results = score_pairs(frame, [20, 25, 30], 1, 0, 2.5)
    pd.testing.assert_frame_equal(frame, original)
    supported = results.query("observation_id == 'valid'")
    np.testing.assert_allclose(supported.signed_wqi_error, 1.33, atol=1e-12)
    unsupported = results.query("observation_id == 'unsupported'")
    assert unsupported.category_agrees.isna().all()
    metrics, confusion = summaries(results)
    assert metrics.attempted.eq(2).all() and metrics.scored.eq(1).all()
    assert metrics.excluded.eq(1).all() and metrics.metric_rows.eq(1).all()
    assert confusion.groupby('assumed_temperature_c')['count'].sum().eq(1).all()

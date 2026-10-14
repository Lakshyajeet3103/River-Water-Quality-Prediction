from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest

from scripts.evaluate_saved_models_unseen import cohort_audit, score


def frames():
    train = pd.DataFrame({'observation_id': ['a'], 'station_id': ['s1'],
                          'scene_id': ['20211116T050101_20211116T050055_T44RQP']})
    new = pd.DataFrame({'observation_id': ['b', 'c'], 'station_id': ['s2', 's2'],
        'station_name': ['new', 'new'], 'sample_date': ['2021-11-16', '2021-11-20'],
        'scene_id': ['20211116T050101_20211116T050055_T44RKS',
                     '20211120T050101_20211120T050055_T44RKS']})
    return train, new


def test_acquisition_overlap_detected_across_tiles():
    train, new = frames()
    audit = cohort_audit(train, new)
    assert audit.eligible_strict_test.tolist() == [False, True]
    assert len(new) == 2


@pytest.mark.parametrize('kind', ['station', 'event', 'duplicate'])
def test_training_overlap_and_duplicates_rejected(kind):
    train, new = frames()
    if kind == 'station':
        new.loc[0, 'station_id'] = 's1'
    elif kind == 'event':
        new.loc[0, 'observation_id'] = 'a'
    else:
        new.loc[1, 'observation_id'] = 'b'
    with pytest.raises(ValueError):
        cohort_audit(train, new)


def test_mape_is_mean_of_individual_percentages():
    result = score([1, 4], [2, 2])
    assert result['mae'] == 1.5
    assert result['mape_percent'] == 75
    with pytest.raises(ValueError):
        score([0, 1], [1, 1])

import pandas as pd
import pytest

from scripts.evaluate_unseen_stations import station_fold


def cohort():
    return pd.DataFrame([{'observation_id': f'{station}-{i}', 'station_id': station,
                          'scene_id': f'{station}-unique-{i}', 'ground_bod': i / 10}
                         for station in ['a', 'b', 'c'] for i in range(12)])


def test_held_out_station_and_answers_never_enter_training():
    data = cohort()
    train, test, purged = station_fold(data, 'a')
    assert set(train.station_id) == {'b', 'c'} and set(test.station_id) == {'a'}
    assert purged.empty and len(test) == 12
    changed = data.copy()
    changed.loc[changed.station_id.eq('a'), 'ground_bod'] = 999
    other_train, _, _ = station_fold(changed, 'a')
    pd.testing.assert_frame_equal(train, other_train)


def test_processing_tile_alias_of_test_acquisition_is_purged():
    data = cohort()
    data.loc[0, 'scene_id'] = '20210101T050000_20210101T050100_T44RQP'
    data.loc[12, 'scene_id'] = '20210101T050000_20210102T050200_T44RQQ'
    train, test, purged = station_fold(data, 'a')
    assert purged.observation_id.tolist() == ['b-0']
    assert 'b-0' not in set(train.observation_id) and len(test) == 12


def test_duplicate_ids_and_training_erased_by_purge_are_rejected():
    data = cohort()
    with pytest.raises(ValueError, match='Duplicate'):
        station_fold(pd.concat([data, data.iloc[:1]], ignore_index=True), 'a')
    data.loc[:, 'scene_id'] = 'shared-scene'
    with pytest.raises(ValueError, match='Insufficient'):
        station_fold(data, 'a')

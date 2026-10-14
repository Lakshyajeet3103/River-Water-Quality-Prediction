import numpy as np
import pandas as pd
import pytest

from river_water_quality.training.satellite_selection import (candidates, inner_folds, select_bands,
                                                               select_chronological_bands, winner)


def training():
    return pd.DataFrame([{'observation_id': f'{station}-{i}', 'station_id': station,
                          'scene_id': f'{station}-scene-{i}', 'ground_bod': i / 10}
                         for station in ['a', 'b', 'c'] for i in range(12)])


def test_inner_station_and_acquisition_are_excluded_from_fitting():
    frame = training()
    frame.loc[0, 'scene_id'] = '20210101T050000_20210101T051000_T44RQP'
    frame.loc[12, 'scene_id'] = '20210101T050000_20210102T051000_T44RQQ'
    station, fitting, validation, purged = next(inner_folds(frame))
    assert station == 'a' and set(fitting.station_id) == {'b', 'c'}
    assert set(validation.station_id) == {'a'} and purged.observation_id.tolist() == ['b-0']
    assert not (set(fitting.observation_id) & set(validation.observation_id))


def test_validation_only_selection_never_passes_ground_features(monkeypatch):
    import river_water_quality.training.satellite_selection as module
    calls = []

    class Estimator:
        n_iter_ = 1

    def fake_fit(frame, features):
        calls.append((set(frame.station_id), list(features)))
        assert all(name.startswith('satellite_') for name in features)
        return {'model': Estimator(), 'warnings': [], 'mean': frame.ground_bod.mean()}

    monkeypatch.setattr(module, 'fit', fake_fit)
    monkeypatch.setattr(module, 'predict', lambda model, frame: np.full(len(frame), model['mean']))
    frame = training()
    for name in candidates()['all_bands']:
        frame[name] = .01
    bands, choice, scores, details, assignments = select_bands(frame)
    assert len(calls) == 3 * len(candidates())
    assert all(len(stations) == 2 for stations, _ in calls)
    assert len(bands) == 1 and choice == 'only_B11'
    assert scores.selected.sum() == 1
    assert details.inner_validation_station_id.nunique() == 3
    assert assignments.groupby('inner_validation_station_id').apply(
        lambda g: set(g.loc[g.assignment.eq('fit'), 'station_id']).isdisjoint(
            set(g.loc[g.assignment.eq('validation'), 'station_id'])), include_groups=False).all()


def test_candidate_tie_order_and_missing_scores():
    scores = pd.DataFrame({'candidate': ['all_bands', 'only_B3', 'only_B2'],
                           'input_count': [10, 1, 1], 'macro_validation_mae': [.5, .5, .5]})
    assert winner(scores) == 'only_B2'
    scores.loc[0, 'macro_validation_mae'] = np.nan
    with pytest.raises(ValueError, match='Invalid'):
        winner(scores)


def chronological():
    frame = pd.DataFrame({'observation_id': [str(i) for i in range(80)],
        'station_id': [str(i % 3) for i in range(80)],
        'scene_id': [f'scene-{i}' for i in range(80)],
        'sample_date': pd.date_range('2021-01-01', periods=80).strftime('%Y-%m-%d'),
        'satellite_local_date': pd.date_range('2021-01-01', periods=80).strftime('%Y-%m-%d'),
        'ground_bod': np.arange(80) / 100})
    for name in candidates()['all_bands']:
        frame[name] = .01
    assignments = frame[['observation_id']].assign(split=['selection_train'] * 60 + ['selection_validation'] * 20)
    return frame, assignments


def test_chronological_selection_fits_only_60_rows_and_rejects_outer_ids(monkeypatch):
    import river_water_quality.training.satellite_selection as module
    frame, assignments = chronological()
    calls = []

    class Estimator:
        n_iter_ = 1

    def fake_fit(data, features):
        calls.append(set(data.observation_id))
        assert all(f.startswith('satellite_') for f in features)
        return {'model': Estimator(), 'warnings': [], 'mean': data.ground_bod.mean()}

    monkeypatch.setattr(module, 'fit', fake_fit)
    monkeypatch.setattr(module, 'predict', lambda model, data: np.full(len(data), model['mean']))
    _, _, scores = select_chronological_bands(frame, assignments)
    assert len(calls) == 17 and all(ids == set(frame.iloc[:60].observation_id) for ids in calls)
    assert scores.selected.sum() == 1
    assignments.loc[0, 'observation_id'] = 'outer-test-event'
    with pytest.raises(ValueError, match='only the outer training'):
        select_chronological_bands(frame, assignments)


def test_chronological_selection_rejects_shared_acquisition():
    frame, assignments = chronological()
    frame.loc[60, 'scene_id'] = frame.loc[59, 'scene_id']
    with pytest.raises(ValueError, match='Acquisition crosses'):
        select_chronological_bands(frame, assignments)

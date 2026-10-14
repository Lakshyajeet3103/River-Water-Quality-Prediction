"""Fixed satellite-band subsets selected using only internal station holdouts."""
import pandas as pd

from river_water_quality.training.sklearn_demonstrations import fit, predict, check_separation
from river_water_quality.training.sklearn_fusion import SATELLITE_FEATURES, TARGET, regression_metrics
from river_water_quality.training.splits import acquisition_group


def candidates():
    groups = {'all_bands': list(SATELLITE_FEATURES)}
    groups.update({f'only_{name.removeprefix("satellite_")}': [name] for name in SATELLITE_FEATURES})
    for name, bands in {
        'visible': ['B2', 'B3', 'B4'],
        'red_edge': ['B5', 'B6', 'B7', 'B8A'],
        'nir': ['B8', 'B8A'],
        'swir': ['B11', 'B12'],
        'visible_red_edge': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        'visible_nir': ['B2', 'B3', 'B4', 'B8', 'B8A'],
    }.items():
        groups[name] = [f'satellite_{band}' for band in bands]
    return groups


def inner_folds(train):
    """No outer test data is received; purge shared acquisitions internally too."""
    if train.station_id.nunique() < 3 or not train.observation_id.is_unique:
        raise ValueError('Need three unique training stations for internal selection')
    for station in sorted(train.station_id.unique()):
        validation = train[train.station_id.eq(station)].copy()
        pool = train[~train.station_id.eq(station)].copy()
        shared = pool.scene_id.map(acquisition_group).isin(set(validation.scene_id.map(acquisition_group)))
        fitting, purged = pool[~shared].copy(), pool[shared].copy()
        if len(fitting) < 10 or fitting.station_id.nunique() < 2:
            raise ValueError('Insufficient internal fitting data after acquisition purge')
        yield station, fitting, validation, purged


def winner(scores):
    """Minimum mean station MAE; deterministic ties favour fewer inputs, then name."""
    if scores.empty or scores.candidate.duplicated().any() or scores.macro_validation_mae.isna().any():
        raise ValueError('Invalid candidate scores')
    return scores.sort_values(['macro_validation_mae', 'input_count', 'candidate']).iloc[0].candidate


def select_bands(train):
    """Return an inner-validation-selected subset, scores, and auditable folds."""
    options = candidates()
    rows, assignments = [], []
    for station, fitting, validation, purged in inner_folds(train):
        for label, frame in [('fit', fitting), ('validation', validation), ('purged_shared_acquisition', purged)]:
            records = frame[['observation_id', 'station_id', 'scene_id']].copy()
            records['inner_validation_station_id'], records['assignment'] = station, label
            assignments.append(records)
        for name, features in options.items():
            model = fit(fitting, features)
            metrics = regression_metrics(validation[TARGET], predict(model, validation[features]))
            rows.append({'candidate': name, 'inner_validation_station_id': station,
                         'input_count': len(features), 'fit_rows': len(fitting), 'purged_rows': len(purged),
                         **metrics, 'iterations': int(model['model'].n_iter_),
                         'training_warnings': '; '.join(model['warnings'])})
    details = pd.DataFrame(rows)
    scores = details.groupby(['candidate', 'input_count'], as_index=False).agg(
        macro_validation_mae=('mae', 'mean'), macro_validation_rmse=('rmse', 'mean'),
        validation_stations=('inner_validation_station_id', 'nunique'))
    choice = winner(scores)
    scores['selected'] = scores.candidate.eq(choice)
    return options[choice], choice, scores, details, pd.concat(assignments, ignore_index=True)


def select_chronological_bands(train, assignments):
    """Select within the saved 60/20 split of the outer 80 training observations."""
    if not train.observation_id.is_unique or not assignments.observation_id.is_unique:
        raise ValueError('Duplicate selection IDs')
    if set(train.observation_id) != set(assignments.observation_id):
        raise ValueError('Selection assignments must cover only the outer training set')
    if set(assignments.split) != {'selection_train', 'selection_validation'}:
        raise ValueError('Invalid internal chronological split labels')
    joined = train.merge(assignments[['observation_id', 'split']], on='observation_id', validate='one_to_one')
    fitting = joined.query("split == 'selection_train'")
    validation = joined.query("split == 'selection_validation'")
    if len(fitting) != 60 or len(validation) != 20:
        raise ValueError('Expected saved 60/20 selection split')
    check_separation(fitting, validation)
    if not set(validation.station_id).issubset(set(fitting.station_id)):
        raise ValueError('Internal validation must use familiar stations')
    options, rows = candidates(), []
    for name, features in options.items():
        model = fit(fitting, features)
        score = regression_metrics(validation[TARGET], predict(model, validation[features]))
        rows.append({'candidate': name, 'features': ';'.join(features), 'input_count': len(features),
                     'validation_mae': score['mae'], 'validation_rmse': score['rmse'],
                     'iterations': int(model['model'].n_iter_), 'warnings': '; '.join(model['warnings'])})
    scores = pd.DataFrame(rows)
    choice = scores.sort_values(['validation_mae', 'input_count', 'candidate']).iloc[0].candidate
    scores['selected'] = scores.candidate.eq(choice)
    return options[choice], choice, scores

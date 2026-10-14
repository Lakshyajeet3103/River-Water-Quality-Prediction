"""Custom classroom index: predicted BOD plus historical station chemistry."""
import pandas as pd

from river_water_quality.wqi import calculate_scenario_wqi, number

METHOD = 'BOD-led water-quality category v1 (custom demonstration)'
WEIGHTS = {'bod': 0.70, 'ph': 0.10, 'do': 0.10, 'fecal': 0.10}
CHEMISTRY = ['ground_ph', 'ground_dissolved_oxygen', 'ground_fecal_coliform']


def category(score):
    score = number(score, 'custom score', 0, 100)
    return 'Best' if score >= 80 else 'Good' if score >= 60 else 'Bad'


def assess(bod, ph, oxygen, fecal):
    # Missing water conditions are explicit fixed demo assumptions, not observations.
    source = calculate_scenario_wqi(
        bod_mg_l=bod, ph=ph, do_mg_l=oxygen, fecal_per_100ml=fecal,
        assumed_temperature_c=25, assumed_pressure_atm=1, assumed_salinity_ppt=0)
    # This custom index explicitly bounds rounded source sub-indices.
    sub = {k: min(100., max(0., v)) for k, v in source['sub_indices'].items()}
    score = sum(WEIGHTS[k] * sub[k] for k in WEIGHTS)
    return {'custom_score': score, 'bod_led_category': category(score)}


def training_profiles(paired, assignments):
    if not paired.observation_id.is_unique or not assignments.observation_id.is_unique:
        raise ValueError('Duplicate observation IDs')
    if set(paired.observation_id) != set(assignments.observation_id):
        raise ValueError('Dataset and split IDs differ')
    if set(assignments.split) != {'train', 'test'}:
        raise ValueError('Expected train/test assignments')
    train = paired.set_index('observation_id').loc[
        assignments.loc[assignments.split.eq('train'), 'observation_id']]
    if train.station_id.isna().any():
        raise ValueError('Missing training station')
    for row in train.itertuples():
        assess(0, row.ground_ph, row.ground_dissolved_oxygen, row.ground_fecal_coliform)
    profiles = train.groupby('station_id')[CHEMISTRY].mean()
    profiles['training_count'] = train.groupby('station_id').size()
    profiles['training_end_date'] = train.groupby('station_id').sample_date.max()
    return profiles.reset_index()


def classify(frame, predicted_bod, profiles):
    if not profiles.station_id.is_unique or len(frame) != len(predicted_bod):
        raise ValueError('Invalid profiles or prediction count')
    lookup = profiles.set_index('station_id')
    rows = []
    for (_, row), bod in zip(frame.iterrows(), predicted_bod):
        if row.station_id not in lookup.index:
            raise ValueError(f'No training chemistry profile for station {row.station_id}')
        p = lookup.loc[row.station_id]
        result = assess(bod, p.ground_ph, p.ground_dissolved_oxygen, p.ground_fecal_coliform)
        # A sensitivity check only: MAE is not a confidence interval.
        low = assess(max(0, bod - 0.591148), p.ground_ph, p.ground_dissolved_oxygen, p.ground_fecal_coliform)
        high = assess(bod + 0.591148, p.ground_ph, p.ground_dissolved_oxygen, p.ground_fecal_coliform)
        rows.append({'station_id': row.station_id, 'predicted_bod_mg_l': float(bod),
                     **result, 'boundary_sensitive': low['bod_led_category'] != high['bod_led_category'],
                     'chemistry_source': 'training-only station averages',
                     **{name: float(p[name]) for name in CHEMISTRY}})
    result = pd.DataFrame(rows)
    if 'observation_id' in frame:
        result.insert(0, 'observation_id', frame.observation_id.to_numpy())
    return result

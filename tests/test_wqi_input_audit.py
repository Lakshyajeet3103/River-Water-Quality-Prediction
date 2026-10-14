"""Protect exact-match auditing against ambiguous or fabricated matches."""
import pandas as pd
import pytest

from scripts.audit_wqi_inputs import join_physical


def frames():
    event = {'station_name': 'Example', 'sample_date': '2021-01-01',
             'latitude': 28.0, 'longitude': 78.0}
    return (pd.DataFrame([{'observation_id': 'one', **event}]),
            pd.DataFrame([{**event, 'water_temperature_c': None}]))


def test_missing_temperature_remains_missing():
    paired, physical = frames()
    result = join_physical(paired, physical)
    assert result.physical_match.iloc[0] == 'both'
    assert pd.isna(result.water_temperature_c.iloc[0])


def test_nearby_coordinate_is_not_exact_match():
    paired, physical = frames()
    physical.loc[0, 'latitude'] += 0.000001
    assert join_physical(paired, physical).physical_match.iloc[0] == 'left_only'


def test_duplicate_source_key_is_rejected():
    paired, physical = frames()
    with pytest.raises(ValueError, match='Ambiguous'):
        join_physical(paired, pd.concat([physical, physical], ignore_index=True))

"""Check ambiguity, missing observations and preservation when joining lab exports."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('prepare_chemical_dataset', Path(__file__).parents[1] / 'scripts/prepare_chemical_dataset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    paired = pd.DataFrame({'observation_id': ['a'], 'station_name': ['station'],
                           'sample_date': ['2021-01-01'], 'latitude': [25.0], 'longitude': [83.0],
                           'ground_bod': [2.5], 'satellite_B2': [0.05]})
    chemical = paired[module.KEYS].copy()
    chemical['Potential of Hydrogen (pH)'] = '7.4'
    chemical['Dissolved oxygen (mg/L)'] = '8.2'
    return paired, chemical


def test_join_preserves_existing_values():
    paired, chemical = inputs()
    enriched, _ = module.join_chemicals(paired, chemical)
    pd.testing.assert_frame_equal(enriched[paired.columns], paired)
    assert enriched.ground_ph.iloc[0] == 7.4
    assert enriched.ground_dissolved_oxygen.iloc[0] == 8.2


def test_duplicate_source_match_is_rejected():
    paired, chemical = inputs()
    with pytest.raises(ValueError, match='Duplicate chemical'):
        module.join_chemicals(paired, pd.concat([chemical, chemical]))


def test_nearby_coordinates_are_not_silently_matched():
    paired, chemical = inputs()
    chemical.loc[0, 'latitude'] += 0.001
    with pytest.raises(ValueError, match='no exact chemical match'):
        module.join_chemicals(paired, chemical)


@pytest.mark.parametrize('value', ['missing', float('nan'), -1.0])
def test_invalid_do_is_not_filled(value):
    paired, chemical = inputs()
    chemical['Dissolved oxygen (mg/L)'] = value
    with pytest.raises(ValueError):
        module.join_chemicals(paired, chemical)

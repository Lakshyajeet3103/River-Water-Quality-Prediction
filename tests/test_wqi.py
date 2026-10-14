import math

import pandas as pd
import pytest

from river_water_quality.wqi import (calculate_wqi, calculate_scenario_wqi, category,
                                     oxygen_solubility_mg_l)
from scripts.calculate_wqi import calculate_rows


@pytest.mark.parametrize('temperature,rounded_mg_l', [(0, 14.62), (20, 9.09), (25, 8.26), (30, 7.56), (40, 6.41)])
def test_freshwater_equilibrium_reference_values(temperature, rounded_mg_l):
    # Independent rounded physical reference points; detects Kelvin/sign/units errors.
    assert oxygen_solubility_mg_l(temperature, 1, 0) == pytest.approx(rounded_mg_l, abs=0.01)


def test_pressure_salinity_effects_and_standard_conditions():
    baseline = oxygen_solubility_mg_l(25, 1, 0)
    assert oxygen_solubility_mg_l(25, 0.8, 0) < baseline
    assert oxygen_solubility_mg_l(25, 1, 35) < baseline
    assert oxygen_solubility_mg_l(25, 1.1, 0) > baseline


def test_hand_calculation_and_bod_substitution_effect():
    args = dict(bod_mg_l=3, ph=7.8, fecal_per_100ml=1000, do_percent_saturation=100)
    result = calculate_wqi(**args)
    assert result['wqi'] == pytest.approx(69.8274)
    assert sum(result['weighted_contributions'].values()) == pytest.approx(result['wqi'])
    changed = calculate_wqi(**{**args, 'bod_mg_l': 4})
    assert result['wqi'] - changed['wqi'] == pytest.approx(1.33)


@pytest.mark.parametrize('score,expected', [(100, 'Good-Excellent'), (63, 'Good-Excellent'),
    (62.999, 'Good'), (50, 'Good'), (49.999, 'Poor'), (38.001, 'Poor'), (38, 'Very Poor'), (0, 'Very Poor')])
def test_category_endpoints(score, expected):
    assert category(score) == expected


def test_rounded_equation_anomaly_is_visible_not_clipped():
    result = calculate_wqi(bod_mg_l=3, ph=7.3, fecal_per_100ml=100, do_percent_saturation=100)
    assert result['sub_indices']['ph'] > 100
    assert result['formula_warnings']


@pytest.mark.parametrize('field,value', [('bod_mg_l', -1), ('ph', 15), ('fecal_per_100ml', 0),
    ('do_percent_saturation', 141), ('ph', math.nan), ('bod_mg_l', None)])
def test_invalid_inputs_never_produce_category(field, value):
    args = dict(bod_mg_l=3, ph=7.8, fecal_per_100ml=100, do_percent_saturation=100)
    with pytest.raises(ValueError):
        calculate_wqi(**{**args, field: value})


def test_scenario_records_assumptions_and_missing_rows_remain_missing():
    args = dict(bod_mg_l=3, ph=7.8, fecal_per_100ml=100, do_mg_l=8,
                assumed_temperature_c=25, assumed_pressure_atm=1, assumed_salinity_ppt=0)
    result = calculate_scenario_wqi(**args)
    assert result['calculation_type'] == 'assumption-based WQI scenario'
    assert result['assumed_temperature_c'] == 25
    assert result['do_percent_saturation'] == pytest.approx(100 * 8 / result['equilibrium_do_mg_l'])
    frame = pd.DataFrame([dict(predicted_bod_mg_l=3, ground_ph=math.nan,
                              ground_fecal_coliform=100, ground_dissolved_oxygen=8)])
    rows = calculate_rows(frame, [20, 25, 30], 1, 0)
    assert len(rows) == 3 and rows.status.eq('invalid_input').all()
    assert rows.wqi.isna().all() and rows.category.isna().all()


def test_high_saturation_is_not_silently_capped():
    frame = pd.DataFrame([dict(predicted_bod_mg_l=3, ground_ph=7.8,
                              ground_fecal_coliform=100, ground_dissolved_oxygen=20)])
    rows = calculate_rows(frame, [25], 1, 0)
    assert rows.status.iloc[0] == 'invalid_input'
    assert pd.isna(rows.wqi.iloc[0])

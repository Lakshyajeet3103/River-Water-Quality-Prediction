"""Synthetic tests for compatibility and explicit near-time pairing."""

import pandas as pd
import pytest

from river_water_quality.data.wqp import result_reason, pair_samples, screen_results


def row(parameter="bod", **changes):
    entry = {"OrganizationIdentifier": "USGS-TEST", "MonitoringLocationIdentifier": "USGS-TEST_SITE",
        "ActivityIdentifier": f"SYNTHETIC_{parameter}", "ActivityStartDate": "2021-01-01",
        "ActivityStartTime/Time": "10:00:00" if parameter == "bod" else "10:05:00",
        "ActivityStartTime/TimeZoneCode": "EST", "ActivityTypeCode": "Sample-Routine", "ActivityMediaName": "Water",
        "ResultStatusIdentifier": "Accepted", "ResultDetectionConditionText": "", "MeasureQualifierCode": "",
        "ResultValueTypeName": "Actual", "CharacteristicName": "Biochemical oxygen demand, standard conditions" if parameter == "bod" else "Fecal Coliform",
        "USGSPCode": "00310" if parameter == "bod" else "31615", "ResultMeasureValue": "2" if parameter == "bod" else "40",
        "ResultMeasure/MeasureUnitCode": "mg/l" if parameter == "bod" else "MPN/100 ml"}
    entry.update(changes)
    return entry


@pytest.mark.parametrize("field,value,reason", [("ResultMeasure/MeasureUnitCode", "cfu/100ml", "incompatible_unit"),
    ("MeasureQualifierCode", "E", "censored_or_qualified"), ("ResultDetectionConditionText", "Not Detected", "censored_or_qualified"),
    ("ResultStatusIdentifier", "Preliminary", "not_accepted"), ("ResultMeasureValue", "<2", "invalid_numeric_value"),
    ("ResultMeasureValue", "nan", "invalid_numeric_value"), ("ActivityTypeCode", "Quality Control Sample-Field Replicate", "not_routine_water_sample")])
def test_incompatible_or_qualified_results_are_not_substituted(field, value, reason):
    assert result_reason(row("fecal", **{field: value}), "fecal_coliform") == reason


def test_valid_units_and_bod_definition():
    assert result_reason(row(), "bod") is None
    assert result_reason(row("fecal"), "fecal_coliform") is None
    assert result_reason(row(USGSPCode="00320"), "bod") == "wrong_bod_definition"


def test_distinct_activity_pair_preserves_time_gap_and_identifiers():
    bod, _ = screen_results(pd.DataFrame([row()]), "bod")
    fecal, _ = screen_results(pd.DataFrame([row("fecal")]), "fecal_coliform")
    pairs, audit = pair_samples(bod, fecal)
    assert len(pairs) == 1 and audit.empty
    assert pairs.ground_offset_minutes.iloc[0] == 5
    assert pairs.ActivityIdentifier_bod.iloc[0] != pairs.ActivityIdentifier_fecal.iloc[0]


@pytest.mark.parametrize("time,zone", [("10:06:00", "EST"), ("10:05:00", "EDT"), ("", "EST")])
def test_near_time_pair_rejects_wrong_time_or_zone(time, zone):
    bod, _ = screen_results(pd.DataFrame([row()]), "bod")
    fecal, _ = screen_results(pd.DataFrame([row("fecal", **{"ActivityStartTime/Time": time, "ActivityStartTime/TimeZoneCode": zone})]), "fecal_coliform")
    pairs, audit = pair_samples(bod, fecal)
    assert pairs.empty and len(audit) == 1


def test_ambiguous_results_not_averaged():
    bod, _ = screen_results(pd.DataFrame([row()]), "bod")
    fecal, _ = screen_results(pd.DataFrame([row("fecal"), row("fecal", ActivityIdentifier="SYNTHETIC_OTHER")]), "fecal_coliform")
    pairs, audit = pair_samples(bod, fecal)
    assert pairs.empty and audit.reason.iloc[0] == "ambiguous_nearest_sample"

"""Conservative USGS WQP compatibility and near-time sample pairing."""

from datetime import datetime
import hashlib

import numpy as np
import pandas as pd


def result_reason(row, parameter):
    """Keep accepted, unqualified numerical measurements in compatible units."""
    if row.get("ActivityTypeCode") != "Sample-Routine" or row.get("ActivityMediaName") != "Water":
        return "not_routine_water_sample"
    if row.get("ResultStatusIdentifier") != "Accepted":
        return "not_accepted"
    if row.get("ResultDetectionConditionText", "").strip() or row.get("MeasureQualifierCode", "").strip():
        return "censored_or_qualified"
    if row.get("ResultValueTypeName") != "Actual":
        return "not_actual_result"
    if parameter == "bod":
        if row.get("CharacteristicName") != "Biochemical oxygen demand, standard conditions" or row.get("USGSPCode") != "00310":
            return "wrong_bod_definition"
        if row.get("ResultMeasure/MeasureUnitCode", "").strip().lower() != "mg/l":
            return "incompatible_unit"
    else:
        if row.get("CharacteristicName") != "Fecal Coliform":
            return "wrong_characteristic"
        if row.get("ResultMeasure/MeasureUnitCode", "").replace(" ", "").lower() != "mpn/100ml":
            return "incompatible_unit"
    value = pd.to_numeric(row.get("ResultMeasureValue"), errors="coerce")
    if pd.isna(value) or not np.isfinite(value) or value < 0:
        return "invalid_numeric_value"
    return None


def screen_results(frame, parameter):
    good, rejected = [], []
    for index, row in frame.iterrows():
        reason = result_reason(row, parameter)
        info = {"source_record": index + 2, "parameter": parameter,
                "station_id": row.MonitoringLocationIdentifier, "activity_id": row.ActivityIdentifier}
        if reason:
            rejected.append({**info, "reason": reason, "raw_value": row.ResultMeasureValue,
                             "raw_unit": row["ResultMeasure/MeasureUnitCode"]})
        else:
            good.append({**row.to_dict(), "source_record": index + 2})
    return pd.DataFrame(good, columns=[*frame.columns, "source_record"]), pd.DataFrame(rejected)


def pair_samples(bod, fecal, max_minutes=5):
    """Pair unique nearest routine samples on same station/day/time-zone.

    Distinct source activities remain distinct in the audit. Ambiguities and reuse
    are rejected, never averaged or silently joined by date alone.
    """
    keys = ["OrganizationIdentifier", "MonitoringLocationIdentifier", "ActivityStartDate"]
    candidates = bod.merge(fecal, on=keys, suffixes=("_bod", "_fecal"))
    pairs, logs, used = [], [], set()
    for activity, group in candidates.groupby("ActivityIdentifier_bod", sort=True):
        ranked = []
        for _, row in group.iterrows():
            if row["ActivityStartTime/TimeZoneCode_bod"] != row["ActivityStartTime/TimeZoneCode_fecal"]:
                continue
            try:
                first = datetime.fromisoformat(f"{row.ActivityStartDate}T{row['ActivityStartTime/Time_bod']}")
                second = datetime.fromisoformat(f"{row.ActivityStartDate}T{row['ActivityStartTime/Time_fecal']}")
            except ValueError:
                continue
            minutes = abs((first - second).total_seconds()) / 60
            if minutes <= max_minutes:
                ranked.append((minutes, row))
        if not ranked:
            logs.append({"bod_activity": activity, "reason": "no_sample_within_time_window"})
            continue
        ranked.sort(key=lambda entry: entry[0])
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            logs.append({"bod_activity": activity, "reason": "ambiguous_nearest_sample"})
            continue
        minutes, row = ranked[0]
        fecal_key = (row.OrganizationIdentifier, row.ActivityIdentifier_fecal)
        if fecal_key in used:
            logs.append({"bod_activity": activity, "reason": "fecal_activity_reuse"})
            continue
        used.add(fecal_key)
        pairs.append({**row.to_dict(), "ground_offset_minutes": minutes,
                      "observation_id": "wqp-" + hashlib.sha256(f"{row.OrganizationIdentifier}|{activity}|{row.ActivityIdentifier_fecal}".encode()).hexdigest()[:20]})
    unmatched = set(bod.ActivityIdentifier) - set(candidates.ActivityIdentifier_bod)
    logs.extend({"bod_activity": activity, "reason": "no_compatible_fecal_on_station_date"} for activity in sorted(unmatched))
    return pd.DataFrame(pairs), pd.DataFrame(logs)

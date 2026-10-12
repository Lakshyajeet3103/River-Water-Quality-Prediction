"""Audited adapter for the user-supplied Uttar Pradesh biological CSV exports."""

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from river_water_quality.data.schemas import validate_ground

PARAMETERS = {
    "Biochemical Oxygen Demand (mg/L)": ("bod", "mg/L"),
    "Chemical Oxygen Demand (mg/L)": ("cod", "mg/L"),
    "Fecal Coliform (MPN/100mL)": ("fecal_coliform", "MPN/100mL"),
    "Total Coliform (MPN/100mL)": ("total_coliform", "MPN/100mL"),
}


@dataclass(frozen=True)
class PreparedCPCB:
    """Numeric records plus an explicit audit of every unused cell/record."""
    ground: pd.DataFrame
    exclusions: pd.DataFrame
    sources: list[dict]


def prepare_cpcb(paths: list[Path]) -> PreparedCPCB:
    """Preserve originals; parse declared day-first timestamps and log missing/text cells."""
    if not paths:
        raise ValueError("Supply at least one CPCB CSV")
    records, exclusions, sources = [], [], []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"Station", "Latitude", "Longitude", "Data Acquisition Time", *PARAMETERS}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing CPCB columns in {path.name}: {sorted(required - set(frame.columns))}")
        dates = pd.to_datetime(frame["Data Acquisition Time"], format="%d-%m-%Y %H:%M", errors="coerce")
        sources.append({"filename": path.name, "sha256": digest, "rows": len(frame),
                        "actual_start": None if dates.dropna().empty else dates.min().isoformat(),
                        "actual_end": None if dates.dropna().empty else dates.max().isoformat()})
        for index, row in frame.iterrows():
            source_record = index + 2
            event_id = f"cpcb-{digest[:16]}-{source_record}"
            lat, lon = pd.to_numeric(row.Latitude, errors="coerce"), pd.to_numeric(row.Longitude, errors="coerce")
            reason = None
            if not row.Station.strip():
                reason = "missing_station"
            elif pd.isna(dates.iloc[index]):
                reason = "invalid_timestamp"
            elif not np.isfinite(lat) or not np.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                reason = "invalid_coordinates"
            if reason:
                exclusions.append([path.name, source_record, event_id, "record", "", reason])
                continue
            identity = f"{row.Station.strip()}|{lat:.8f}|{lon:.8f}"
            station_id = "cpcb-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
            for column, (parameter, unit) in PARAMETERS.items():
                raw = row[column]
                value = pd.to_numeric(raw, errors="coerce")
                if pd.isna(value) or not np.isfinite(value):
                    reason = "missing_measurement" if raw.strip() in ("", "-", "NA", "N/A", "NaN", "nan") else "non_numeric_or_nonfinite"
                    exclusions.append([path.name, source_record, event_id, parameter, raw, reason])
                    continue
                records.append({"observation_id": event_id, "station_id": station_id,
                    "sample_date": dates.iloc[index].date().isoformat(), "latitude": float(lat), "longitude": float(lon),
                    "parameter": parameter, "value": float(value), "unit": unit, "quality_flag": "source_not_reported",
                    "station_name": row.Station, "source_file": path.name, "source_record": source_record,
                    "source_timestamp": row["Data Acquisition Time"], "river_label": row.get("River", ""),
                    "raw_measurement": raw})
    if not records:
        raise ValueError("No numeric CPCB observations remain; inspect source records")
    ground = pd.DataFrame(records)
    validate_ground(ground)
    return PreparedCPCB(ground, pd.DataFrame(exclusions, columns=["source_file", "source_record", "observation_id", "parameter", "raw_value", "reason"]), sources)

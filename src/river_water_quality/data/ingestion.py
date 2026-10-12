"""Local CSV adapters with explicit provenance; no network or output mutations."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal
import csv
import hashlib
import io
import json

import pandas as pd

from river_water_quality.data.schemas import (
    GROUND_COLUMNS, GROUND_NUMERIC, SATELLITE_COLUMNS, SATELLITE_NUMERIC,
    validate_ground, validate_satellite,
)


@dataclass(frozen=True)
class IngestedDataset:
    """Validated observations with source metadata and hashes for traceability."""

    data: pd.DataFrame
    provenance: dict[str, Any]
    source_sha256: str
    manifest_sha256: str


def load_dataset(
    csv_path: str | Path,
    manifest_path: str | Path,
    *,
    kind: Literal["cpcb", "usgs", "sentinel2"],
) -> IngestedDataset:
    """Read a local UTF-8 CSV and JSON manifest, rejecting invalid rows.

    Column mappings map source names to internal names. Numeric conversion is
    strict; identifiers, dates, flags, extra columns, and units stay unchanged.
    This does not convert units, calculate targets, filter flags, or match data.
    """
    if kind not in ("cpcb", "usgs", "sentinel2"):
        raise ValueError("kind must be cpcb, usgs, or sentinel2")
    source_path = Path(csv_path)
    metadata_path = Path(manifest_path)
    manifest_bytes = metadata_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Provenance manifest must be a JSON object")
    for field in ("dataset_id", "provider", "source", "access_date", "license", "coordinate_reference_system", "timestamp_convention"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Provenance {field} must be a non-empty string")
    if manifest["provider"] != kind:
        raise ValueError(f"Provenance provider must be {kind}")
    if manifest["coordinate_reference_system"] != "EPSG:4326":
        raise ValueError("Coordinates must be explicitly supplied in EPSG:4326; reprojection is not automatic")
    try:
        if date.fromisoformat(manifest["access_date"]).isoformat() != manifest["access_date"]:
            raise ValueError()
    except ValueError as exc:
        raise ValueError("Provenance access_date must be a valid ISO date") from exc
    units = manifest.get("units")
    if not isinstance(units, dict) or not units or any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
        for k, v in units.items()
    ):
        raise ValueError("Provenance units must map parameter/band names to non-empty unit strings")
    mapping = manifest.get("column_mapping", {})
    if not isinstance(mapping, dict) or any(
        not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
        for k, v in mapping.items()
    ):
        raise ValueError("column_mapping must map source column names to internal names")
    source_bytes = source_path.read_bytes()
    # pandas otherwise silently renames duplicate headers; check original headers.
    text = source_bytes.decode("utf-8-sig")
    header = next(csv.reader(io.StringIO(text)), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("CSV must have a non-empty header with unique column names")
    if set(mapping) - set(header):
        raise ValueError(f"Mapped source columns not found: {sorted(set(mapping) - set(header))}")
    renamed = [mapping.get(c, c) for c in header]
    if len(renamed) != len(set(renamed)):
        raise ValueError("column_mapping creates duplicate column names")
    records = csv.reader(io.StringIO(text))
    next(records)
    for number, row in enumerate(records, start=2):
        if len(row) != len(header):
            raise ValueError(f"CSV record {number} has {len(row)} fields; expected {len(header)}")
    data = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, skip_blank_lines=False).rename(columns=mapping)
    satellite = kind == "sentinel2"
    columns = SATELLITE_COLUMNS if satellite else GROUND_COLUMNS
    numeric = SATELLITE_NUMERIC if satellite else GROUND_NUMERIC
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for column in numeric:
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid numeric values in {column}") from exc
    (validate_satellite if satellite else validate_ground)(data)
    feature = "band" if satellite else "parameter"
    for name, unit in zip(data[feature], data["unit"]):
        if units.get(name) != unit:
            raise ValueError(f"Unit for {feature} {name!r} does not match provenance units")
    return IngestedDataset(data, manifest, hashlib.sha256(source_bytes).hexdigest(), hashlib.sha256(manifest_bytes).hexdigest())

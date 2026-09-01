"""Canonical columns and validation for Phase 1 data interfaces."""

from collections.abc import Iterable

import pandas as pd

from .base import SchemaError

SENTINEL2_COLUMNS = (
    "observation_time",
    "latitude",
    "longitude",
    "tile_id",
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
)

CPCB_COLUMNS = (
    "observation_time",
    "station_id",
    "latitude",
    "longitude",
    "parameter",
    "value",
    "unit",
)

USGS_COLUMNS = (
    "observation_time",
    "site_id",
    "latitude",
    "longitude",
    "parameter",
    "value",
    "unit",
)


def validate_schema(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise SchemaError when required columns are missing or duplicated."""
    if not isinstance(frame, pd.DataFrame):
        raise SchemaError("Expected a pandas DataFrame")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise SchemaError(f"Duplicate columns: {duplicates}")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

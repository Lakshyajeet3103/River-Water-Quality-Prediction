"""Internal interchange contracts, not claims about provider export formats."""

from datetime import date, datetime

import numpy as np
import pandas as pd

from river_water_quality.data.validation import validate_observations


GROUND_COLUMNS = (
    "observation_id", "station_id", "sample_date", "latitude", "longitude",
    "parameter", "value", "unit", "quality_flag",
)
SATELLITE_COLUMNS = (
    "scene_id", "station_id", "acquired_at", "latitude", "longitude",
    "band", "value", "unit", "product_id", "cloud_fraction", "water_fraction",
    "pixel_count",
)
GROUND_NUMERIC = ("latitude", "longitude", "value")
SATELLITE_NUMERIC = (
    "latitude", "longitude", "value", "cloud_fraction", "water_fraction", "pixel_count",
)


def _validate(data: pd.DataFrame, columns: tuple[str, ...], numeric: tuple[str, ...], key: list[str]) -> None:
    validate_observations(
        data, required_columns=columns,
        column_types={c: "numeric" if c in numeric else "string" for c in columns},
        observation_key=key,
    )
    for column in numeric:
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numeric values")
    for column, lower, upper in (("latitude", -90, 90), ("longitude", -180, 180)):
        if not data[column].between(lower, upper).all():
            raise ValueError(f"{column} must be between {lower} and {upper} (WGS84 degrees)")


def validate_ground(data: pd.DataFrame) -> None:
    """Validate long-format ground observations, retaining flags and replicates.

    observation_id identifies an actual sampling event; repeated parameters in
    that event are rejected. Distinct real replicates need distinct source IDs.
    No chemical ranges or acceptance thresholds are invented.
    """
    _validate(data, GROUND_COLUMNS, GROUND_NUMERIC, ["observation_id", "parameter"])
    for value in data["sample_date"]:
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError()
        except ValueError as exc:
            raise ValueError("sample_date must be a valid ISO date YYYY-MM-DD") from exc
    metadata = ["station_id", "sample_date", "latitude", "longitude"]
    if data.groupby("observation_id")[metadata].nunique().gt(1).any().any():
        raise ValueError("Conflicting station/date/coordinates within observation_id")


def validate_satellite(data: pd.DataFrame) -> None:
    """Validate extracted long-format spectral features, without reading rasters."""
    _validate(data, SATELLITE_COLUMNS, SATELLITE_NUMERIC, ["station_id", "scene_id", "band"])
    for value in data["acquired_at"]:
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError()
        except ValueError as exc:
            raise ValueError("acquired_at must be an ISO timestamp with an explicit UTC offset") from exc
    for column in ("cloud_fraction", "water_fraction"):
        if not data[column].between(0, 1).all():
            raise ValueError(f"{column} must be between 0 and 1")
    counts = data["pixel_count"]
    if not ((counts > 0) & (counts % 1 == 0)).all():
        raise ValueError("pixel_count must contain positive integers")
    metadata = ["acquired_at", "latitude", "longitude", "product_id", "cloud_fraction", "water_fraction", "pixel_count"]
    if data.groupby(["station_id", "scene_id"])[metadata].nunique().gt(1).any().any():
        raise ValueError("Conflicting metadata within station_id/scene_id")

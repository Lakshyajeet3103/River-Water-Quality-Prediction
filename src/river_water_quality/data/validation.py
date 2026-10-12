"""Explicit, non-mutating checks for observation tables."""

from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd
from pandas.api import types


def validate_observations(
    data: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    column_types: Mapping[str, str] | None = None,
    observation_key: Sequence[str] = (),
    duplicate_policy: Literal["error", "allow"] = "error",
) -> None:
    """Raise on invalid observations without filling, converting, or deleting rows.

    Types supported: numeric, string, datetime, boolean. Missing values in
    required fields are forbidden. Duplicate keys are rejected unless explicitly
    allowed; callers must choose keys appropriate to their dataset schema.
    """
    if duplicate_policy not in ("error", "allow"):
        raise ValueError("duplicate_policy must be error or allow")
    if not data.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed")
    if data.empty:
        raise ValueError("Observation table must contain at least one row")
    checks = {
        "numeric": lambda s: types.is_numeric_dtype(s) and not types.is_bool_dtype(s),
        "string": lambda s: s.dropna().map(lambda v: isinstance(v, str)).all(),
        "datetime": lambda s: types.is_datetime64_any_dtype(s),
        "boolean": lambda s: types.is_bool_dtype(s),
    }
    column_types = column_types or {}
    unknown = set(column_types.values()) - set(checks)
    if unknown:
        raise ValueError(f"Unsupported column types: {sorted(unknown)}")
    expected = set(required_columns) | set(column_types) | set(observation_key)
    missing = expected - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for column in set(required_columns) | set(observation_key):
        values = data[column]
        blank = values.map(lambda v: isinstance(v, str) and not v.strip())
        invalid = values.isna() | blank
        if invalid.any():
            raise ValueError(f"Missing required fields in {column}: {int(invalid.sum())} rows")
    for column, kind in column_types.items():
        if not checks[kind](data[column]):
            raise ValueError(f"Column {column} must have type {kind}; got {data[column].dtype}")
    if observation_key and duplicate_policy == "error":
        duplicates = data.duplicated(subset=list(observation_key), keep=False)
        if duplicates.any():
            raise ValueError(f"Duplicate observations for {list(observation_key)}: {int(duplicates.sum())} rows")

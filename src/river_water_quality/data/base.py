"""Base protocol shared by Sentinel-2, CPCB and USGS adapters."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class SchemaError(ValueError):
    """Raised when a source does not provide the required standardized columns."""


class DataSource(ABC):
    """Small adapter contract for a water-quality data source.

    Adapters return pandas DataFrames with a stable schema. Network access and
    source-specific authentication are intentionally left to concrete adapters.
    """

    name: str
    columns: tuple[str, ...]

    @abstractmethod
    def load(self, **kwargs: Any) -> pd.DataFrame:
        """Load source data and return it using the standardized schema."""
        raise NotImplementedError

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate and return a frame without modifying its values."""
        from .schemas import validate_schema

        validate_schema(frame, self.columns)
        return frame

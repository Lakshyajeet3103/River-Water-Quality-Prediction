"""Deterministic synthetic fixtures for interface/unit tests only."""

import pandas as pd


def sentinel2_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_time": pd.to_datetime(["2026-01-01T10:00:00Z", "2026-01-02T10:00:00Z"]),
            "latitude": [12.97, 13.01],
            "longitude": [80.25, 80.28],
            "tile_id": ["TILE_TEST_A", "TILE_TEST_B"],
            "blue": [0.10, 0.11],
            "green": [0.15, 0.16],
            "red": [0.12, 0.13],
            "nir": [0.35, 0.36],
            "swir1": [0.20, 0.21],
            "swir2": [0.18, 0.19],
        }
    )


def cpcb_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_time": pd.to_datetime(["2026-01-01T10:15:00Z", "2026-01-02T10:15:00Z"]),
            "station_id": ["CPCB_TEST_01", "CPCB_TEST_02"],
            "latitude": [12.97, 13.01],
            "longitude": [80.25, 80.28],
            "parameter": ["pH", "dissolved_oxygen"],
            "value": [7.2, 6.8],
            "unit": ["pH", "mg/L"],
        }
    )


def usgs_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_time": pd.to_datetime(["2026-01-01T11:00:00Z", "2026-01-02T11:00:00Z"]),
            "site_id": ["USGS_TEST_01", "USGS_TEST_02"],
            "latitude": [40.00, 40.01],
            "longitude": [-75.00, -75.01],
            "parameter": ["temperature", "specific_conductance"],
            "value": [12.5, 420.0],
            "unit": ["degC", "uS/cm"],
        }
    )

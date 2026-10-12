"""Synthetic fixtures for software validation only; not research observations."""

import pandas as pd
import pytest

from river_water_quality.data.validation import validate_observations


@pytest.fixture
def observations():
    return pd.DataFrame({
        "station_id": ["TEST_A", "TEST_B"],
        "observed_at": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "value": [1.0, 2.0],
    })


def check(data, **kwargs):
    validate_observations(
        data,
        required_columns=["station_id", "observed_at", "value"],
        column_types={"station_id": "string", "observed_at": "datetime", "value": "numeric"},
        observation_key=["station_id", "observed_at"],
        **kwargs,
    )


def test_valid_data_is_not_changed(observations):
    original = observations.copy(deep=True)
    check(observations)
    pd.testing.assert_frame_equal(observations, original)


def test_missing_column(observations):
    with pytest.raises(ValueError, match="Missing required columns.*value"):
        check(observations.drop(columns="value"))


@pytest.mark.parametrize("invalid", [None, "", "  "])
def test_missing_required_field(observations, invalid):
    observations.loc[0, "station_id"] = invalid
    with pytest.raises(ValueError, match="Missing required fields in station_id"):
        check(observations)


def test_wrong_type(observations):
    observations["value"] = ["1", "2"]
    with pytest.raises(ValueError, match="value must have type numeric"):
        check(observations)


def test_duplicates_have_explicit_policy(observations):
    data = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate observations"):
        check(data)
    check(data, duplicate_policy="allow")
    assert len(data) == 3


def test_unknown_policy(observations):
    with pytest.raises(ValueError, match="duplicate_policy"):
        check(observations, duplicate_policy="drop")


def test_empty_table(observations):
    with pytest.raises(ValueError, match="at least one row"):
        check(observations.iloc[:0])


def test_duplicate_column_labels(observations):
    observations.columns = ["station_id", "value", "value"]
    with pytest.raises(ValueError, match="Duplicate column names"):
        check(observations)


def test_unknown_type(observations):
    with pytest.raises(ValueError, match="Unsupported column types"):
        validate_observations(observations, required_columns=[], column_types={"value": "floatish"})

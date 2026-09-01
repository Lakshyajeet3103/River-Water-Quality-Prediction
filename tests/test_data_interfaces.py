from abc import ABC

import pandas as pd
import pytest

from river_water_quality.data.base import DataSource, SchemaError
from river_water_quality.data.fixtures import cpcb_fixture, sentinel2_fixture, usgs_fixture
from river_water_quality.data.schemas import CPCB_COLUMNS, SENTINEL2_COLUMNS, USGS_COLUMNS, validate_schema


@pytest.mark.parametrize(
    ("fixture", "columns"),
    [
        (sentinel2_fixture, SENTINEL2_COLUMNS),
        (cpcb_fixture, CPCB_COLUMNS),
        (usgs_fixture, USGS_COLUMNS),
    ],
)
def test_synthetic_fixtures_match_source_schemas(fixture, columns):
    frame = fixture()
    validate_schema(frame, columns)
    assert not frame.empty


def test_schema_validation_reports_missing_columns():
    with pytest.raises(SchemaError, match="Missing required columns"):
        validate_schema(pd.DataFrame({"station_id": ["x"]}), CPCB_COLUMNS)


def test_data_source_is_abstract():
    assert issubclass(DataSource, ABC)
    with pytest.raises(TypeError):
        DataSource()

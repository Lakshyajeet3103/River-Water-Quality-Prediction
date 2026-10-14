"""Regenerate UI test fixtures from real frozen API results, never runtime demo data."""

import json
from pathlib import Path
from river_water_quality.api import (
    catalog,
    predict,
    PredictionRequest,
    evaluation,
    data,
    oxygen,
    OxygenRequest,
)


def main():
    root = Path(__file__).resolve().parents[1]
    c = catalog()
    first = next(o for o in c["observations"] if o["has_shap"])
    oid = first["observation_id"]
    fixture = dict(
        catalog=c,
        evaluations={
            cohort: evaluation(cohort)
            for cohort in ["familiar_20", "new_stations_strict_26"]
        },
        data=data(oid),
        predictions={
            m["id"]: predict(PredictionRequest(observation_id=oid, model_id=m["id"]))
            for m in c["models"]
        },
        oxygen=oxygen(OxygenRequest(observation_id=oid, model_id="satellite_ratios")),
    )
    (root / "frontend/src/demo.test-fixture.json").write_text(
        json.dumps(fixture), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

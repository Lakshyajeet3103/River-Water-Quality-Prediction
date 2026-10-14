"""Acceptance checks against the real frozen demonstration artifacts."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from river_water_quality.api import app, MODELS, SHAP, read
from river_water_quality.bod_category import assess

client = TestClient(app)


@pytest.fixture(scope="module")
def catalog():
    response = client.get("/api/catalog")
    assert response.status_code == 200
    return response.json()


def test_catalog_and_exact_evaluation(catalog):
    assert len(catalog["observations"]) == 46
    assert sum(o["has_shap"] for o in catalog["observations"]) == 7
    for cohort, n in [("familiar_20", 20), ("new_stations_strict_26", 26)]:
        response = client.get("/api/evaluation", params={"cohort": cohort})
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == n
        for mid, column in [
            ("selected_four_inputs", "selected_four_inputs"),
            ("all_components", "network_all_components"),
            ("satellite_ratios", "satellite_ratios"),
        ]:
            metric = next(m for m in data["metrics"] if m["model"] == column)
            actual = np.array([p["actual_bod_mg_l"] for p in data["predictions"]])
            predicted = np.array([p[column + "_bod_mg_l"] for p in data["predictions"]])
            assert metric["mae"] == pytest.approx(
                np.abs(actual - predicted).mean(), abs=1e-7
            )
        csv = client.get("/api/evaluation.csv", params={"cohort": cohort})
        assert len(csv.text.splitlines()) == n + 1


@pytest.mark.parametrize("mid", [m["id"] for m in MODELS])
def test_models_shap_category_and_live(catalog, mid):
    examples = [o for o in catalog["observations"] if o["has_shap"]]
    artifact = read(SHAP / (mid + "_explanations.csv"))
    for o in examples:
        body = dict(model_id=mid, observation_id=o["observation_id"])
        response = client.post("/api/predict", json=body)
        assert response.status_code == 200, response.text
        saved = response.json()
        explanation = saved["shap"]
        assert explanation["observation_id"] == o["observation_id"]
        assert explanation["predicted_bod_mg_l"] == pytest.approx(
            saved["predicted_bod_mg_l"], abs=1e-5
        )
        assert explanation["reference_bod_mg_l"] + sum(
            v for k, v in explanation.items() if k.startswith("shap_")
        ) == pytest.approx(saved["predicted_bod_mg_l"], abs=1e-5)
        assert (
            artifact.loc[artifact.observation_id.eq(o["observation_id"])].shape[0] == 1
        )
        live = client.post("/api/predict", json={**body, "source": "live"})
        assert live.status_code == 200, live.text
        assert live.json()["predicted_bod_mg_l"] == pytest.approx(
            saved["predicted_bod_mg_l"], abs=1e-5
        )
        category = saved["category"]
        if o["cohort"] == "familiar_20":
            chemistry = category["chemistry"]
            expected = assess(
                saved["predicted_bod_mg_l"],
                chemistry["ground_ph"]["value"],
                chemistry["ground_dissolved_oxygen"]["value"],
                chemistry["ground_fecal_coliform"]["value"],
            )
            assert category["custom_score"] == pytest.approx(expected["custom_score"])
        else:
            assert not category["available"]
        if mid == "all_components":
            assert live.json()["gate"]["satellite"] + live.json()["gate"][
                "ground"
            ] == pytest.approx(1)
    nonexample = next(o for o in catalog["observations"] if not o["has_shap"])
    assert (
        client.post(
            "/api/predict",
            json=dict(model_id=mid, observation_id=nonexample["observation_id"]),
        ).json()["shap"]
        is None
    )


@pytest.mark.parametrize("mid", [m["id"] for m in MODELS])
def test_oxygen_and_matching(catalog, mid):
    o = next(o for o in catalog["observations"] if o["has_shap"])
    response = client.post(
        "/api/oxygen",
        json=dict(model_id=mid, observation_id=o["observation_id"], days=5),
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["curves"]) == 101
    first = data["curves"][0]
    assert first["pinn_oxygen"] == pytest.approx(o["ground_dissolved_oxygen"], abs=1e-5)
    assert first["reference_oxygen"] == pytest.approx(o["ground_dissolved_oxygen"])
    matched = client.get("/api/data/" + o["observation_id"]).json()
    assert len(matched["ratios"]) == 5
    assert matched["satellite_thumbnail"] is None
    assert (
        client.post(
            "/api/oxygen",
            json=dict(model_id=mid, observation_id=o["observation_id"], days=6),
        ).status_code
        == 422
    )


def test_validation(catalog):
    oid = catalog["observations"][0]["observation_id"]
    assert (
        client.post(
            "/api/predict", json={"model_id": "fake", "observation_id": oid}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/predict",
            json={"model_id": "satellite_ratios", "observation_id": "fake"},
        ).status_code
        == 404
    )
    assert client.get("/api/evaluation?cohort=fake").status_code == 422


def test_missing_ground_input_does_not_block_satellite_model(catalog, monkeypatch):
    import river_water_quality.api as api

    row = api.observation(catalog["observations"][0]["observation_id"]).copy()
    row["ground_fecal_coliform"] = np.nan
    monkeypatch.setattr(api, "observation", lambda oid: row)
    body = dict(observation_id=row.observation_id, model_id="selected_four_inputs")
    response = client.post("/api/predict", json=body)
    assert response.status_code == 422
    assert "ground_fecal_coliform" in response.json()["detail"]
    assert (
        client.post(
            "/api/predict", json={**body, "model_id": "satellite_ratios"}
        ).status_code
        == 200
    )

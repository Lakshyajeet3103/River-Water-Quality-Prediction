"""Frozen evaluation demo API; no Earth Engine authentication required."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from river_water_quality.bod_category import assess, CHEMISTRY, WEIGHTS

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/saved_models_unseen_results"
SHAP = ROOT / "docs/three_model_shap_results"
PROVENANCE = json.loads((SHAP / "provenance.json").read_text())
MODELS = [
    dict(
        id="selected_four_inputs",
        name="Selected-feature combined",
        trait="Lowest error on our familiar-station test",
        limitation="Requires current fecal coliform; a single combined-input network",
    ),
    dict(
        id="all_components",
        name="Two-branch fusion",
        trait="Separate satellite and chemical branches with a learned gate",
        limitation="Requires ground measurements; experimental components do not guarantee better accuracy",
    ),
    dict(
        id="satellite_ratios",
        name="Satellite-only with band ratios",
        trait="BOD prediction without current chemical measurements",
        limitation="Higher prediction error",
    ),
]
for m in MODELS:
    m.update(
        version=PROVENANCE["models"][m["id"]]["model_sha256"],
        input_requirements=PROVENANCE["models"][m["id"]]["input_features"],
        output_unit="mg/L",
        input_units={
            k: (
                "surface reflectance"
                if k.startswith("satellite_")
                else (
                    "days"
                    if k == "temporal_offset_days"
                    else (
                        "pH"
                        if k == "ground_ph"
                        else (
                            "mg/L" if k == "ground_dissolved_oxygen" else "count/100 mL"
                        )
                    )
                )
            )
            for k in PROVENANCE["models"][m["id"]]["input_features"]
        },
    )
app = FastAPI(title="RIVEROS research demo", version="2.0.0")


def read(path):
    if not path.is_file():
        raise HTTPException(
            503,
            f"Saved artifact unavailable: {path.relative_to(ROOT)}. Run scripts/restore_demo_bundle.py.",
        )
    return pd.read_csv(path)


def records(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def clean(value):
    return json.loads(
        json.dumps(
            value,
            default=lambda x: x.item() if hasattr(x, "item") else str(x),
            allow_nan=False,
        )
    )


@lru_cache
def observations_frame():
    paired = read(ROOT / "outputs/india_100_chemical_dataset_20261014/paired.csv")
    new = read(ROOT / "outputs/new_station_matches_v2_20261014/new_paired.csv")
    results = read(RESULTS / "predictions.csv")
    ids = results.loc[
        results.cohort.isin(["familiar_20", "new_stations_strict_26"]),
        ["observation_id", "cohort"],
    ]
    return pd.concat([paired, new], ignore_index=True).merge(
        ids, on="observation_id", validate="one_to_one"
    )


def observation(oid):
    found = observations_frame().loc[lambda x: x.observation_id.eq(oid)]
    if found.empty:
        raise HTTPException(404, "Unknown saved evaluation observation")
    return found.iloc[0]


def model_info(mid):
    model = next((m for m in MODELS if m["id"] == mid), None)
    if model is None:
        raise HTTPException(422, "Choose one of the three supported models")
    return model


@app.get("/api/catalog")
def catalog():
    rows = observations_frame()
    examples = set(read(SHAP / "explained_examples.csv").observation_id)
    return dict(
        models=MODELS,
        observations=records(
            rows.assign(has_shap=rows.observation_id.isin(examples), split="test")
        ),
        units={
            "ground_ph": "pH",
            "ground_fecal_coliform": "count/100 mL",
            "ground_dissolved_oxygen": "mg/L",
            "temporal_offset_days": "days",
        },
    )


@app.get("/api/evaluation")
def evaluation(
    cohort: Literal["familiar_20", "new_stations_strict_26"] = "familiar_20",
):
    metrics = read(RESULTS / "metrics.csv")
    predictions = read(RESULTS / "predictions.csv")
    return dict(
        metrics=records(metrics.loc[metrics.cohort.eq(cohort)]),
        predictions=records(predictions.loc[predictions.cohort.eq(cohort)]),
        source="docs/saved_models_unseen_results",
    )


@app.get("/api/evaluation.csv")
def download(cohort: Literal["familiar_20", "new_stations_strict_26"] = "familiar_20"):
    frame = read(RESULTS / "predictions.csv")
    return Response(
        frame.loc[frame.cohort.eq(cohort)].to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{cohort}.csv"'},
    )


@lru_cache
def frozen_models():
    import sys

    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from three_model_support import load

    return load()[4]


def category_for(row, bod):
    profile_path = ROOT / "docs/bod_led_category_results/station_profiles.csv"
    if not profile_path.is_file():
        return dict(
            available=False, note="Saved category chemistry profiles unavailable."
        )
    profiles = read(profile_path)
    profile = profiles.loc[profiles.station_id.eq(row.station_id)]
    if profile.empty:
        return dict(
            available=False,
            note="No training-only station chemistry profile for this unseen station; category unavailable.",
        )
    p = profile.iloc[0]
    return dict(
        available=True,
        **assess(bod, p.ground_ph, p.ground_dissolved_oxygen, p.ground_fecal_coliform),
        predicted_bod_mg_l=bod,
        chemistry={
            k: dict(value=float(p[k]), source="training-only station average")
            for k in CHEMISTRY
        },
        weights=WEIGHTS,
        thresholds="Best: score >= 80; Good: 60 <= score < 80; Bad: score < 60",
        note="Custom demonstration indicator; not an independently validated full WQI.",
    )


class PredictionRequest(BaseModel):
    observation_id: str
    model_id: str
    source: Literal["saved", "live"] = "saved"


@app.post("/api/predict")
def predict(request: PredictionRequest):
    info = model_info(request.model_id)
    row = observation(request.observation_id)
    missing = [k for k in info["input_requirements"] if k not in row or pd.isna(row[k])]
    if missing:
        raise HTTPException(422, "Missing required inputs: " + ", ".join(missing))
    gate = None
    if request.source == "live":
        try:
            model = frozen_models()[request.model_id]
            bod = float(model["predict"](row.to_frame().T)[0])
            if request.model_id == "all_components":
                import torch
                from river_water_quality.training.experimental_network import (
                    predict_network,
                )

                out = predict_network(
                    torch.load(model["path"], map_location="cpu", weights_only=True),
                    row.to_frame().T,
                )
                gate = dict(
                    satellite=float(out["satellite_weight"].ravel()[0]),
                    ground=float(out["ground_weight"].ravel()[0]),
                )
        except (ImportError, OSError, ValueError) as exc:
            raise HTTPException(503, f"Live model unavailable: {exc}") from exc
    else:
        frame = read(RESULTS / "predictions.csv")
        item = frame.loc[
            frame.observation_id.eq(request.observation_id)
            & frame.cohort.eq(row.cohort)
        ].iloc[0]
        prefix = (
            "network_all_components"
            if request.model_id == "all_components"
            else request.model_id
        )
        bod = float(item[prefix + "_bod_mg_l"])
        if request.model_id == "all_components":
            saved_gates = read(
                ROOT
                / "outputs/experimental_network_20261014/all_components_predictions.csv"
            )
            matched_gate = saved_gates.loc[
                saved_gates.observation_id.eq(request.observation_id)
            ]
            if not matched_gate.empty:
                g = matched_gate.iloc[0]
                if abs(float(g.predicted_bod_mg_l) - bod) <= 1e-5:
                    gate = dict(
                        satellite=float(g.satellite_weight),
                        ground=float(g.ground_weight),
                    )
    explanation_path = SHAP / (request.model_id + "_explanations.csv")
    explanation = []
    if explanation_path.is_file():
        explanations = read(explanation_path)
        explanation = records(
            explanations.loc[explanations.observation_id.eq(request.observation_id)]
        )
    shap = explanation[0] if explanation else None
    if shap and abs(shap["predicted_bod_mg_l"] - bod) > 1e-5:
        shap = None
    return clean(
        dict(
            model=info,
            observation_id=request.observation_id,
            result_source=(
                "Live prediction"
                if request.source == "live"
                else "Saved evaluation result"
            ),
            predicted_bod_mg_l=bod,
            measured_bod_mg_l=float(row.ground_bod),
            difference_mg_l=bod - float(row.ground_bod),
            inputs={k: float(row[k]) for k in info["input_requirements"]},
            category=category_for(row, bod),
            shap=shap,
            gate=gate,
        )
    )


@app.get("/api/data/{oid}")
def data(oid: str):
    row = observation(oid)
    from river_water_quality.training.satellite_refinements import features

    ratios = features(row.to_frame().T, "bands_ratios").iloc[0]
    return dict(
        observation=records(row.to_frame().T)[0],
        ratios={
            k: float(v) for k, v in ratios.items() if not k.startswith("satellite_")
        },
        satellite_thumbnail=None,
        thumbnail_note="Satellite thumbnail unavailable",
        source="Sentinel-2 matched spectral extraction; map tiles are a separate OpenStreetMap background",
        extraction_note="Extraction geometry is not included in the saved record; distance and pixel count are provided where available.",
        river="Ganga" if "GANGA" in str(row.station_name).upper() else None,
    )


class OxygenRequest(PredictionRequest):
    days: float = 5


@app.post("/api/oxygen")
def oxygen(request: OxygenRequest):
    if not np.isfinite(request.days) or not 0 <= request.days <= 5:
        raise HTTPException(422, "Simulation time must be between zero and five days")
    row = observation(request.observation_id)
    if pd.isna(row.ground_dissolved_oxygen):
        raise HTTPException(422, "Initial measured dissolved oxygen is required")
    result = predict(request)
    try:
        import torch
        from river_water_quality.training.oxygen_pinn import (
            predict as simulate,
            analytical,
            initial_demand,
        )

        path = ROOT / "docs/oxygen_pinn_results/oxygen_pinn.pt"
        provenance = json.loads((path.parent / "provenance.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["model_sha256"]:
            raise ValueError("Oxygen checkpoint hash mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        t = np.linspace(0, request.days, 101)
        bod, do = result["predicted_bod_mg_l"], float(row.ground_dissolved_oxygen)
        curve = simulate(checkpoint, t, bod, do)
        demand, ref = analytical(t, initial_demand(bod), do)
        return clean(
            dict(
                inputs=dict(predicted_bod_mg_l=bod, initial_measured_do_mg_l=do),
                assumptions=provenance["assumptions"],
                extrapolation=not (
                    checkpoint["training_l0_range"][0]
                    <= float(initial_demand(bod))
                    <= checkpoint["training_l0_range"][1]
                    and checkpoint["training_do_range"][0]
                    <= do
                    <= checkpoint["training_do_range"][1]
                ),
                curves=[
                    dict(
                        day=float(day),
                        pinn_oxygen=float(p[1]),
                        reference_oxygen=float(o),
                        pinn_demand=float(p[0]),
                        reference_demand=float(l),
                    )
                    for day, p, o, l in zip(t, curve, ref, demand)
                ],
            )
        )
    except (ImportError, OSError, ValueError) as exc:
        raise HTTPException(503, f"Oxygen simulation unavailable: {exc}") from exc


@app.get("/api/health")
def health():
    return dict(status="ok")

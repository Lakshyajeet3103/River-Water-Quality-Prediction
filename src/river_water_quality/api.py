from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2].parent
DEMO_DATA = ROOT / "data" / "interim" / "dashboard_demo"
ARTIFACTS = ROOT / "artifacts" / "restored-demo" / "outputs"

app = FastAPI(
    title="RIVEROS API",
    description="Read-only API for river water-quality observations and experiment reports.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Data file not found: {path.relative_to(ROOT)}",
        )
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read data file: {path.name}") from exc


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


@app.get("/")
def root():
    return {
        "name": "RIVEROS API",
        "status": "ok",
        "docs": "/docs",
        "endpoints": [
            "/api/health",
            "/api/summary",
            "/api/stations",
            "/api/observations",
            "/api/satellite",
            "/api/model-metrics",
            "/api/model-predictions",
        ],
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "data_directory_exists": DEMO_DATA.is_dir(),
        "artifacts_directory_exists": ARTIFACTS.is_dir(),
    }


@app.get("/api/summary")
def summary():
    paired = load_csv(DEMO_DATA / "paired.csv")
    stations = load_csv(DEMO_DATA / "station_summary.csv")

    dates = pd.to_datetime(paired.get("sample_date"), errors="coerce")
    return {
        "paired_observations": len(paired),
        "station_count": int(paired["station_id"].nunique()) if "station_id" in paired else 0,
        "date_start": dates.min().date().isoformat() if dates.notna().any() else None,
        "date_end": dates.max().date().isoformat() if dates.notna().any() else None,
        "matched_events": int(stations["matched"].sum()) if "matched" in stations else len(paired),
        "unmatched_events": int(stations["unmatched"].sum()) if "unmatched" in stations else None,
        "source": "data/interim/dashboard_demo",
        "data_status": "pilot dataset; not representative of all river stations",
    }


@app.get("/api/stations")
def stations():
    df = load_csv(DEMO_DATA / "station_summary.csv")
    return {"count": len(df), "items": records(df)}


@app.get("/api/observations")
def observations(
    station_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
):
    df = load_csv(DEMO_DATA / "paired.csv")

    if station_id and "station_id" in df:
        df = df[df["station_id"].astype(str) == station_id]

    if start_date or end_date:
        dates = pd.to_datetime(df["sample_date"], errors="coerce")
        if start_date:
            dates_start = pd.to_datetime(start_date, errors="coerce")
            if pd.isna(dates_start):
                raise HTTPException(status_code=400, detail="Invalid start_date; use YYYY-MM-DD.")
            df = df[dates >= dates_start]
        if end_date:
            dates_end = pd.to_datetime(end_date, errors="coerce")
            if pd.isna(dates_end):
                raise HTTPException(status_code=400, detail="Invalid end_date; use YYYY-MM-DD.")
            df = df[dates <= dates_end]

    return {"count": len(df), "items": records(df.head(limit))}


@app.get("/api/satellite")
def satellite():
    df = load_csv(DEMO_DATA / "satellite.csv")
    return {"count": len(df), "items": records(df)}


@app.get("/api/model-metrics")
def model_metrics():
    path = ARTIFACTS / "fusion_sklearn_001" / "metrics.csv"
    df = load_csv(path)
    return {
        "experiment": "fusion_sklearn_001",
        "status": "exploratory; only 3 test observations",
        "target": "BOD",
        "unit": "mg/L",
        "limitations": [
            "Very small pilot dataset; metrics do not establish generalization.",
            "Prediction requires ground fecal-coliform measurement.",
            "This is retrospective estimation, not forecasting.",
        ],
        "count": len(df),
        "items": records(df),
    }


@app.get("/api/model-predictions")
def model_predictions(limit: int = Query(default=100, ge=1, le=1000)):
    path = ARTIFACTS / "fusion_sklearn_001" / "test_predictions.csv"
    df = load_csv(path)
    return {
        "experiment": "fusion_sklearn_001",
        "split": "test",
        "count": len(df),
        "items": records(df.head(limit)),
        "note": "Saved test-set predictions; not new live predictions.",
    }

"""Scikit-learn satellite + ground-data fusion baseline."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


SATELLITE_FEATURES = [f"satellite_{band}" for band in
                      ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")]
GROUND_FEATURES = ["ground_fecal_coliform"]
FEATURES = SATELLITE_FEATURES + GROUND_FEATURES
TARGET = "ground_bod"


def prepare_data(paired_csv, split_csv):
    """Join features to pre-existing split assignments without resplitting."""
    paired = pd.read_csv(paired_csv)
    splits = pd.read_csv(split_csv)

    required_paired = {"observation_id", *FEATURES, TARGET}
    required_splits = {"observation_id", "split"}
    if missing := required_paired - set(paired.columns):
        raise ValueError(f"paired.csv missing columns: {sorted(missing)}")
    if missing := required_splits - set(splits.columns):
        raise ValueError(f"split assignments missing columns: {sorted(missing)}")
    if paired["observation_id"].duplicated().any():
        raise ValueError("paired.csv must have one row per observation_id")
    if splits["observation_id"].duplicated().any():
        raise ValueError("split assignments must have one row per observation_id")

    data = paired.merge(
        splits[["observation_id", "split"]],
        on="observation_id", how="inner", validate="one_to_one"
    )
    data = data[data["split"].isin(["train", "validation", "test"])].copy()
    if data.empty:
        raise ValueError("No paired rows matched the saved split assignments")

    numeric = data[FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Model features and target must be finite and non-missing")
    if (numeric["ground_fecal_coliform"] < 0).any():
        raise ValueError("Fecal-coliform values must be non-negative")
    if (numeric[TARGET] < 0).any():
        raise ValueError("BOD target values must be non-negative")

    data[FEATURES + [TARGET]] = numeric
    for part in ("train", "validation", "test"):
        if (data["split"] == part).sum() == 0:
            raise ValueError(f"Saved split has no rows: {part}")
    return data


def regression_metrics(actual, predicted):
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else None,
    }


def train_fusion(paired_csv, split_csv, output_dir, seed=42):
    """Train using train rows only; validation/test rows are evaluation-only."""
    data = prepare_data(paired_csv, split_csv)
    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]
    test = data[data["split"] == "test"]

    if len(train) < 3:
        raise ValueError("At least 3 training rows are required for this baseline")

    # Log transform reduces the extreme scale of fecal-coliform counts.
    def transform_features(frame):
        x = frame[FEATURES].to_numpy(dtype=float).copy()
        x[:, -1] = np.log1p(x[:, -1])
        return x

    x_scaler = StandardScaler().fit(transform_features(train))
    y_scaler = StandardScaler().fit(train[[TARGET]].to_numpy(dtype=float))

    def x_values(frame):
        return x_scaler.transform(transform_features(frame))

    def y_values(frame):
        return y_scaler.transform(frame[[TARGET]].to_numpy(dtype=float)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="lbfgs",
        alpha=1.0,
        max_iter=3000,
        random_state=seed,
    )
    model.fit(x_values(train), y_values(train))

    def predict(frame):
        scaled = model.predict(x_values(frame)).reshape(-1, 1)
        return np.maximum(0.0, y_scaler.inverse_transform(scaled).ravel())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        pred = predict(frame)
        metric_rows.append({
            "model": "sklearn_mlp_fusion",
            "split": name,
            **regression_metrics(frame[TARGET].to_numpy(), pred),
        })
    test_pred = predict(test)
    mean_baseline = np.full(len(test), float(train[TARGET].mean()))
    metric_rows.append({
        "model": "training_mean_baseline",
        "split": "test",
        **regression_metrics(test[TARGET].to_numpy(), mean_baseline),
    })

    pd.DataFrame(metric_rows).to_csv(out / "metrics.csv", index=False)
    columns = [c for c in (
        "observation_id", "station_id", "sample_date", "scene_id", TARGET, "split"
    ) if c in test.columns]
    predictions = test[columns].copy().rename(columns={TARGET: "actual_bod_mg_l"})
    predictions["predicted_bod_mg_l"] = test_pred
    predictions.to_csv(out / "test_predictions.csv", index=False)
    data[["observation_id", "split"]].to_csv(out / "split_assignments.csv", index=False)

    joblib.dump({
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "features": FEATURES,
        "target": TARGET,
        "seed": seed,
    }, out / "fusion_model.joblib")

    report = {
        "model": "scikit-learn MLPRegressor fusion baseline",
        "seed": seed,
        "feature_columns": FEATURES,
        "target_column": TARGET,
        "rows": {part: int((data["split"] == part).sum())
                 for part in ("train", "validation", "test")},
        "training_only_scaling": True,
        "split_source": str(split_csv),
        "limitations": [
            "Small pilot dataset; metrics are exploratory, not evidence of generalization.",
            "Ground fecal-coliform measurement is required at prediction time.",
            "Date-level satellite matching is not forecasting.",
            "MLP architecture differs from the previous PyTorch experiment.",
        ],
    }
    (out / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return pd.DataFrame(metric_rows)

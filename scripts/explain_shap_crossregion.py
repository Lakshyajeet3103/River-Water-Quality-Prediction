import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(".")
MODEL_PATH = ROOT / "artifacts/restored-demo/outputs/fusion_sklearn_001/fusion_model.joblib"
INDIA_PATH = ROOT / "artifacts/restored-demo/outputs/hybrid_demo_001/paired.csv"
SPLIT_PATH = ROOT / "artifacts/restored-demo/outputs/fusion_sklearn_001/split_assignments.csv"
US_PATH = ROOT / "artifacts/restored-demo/outputs/crossregion_demo_001/us_paired.csv"
OUT = ROOT / "artifacts/restored-demo/outputs/shap_crossregion_001"
OUT.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

artifact = joblib.load(MODEL_PATH)
features = artifact["features"]
target = artifact["target"]
model = artifact["model"]
x_scaler = artifact["x_scaler"]
y_scaler = artifact["y_scaler"]

india = pd.read_csv(INDIA_PATH)
splits = pd.read_csv(SPLIT_PATH)
us = pd.read_csv(US_PATH)

if not set(features + [target]).issubset(india.columns):
    raise ValueError("India dataset is missing required model columns")
if not set(features).issubset(us.columns):
    raise ValueError("US dataset is missing required model features")

india = india.merge(
    splits[["observation_id", "split"]],
    on="observation_id",
    how="inner",
    validate="one_to_one",
)
background_frame = india.loc[india["split"] == "train", features].copy()
us_features = us[features].copy()

if len(background_frame) < 2 or len(us_features) == 0:
    raise ValueError("Insufficient India training background or US observations")
if set(india["observation_id"]) & set(us["observation_id"]):
    raise ValueError("India and US observation IDs overlap")
if not np.isfinite(background_frame.to_numpy(dtype=float)).all():
    raise ValueError("India background contains missing or non-finite features")
if not np.isfinite(us_features.to_numpy(dtype=float)).all():
    raise ValueError("US features contain missing or non-finite values")
if (background_frame["ground_fecal_coliform"] < 0).any():
    raise ValueError("India fecal-coliform values must be non-negative")
if (us_features["ground_fecal_coliform"] < 0).any():
    raise ValueError("US fecal-coliform values must be non-negative")

def predict_bod(raw_values):
    """Predict BOD in mg/L using the saved model and saved scalers."""
    x = np.asarray(raw_values, dtype=float).copy()
    x[:, -1] = np.log1p(x[:, -1])
    x_scaled = x_scaler.transform(x)
    y_scaled = model.predict(x_scaled).reshape(-1, 1)
    return np.maximum(
        0.0, y_scaler.inverse_transform(y_scaled).ravel()
    )

background = background_frame.to_numpy(dtype=float)
evaluation = us_features.to_numpy(dtype=float)

print("Model: saved scikit-learn MLP (no retraining)")
print("SHAP version:", shap.__version__)
print("India training background:", len(background))
print("US observations to explain:", len(evaluation))
print("Features:", len(features))
print("Calculating SHAP values; this may take a little while...")

explainer = shap.KernelExplainer(predict_bod, background)
shap_values = np.asarray(
    explainer.shap_values(evaluation, nsamples=200)
)
if shap_values.ndim == 3:
    shap_values = shap_values[..., 0]
shap_values = np.asarray(shap_values).reshape(len(evaluation), len(features))

predictions = predict_bod(evaluation)
base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
reconstructed = base_value + shap_values.sum(axis=1)

# Save per-observation predictions and their feature contributions.
contributions = pd.DataFrame(shap_values, columns=features)
contributions.insert(0, "observation_id", us["observation_id"].astype(str).to_numpy())
contributions.to_csv(OUT / "shap_contributions.csv", index=False)

importance = pd.DataFrame({
    "feature": features,
    "mean_absolute_shap": np.mean(np.abs(shap_values), axis=0),
}).sort_values("mean_absolute_shap", ascending=False)
importance.to_csv(OUT / "shap_feature_importance.csv", index=False)

details = pd.DataFrame({
    "observation_id": us["observation_id"].astype(str),
    "station_id": us["station_id"].astype(str) if "station_id" in us else "",
    "sample_date": us["sample_date"].astype(str) if "sample_date" in us else "",
    "actual_bod_mg_l": pd.to_numeric(us[target], errors="coerce")
        if target in us else np.nan,
    "predicted_bod_mg_l": predictions,
    "shap_base_value_mg_l": base_value,
    "shap_sum_reconstructed_mg_l": reconstructed,
    "additivity_residual_mg_l": predictions - reconstructed,
})
details.to_csv(OUT / "shap_predictions.csv", index=False)

# Global feature-importance bar plot.
plt.figure()
plt.barh(
    importance["feature"].iloc[::-1],
    importance["mean_absolute_shap"].iloc[::-1],
)
plt.xlabel("Mean absolute SHAP value (BOD mg/L)")
plt.title("SHAP feature importance — US evaluation observations")
plt.tight_layout()
plt.savefig(OUT / "shap_feature_importance.png", dpi=180)
plt.close()

# Per-observation SHAP contribution heatmap.
fig, ax = plt.subplots(figsize=(12, 4.8))
limit = max(float(np.max(np.abs(shap_values))), 1e-9)
im = ax.imshow(shap_values, aspect="auto", cmap="coolwarm",
               vmin=-limit, vmax=limit)
ax.set_xticks(range(len(features)))
ax.set_xticklabels(features, rotation=65, ha="right")
ax.set_yticks(range(len(us)))
ax.set_yticklabels(us["observation_id"].astype(str))
ax.set_title("SHAP contributions by US observation")
fig.colorbar(im, ax=ax, label="Contribution to predicted BOD (mg/L)")
fig.tight_layout()
fig.savefig(OUT / "shap_contribution_heatmap.png", dpi=180)
plt.close(fig)

report = {
    "method": "SHAP KernelExplainer",
    "model_type": type(model).__name__,
    "model_artifact": str(MODEL_PATH),
    "model_sha256": sha256(MODEL_PATH),
    "india_background_rows": int(len(background)),
    "india_background_split": "train only",
    "us_observations_explained": int(len(us)),
    "feature_count": int(len(features)),
    "features": list(features),
    "target": target,
    "prediction_units": "mg/L BOD",
    "kernel_explainer_nsamples": 200,
    "model_retrained": False,
    "us_data_used_for_training_or_background": False,
    "mean_absolute_additivity_residual_mg_l": float(
        np.mean(np.abs(predictions - reconstructed))
    ),
    "maximum_absolute_additivity_residual_mg_l": float(
        np.max(np.abs(predictions - reconstructed))
    ),
    "limitations": [
        "Only four US observations are explained.",
        "The India training background contains only six observations.",
        "SHAP describes model behavior, not causal effects.",
        "The model requires fecal-coliform measurements as an input.",
        "Cross-region results are exploratory and may reflect domain shift.",
        "Kernel SHAP estimates can vary with sampling and background selection.",
    ],
}
(OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\nSHAP run completed.")
print("Output directory:", OUT.resolve())
print("\nFeature importance:")
print(importance.to_string(index=False))
print("\nPredictions and additivity checks:")
print(details.to_string(index=False))
print("\nSaved report:", OUT / "report.json")

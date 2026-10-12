"""Train-only preprocessing, validation-only selection, and held-out testing."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from river_water_quality.training.dataset import BaselineDataset
from river_water_quality.training.splits import assign_splits


@dataclass(frozen=True)
class BaselineResults:
    """Selected fitted models, validation/test metrics, and held-out predictions."""

    models: dict[str, Any]
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    assignments: pd.DataFrame


def regression_metrics(actual: pd.Series, prediction: np.ndarray) -> dict[str, float | None]:
    """Return MAE/RMSE and R2; constant or too-small targets have undefined R2."""
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)) if len(actual) >= 2 and actual.nunique() > 1 else None,
    }


def train_baselines(
    data: BaselineDataset, split_settings: dict, *, seed: int,
    ridge_alpha: float = 1.0, forest_trees: int = 100, forest_min_leaf: int = 2,
) -> BaselineResults:
    """Select Ridge/forest on validation MAE for each modality, leaving test untouched.

    Scaling is fitted inside each Ridge pipeline on training only. A training-mean
    DummyRegressor is reported separately. Fixed hyperparameters are not tuned on
    test data, and selected models are not refitted on validation data.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if isinstance(ridge_alpha, bool) or not isinstance(ridge_alpha, (int, float)) or not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be finite and positive")
    if type(forest_trees) is not int or forest_trees < 1 or type(forest_min_leaf) is not int or forest_min_leaf < 1:
        raise ValueError("Forest trees and minimum leaf size must be positive integers")
    assignments = assign_splits(data.metadata, split_settings)
    if not assignments.index.equals(data.metadata.index):
        raise ValueError("Split assignments must align exactly with the baseline cohort")
    if not assignments.split.isin(["train", "validation", "test", "excluded"]).all():
        raise ValueError("Invalid split assignment")
    ids = {fold: assignments.index[assignments.split == fold] for fold in ("train", "validation", "test")}
    if any(len(events) < 2 for events in ids.values()):
        raise ValueError("Each split requires at least two events")
    # Independently check shared-scene leakage before fitting.
    joined = data.metadata.join(assignments)
    if joined[joined.split != "excluded"].groupby("scene_id").split.nunique().gt(1).any():
        raise ValueError("Shared satellite scene crosses splits")
    for view in (data.ground, data.satellite):
        if not view.index.equals(data.metadata.index) or view.empty or not np.isfinite(view.to_numpy()).all():
            raise ValueError("Feature views must be finite and aligned with the cohort")
    if not data.target.index.equals(data.metadata.index) or not np.isfinite(data.target.to_numpy()).all():
        raise ValueError("Target must be finite and aligned with the cohort")
    if f"ground::{data.target.name}" in data.ground.columns:
        raise ValueError("Target leakage: target cannot be a predictor")
    records, predictions, selected = [], [], {}
    y_train = data.target.loc[ids["train"]]
    y_validation = data.target.loc[ids["validation"]]
    y_test = data.target.loc[ids["test"]]
    def record(modality: str, model_name: str, split: str, actual: pd.Series, values: np.ndarray) -> None:
        records.append({"modality": modality, "model": model_name, "split": split,
                        "n_events": len(actual), **regression_metrics(actual, values)})
    dummy = DummyRegressor(strategy="mean").fit(np.zeros((len(y_train), 1)), y_train)
    selected["mean_reference"] = dummy
    for fold, actual in (("validation", y_validation), ("test", y_test)):
        values = dummy.predict(np.zeros((len(actual), 1)))
        record("reference", "training_mean", fold, actual, values)
        if fold == "test":
            for event, truth, prediction in zip(actual.index, actual, values):
                predictions.append([event, "reference", "training_mean", truth, prediction])
    for modality, view in (("ground", data.ground), ("satellite", data.satellite)):
        models = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha, solver="svd")),
            "random_forest": RandomForestRegressor(n_estimators=forest_trees,
                min_samples_leaf=forest_min_leaf, random_state=seed, n_jobs=1),
        }
        scores = []
        for name, model in models.items():
            model.fit(view.loc[ids["train"]], y_train)
            values = model.predict(view.loc[ids["validation"]])
            record(modality, name, "validation", y_validation, values)
            scores.append((mean_absolute_error(y_validation, values), name))
        _, chosen = min(scores)
        selected[modality] = models[chosen]
        values = selected[modality].predict(view.loc[ids["test"]])
        record(modality, chosen, "test", y_test, values)
        for event, truth, prediction in zip(y_test.index, y_test, values):
            predictions.append([event, modality, chosen, truth, prediction])
    return BaselineResults(selected, pd.DataFrame(records),
                           pd.DataFrame(predictions, columns=["observation_id", "modality", "model", "actual", "predicted"]),
                           assignments)

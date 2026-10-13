"""Small CPCB-only BOD forecaster with strictly earlier station history."""

from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from river_water_quality.training.baselines import regression_metrics

FEATURES = ["last_bod", "mean_last_3", "days_ahead", "month_sin", "month_cos"]
MAX_HORIZON = 90
MAX_HISTORY_AGE = 180


def iso_date(value):
    """Reject ambiguous dates and timestamps."""
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD")
    return pd.Timestamp(value)


def validate_history(history):
    """Validate user history without silently filling or merging measurements."""
    needed = {"station_id", "sample_date", "bod_mg_l"}
    if not needed.issubset(history.columns) or history.empty:
        raise ValueError("History needs station_id, sample_date, bod_mg_l and at least one row")
    frame = history.copy()
    if frame.station_id.map(lambda x: not isinstance(x, str) or not x.strip() or x != x.strip()).any():
        raise ValueError("station_id must be nonempty text without surrounding spaces")
    frame["sample_date"] = frame.sample_date.map(iso_date)
    if frame.bod_mg_l.map(lambda x: isinstance(x, bool)).any():
        raise ValueError("BOD must be numeric, not boolean")
    frame["bod_mg_l"] = pd.to_numeric(frame.bod_mg_l, errors="raise")
    if not np.isfinite(frame.bod_mg_l).all() or frame.bod_mg_l.lt(0).any():
        raise ValueError("BOD must be finite and nonnegative, in mg/L")
    if frame.duplicated(["station_id", "sample_date"]).any():
        raise ValueError("Duplicate station/date readings require source review")
    return frame.sort_values(["station_id", "sample_date"]).reset_index(drop=True)


def feature_row(prior, when):
    """One-step features shared by training and inference; current value is absent."""
    prior = prior[prior.sample_date < when].sort_values("sample_date").tail(3)
    if len(prior) < 3:
        return None, "fewer_than_3_prior_readings"
    days = (when - prior.sample_date.iloc[-1]).days
    if not 1 <= days <= MAX_HORIZON:
        return None, "latest_reading_outside_1_to_90_days"
    if (when - prior.sample_date.iloc[0]).days > MAX_HISTORY_AGE:
        return None, "three_readings_not_within_180_days"
    angle = 2 * np.pi * (when.month - 1) / 12
    return {"last_bod": float(prior.bod_mg_l.iloc[-1]),
            "mean_last_3": float(prior.bod_mg_l.mean()), "days_ahead": days,
            "month_sin": float(np.sin(angle)), "month_cos": float(np.cos(angle))}, None


def make_examples(history):
    """Audit every BOD row, using only observations strictly earlier than its date."""
    history = validate_history(history)
    rows, exclusions = [], []
    for station, group in history.groupby("station_id", sort=True):
        for index in range(len(group)):
            target = group.iloc[index]
            info = {"station_id": station, "sample_date": target.sample_date.date().isoformat(),
                    "observation_id": target.get("observation_id", f"{station}:{target.sample_date.date()}")}
            features, reason = feature_row(group.iloc[:index], target.sample_date)
            if reason:
                exclusions.append({**info, "reason": reason})
            else:
                rows.append({**info, **features, "actual_bod_mg_l": float(target.bod_mg_l),
                             "last_sample_date": group.iloc[index - 1].sample_date.date().isoformat()})
    if not rows:
        raise ValueError("No forecasting examples remain; each needs three recent earlier readings")
    return pd.DataFrame(rows), pd.DataFrame(exclusions, columns=["station_id", "sample_date", "observation_id", "reason"])


def model_predict(bundle, features):
    if bundle["selected_model"] == "last_reading":
        return features.last_bod.to_numpy(dtype=float)
    if bundle["selected_model"] == "training_mean":
        return np.full(len(features), bundle["training_mean"])
    if bundle["selected_model"] == "random_forest":
        return bundle["model"].predict(features[FEATURES])
    raise ValueError("Unknown MVP model")


def fit_mvp(history, train_end="2019-12-31", validation_end="2020-12-31"):
    """Fixed train/validation/test dates; selection uses validation MAE only.

    Evaluation is rolling one-step: an earlier observed test reading may be used
    as history for a later test forecast, but never for fitting or model selection.
    """
    if iso_date(train_end) >= iso_date(validation_end):
        raise ValueError("train_end must precede validation_end")
    examples, exclusions = make_examples(history)
    examples["split"] = np.where(examples.sample_date <= train_end, "train",
                                 np.where(examples.sample_date <= validation_end, "validation", "test"))
    trained_stations = sorted(examples.loc[examples.split.eq("train"), "station_id"].unique())
    unseen = ~examples.station_id.isin(trained_stations)
    extra = examples.loc[unseen, ["station_id", "sample_date", "observation_id"]].assign(reason="station_absent_from_training")
    exclusions = pd.concat([exclusions, extra], ignore_index=True)
    examples = examples.loc[~unseen].reset_index(drop=True)
    folds = {fold: examples[examples.split.eq(fold)] for fold in ("train", "validation", "test")}
    for fold, minimum in (("train", 30), ("validation", 10), ("test", 10)):
        if len(folds[fold]) < minimum:
            raise ValueError(f"{fold} needs at least {minimum} eligible examples; check dates/history")
    train, validation, test = (folds[name] for name in ("train", "validation", "test"))
    forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=1)
    forest.fit(train[FEATURES], train.actual_bod_mg_l)
    bundle = {"format_version": 1, "target": "bod", "unit": "mg/L", "features": FEATURES,
              "model": forest, "training_mean": float(train.actual_bod_mg_l.mean()),
              "station_ids": trained_stations, "train_end": train_end, "validation_end": validation_end,
              "max_horizon_days": MAX_HORIZON, "max_history_age_days": MAX_HISTORY_AGE}
    metrics = []
    for name in ("training_mean", "last_reading", "random_forest"):
        predicted = model_predict({**bundle, "selected_model": name}, validation)
        metrics.append({"split": "validation", "model": name, "n": len(validation),
                        **regression_metrics(validation.actual_bod_mg_l, predicted)})
    # Prefer the simpler predictor on a tie; test outcomes cannot influence choice.
    priority = {"last_reading": 0, "training_mean": 1, "random_forest": 2}
    selected = min(metrics, key=lambda row: (row["mae"], priority[row["model"]]))["model"]
    bundle["selected_model"] = selected
    predictions = test[["station_id", "sample_date", "observation_id", "last_sample_date", "days_ahead", "actual_bod_mg_l"]].copy()
    for name in dict.fromkeys((selected, "last_reading", "training_mean")):
        predicted = model_predict({**bundle, "selected_model": name}, test)
        metrics.append({"split": "test", "model": name, "n": len(test),
                        **regression_metrics(test.actual_bod_mg_l, predicted)})
        predictions[f"{name}_bod_mg_l"] = predicted
    predictions["predicted_bod_mg_l"] = model_predict(bundle, test)
    return bundle, pd.DataFrame(metrics), predictions, examples, exclusions


def predict_mvp(bundle, history, forecast_date):
    """Predict from supplied past measurements; refuse unsupported or stale inputs."""
    if bundle.get("format_version") != 1 or bundle.get("features") != FEATURES:
        raise ValueError("Unsupported model format or feature order; retrain the MVP")
    when = iso_date(forecast_date)
    if when <= iso_date(bundle["validation_end"]):
        raise ValueError("Forecast date must be after the model's validation period")
    history = validate_history(history)
    if (history.sample_date >= when).any():
        raise ValueError("All history dates must be strictly before the forecast date")
    rows = []
    for station, group in history.groupby("station_id", sort=True):
        if station not in bundle["station_ids"]:
            raise ValueError(f"Unknown station {station}; use a station represented in training")
        features, reason = feature_row(group, when)
        if reason:
            raise ValueError(f"Cannot forecast {station}: {reason}")
        value = model_predict(bundle, pd.DataFrame([features]))[0]
        rows.append({"station_id": station, "forecast_date": forecast_date,
                     "last_sample_date": group.sample_date.iloc[-1].date().isoformat(),
                     "last_bod_mg_l": features["last_bod"], "days_ahead": features["days_ahead"],
                     "predicted_bod_mg_l": float(value), "model": bundle["selected_model"]})
    return pd.DataFrame(rows)

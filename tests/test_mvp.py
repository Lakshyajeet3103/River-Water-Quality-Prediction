"""Synthetic forecasting tests; real accuracy is reported separately."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from river_water_quality.mvp import FEATURES, fit_mvp, make_examples, predict_mvp, validate_history


@pytest.fixture
def history():
    return pd.DataFrame([{"station_id": f"TEST_{station}", "sample_date": when.date().isoformat(),
                          "bod_mg_l": float(station + 2 + index % 4)}
                         for station in range(3)
                         for index, when in enumerate(pd.date_range("2018-01-01", "2021-12-01", freq="MS"))])


def test_current_and_future_targets_do_not_enter_features(history):
    before, _ = make_examples(history)
    changed = history.copy()
    changed.loc[(changed.station_id == "TEST_0") & (changed.sample_date >= "2020-06-01"), "bod_mg_l"] = 999
    after, _ = make_examples(changed)
    eligible = before.sample_date <= "2020-06-01"
    pd.testing.assert_frame_equal(before.loc[eligible, FEATURES], after.loc[eligible, FEATURES])
    assert (before.last_sample_date < before.sample_date).all()


def test_station_histories_do_not_mix(history):
    before, _ = make_examples(history)
    changed = history.copy()
    changed.loc[changed.station_id == "TEST_0", "bod_mg_l"] = 777
    after, _ = make_examples(changed)
    pd.testing.assert_frame_equal(before[before.station_id == "TEST_1"], after[after.station_id == "TEST_1"])


def test_test_labels_do_not_select_or_fit_model(history):
    bundle, metrics, _, examples, _ = fit_mvp(history)
    changed = history.copy()
    changed.loc[changed.sample_date >= "2021-01-01", "bod_mg_l"] = 999
    other, scores, _, _, _ = fit_mvp(changed)
    assert bundle["selected_model"] == other["selected_model"]
    pd.testing.assert_frame_equal(metrics.query("split == 'validation'"), scores.query("split == 'validation'"))
    np.testing.assert_array_equal(bundle["model"].predict(examples[FEATURES]), other["model"].predict(examples[FEATURES]))


def test_inference_equals_evaluation_and_serialization(history, tmp_path):
    import joblib
    bundle, _, predictions, _, _ = fit_mvp(history)
    joblib.dump(bundle, tmp_path / "model.joblib")
    loaded = joblib.load(tmp_path / "model.joblib")
    row = predictions.iloc[0]
    prior = history[(history.station_id == row.station_id) & (history.sample_date < row.sample_date)]
    result = predict_mvp(loaded, prior, row.sample_date)
    assert result.predicted_bod_mg_l.iloc[0] == pytest.approx(row.predicted_bod_mg_l)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True, "bad"])
def test_invalid_bod_is_rejected(history, value):
    history["bod_mg_l"] = history.bod_mg_l.astype(object)
    history.loc[0, "bod_mg_l"] = value
    with pytest.raises((ValueError, TypeError)):
        validate_history(history)


def test_duplicate_dates_rejected(history):
    with pytest.raises(ValueError, match="Duplicate"):
        make_examples(pd.concat([history, history.iloc[:1]]))


@pytest.mark.parametrize("change,match", [("future", "strictly before"), ("stale", "90_days"),
                                          ("unknown", "Unknown station"), ("few", "3_prior"), ("past", "validation")])
def test_unsupported_predictions_fail(history, change, match):
    bundle, *_ = fit_mvp(history)
    prior = history[history.station_id == "TEST_0"].tail(3).copy()
    when = "2022-01-01"
    if change == "future":
        prior.loc[prior.index[-1], "sample_date"] = when
    elif change == "stale":
        when = "2023-01-01"
    elif change == "unknown":
        prior["station_id"] = "UNKNOWN"
    elif change == "few":
        prior = prior.tail(2)
    elif change == "past":
        when = "2019-01-01"
    with pytest.raises(ValueError, match=match):
        predict_mvp(bundle, prior, when)


def test_cli_train_predict_and_refuse_overwrite(history, tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    from river_water_quality.data.cpcb import PARAMETERS
    raw = pd.DataFrame({"Station": history.station_id, "Latitude": 25.0, "Longitude": 83.0,
                        "Data Acquisition Time": pd.to_datetime(history.sample_date).dt.strftime("%d-%m-%Y 08:30"),
                        **{name: "" for name in PARAMETERS}})
    raw["Biochemical Oxygen Demand (mg/L)"] = history.bod_mg_l
    source = tmp_path / "synthetic.csv"
    raw.to_csv(source, index=False)
    out = tmp_path / "model"
    root = Path(__file__).parents[1]
    command = [sys.executable, "scripts/mvp.py", "train", "--input", str(source), "--output", str(out)]
    trained = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert trained.returncode == 0, trained.stderr
    report = json.loads((out / "report.json").read_text())
    predicted = tmp_path / "predictions.csv"
    command = [sys.executable, "scripts/mvp.py", "predict", "--model", str(out / "model.joblib"),
               "--history", str(out / "example_history.csv"), "--date", report["example_forecast_date"], "--output", str(predicted)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert pd.read_csv(predicted).predicted_bod_mg_l.notna().all()
    before = predicted.read_bytes()
    refused = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert refused.returncode != 0 and "already exists" in refused.stderr
    assert predicted.read_bytes() == before

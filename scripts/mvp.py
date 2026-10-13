"""Train the CPCB-only MVP or predict BOD from recent station history."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import sklearn

from river_water_quality.data.cpcb import prepare_cpcb
from river_water_quality.mvp import fit_mvp, predict_mvp


def read_history(path):
    return pd.read_csv(path, dtype={"station_id": str, "sample_date": str})


def train(args):
    out = Path(args.output)
    if out.exists():
        raise ValueError("Output already exists; choose a new directory")
    prepared = prepare_cpcb([Path(path) for path in args.input])
    bod = prepared.ground[prepared.ground.parameter.eq("bod")].copy()
    negative = bod.value.lt(0)
    rejected = bod.loc[negative, ["observation_id", "station_id", "sample_date"]].assign(reason="negative_bod")
    history = bod.loc[~negative, ["observation_id", "station_id", "station_name", "sample_date", "value"]].rename(columns={"value": "bod_mg_l"})
    bundle, metrics, predictions, examples, exclusions = fit_mvp(history, args.train_end, args.validation_end)
    exclusions = pd.concat([exclusions, rejected], ignore_index=True)
    limits = ["User-supplied CPCB-labelled files: original URL, license, laboratory method and source QC remain unverified.",
              "Date-only day-first source parsing; sampling-time meaning is unverified.",
              "Evaluates later readings at known stations, not new rivers/stations or satellite prediction.",
              "Rolling one-step forecasts use earlier observed readings as they become available; this is not a year-ahead forecast.",
              "Needs three prior BOD readings within 180 days and a latest reading 1-90 days before the forecast.",
              "Historical 2018-2021 data do not establish present-day accuracy or drinking-water safety."]
    report = {"task": "CPCB-only future station BOD estimate", "unit": "mg/L", "selected_model": bundle["selected_model"],
              "selection": "Lowest 2020 validation MAE by default; held-out test does not select the model",
              "train_end": args.train_end, "validation_end": args.validation_end,
              "split_counts": examples.split.value_counts().to_dict(),
              "source_bod_readings": len(bod), "eligible_examples": len(examples),
              "forecast_exclusions": exclusions.reason.value_counts().to_dict(),
              "raw_adapter_exclusions": prepared.exclusions.reason.value_counts().to_dict(),
              "training_stations": len(bundle["station_ids"]), "sources": prepared.sources,
              "features": bundle["features"], "seed": 42, "metrics": json.loads(metrics.to_json(orient="records")),
              "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "pandas": pd.__version__},
              "run_utc": datetime.now(timezone.utc).isoformat(), "limitations": limits}
    out.mkdir(parents=True, exist_ok=False)
    bundle["limitations"] = limits
    joblib.dump(bundle, out / "model.joblib")
    report["model_sha256"] = hashlib.sha256((out / "model.joblib").read_bytes()).hexdigest()
    for name, frame in (("history", history), ("metrics", metrics), ("test_predictions", predictions),
                        ("examples", examples), ("forecast_exclusions", exclusions), ("source_exclusions", prepared.exclusions)):
        frame.to_csv(out / f"{name}.csv", index=False)
    history[["station_id", "station_name"]].drop_duplicates().to_csv(out / "stations.csv", index=False)
    # A real held-out example with precisely the earlier readings used by inference.
    demo = predictions.sort_values(["sample_date", "station_id"]).iloc[0]
    demo_history = history[(history.station_id == demo.station_id) & (history.sample_date < demo.sample_date)].sort_values("sample_date").tail(3)
    demo_history.to_csv(out / "example_history.csv", index=False)
    report["example_forecast_date"] = demo.sample_date
    report["example_actual_bod_mg_l"] = float(demo.actual_bod_mg_l)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"\nSelected: {bundle['selected_model']}. Saved model and report to {out}")
    print(f"Example forecast date: {demo.sample_date}; history: {out / 'example_history.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Prepare raw CPCB CSVs, evaluate and save a model")
    training.add_argument("--input", nargs="+", required=True)
    training.add_argument("--output", default="outputs/mvp")
    training.add_argument("--train-end", default="2019-12-31")
    training.add_argument("--validation-end", default="2020-12-31")
    prediction = commands.add_parser("predict", help="Predict from three or more earlier BOD readings")
    prediction.add_argument("--model", default="outputs/mvp/model.joblib")
    prediction.add_argument("--history", required=True)
    prediction.add_argument("--date", required=True)
    prediction.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "train":
            train(args)
        else:
            out = Path(args.output)
            if out.exists():
                raise ValueError("Prediction output already exists; choose a new filename")
            bundle = joblib.load(args.model)
            predictions = predict_mvp(bundle, read_history(args.history), args.date)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("x", encoding="utf-8", newline="") as stream:
                predictions.to_csv(stream, index=False)
            print(predictions.to_string(index=False))
            print(f"Saved: {out}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"MVP failed: {exc}\n")


if __name__ == "__main__":
    main()

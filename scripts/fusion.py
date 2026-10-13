"""Train or run the small dual-branch BOD proof of concept."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from river_water_quality.training.fusion import SATELLITE, GROUND, predict_fusion, train_fusion, load_paired_data


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def train(args):
    source, out = Path(args.data), Path(args.output)
    if out.exists():
        raise ValueError("Output directory exists; choose a new run directory")
    data, hashes, provenance = load_paired_data(source)
    split = {"mode": "chronological", "train_end_date": args.train_end,
             "validation_end_date": args.validation_end, "gap_days": 7}
    checkpoint, metrics, predictions, assignments, history = train_fusion(data, split, epochs=args.epochs)
    limitations = ["Tiny proof of concept: only 14 available real pairs, not evidence of dependable prediction accuracy.",
                  "Ground branch requires same-event fecal coliform; this is retrospective BOD estimation, not forecasting or satellite-only prediction.",
                  "Both branches use tabular features; satellite inputs are ROI band means, not raw image patches.",
                  "Known-station chronological split does not establish cross-region generalization.",
                  "Original source/QC/time convention and possible bridge/bank mixed pixels retain the hybridization limitations."]
    checkpoint["limitations"] = limitations
    checkpoint["input_hashes"] = hashes
    out.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, out / "model.pt")
    metrics.to_csv(out / "metrics.csv", index=False)
    predictions.to_csv(out / "test_predictions.csv")
    data.metadata.join(assignments).to_csv(out / "split_assignments.csv")
    history.to_csv(out / "training_history.csv", index=False)
    data.exclusions.to_csv(out / "cohort_exclusions.csv", index=False)
    test_ids = assignments.index[assignments.split.eq("test")]
    example = data.satellite.loc[test_ids].rename(columns=lambda c: c.replace("::", "_"))
    example = example.join(data.ground.loc[test_ids].rename(columns=lambda c: c.replace("::", "_")))
    example.to_csv(out / "example_inputs.csv")
    report = {"scope": "Dual-branch fusion proof of concept", "architecture": checkpoint["architecture"],
              "parameter_count": checkpoint["parameter_count"], "target": "bod", "unit": "mg/L",
              "input_units": {"satellite_bands": "reflectance", "fecal_coliform": "MPN/100mL"},
              "preprocessing": "train-only per-feature standardization; log1p fecal coliform before standardization",
              "split": split, "split_counts": assignments.split.value_counts().to_dict(),
              "selected_epoch": checkpoint["best_epoch"], "epochs_run": len(history),
              "training": {"optimizer": "Adam", "learning_rate": 0.01, "weight_decay": 0.001,
                           "loss": "MSE", "selection": "lowest validation MAE", "patience": 60,
                           "max_epochs": args.epochs, "seed": 42, "refit_after_selection": False},
              "metrics": json.loads(metrics.to_json(orient="records")), "input_hashes": hashes,
              "model_sha256": hashlib.sha256((out / "model.pt").read_bytes()).hexdigest(),
              "torch_version": str(torch.__version__), "source_report": provenance, "limitations": limitations}
    save_json(out / "report.json", report)
    print(metrics.to_string(index=False))
    print(f"Saved fusion model to {out / 'model.pt'}; selected epoch {checkpoint['best_epoch']}")


def predict(args):
    out = Path(args.output)
    if out.exists():
        raise ValueError("Output file exists; choose a new filename")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    frame = pd.read_csv(args.input, dtype={"observation_id": str})
    active = checkpoint.get("selected_bands", SATELLITE) + (GROUND if checkpoint.get("use_ground", True) else [])
    columns = [c.replace("::", "_") for c in active]
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"Input requires columns: {', '.join(columns)}")
    spectral = pd.DataFrame(0.0, index=frame.index, columns=SATELLITE)
    laboratory = pd.DataFrame(0.0, index=frame.index, columns=GROUND)
    for name in active:
        (spectral if name in SATELLITE else laboratory)[name] = frame[name.replace("::", "_")]
    values = predict_fusion(checkpoint, spectral, laboratory)
    result = frame[[c for c in ("observation_id", "station_id", "sample_date") if c in frame]].copy()
    result["predicted_bod_mg_l"] = values
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)
    print(result.to_string(index=False))
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--data", default="outputs/hybrid_demo_001")
    p.add_argument("--output", default="outputs/fusion_demo_001")
    p.add_argument("--train-end", default="2021-03-31")
    p.add_argument("--validation-end", default="2021-09-30")
    p.add_argument("--epochs", type=int, default=500)
    p = commands.add_parser("predict")
    p.add_argument("--model", default="outputs/fusion_demo_001/model.pt")
    p.add_argument("--input", default="outputs/fusion_demo_001/example_inputs.csv")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        (train if args.command == "train" else predict)(args)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Fusion failed: {exc}\n")


if __name__ == "__main__":
    main()

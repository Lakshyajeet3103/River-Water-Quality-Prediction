"""Run the small, auditable spectral/ground input selection experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from river_water_quality.training.feature_selection import select_features
from river_water_quality.training.fusion import GROUND, load_paired_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="outputs/hybrid_demo_001")
    parser.add_argument("--output", default="outputs/feature_selection_001")
    parser.add_argument("--train-end", default="2021-03-31")
    parser.add_argument("--validation-end", default="2021-09-30")
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    try:
        out = Path(args.output)
        if out.exists():
            raise ValueError("Output directory exists; choose a new directory")
        data, hashes, provenance = load_paired_data(args.data)
        split = {"mode": "chronological", "train_end_date": args.train_end,
                 "validation_end_date": args.validation_end, "gap_days": 7}
        checkpoint, ranking, candidates, metrics, predictions, assignments = select_features(data, split, epochs=args.epochs)
        active = checkpoint["selected_bands"] + (GROUND if checkpoint["use_ground"] else [])
        limitations = ["Six training, five validation and three test events: rankings and comparisons are unstable.",
            "Only fecal coliform is available as a ground predictor; its inclusion can be tested, but different chemical parameters cannot be ranked.",
            "BOD is the answer, not a predictor. Univariate correlation measures association, not causality or independent usefulness.",
            "Input masking removes excluded standardized values from the computation, but does not prune neural-network parameters.",
            "The test events were already evaluated in the earlier fusion demonstration; this is a retrospective experiment, not a fresh untouched final test.",
            "Original source/QC/time and bridge/bank mixture limitations remain."]
        checkpoint["limitations"] = limitations
        checkpoint["input_hashes"] = hashes
        out.mkdir(parents=True, exist_ok=False)
        torch.save(checkpoint, out / "model.pt")
        for name, frame in (("feature_ranking", ranking), ("validation_candidates", candidates), ("metrics", metrics)):
            frame.to_csv(out / f"{name}.csv", index=False)
        predictions.to_csv(out / "test_predictions.csv")
        data.metadata.join(assignments).to_csv(out / "split_assignments.csv")
        example = data.satellite.join(data.ground).loc[assignments.split.eq("test"), active]
        example.rename(columns=lambda c: c.replace("::", "_")).to_csv(out / "example_inputs.csv")
        report = {"scope": "Small spectral/ground feature-selection demonstration", "method": checkpoint["selection_method"],
            "selected_candidate": checkpoint["selection_candidate"], "selected_inputs": active,
            "target": "bod", "target_unit": "mg/L", "split": split,
            "split_counts": assignments.split.value_counts().to_dict(), "selected_epoch": checkpoint["best_epoch"],
            "seed": checkpoint["seed"], "max_epochs": args.epochs,
            "ranking_data": "training only", "candidate_selection": "validation MAE only; six predeclared sets",
            "metrics": json.loads(metrics.to_json(orient="records")), "input_hashes": hashes,
            "model_sha256": hashlib.sha256((out / "model.pt").read_bytes()).hexdigest(),
            "torch_version": str(torch.__version__), "source_report": provenance, "limitations": limitations}
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(candidates[["candidate", "validation_mae", "selected"]].to_string(index=False))
        print("\nSelected inputs:", ", ".join(active))
        print(metrics.to_string(index=False))
        print(f"Saved selected-input model and reports to {out}")
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Feature selection failed: {exc}\n")


if __name__ == "__main__":
    main()

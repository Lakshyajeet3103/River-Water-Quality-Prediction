"""Evaluate a frozen India-trained model on real paired USGS observations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from river_water_quality.data.hybrid import paired_table
from river_water_quality.data.matching import MatchingResult
from river_water_quality.training.baselines import regression_metrics
from river_water_quality.training.explain import explain_fusion
from river_water_quality.training.fusion import load_paired_data, predict_fusion, raw_features, SATELLITE, GROUND
from river_water_quality.training.splits import assign_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--us-data", default="outputs/us_matching_001")
    parser.add_argument("--india-data", default="outputs/hybrid_demo_001")
    parser.add_argument("--model", default="outputs/fusion_demo_001/model.pt")
    parser.add_argument("--output", default="outputs/crossregion_demo_001")
    args = parser.parse_args()
    try:
        out = Path(args.output)
        if out.exists():
            raise ValueError("Output exists; choose a new directory")
        model_path = Path(args.model)
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        india, india_hashes, _ = load_paired_data(args.india_data)
        if checkpoint.get("input_hashes") != india_hashes:
            raise ValueError("India reference inputs do not match saved training checkpoint hashes")
        us, us_hashes, us_report = load_paired_data(args.us_data, timezone="America/New_York")
        if not us.metadata.station_id.str.startswith("USGS-").all():
            raise ValueError("Cross-region case study requires USGS-origin station identifiers")
        assignments = assign_splits(india.metadata, checkpoint["split_settings"])
        train = assignments.split.eq("train")
        baseline = float(india.target.loc[train].mean())
        predicted = predict_fusion(checkpoint, us.satellite, us.ground)
        explanations = explain_fusion(checkpoint, us.satellite, us.ground)
        rows = us.metadata.copy()
        rows["actual_bod_mg_l"] = us.target
        rows["predicted_bod_mg_l"] = predicted
        rows["india_training_mean_bod_mg_l"] = baseline
        rows["absolute_error_mg_l"] = np.abs(us.target.to_numpy() - predicted)
        rows = rows.join(explanations.drop(columns="predicted_bod_mg_l"))
        ispectral, iground = raw_features(india.satellite.loc[train], india.ground.loc[train])
        uspectral, uground = raw_features(us.satellite, us.ground)
        spectral_active = np.array([name in checkpoint.get("selected_bands", SATELLITE) for name in SATELLITE])
        rows["spectral_features_outside_india_training_range"] = (((uspectral < ispectral.min(axis=0)) |
            (uspectral > ispectral.max(axis=0))) & spectral_active).sum(axis=1)
        rows["fecal_outside_india_training_range"] = ((uground[:, 0] < iground[:, 0].min()) |
            (uground[:, 0] > iground[:, 0].max())) & checkpoint.get("use_ground", True)
        rows["bod_outside_india_training_range"] = (us.target < india.target.loc[train].min()) | (us.target > india.target.loc[train].max())
        metrics = [{"model": "frozen_india_fusion", "region": "US", "n": len(rows), **regression_metrics(us.target, predicted)},
                   {"model": "india_training_mean", "region": "US", "n": len(rows),
                    **regression_metrics(us.target, np.full(len(rows), baseline))}]
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != model_hash:
            raise ValueError("Frozen model file changed during evaluation")
        ground = pd.read_csv(Path(args.us_data) / "ground.csv")
        spectral = pd.read_csv(Path(args.us_data) / "satellite.csv")
        matches = pd.read_csv(Path(args.us_data) / "matches.csv")
        paired = paired_table(MatchingResult(matches, ground, spectral, pd.DataFrame(), pd.DataFrame(), {}))
        limitations = ["Tiny convenience subset; do not generalize to US rivers or claim robustness from these errors.",
            "US BOD is five-day/20C and fecal is presumptive MPN; CPCB laboratory methods remain unknown. Unit compatibility does not establish full method equivalence.",
            "BOD and fecal are distinct routine samples collected within five minutes; this is an explicit near-time pairing assumption.",
            "US station coordinates were transformed from NAD83; source location uncertainty, bridges, banks and SCL misclassification may remain.",
            "Savannah site is a tidal stream; rivers are not assumed clean. Sampling dates may predate the India training period; this is geographic transfer, not forecasting.",
            "Integrated gradients explain this model relative to its training-mean transformed inputs; they are not SHAP, causal effects or water-safety conclusions.",
            "No US data are used for fitting, scaling, checkpoint selection or adaptation. No US-labelled recalibration is performed."]
        report = {"scope": "Exploratory frozen India-to-US evaluation with integrated-gradients explanations",
            "us_events": len(rows), "us_stations": int(us.metadata.station_id.nunique()), "metrics": metrics,
            "india_input_hashes": india_hashes, "us_input_hashes": us_hashes, "frozen_model_sha256": model_hash,
            "model_unchanged_after_evaluation": True, "attribution": {"method": "integrated gradients", "steps": 512,
                "reference": "zero standardized inputs: India training mean reflectance and mean log1p fecal",
                "maximum_absolute_integration_residual_mg_l": float(rows.integration_residual_mg_l.abs().max())},
            "us_source_report": us_report, "limitations": limitations}
        out.mkdir(parents=True, exist_ok=False)
        rows.to_csv(out / "predictions_explanations.csv")
        pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
        paired.to_csv(out / "us_paired.csv", index=False)
        example = us.satellite.join(us.ground).rename(columns=lambda c: c.replace("::", "_"))
        example.to_csv(out / "example_inputs.csv")
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(rows[["station_id", "sample_date", "actual_bod_mg_l", "predicted_bod_mg_l", "absolute_error_mg_l",
                    "satellite_contribution_mg_l", "ground_contribution_mg_l"]].to_string())
        print(pd.DataFrame(metrics).to_string(index=False))
        print(f"Frozen cross-region comparison saved to {out}")
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Cross-region evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()

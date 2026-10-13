"""Training-only correlation ranking and validation-only small feature search."""

import numpy as np
import pandas as pd

from river_water_quality.training.baselines import regression_metrics
from river_water_quality.training.fusion import SATELLITE, GROUND, raw_features, train_fusion, predict_fusion
from river_water_quality.training.splits import assign_splits


def rank_features(data, assignments):
    spectral, laboratory = raw_features(data.satellite, data.ground)
    train = assignments.split.eq("train").to_numpy()
    values = np.concatenate((spectral[train], laboratory[train]), axis=1)
    target = data.target.to_numpy(dtype=float)[train]
    yc = target - target.mean()
    rows = []
    for index, name in enumerate(SATELLITE + GROUND):
        xc = values[:, index] - values[:, index].mean()
        denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
        correlation = float(np.dot(xc, yc) / denominator) if denominator > 0 else 0.0
        rows.append({"feature": name, "modality": "satellite" if name in SATELLITE else "ground",
                     "train_correlation": correlation, "score": abs(correlation), "n_train": int(train.sum())})
    return pd.DataFrame(rows).sort_values(["score", "feature"], ascending=[False, True]).reset_index(drop=True)


def select_features(data, split_settings, *, epochs=500):
    assignments = assign_splits(data.metadata, split_settings)
    ranking = rank_features(data, assignments)
    ordered = ranking.loc[ranking.modality.eq("satellite"), "feature"].tolist()
    candidates, checkpoints = [], {}
    for count in (3, 5, 10):
        for use_ground in (False, True):
            selected = [name for name in SATELLITE if name in ordered[:count]]
            key = f"top_{count}_bands_ground_{int(use_ground)}"
            checkpoint, metrics, _, _, history = train_fusion(data, split_settings, epochs=epochs,
                selected_bands=selected, use_ground=use_ground, evaluate_test=False)
            checkpoints[key] = checkpoint
            candidates.append({"candidate": key, "satellite_count": count, "use_ground": use_ground,
                "total_inputs": count + int(use_ground), "selected_bands": ";".join(selected),
                "validation_mae": float(metrics.mae.iloc[0]), "selected_epoch": checkpoint["best_epoch"],
                "epochs_run": len(history)})
    table = pd.DataFrame(candidates)
    choice = table.sort_values(["validation_mae", "total_inputs", "candidate"]).iloc[0].candidate
    checkpoint = checkpoints[choice]
    checkpoint["selection_method"] = "train-only absolute Pearson correlation; top 3/5/10 bands x ground off/on; minimum validation MAE"
    checkpoint["selection_candidate"] = choice
    table["selected"] = table.candidate.eq(choice)
    ranking["selected"] = ranking.feature.isin(checkpoint["selected_bands"] + (GROUND if checkpoint["use_ground"] else []))
    test = assignments.split.eq("test")
    predictions = data.metadata.loc[test].copy()
    predictions["actual_bod_mg_l"] = data.target.loc[test]
    metrics = []
    # Only the chosen model and the declared all-input reference reach test scoring.
    for name, model in (("selected_features", checkpoint), ("all_features_reference", checkpoints["top_10_bands_ground_1"])):
        for fold in ("validation", "test"):
            mask = assignments.split.eq(fold)
            values = predict_fusion(model, data.satellite.loc[mask], data.ground.loc[mask])
            metrics.append({"model": name, "split": fold, "n": int(mask.sum()),
                            **regression_metrics(data.target.loc[mask], values)})
            if fold == "test":
                predictions[f"{name}_bod_mg_l"] = values
    return checkpoint, ranking, table, pd.DataFrame(metrics), predictions, assignments

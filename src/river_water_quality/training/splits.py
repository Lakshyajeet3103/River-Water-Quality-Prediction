"""Explicit station or buffered chronological splits, never random row splits."""

from datetime import date, timedelta
import re

import pandas as pd


def acquisition_group(scene_id: str) -> str:
    """Conservatively group Sentinel-2 tile/processing aliases by sensing time.

    Supports bare catalog IDs and full asset paths. Retain exact scene identity
    for other providers; their acquisition grouping needs a provider-specific rule.
    """
    name = scene_id.rsplit("/", 1)[-1]
    sentinel = re.fullmatch(r"(\d{8}T\d{6})_\d{8}T\d{6}_T\d{2}[A-Z]{3}", name)
    return f"sentinel2:{sentinel.group(1)}" if sentinel else f"scene:{scene_id}"


def assign_splits(metadata: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Assign all cohort events, auditing omissions and blocking scene leakage.

    Chronological mode uses the entire sample/satellite date interval, so scenes
    crossing boundaries are excluded. Station mode requires explicit disjoint lists.
    Any scene ID or recognized Sentinel-2 sensing time spanning retained folds
    is rejected conservatively, including overlapping tile/processing aliases.
    """
    if not isinstance(settings, dict):
        raise ValueError("baseline.split must be a mapping")
    required = {"station_id", "sample_date", "scene_id", "satellite_local_date"}
    if not required.issubset(metadata.columns) or metadata.empty or not metadata.index.is_unique:
        raise ValueError("Split metadata requires unique event IDs and station/date/scene columns")
    mode = settings.get("mode")
    if mode == "station_holdout":
        groups = {}
        used = set()
        for fold in ("train", "validation", "test"):
            stations = settings.get(f"{fold}_stations")
            if not isinstance(stations, list) or not stations or any(not isinstance(s, str) or not s.strip() for s in stations) or len(set(stations)) != len(stations):
                raise ValueError(f"{fold}_stations must be a non-empty unique list of IDs")
            if set(stations) & used:
                raise ValueError("Station holdout lists must be disjoint")
            groups.update({station: fold for station in stations})
            used.update(stations)
        assignments = [(event, groups.get(row.station_id, "excluded"),
                        "" if row.station_id in groups else "unassigned_station")
                       for event, row in metadata.iterrows()]
    elif mode == "chronological":
        try:
            train_end = date.fromisoformat(settings["train_end_date"])
            validation_end = date.fromisoformat(settings["validation_end_date"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Chronological split requires valid train_end_date and validation_end_date") from exc
        gap = settings.get("gap_days")
        if type(gap) is not int or gap < 0 or validation_end <= train_end + timedelta(days=gap):
            raise ValueError("Chronological gap_days must be nonnegative with ordered cutoffs")
        assignments = []
        for event, row in metadata.iterrows():
            first, last = sorted((date.fromisoformat(row.sample_date), date.fromisoformat(row.satellite_local_date)))
            if last <= train_end:
                fold = "train"
            elif first > train_end + timedelta(days=gap) and last <= validation_end:
                fold = "validation"
            elif first > validation_end + timedelta(days=gap):
                fold = "test"
            else:
                fold = "excluded"
            assignments.append((event, fold, "temporal_boundary_or_gap" if fold == "excluded" else ""))
    else:
        raise ValueError("baseline.split.mode must be station_holdout or chronological")
    result = pd.DataFrame(assignments, columns=["observation_id", "split", "reason"]).set_index("observation_id")
    for fold in ("train", "validation", "test"):
        if (result.split == fold).sum() < 2:
            raise ValueError(f"{fold} requires at least two complete events; collect more data or revise split")
    retained = metadata.join(result).query("split != 'excluded'")
    if retained.groupby("scene_id").split.nunique().gt(1).any():
        raise ValueError("Shared satellite scene crosses splits; revise grouping or temporal boundaries")
    groups = retained.scene_id.map(acquisition_group)
    if retained.groupby(groups).split.nunique().gt(1).any():
        raise ValueError("Shared Sentinel-2 acquisition crosses splits through different tile/processing IDs; revise grouping or temporal boundaries")
    return result

"""Build explicit, unit-checked feature views from Phase 2 relational outputs."""

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from river_water_quality.data.schemas import validate_ground, validate_satellite
from river_water_quality.data.validation import validate_observations


@dataclass(frozen=True)
class BaselineDataset:
    """Shared complete-case cohort for fair modality comparisons, plus exclusions."""

    metadata: pd.DataFrame
    target: pd.Series
    ground: pd.DataFrame
    satellite: pd.DataFrame
    exclusions: pd.DataFrame


def build_baseline_dataset(
    matches: pd.DataFrame, ground: pd.DataFrame, satellite: pd.DataFrame,
    *, target_parameter: str, target_unit: str,
    ground_units: dict[str, str], satellite_units: dict[str, str], timezone: str,
) -> BaselineDataset:
    """Use explicit predictors, rejecting the target as a ground predictor.

    Missing targets/predictors are logged, not imputed. Both modalities use the
    same retained events. No WQI computation or automatic feature selection occurs.
    Extra source columns remain outside the numeric predictors.
    """
    if not isinstance(target_parameter, str) or not target_parameter.strip():
        raise ValueError("An explicit target_parameter is required")
    if not isinstance(target_unit, str) or not target_unit.strip():
        raise ValueError("An explicit target_unit is required")
    for name, mapping in (("ground_units", ground_units), ("satellite_units", satellite_units)):
        if not isinstance(mapping, dict) or not mapping or any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
            for k, v in mapping.items()
        ):
            raise ValueError(f"{name} must explicitly map features to units")
    if target_parameter in ground_units:
        raise ValueError("Target leakage: target parameter cannot be a ground predictor")
    zone = ZoneInfo(timezone)
    validate_ground(ground)
    match_columns = ["observation_id", "station_id", "sample_date", "scene_id", "acquired_at", "satellite_local_date"]
    validate_observations(matches, required_columns=match_columns,
                          column_types={c: "string" for c in match_columns},
                          observation_key=["observation_id"])
    if "observation_id" not in satellite.columns:
        raise ValueError("Matched satellite rows must include observation_id")
    expected_ids = set(matches.observation_id)
    if set(ground.observation_id) != expected_ids or set(satellite.observation_id) != expected_ids:
        raise ValueError("Event IDs must agree across matches, ground, and satellite tables")
    for feature, unit in {target_parameter: target_unit, **ground_units}.items():
        rows = ground[ground.parameter == feature]
        if not rows.empty and not rows.unit.eq(unit).all():
            raise ValueError(f"Ground unit mismatch for {feature}")
    for feature, unit in satellite_units.items():
        rows = satellite[satellite.band == feature]
        if not rows.empty and not rows.unit.eq(unit).all():
            raise ValueError(f"Satellite unit mismatch for {feature}")
    metadata, targets, ground_values, satellite_values, exclusions = [], [], [], [], []
    ground_groups = {key: rows for key, rows in ground.groupby("observation_id", sort=False)}
    satellite_groups = {key: rows for key, rows in satellite.groupby("observation_id", sort=False)}
    for match in matches.sort_values("observation_id").itertuples():
        chemical = ground_groups[match.observation_id]
        spectral = satellite_groups[match.observation_id]
        validate_satellite(spectral.drop(columns="observation_id"))
        if not chemical.station_id.eq(match.station_id).all() or not chemical.sample_date.eq(match.sample_date).all():
            raise ValueError("Ground metadata disagrees with match links")
        if not spectral.station_id.eq(match.station_id).all() or not spectral.scene_id.eq(match.scene_id).all() or not spectral.acquired_at.eq(match.acquired_at).all():
            raise ValueError("Satellite metadata disagrees with match links")
        local_date = datetime.fromisoformat(match.acquired_at.replace("Z", "+00:00")).astimezone(zone).date().isoformat()
        if local_date != match.satellite_local_date:
            raise ValueError("Satellite local date disagrees with timestamp/timezone")
        if date.fromisoformat(match.sample_date).isoformat() != match.sample_date:
            raise ValueError("Invalid sample_date")
        chemistry = chemical.set_index("parameter").value
        spectra = spectral.set_index("band").value
        if target_parameter not in chemistry.index:
            exclusions.append([match.observation_id, "missing_target"])
        elif not set(ground_units).issubset(chemistry.index):
            exclusions.append([match.observation_id, "missing_ground_predictor"])
        elif not set(satellite_units).issubset(spectra.index):
            exclusions.append([match.observation_id, "missing_satellite_predictor"])
        else:
            metadata.append([match.observation_id, match.station_id, match.sample_date, match.scene_id, local_date])
            targets.append(float(chemistry[target_parameter]))
            ground_values.append([float(chemistry[f]) for f in ground_units])
            satellite_values.append([float(spectra[f]) for f in satellite_units])
    if not metadata:
        raise ValueError("No complete baseline events remain; check target/predictors and coverage")
    frame = pd.DataFrame(metadata, columns=["observation_id", "station_id", "sample_date", "scene_id", "satellite_local_date"]).set_index("observation_id")
    return BaselineDataset(
        frame, pd.Series(targets, index=frame.index, name=target_parameter),
        pd.DataFrame(ground_values, index=frame.index, columns=[f"ground::{f}" for f in ground_units]),
        pd.DataFrame(satellite_values, index=frame.index, columns=[f"satellite::{f}" for f in satellite_units]),
        pd.DataFrame(exclusions, columns=["observation_id", "reason"]),
    )

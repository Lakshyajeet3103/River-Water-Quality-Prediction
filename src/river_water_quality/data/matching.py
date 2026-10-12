"""Auditable date-level matching of validated local observations and features."""

from dataclasses import dataclass
from datetime import date, datetime
from math import asin, cos, radians, sin, sqrt, isfinite
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from river_water_quality.data.schemas import validate_ground, validate_satellite


@dataclass(frozen=True)
class MatchingSettings:
    """Explicit experimental choices, not scientifically established defaults."""

    timezone: str
    max_distance_m: float
    max_cloud_fraction: float
    min_water_fraction: float
    accepted_ground_flags: tuple[str, ...]
    required_bands: tuple[str, ...]
    direction: str
    primary_window_days: int = 3
    max_window_days: int = 7

    def __post_init__(self) -> None:
        """Reject missing or invalid settings before matching any observations."""
        if not isinstance(self.timezone, str) or not self.timezone.strip():
            raise ValueError("matching.timezone must be an explicit timezone name")
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("matching.timezone must be a valid IANA timezone") from exc
        for field in ("max_distance_m", "max_cloud_fraction", "min_water_fraction"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"matching.{field} must be a finite number")
        if self.max_distance_m < 0:
            raise ValueError("matching.max_distance_m must be nonnegative")
        for field in ("max_cloud_fraction", "min_water_fraction"):
            if not 0 <= getattr(self, field) <= 1:
                raise ValueError(f"matching.{field} must be between 0 and 1")
        for field in ("primary_window_days", "max_window_days"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"matching.{field} must be a nonnegative integer")
        if self.max_window_days < self.primary_window_days:
            raise ValueError("matching.max_window_days must be >= primary_window_days")
        for field in ("accepted_ground_flags", "required_bands"):
            values = getattr(self, field)
            if not isinstance(values, (tuple, list)) or not values or any(
                not isinstance(v, str) or not v.strip() for v in values
            ) or len(set(values)) != len(values):
                raise ValueError(f"matching.{field} must be a non-empty unique list of strings")
            object.__setattr__(self, field, tuple(values))
        if self.direction not in ("retrospective", "prior_dates_only"):
            raise ValueError("matching.direction must be retrospective or prior_dates_only")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MatchingSettings":
        """Read the matching section, requiring all non-default choices explicitly."""
        settings = config.get("matching")
        if not isinstance(settings, dict):
            raise ValueError("matching must be a configuration mapping")
        try:
            return cls(**settings)
        except TypeError as exc:
            raise ValueError(f"Invalid matching settings: {exc}") from exc


@dataclass(frozen=True)
class MatchingResult:
    """Separate relational outputs avoid duplicating chemistry across bands."""

    matches: pd.DataFrame
    ground: pd.DataFrame
    satellite: pd.DataFrame
    exclusions: pd.DataFrame
    candidates: pd.DataFrame
    summary: dict[str, Any]


def distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Return spherical great-circle distance in metres (mean radius 6371008.8 m)."""
    delta_lat = radians(lat_b - lat_a)
    delta_lon = radians(lon_b - lon_a)
    a = sin(delta_lat / 2)**2 + cos(radians(lat_a)) * cos(radians(lat_b)) * sin(delta_lon / 2)**2
    return 2 * 6371008.8 * asin(sqrt(min(1.0, max(0.0, a))))


def match_observations(
    ground: pd.DataFrame, satellite: pd.DataFrame, settings: MatchingSettings,
) -> MatchingResult:
    """Select one usable scene per event, logging every rejected ground row.

    Match exact station IDs first, then distance, direction/window, quality, and
    required-band availability. Rank by absolute day offset, distance, UTC time,
    and scene ID. Future dates are allowed only in explicit retrospective mode.
    Date-only ground samples are not converted to invented observation times.
    Inputs are never changed. Invalid inputs fail rather than silently disappearing.
    """
    validate_ground(ground)
    validate_satellite(satellite)
    if "observation_id" in satellite.columns:
        raise ValueError("Satellite input observation_id conflicts with the output linking field")
    zone = ZoneInfo(settings.timezone)
    match_columns = ["observation_id", "station_id", "sample_date", "scene_id", "acquired_at",
                     "satellite_local_date", "temporal_offset_days", "distance_m", "search_tier"]
    candidate_columns = ["observation_id", "station_id", "scene_id", "temporal_offset_days", "distance_m", "status"]
    exclusion_columns = ["observation_id", "station_id", "parameter", "reason"]
    matches, exclusions, candidates = [], [], []
    matched_ground, matched_satellite = [], []
    scenes = {(station, scene): rows for (station, scene), rows in satellite.groupby(["station_id", "scene_id"], sort=True)}
    for observation_id, event in ground.groupby("observation_id", sort=True):
        accepted = event[event.quality_flag.isin(settings.accepted_ground_flags)]
        for row in event[~event.quality_flag.isin(settings.accepted_ground_flags)].itertuples():
            exclusions.append([observation_id, row.station_id, row.parameter, "ground_quality_flag"])
        if accepted.empty:
            continue
        sample = accepted.iloc[0]
        sample_date = date.fromisoformat(sample.sample_date)
        usable = []
        event_audit = []
        for (station_id, scene_id), rows in scenes.items():
            if station_id != sample.station_id:
                continue
            scene = rows.iloc[0]
            timestamp = datetime.fromisoformat(scene.acquired_at.replace("Z", "+00:00"))
            local_date = timestamp.astimezone(zone).date()
            offset = (local_date - sample_date).days
            distance = distance_m(sample.latitude, sample.longitude, scene.latitude, scene.longitude)
            if distance > settings.max_distance_m:
                status = "outside_spatial_tolerance"
            elif settings.direction == "prior_dates_only" and offset >= 0:
                status = "not_prior_date"
            elif abs(offset) > settings.max_window_days:
                status = "outside_temporal_window"
            elif scene.cloud_fraction > settings.max_cloud_fraction:
                status = "cloud_fraction"
            elif scene.water_fraction < settings.min_water_fraction:
                status = "water_fraction"
            elif not set(settings.required_bands).issubset(rows.band):
                status = "missing_required_bands"
            else:
                status = "eligible_not_selected"
                usable.append((abs(offset), distance, timestamp.timestamp(), scene_id, offset, local_date, rows))
            audit = [observation_id, station_id, scene_id, offset, distance, status]
            event_audit.append(audit)
        if not usable:
            reason = "no_station_candidate" if not event_audit else "no_usable_scene"
            for row in accepted.itertuples():
                exclusions.append([observation_id, row.station_id, row.parameter, reason])
        else:
            best = min(usable, key=lambda item: item[:4])
            absolute_offset, distance, _, scene_id, offset, local_date, rows = best
            tier = "primary" if absolute_offset <= settings.primary_window_days else "fallback"
            matches.append([observation_id, sample.station_id, sample.sample_date, scene_id,
                            rows.iloc[0].acquired_at, local_date.isoformat(), offset, distance, tier])
            matched_ground.append(accepted.copy())
            features = rows.copy()
            features.insert(0, "observation_id", observation_id)
            matched_satellite.append(features)
            for audit in event_audit:
                if audit[2] == scene_id:
                    audit[-1] = "selected"
        candidates.extend(event_audit)
    matches_frame = pd.DataFrame(matches, columns=match_columns)
    exclusions_frame = pd.DataFrame(exclusions, columns=exclusion_columns)
    ground_frame = pd.concat(matched_ground, ignore_index=True) if matched_ground else ground.iloc[:0].copy()
    satellite_frame = pd.concat(matched_satellite, ignore_index=True) if matched_satellite else pd.DataFrame(columns=["observation_id", *satellite.columns])
    summary = {
        "input_ground_rows": len(ground), "input_ground_events": ground.observation_id.nunique(),
        "matched_events": len(matches_frame), "matched_ground_rows": len(ground_frame),
        "excluded_ground_rows": len(exclusions_frame),
        "exclusion_rate_rows": len(exclusions_frame) / len(ground),
        "exclusion_reasons": exclusions_frame.reason.value_counts().to_dict(),
        "candidate_statuses": pd.Series([row[-1] for row in candidates], dtype=str).value_counts().to_dict(),
    }
    return MatchingResult(matches_frame, ground_frame, satellite_frame, exclusions_frame,
                          pd.DataFrame(candidates, columns=candidate_columns), summary)

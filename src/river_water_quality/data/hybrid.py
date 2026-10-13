"""Small deterministic demonstration subsets and lossless event-level pairing."""

import pandas as pd

from river_water_quality.data.schemas import validate_ground


def choose_events(ground, stations, year, months):
    validate_ground(ground)
    if not stations or len(stations) > 8 or len(set(stations)) != len(stations):
        raise ValueError("Supply 1-8 unique station IDs")
    if not months or len(set(months)) != len(months) or any(type(m) is not int or not 1 <= m <= 12 for m in months):
        raise ValueError("Supply unique calendar months 1-12")
    if len(stations) * len(months) > 48:
        raise ValueError("Demonstration is bounded to 48 planned events")
    bod = ground[ground.parameter.eq("bod")].copy()
    picks, log = [], []
    for station in stations:
        if station not in set(bod.station_id):
            raise ValueError(f"No BOD data for station {station}")
        for month in months:
            prefix = f"{year:04d}-{month:02d}"
            candidates = bod[bod.station_id.eq(station) & bod.sample_date.str.startswith(prefix)].copy()
            if candidates.empty:
                log.append({"station_id": station, "month": prefix, "status": "no_ground_observation"})
                continue
            candidates["date_distance"] = (pd.to_datetime(candidates.sample_date) - pd.Timestamp(f"{prefix}-15")).abs().dt.days
            chosen = candidates.sort_values(["date_distance", "sample_date", "observation_id"]).iloc[0]
            picks.append(chosen.observation_id)
            log.append({"station_id": station, "month": prefix, "status": "selected", "sample_date": chosen.sample_date,
                        "observation_id": chosen.observation_id})
    if not picks:
        raise ValueError("No ground events in requested months")
    return ground[ground.observation_id.isin(picks)].copy(), pd.DataFrame(log)


def paired_table(result):
    """One row per matched event; use pivot (never average duplicate measurements)."""
    if result.matches.empty:
        return result.matches.copy()
    chemical = result.ground.pivot(index="observation_id", columns="parameter", values="value").add_prefix("ground_")
    spectral = result.satellite.pivot(index="observation_id", columns="band", values="value").add_prefix("satellite_")
    locations = result.ground[["observation_id", "station_name", "latitude", "longitude"]].drop_duplicates()
    quality_columns = ["observation_id", "cloud_fraction", "water_fraction", "pixel_count"]
    if "coverage_fraction" in result.satellite:
        quality_columns.append("coverage_fraction")
    quality = result.satellite[quality_columns].drop_duplicates()
    table = result.matches.merge(locations, on="observation_id", validate="one_to_one")
    table = table.merge(quality, on="observation_id", validate="one_to_one")
    table = table.merge(chemical, on="observation_id", validate="one_to_one")
    table = table.merge(spectral, on="observation_id", validate="one_to_one")
    if len(table) != len(result.matches):
        raise ValueError("Pairing lost matched events")
    return table.sort_values(["station_id", "sample_date"]).reset_index(drop=True)

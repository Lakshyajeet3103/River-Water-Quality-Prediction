"""Bounded Sentinel-2 ROI feature extraction using explicit SCL masking."""

from datetime import date, datetime, timedelta, timezone
from math import isfinite
from zoneinfo import ZoneInfo

COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


def candidate_reason(stats: dict, means: dict, bands: list[str], settings: dict) -> str | None:
    """Evaluate pilot ROI quality without substituting unavailable measurements."""
    # Validate before arithmetic: missing data must not become a clear pixel,
    # and permissive coverage settings must never cause division by zero.
    keys = ("total", "covered", "water", "cloud", "usable")
    if any(isinstance(stats.get(key), bool) or not isinstance(stats.get(key), (int, float))
           or not isfinite(stats[key]) or stats[key] < 0 or stats[key] % 1 != 0 for key in keys):
        return "invalid_pixel_counts"
    total, covered = stats["total"], stats["covered"]
    if not (stats["usable"] <= stats["water"] <= covered <= total) or stats["cloud"] + stats["water"] > covered:
        return "invalid_pixel_counts"
    if total == 0 or covered == 0 or covered / total < settings["min_coverage_fraction"]:
        return "insufficient_scene_coverage"
    if stats["cloud"] / covered > settings["max_cloud_fraction"]:
        return "cloud_fraction"
    if stats["water"] / covered < settings["min_water_fraction"]:
        return "water_fraction"
    if stats["usable"] < settings["min_pixels"]:
        return "insufficient_valid_water_pixels"
    if any(isinstance(means.get(b), bool) or not isinstance(means.get(b), (int, float))
           or not isfinite(means[b]) for b in bands):
        return "missing_spectral_value"
    return None


def validate_extraction_settings(settings: dict) -> None:
    """Validate pilot settings before creating network requests."""
    if not isinstance(settings.get("project_id"), str) or not settings["project_id"].strip():
        raise ValueError("project_id is required")
    ZoneInfo(settings["timezone"])
    for name in ("radius_m", "scale_m"):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    if settings["scale_m"] != 20:
        raise ValueError("This pilot uses a common 20m B5 projection grid")
    for name in ("max_cloud_fraction", "min_water_fraction", "min_coverage_fraction"):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    for name in ("primary_window_days", "max_window_days", "min_pixels", "max_events", "max_candidates"):
        value = settings.get(name)
        if type(value) is not int or value < (0 if "window" in name else 1):
            raise ValueError(f"Invalid {name}")
    if settings["max_window_days"] < settings["primary_window_days"]:
        raise ValueError("max_window_days must be >= primary_window_days")
    bands = settings.get("bands")
    allowed = {"B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"}
    if not isinstance(bands, list) or not bands or len(set(bands)) != len(bands) or not set(bands).issubset(allowed):
        raise ValueError("bands must be a unique list of supported spectral bands")


def extract_event(ee, event: dict, settings: dict) -> tuple[list[dict], list[dict]]:
    """Query one event's bounded date/ROI window and select the closest usable scene."""
    zone = ZoneInfo(settings["timezone"])
    sample = date.fromisoformat(event["sample_date"])
    window = settings["max_window_days"]
    start = datetime.combine(sample - timedelta(days=window), datetime.min.time(), zone)
    end = datetime.combine(sample + timedelta(days=window + 1), datetime.min.time(), zone)
    roi = ee.Geometry.Point([float(event["longitude"]), float(event["latitude"])]).buffer(settings["radius_m"])
    collection = ee.ImageCollection(COLLECTION).filterBounds(roi).filterDate(start.isoformat(), end.isoformat())
    count = collection.size().getInfo()
    base = {"observation_id": event["observation_id"], "station_id": event["station_id"], "sample_date": event["sample_date"]}
    if count > settings["max_candidates"]:
        return [], [{**base, "status": "candidate_cap_exceeded", "candidate_count": count}]
    if not count:
        return [], [{**base, "status": "no_scene_in_window", "candidate_count": 0}]
    metadata = collection.toList(count).map(lambda image: ee.Image(image).toDictionary(["system:index", "system:time_start", "PRODUCT_ID"])).getInfo()
    def rank(item):
        stamp = datetime.fromtimestamp(item["system:time_start"] / 1000, timezone.utc)
        return (abs((stamp.astimezone(zone).date() - sample).days), item["system:time_start"], item["system:index"])
    audits = []
    for item in sorted(metadata, key=rank):
        scene_id = item["system:index"]
        image = ee.Image(f"{COLLECTION}/{scene_id}")
        stamp = datetime.fromtimestamp(item["system:time_start"] / 1000, timezone.utc)
        offset = (stamp.astimezone(zone).date() - sample).days
        projection = image.select("B5").projection()
        scl = image.select("SCL")
        water = scl.eq(6)
        cloud = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
        spectral = image.select(settings["bands"]).multiply(0.0001)
        valid = spectral.mask().reduce(ee.Reducer.min()).And(water)
        counts = ee.Image.constant(1).rename("total").addBands([
            scl.mask().unmask(0).rename("covered"), cloud.unmask(0).rename("cloud"),
            water.unmask(0).rename("water"), valid.unmask(0).rename("usable")])
        query = ee.Dictionary({
            "counts": counts.reduceRegion(ee.Reducer.sum().unweighted(), roi, scale=20, crs=projection, maxPixels=10000),
            "means": spectral.updateMask(valid).reduceRegion(ee.Reducer.mean().unweighted(), roi, scale=20, crs=projection, maxPixels=10000),
            "scl_histogram": scl.reduceRegion(ee.Reducer.frequencyHistogram().unweighted(), roi, scale=20, crs=projection, maxPixels=10000)})
        values = query.getInfo()
        stats, means = values["counts"], values["means"]
        reason = candidate_reason(stats, means, settings["bands"], settings)
        audits.append({**base, "scene_id": scene_id, "temporal_offset_days": offset,
            "status": reason or "selected", **{f"roi_{key}": value for key, value in stats.items()},
            **{f"scl_class_{code}_pixels": (values["scl_histogram"].get("SCL") or {}).get(str(code), 0)
               for code in range(1, 12)}})
        if reason:
            continue
        covered = stats["covered"]
        rows = [{"station_id": event["station_id"], "scene_id": scene_id,
            "acquired_at": stamp.isoformat(), "latitude": float(event["latitude"]), "longitude": float(event["longitude"]),
            "band": band, "value": means[band], "unit": "reflectance",
            "product_id": item.get("PRODUCT_ID") or f"{COLLECTION}/{scene_id}",
            "cloud_fraction": stats["cloud"] / covered, "water_fraction": stats["water"] / covered,
            "pixel_count": int(stats["usable"]), "coverage_fraction": covered / stats["total"],
            "source_observation_id": event["observation_id"], "sample_date": event["sample_date"],
            "temporal_offset_days": offset, "search_tier": "primary" if abs(offset) <= settings["primary_window_days"] else "fallback"}
            for band in settings["bands"]]
        return rows, audits
    audits.append({**base, "status": "no_usable_scene"})
    return [], audits

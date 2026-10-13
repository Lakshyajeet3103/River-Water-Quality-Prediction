"""Inspect one clear candidate per pilot event without changing sampling locations."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

from river_water_quality.data.sentinel import COLLECTION, validate_extraction_settings
from river_water_quality.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_csv")
    parser.add_argument("audit_csv")
    parser.add_argument("--config", default="configs/extraction_pilot.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--event-names", action="store_true", help="Include dates in image filenames for repeated stations")
    args = parser.parse_args()
    settings = load_yaml(args.config)
    validate_extraction_settings(settings)
    ground = pd.read_csv(args.ground_csv)
    audit = pd.read_csv(args.audit_csv)
    events = ground.drop_duplicates("observation_id")
    if len(events) > settings["max_events"]:
        raise ValueError("Diagnostic input exceeds max_events")
    out = Path(args.output)
    if out.exists():
        raise ValueError("Choose a new output directory")
    import ee
    ee.Initialize(project=settings["project_id"])
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for event in events.to_dict("records"):
        candidates = audit[audit.observation_id == event["observation_id"]].copy()
        candidates = candidates[candidates.scene_id.notna() & (candidates.roi_covered > 0)]
        candidates["cloud_fraction"] = candidates.roi_cloud / candidates.roi_covered
        candidates = candidates[candidates.cloud_fraction <= settings["max_cloud_fraction"]]
        candidates["distance_days"] = candidates.temporal_offset_days.abs()
        candidates["selected_priority"] = candidates.status.ne("selected").astype(int)
        candidates = candidates.sort_values(["selected_priority", "distance_days", "temporal_offset_days", "scene_id"])
        if candidates.empty:
            results.append({"observation_id": event["observation_id"], "status": "no_clear_audited_candidate"})
            continue
        scene = candidates.iloc[0].scene_id
        image = ee.Image(f"{COLLECTION}/{scene}")
        point = ee.Geometry.Point([event["longitude"], event["latitude"]])
        roi = point.buffer(settings["radius_m"])
        context = point.buffer(500)
        projection = image.select("B5").projection()
        scl = image.select("SCL")
        # Geometries are pixel centres, not verified sampling locations.
        water = scl.updateMask(scl.eq(6)).sample(region=context, projection=projection,
                                               geometries=True, dropNulls=True)
        water = water.map(lambda feature: feature.set("distance_m", feature.geometry().distance(point, 1)))
        nearest = water.sort("distance_m").limit(1)
        query = ee.Dictionary({
            "roi_scl_counts": scl.reduceRegion(ee.Reducer.frequencyHistogram().unweighted(), roi,
                                                crs=projection, maxPixels=10000),
            "context_water_pixels": water.size(),
            "nearest_scl_water_pixel": nearest.toList(1).map(lambda feature: ee.Dictionary({
                "distance_m": ee.Feature(feature).get("distance_m"),
                "coordinates": ee.Feature(feature).geometry().coordinates()})),
            "grid": projection,
        }).getInfo()
        result = {"observation_id": event["observation_id"], "station_id": event["station_id"],
                  "station_name": event.get("station_name"), "scene_id": scene,
                  "latitude": event["latitude"], "longitude": event["longitude"],
                  "sampling_radius_m": settings["radius_m"], "context_radius_m": 500, **query}
        results.append(result)
        prefix = f"{event['station_id']}_{event['sample_date']}" if args.event_names else event["station_id"]
        # Yellow outline = original sampling ROI; red centre = supplied coordinate.
        border = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(roi)]), 1, 2).selfMask()
        marker = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(point.buffer(5))]), 1, 2).selfMask()
        rgb = image.visualize(bands=["B4", "B3", "B2"], min=0, max=3000)
        classes = scl.visualize(min=0, max=11, palette=["000000", "ff0004", "868686", "774b0a",
                                   "10d22c", "ffff52", "0000ff", "818181", "c0c0c0", "f1f1f1", "bac5eb", "52fff9"])
        nir = image.visualize(bands=["B8", "B4", "B3"], min=0, max=3000)
        for label, visual in (("rgb", rgb), ("scl", classes), ("nir", nir)):
            visual = visual.blend(border.visualize(palette=["ffff00"])).blend(marker.visualize(palette=["ff0000"]))
            url = visual.getThumbURL({"region": context.bounds(), "dimensions": 600, "format": "png"})
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            (out / f"{prefix}_{label}.png").write_bytes(response.content)
        (out / "diagnostics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Diagnosed {prefix}: {scene}", flush=True)
    manifest = {"collection": COLLECTION, "settings": settings, "earthengine_version": ee.__version__,
                "inputs": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                           for path in (args.ground_csv, args.audit_csv, args.config)},
                "warning": "Nearest SCL water centre is diagnostic only; never an approved coordinate correction."}
    (out / "diagnostics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

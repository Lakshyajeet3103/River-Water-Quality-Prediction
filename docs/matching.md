# Phase 2: local spatial and temporal matching

This implementation matches Phase 1 ground observations to already extracted
Sentinel-2 features. It does not download imagery, process rasters, establish
scientific thresholds, or verify that coordinates lie on a river. No real data
has been supplied. Tests use synthetic software fixtures only.

## Explicit study configuration

Fill the `matching` section in `configs/base.yaml` after reviewing source metadata:

| Setting | Meaning |
|---|---|
| timezone | IANA timezone corresponding to ground dates |
| max_distance_m | Inclusive coordinate-distance tolerance in metres |
| max_cloud_fraction | Inclusive maximum ROI cloud fraction |
| min_water_fraction | Inclusive minimum ROI water fraction |
| accepted_ground_flags | Exact source flags accepted by the study |
| required_bands | Bands required in every selected scene |
| direction | retrospective or prior_dates_only |
| primary_window_days | Inclusive primary calendar-day window; initial scaffold is 3 |
| max_window_days | Inclusive maximum calendar-day window; initial scaffold is 7 |

The 3/7-day windows follow the existing proposal; they are configurable choices,
not validated scientific constants. No distance/cloud/water/flag/band thresholds
are supplied. Foundation configuration can load with unresolved matching settings;
`MatchingSettings.from_config(config)` and the matching CLI reject them. Partition
multi-timezone inputs into appropriate source groups. tzdata is explicitly declared
for Windows IANA timezone support.

## Selection and exclusion rules

1. Validate both inputs; invalid records cause errors rather than disappearing.
2. Exclude ground rows outside the quality-flag allowlist; accepted parameters
   within the same sampling event remain eligible.
3. Consider only exact station IDs, then enforce coordinate-distance tolerance.
4. Convert the satellite timestamp to the configured timezone; record signed offset
   as satellite calendar date minus ground sample date. Never invent a sampling hour.
5. Apply direction, maximum window, cloud/water thresholds, and band requirements.
6. Choose the smallest absolute day offset, breaking ties by distance, UTC timestamp,
   and scene ID. Every eligible primary-window scene precedes fallback scenes.
7. Log selected/unselected candidates, the first failed rule per rejected candidate,
   and every excluded ground row. No usable scene means an explicit exclusion.

`retrospective` allows future dates and cannot establish operational prediction.
`prior_dates_only` requires a strictly earlier calendar date; same-day imagery is
excluded because sampling times are unknown. Publication/availability times are
not checked. Offsets are days, not elapsed hours. Distance is a mean-radius spherical
great-circle approximation (6371008.8 m), not an ellipsoidal geodesic or footprint test.

## Local command and outputs

After setting study choices, run from the repository root:

```powershell
.\.venv-check\Scripts\python.exe scripts/match_datasets.py data/raw/cpcb/measurements.csv data/raw/cpcb/measurements.provenance.json data/raw/sentinel2/features.csv data/raw/sentinel2/features.provenance.json --config configs/base.yaml --output data/interim/matching_run_001
```

These filenames are examples, not included data. Use `--ground-kind usgs` for a
compatible USGS input; no pre-matched dataset is assumed. Output must be a new
directory; existing directories are never overwritten. Computation precedes writes,
but a filesystem write failure may leave partial outputs; use a new run path.

| File | Contents |
|---|---|
| matches.csv | One event/scene link, dates, signed offset, distance, search tier |
| ground.csv | Accepted matched chemical rows with original columns |
| satellite.csv | Selected spectral rows linked by observation_id |
| exclusions.csv | Every excluded ground row's ID, station, parameter, reason |
| candidates.csv | Each same-station event/scene candidate and its selection/rejection status |
| run.json | Configuration, source provenance, hashes, versions, diagnostic counts |

Separate tables avoid a chemical-parameter/band Cartesian product. All-excluded
runs still produce headers and an audit report; completely empty inputs remain
invalid under Phase 1 contracts. Inputs are unchanged. Python callers use
`match_observations(ground, satellite, settings)` with persistence kept separate.

## Leakage and limitations

Multiple events may reuse one scene. Group shared scenes appropriately when forming
station/region/time evaluation splits, and use temporal buffers at fold boundaries.
Retrospective future imagery must not leak into prospective prediction experiments.
Chemical target values must be kept distinct from input measurements. Mixed products,
units, methods, flags, coordinates, and extraction masks require source review.
Processing counts/exclusions are diagnostics, not accuracy metrics or research evidence.
No splits, targets, normalization, models, or cross-region evaluation are included.

Phase 2 software is implemented and tested; source compatibility and scientific
suitability remain unverified. Phase 3 baselines require real data, a justified
target, and a leakage-safe evaluation design after review.

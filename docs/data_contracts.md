# Phase 1 local data contracts

These are **project interchange schemas**, not verified CPCB/USGS export schemas.
No real dataset has been supplied or downloaded. Provider compatibility must be
checked against a specific source file before these adapters support research use.
All current test fixtures are clearly labelled synthetic software tests.

## Ground measurements: CPCB and USGS

UTF-8 CSV, one measured parameter per sampling event per row:

| Column | Requirement |
|---|---|
| observation_id | Source sampling-event identifier; distinct real replicates need distinct IDs |
| station_id | Source station identifier, read as text to preserve leading zeros |
| sample_date | Actual local sampling date in YYYY-MM-DD, with date convention documented |
| latitude, longitude | Finite WGS84 degrees in valid coordinate ranges |
| parameter | Documented parameter name/code; no assumed universal provider vocabulary |
| value | Finite numeric measurement; censoring such as `<1` requires a later explicit adapter |
| unit | Explicit unit string matching the manifest for that parameter |
| quality_flag | Explicit source flag or documented `not_reported`; no automatic acceptance |

The key is `(observation_id, parameter)`. An event must have consistent station,
date, and coordinates. Multiple parameters may share an event; distinct sampling
events on the same station/day are preserved. Retain source timestamps, methods,
sample types, detection limits, and other fields in extra columns. Do not invent
sampling IDs, values, flags, or observation times. If source IDs are absent, document
a deterministic identifier construction before preparing an interchange file.
Source date parsing and time-zone interpretation must be explicit upstream steps.
Date-only observations remain dates; they are never converted to invented midnight
timestamps. Values for different units or methods are not silently harmonized.

## Sentinel-2 extracted features

This phase accepts tabular features only, not imagery/raster files. One band per
station/scene row:

| Column | Requirement |
|---|---|
| scene_id, station_id | Source scene and station identifiers |
| acquired_at | Valid ISO timestamp with Z or explicit UTC offset |
| latitude, longitude | Finite WGS84 extraction coordinates |
| band | Explicit band name, matching the manifest unit mapping |
| value, unit | Finite spectral value and declared unit; no implicit DN/reflectance scaling |
| product_id | Product/version identifier; document processing level and scale/offset |
| cloud_fraction, water_fraction | Explicit ROI fractions in [0,1], with calculation methods documented |
| pixel_count | Positive integer contributing-pixel count |

The key is `(station_id, scene_id, band)`. Scene metadata must be consistent across
bands for each station/scene. Fractions/counts are defined here for a common ROI;
band-dependent masks/counts require a future schema extension rather than fabricated
common values. There are no automatic cloud thresholds, band assumptions, water
masks, quality exclusions, resampling, or satellite/ground matching in Phase 1.

## Mandatory provenance sidecar

Provide a JSON file alongside each CSV with the following fields. This template
contains placeholders, not real provenance; replace each placeholder after verifying
the actual dataset. `provider` must match `cpcb`, `usgs`, or `sentinel2`.

```json
{
  "dataset_id": "<specific dataset/export identifier>",
  "provider": "cpcb",
  "source": "<actual source URL/report and export reference>",
  "access_date": "<YYYY-MM-DD>",
  "license": "<verified license or explicitly documented unresolved access terms>",
  "coordinate_reference_system": "EPSG:4326",
  "timestamp_convention": "<source date convention/timezone and conversion rules>",
  "units": {"<actual parameter code>": "<actual unit>"},
  "column_mapping": {"<source column name>": "<internal column name>"}
}
```

Use an empty column_mapping object when the headers are already canonical. Record
additional metadata for measurement methods, quality-flag meanings, source missing
markers, processing steps, reflectance scale/offset, extraction footprints, and
coordinate provenance. Manifest validation checks structure and unit consistency;
it cannot establish truth, scientific validity, or legal access permissions.

## Local validation

From the repository root, after manually preparing verified interchange inputs:

```powershell
.\.venv-check\Scripts\python.exe scripts/validate_dataset.py cpcb data/raw/cpcb/measurements.csv data/raw/cpcb/measurements.provenance.json
.\.venv-check\Scripts\python.exe scripts/validate_dataset.py sentinel2 data/raw/sentinel2/features.csv data/raw/sentinel2/features.provenance.json
```

These filenames are examples; no such real files are included. The CLI reports
validated row counts and source/manifest SHA256 hashes, with no output file writes.
Python callers use `load_dataset(csv_path, manifest_path, kind=...)`, which returns
an IngestedDataset containing a table, provenance, and hashes. Known numeric columns
are converted strictly; extra columns, identifiers, flags, and date strings are
preserved. Original source files are never changed. Empty tables, duplicate headers,
bad CSV record widths, missing required fields, unsupported CRS, unit mismatches,
nonfinite values, and duplicate observations produce errors. Nothing is filled or
silently removed. Keep raw inputs and derived data in ignored directories.

## Remaining work and Phase 2

Confirm actual CPCB/USGS source files, parameter definitions, methods, units, licenses,
coordinates, dates, and flags. Add source-specific parsing and censoring rules only
after reviewing these files. USGS availability does not establish a pre-matched
satellite dataset. Phase 2 now implements local spatial/temporal matching, quality
exclusion logs, and temporal offsets; see [matching documentation](matching.md).
Region/station/time splits and preprocessing
must prevent leakage before any model evaluation. WQI and class definitions remain
unresolved, and this phase establishes no experimental metrics or generalization.

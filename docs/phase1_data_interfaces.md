# Phase 1 Data Interfaces

Phase 1 defines stable software-facing schemas for the three planned data sources. It does not download or commit real observations.

## Standardized schemas

### Sentinel-2
`observation_time`, `latitude`, `longitude`, `tile_id`, `blue`, `green`, `red`, `nir`, `swir1`, `swir2`

### CPCB
`observation_time`, `station_id`, `latitude`, `longitude`, `parameter`, `value`, `unit`

### USGS
`observation_time`, `site_id`, `latitude`, `longitude`, `parameter`, `value`, `unit`

All adapters are expected to expose a `load()` method returning a pandas DataFrame and a `validate()` method that checks the canonical schema.

## Synthetic fixtures

`src/river_water_quality/data/fixtures.py` provides tiny deterministic DataFrames for unit tests. These fixtures are intentionally synthetic and must not be used as experimental evidence.

## Scope boundary

This phase establishes contracts and test data only. Real Sentinel-2, CPCB, and USGS acquisition, cleaning, spatial matching, temporal matching, and modeling remain future work.

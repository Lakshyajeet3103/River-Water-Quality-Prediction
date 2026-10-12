# Dataset Documentation

This document will be completed as real datasets are integrated.

Phase 1 provides [internal CSV contracts and provenance requirements](data_contracts.md)
for local CPCB, USGS, and Sentinel-2 feature adapters. These are not verified provider
export schemas. No real source has been selected, supplied, or downloaded; the fields
below remain unresolved. USGS data must not be assumed to contain matched imagery.

## Sentinel-2

- Source:
- Access method:
- Relevant bands/metadata:
- Spatial resolution:
- Temporal information:
- Preprocessing:
- License/usage constraints:

## CPCB Ground Stations

- Source:
- Access method:
- Relevant fields:
- Spatial information:
- Temporal information:
- Water-quality parameters:
- Preprocessing:
- License/usage constraints:

## USGS

- Source:
- Access method:
- Selected river/site:
- Relevant fields:
- Spatial information:
- Temporal information:
- Preprocessing:
- License/usage constraints:

## Data Integrity Notes

- Real experimental data must never be replaced by synthetic fixtures.
- Every preprocessing step must be documented.
- Train/validation/test splits must be checked for spatial and temporal leakage.

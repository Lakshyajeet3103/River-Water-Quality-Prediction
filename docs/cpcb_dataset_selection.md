# CPCB Dataset Selection

## Purpose

This document records the CPCB ground-station dataset selected for the first real-data stage of the project. The selection is intentionally limited to a recent, manageable regional dataset so that the CPCB-to-Sentinel-2 alignment pipeline can be validated before expanding to additional Indian regions.

## Selected dataset

**Producer:** Central Pollution Control Board (CPCB)

**Portal:** National Water Data Portal (NWDP), National Water Informatics Centre (NWIC)

**Dataset family:** Surface Water Quality (Manual - Chemical Parameters)

**Initial region:** Uttar Pradesh

**Initial period:** 2021-2025

**Resource:** `Surface Water Quality Chemical Parameters CPCB Uttar Pradesh (2021 - 2025) Manual.csv`

**Source page:** https://www.nwdp.nwic.gov.in/en/dataset/surface-water-quality-manual-chemical-parameters-cpcb

## Why this dataset was selected

1. **Official source:** The dataset is published by CPCB through the Government of India's National Water Data Portal.
2. **Recent observations:** The 2021-2025 period is recent enough to provide useful temporal overlap with Sentinel-2 observations.
3. **Geospatial information:** The portal describes records as containing station identifiers and geographic details, which are required for spatial matching with satellite observations.
4. **Water-quality targets:** The chemical dataset contains measurements such as pH, dissolved oxygen, ammonia, chloride, sulphate, total dissolved solids, fluoride, and selected heavy metals.
5. **Manageable first region:** Starting with Uttar Pradesh allows the data pipeline to be validated on a defined regional subset before expanding to other states or river systems.

## Scope of Phase 1.2

Phase 1.2 establishes the **authoritative CPCB source and initial regional scope**. It does not commit the full raw dataset to GitHub.

The real raw CSV must remain under `data/raw/cpcb/`, which is ignored by Git. Only documentation, metadata, tests, and code belong in the repository.

## Next inspection tasks

Before implementing the ingestion loader, inspect the downloaded CSV and record:

- exact column names and spelling
- row count
- date format and date coverage
- station identifier format
- latitude/longitude fields and coordinate validity
- river/water-body and location fields
- units for every measurement
- missing-value conventions
- duplicate records
- parameter coverage across stations and dates

The canonical internal schema will be defined **after this inspection**, rather than assuming column names or units from the portal description.

## Related CPCB datasets

CPCB also publishes separate manual physical-parameter and biological-parameter datasets. These are not part of the initial Phase 1.2 ingestion scope. They can be evaluated later if they provide useful targets or features for the multimodal water-quality task.

## Research integrity

The source dataset is real observational data. No synthetic records will be used as experimental evidence. Any synthetic data used in tests must remain limited to software/unit testing.

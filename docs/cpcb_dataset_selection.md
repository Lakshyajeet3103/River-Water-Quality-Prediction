# CPCB Dataset Selection

## Purpose

This document records the CPCB ground-station dataset selected for the first real-data stage of the project. The selection is intentionally limited to a recent, manageable regional dataset so that the CPCB-to-Sentinel-2 alignment pipeline can be validated before expanding to additional Indian regions.

## Selected dataset

**Producer:** Central Pollution Control Board (CPCB)

**Portal:** National Water Data Portal (NWDP), National Water Informatics Centre (NWIC)

**Dataset family:** Surface Water Quality (Manual - Chemical Parameters)

**Initial region:** Uttar Pradesh

**Initial data scope:** 2018-2021

**Resources inspected:**
- `Surface Water Quality Chemical Parameters CPCB Uttar Pradesh (1961 - 2020) Manual.csv` — actual observations inspected: 2018-2020
- `Surface Water Quality Chemical Parameters CPCB Uttar Pradesh (2021 - 2025) Manual.csv` — actual observations inspected: 2021

**Source page:** https://www.nwdp.nwic.gov.in/en/dataset/surface-water-quality-manual-chemical-parameters-cpcb

## Why this scope was selected

1. **Official source:** The data is published by CPCB through the Government of India's National Water Data Portal.
2. **Observed temporal coverage:** The inspected resources provide actual Uttar Pradesh observations from 2018 through 2021, giving a larger temporal range than the 2021-only portion of the newer resource.
3. **Geospatial information:** The data contains station identifiers and geographic coordinates required for spatial matching with satellite observations.
4. **Water-quality targets:** The chemical dataset contains measurements such as pH, dissolved oxygen, ammonia, chloride, sulphate, total dissolved solids, fluoride, and selected heavy metals.
5. **Manageable first region:** Uttar Pradesh provides a defined regional dataset for validating the data pipeline before expanding to other states or river systems.

## Inspection and compatibility results

The two Uttar Pradesh resources were inspected and compared before combining them.

### 2018-2020 resource

- **Rows:** 3,602
- **Columns:** 52
- **Date coverage:** 2018-01-01 to 2020-12-30
- **Unique stations:** 171
- **Exact duplicate rows:** 0
- **Duplicate station + timestamp records:** 0

### 2021 resource

- **Rows:** 1,456
- **Columns:** 52
- **Date coverage:** 2021-01-01 to 2021-12-30
- **Unique stations:** 151
- **Unique coordinate pairs:** 151
- **Missing latitude/longitude:** 0
- **Exact duplicate rows:** 0
- **Duplicate station + timestamp records:** 0
- **pH observations:** 1,449
- **Dissolved oxygen observations:** 1,353

### Combined scope

The two resources have the same 52-column structure and are compatible for concatenation. The combined Uttar Pradesh dataset contains **5,058 rows** covering **2018-2021**.

The combined dataset is kept as a local raw-data artifact under `data/raw/cpcb/` and is not committed to GitHub. The raw-data directory is ignored by Git.

Missing chemical measurements are preserved and will be handled according to the target parameter during preprocessing. Values will not be fabricated.

## Scope of Phase 1.2

Phase 1.2 establishes the **authoritative CPCB source, initial regional scope, observed data characteristics, and compatibility of the selected resources**. It does not commit the full raw dataset to GitHub and does not implement the ingestion loader.

The real raw CSV remains under `data/raw/cpcb/`, which is ignored by Git. Only documentation, metadata, tests, and code belong in the repository.

## Canonical schema

The two inspected resources use the same 52-column structure. A canonical internal schema will be defined in the ingestion/validation phase after documenting the exact parameter names, units, date representation, station identifiers, coordinate fields, and missing-value conventions.

For the first modeling/alignment experiments, pH and dissolved oxygen are strong candidate targets because they have substantially higher observation coverage than many of the other chemical parameters.

## Related CPCB datasets

CPCB also publishes separate manual physical-parameter and biological-parameter datasets. These are not part of the initial Phase 1.2 ingestion scope. They can be evaluated later if they provide useful targets or features for the multimodal water-quality task.

## Research integrity

The source dataset is real observational data. No synthetic records will be used as experimental evidence. Any synthetic data used in tests must remain limited to software/unit testing.

# CPCB Dataset Selection

## Purpose

This document records the CPCB ground-station dataset selected for the first real-data stage of the project. The selection is intentionally limited to a recent, manageable regional dataset so that the CPCB-to-Sentinel-2 alignment pipeline can be validated before expanding to additional Indian regions.

## Selected dataset

**Producer:** Central Pollution Control Board (CPCB)

**Portal:** National Water Data Portal (NWDP), National Water Informatics Centre (NWIC)

**Dataset family:** Surface Water Quality (Manual - Chemical Parameters)

**Initial region:** Uttar Pradesh

**Portal resource period:** 2021-2025

**Actual observations inspected:** 2021 only

**Resource:** `Surface Water Quality Chemical Parameters CPCB Uttar Pradesh (2021 - 2025) Manual.csv`

**Source page:** https://www.nwdp.nwic.gov.in/en/dataset/surface-water-quality-manual-chemical-parameters-cpcb

## Why this dataset was selected

1. **Official source:** The dataset is published by CPCB through the Government of India's National Water Data Portal.
2. **Recent observations:** The selected resource is intended for the 2021-2025 period; the downloaded Uttar Pradesh resource currently contains observations from 2021.
3. **Geospatial information:** The data contains station identifiers and geographic coordinates required for spatial matching with satellite observations.
4. **Water-quality targets:** The chemical dataset contains measurements such as pH, dissolved oxygen, ammonia, chloride, sulphate, total dissolved solids, fluoride, and selected heavy metals.
5. **Manageable first region:** Starting with Uttar Pradesh allows the data pipeline to be validated on a defined regional subset before expanding to other states or river systems.

## Initial inspection results

The downloaded Uttar Pradesh resource was inspected before defining the canonical schema.

- **Rows:** 1,456
- **Columns:** 52
- **Unique stations:** 151
- **Unique coordinate pairs:** 151
- **Missing latitude/longitude:** 0
- **Date coverage:** 2021-01-01 to 2021-12-30
- **Invalid dates:** 0
- **Exact duplicate rows:** 0
- **Duplicate station + timestamp records:** 0
- **pH observations:** 1,449
- **Dissolved oxygen observations:** 1,353

The file contains many additional chemical-parameter columns with sparse or completely missing observations. Missing values will be preserved and handled according to the target parameter during preprocessing; values will not be fabricated.

The downloaded file is labelled by the portal as a 2021-2025 resource, but the actual observations inspected are from 2021 only. This distinction will be preserved in project documentation and experimental reporting.

## Scope of Phase 1.2

Phase 1.2 establishes the **authoritative CPCB source, initial regional scope, and observed data characteristics**. It does not commit the full raw dataset to GitHub.

The real raw CSV remains under `data/raw/cpcb/`, which is ignored by Git. Only documentation, metadata, tests, and code belong in the repository.

Additional Uttar Pradesh CPCB resources, including the available 2018-2020 data, will be evaluated separately for schema compatibility before any datasets are combined.

## Canonical schema decision

The canonical internal schema will be defined after comparing the available CPCB resources and confirming their actual column names, units, station identifiers, date formats, and missing-value conventions.

For the first modeling/alignment experiment, pH and dissolved oxygen are the strongest candidate targets because they have substantially higher observation coverage in the inspected 2021 resource.

## Related CPCB datasets

CPCB also publishes separate manual physical-parameter and biological-parameter datasets. These are not part of the initial Phase 1.2 ingestion scope. They can be evaluated later if they provide useful targets or features for the multimodal water-quality task.

## Research integrity

The source dataset is real observational data. No synthetic records will be used as experimental evidence. Any synthetic data used in tests must remain limited to software/unit testing.

# CPCB preparation and live Sentinel-2 pilot — 2026-10-13

The original user-provided CSVs remain unchanged. Prepared data, exclusion audits,
and real Earth Engine outputs are under ignored `data/interim` directories.
This is a feasibility pilot, not a trained model or scientific evaluation.

Follow-up: [location diagnosis and handoff](pilot_diagnosis.md) explains the two
failed ROIs, new validation/leakage guards, and the verified `sentinel_pilot_003`
rerun. The latest suite has 173 passing tests.

## Ground data preparation

`scripts/prepare_cpcb.py` adapts the two supplied Uttar Pradesh biological exports.
Column headers define parameter names/units; timestamps are explicitly parsed as
DD-MM-YYYY HH:MM. Source clock times remain as original strings; only displayed
calendar dates are used. Each event ID derives from the raw-file hash and CSV record;
station IDs derive from station name and coordinates. These are internal identifiers,
not invented monitoring-station identifiers attributed to CPCB.

The 5,016 source records contain 9,916 finite numeric parameter cells:

| Parameter | Numeric cells |
|---|---:|
| BOD | 4,538 |
| Fecal coliform | 4,114 |
| Total coliform | 1,264 |
| COD | 0 |

There are 4,836 events and 161 stations with at least one numeric measurement.
The raw files contain 174 station names; stations/events with no numeric parameters
are not represented as measured observations. Every omitted parameter cell appears
in `exclusions.csv`: 9,701 missing cells and 447 nonnumeric/nonfinite cells.
Labels such as Clear are retained in the audit, never converted to numeric values.

Outputs in `data/interim/cpcb_preparation_001` include ground.csv, stations.csv,
exclusions.csv, pilot_ground.csv, provenance sidecars, and preparation.json with
raw-source hashes and actual 2018–2021 date coverage. No measurement is filled or
clipped. `source_not_reported` flags mean absent source QC, not certified good quality.
Original source URL, license, download date, CRS, timezone, and the meaning of the
uniform 08:30 timestamps remain unverified. Preparation's access_date is the local
inspection date, not a fabricated original download date.

## Pilot choices

Three 2021 BOD events nearest March 15 were chosen deterministically from station
names referring to Ganga: Trighat/Ghazipur, Varanasi upstream/Assighat, and upstream
Anoopshahar. Names are provisional river hints, not verified river membership.
The selected dates are March 18, March 18, and March 8 respectively.

`configs/extraction_pilot.yaml` records the authorized Cloud project and choices:
60m radius, common 20m B5 grid, ten bands, a maximum seven-calendar-day search,
primary three-day tier, maximum cloud/shadow fraction 0.2, minimum water fraction
0.2, minimum valid scene coverage 0.9, and at least three valid water pixels.
These are operational pilot settings, not scientifically validated defaults.
Asia/Kolkata is provisional for source calendar dates; actual acquisition timestamps
come from Earth Engine in UTC. The script refuses more than three events and more
than 24 scenes per event rather than silently truncating the search.

Reflectance is scaled by 0.0001 according to the
[official collection catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED).
SCL class 6 supplies the water mask; classes 3/8/9/10 contribute to the cloud/shadow
fraction. Common band validity is combined with the water mask before reducing
spectral means. Native 10/20m bands use default nearest resampling on the common
grid. ROI fractions use covered SCL pixels; scene coverage is reported separately.
SCL water classification and these masks still require visual/scientific review.

Counts and means use unweighted reducers, with pixel centres inside the ROI,
following [Earth Engine's reducer weighting guidance](https://developers.google.com/earth-engine/guides/reducers_weighting).
Fractional counts are rejected rather than rounded into supposed pixel counts.
The initial sentinel_pilot_001/pilot_matching_001 runs used edge-weighted counts
and are **superseded**; do not use those outputs for further work. Corrected runs
are sentinel_pilot_002 and pilot_matching_002.

Corrected live result: one of three events passed. The Anoopshahar event on
2021-03-08 matched a 2021-03-07 acquisition (offset -1 day, primary tier), with
10 spectral bands and 22 common-valid water pixels out of 30 covered ROI pixels.
ROI cloud/shadow fraction was zero. The other two events had no usable scene under
the configured rules; audits include 14 water-fraction and one cloud-fraction
candidate rejection. Two ground parameter rows were matched and four were excluded.
These counts establish execution/feasibility only, not prediction accuracy.

Verification: all 152 software tests pass; raw input hashes remain unchanged and
Git ignores the source datasets and generated outputs.

## Reproduce into new directories

Run from the repository root; existing outputs cannot be overwritten:

```powershell
.\.venv-check\Scripts\python.exe scripts/prepare_cpcb.py --input data/raw/cpcb/swq_biological_parameter_manual_cpcb_up_2021_2025.csv data/raw/cpcb/swq_manual_biological_parameters_cpcb_up_1961_2020.csv --output data/interim/cpcb_preparation_002
.\.venv-check\Scripts\python.exe scripts/extract_sentinel_pilot.py data/interim/cpcb_preparation_002/pilot_ground.csv data/interim/cpcb_preparation_002/pilot_ground.provenance.json --output data/interim/sentinel_pilot_003
.\.venv-check\Scripts\python.exe scripts/match_datasets.py data/interim/cpcb_preparation_002/pilot_ground.csv data/interim/cpcb_preparation_002/pilot_ground.provenance.json data/interim/sentinel_pilot_003/features.csv data/interim/sentinel_pilot_003/features.provenance.json --config configs/pilot_matching.yaml --output data/interim/pilot_matching_003
```

The extractor reuses existing login and never starts an implicit authentication
flow. Each event's candidates are ranked by calendar-day distance, acquisition
timestamp, and scene ID; processing stops at the first scene passing every rule.
Rejected candidate diagnostics are saved in audit.csv. Duplicate station/scene/band
features are emitted once; source event links remain in the audit. Query errors are
recorded separately and never represented as no-image observations or fabricated
features. If all events fail quality checks, the feature table is empty with headers,
and Phase 1 validation/matching deliberately refuses it as an observation dataset.

## Before expanding

Review station coordinates against river imagery, inspect the two failed ROIs and
SCL masks, confirm timestamp/CRS assumptions, and supply the original download
source/usage terms. Do not relax thresholds simply to increase matches. A three-event
pilot is insufficient for baseline training, accuracy reporting, or generalization.

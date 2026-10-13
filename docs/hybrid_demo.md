# Functionality 1: satellite-ground hybridization demonstration

The small real-data demonstration is implemented and executed: original CPCB
measurements are paired with Sentinel-2 surface-reflectance features by monitoring
station and sampling date. Its output is a combined dataset, not a prediction model.

## Result

| Station | Sampling events attempted | Matched | Unmatched |
|---|---:|---:|---:|
| Ganga upstream, Anoopshahar | 6 | 4 | 2 |
| Ganga downstream, Varanasi (Malviya Bridge) | 6 | 6 | 0 |
| Ganga, Madhya Ganga barrage, Bijnor | 6 | 4 | 2 |
| **Total** | **18** | **14** | **4** |

The 14 paired rows contain measured BOD (mg/L), fecal coliform (MPN/100mL), and
ten Sentinel-2 bands in scaled reflectance, along with station/date/scene links
and quality metadata. There are 14 distinct scene IDs and 140 spectral values.
Thirteen pairs fall within three calendar days; one uses the seven-day fallback
(Varanasi, ground 2021-01-16, image 2021-01-20, +4 days).

Four July/September events at Anoopshahar and Bijnor had no candidate passing the
cloud/shadow limit. The full extraction records 32 rejected candidate images and
four final no-usable-scene outcomes, with no query errors. Candidate-image counts
are not counts of independent sampling events.

All 196 software tests passed. An offline rebuild from saved features produced
the identical paired table. Checks confirmed unique event rows, unchanged measured
values, all ten bands, configured quality/time limits, and audit coverage of every
attempted event. Original raw-file SHA256 hashes are unchanged.

## What to show

Open `outputs/hybrid_demo_001/paired.csv` for the combined data. Each row is one
event, not ten duplicated chemical measurements. Ground columns start with
`ground_`; spectral columns start with `satellite_`. Empty measurements are never
invented. `event_outcomes.csv` covers all 18 attempts, and `station_summary.csv`
shows the counts above. `run.json` records settings, hashes, units and limitations.

Example: Anoopshahar on 2021-03-08 has BOD 1.3 mg/L and fecal coliform 720 MPN/100mL,
paired to the 2021-03-07 image (one day earlier), using 22 common-valid water pixels.
The other ten feature values can be inspected directly in that row of paired.csv.

The three station RGB/SCL image pairs are in
`data/interim/hybrid_location_review_001`. Yellow outlines mark the supplied
60 m sampling circle and red marks the original coordinate. These images show
the river context and nearby crossings; they do not independently verify where
the physical samples were collected.

Suggested demonstration statement:

> We implemented satellite-ground data hybridization for three Ganga monitoring
> stations. We attempted 18 sampling dates and produced 14 paired records containing
> CPCB measurements and ten Sentinel-2 bands. Four events were excluded by the
> image-quality rules. Every pair retains its source date, scene and quality metadata.

## Selection and processing

Eight candidate stations were screened using March 2021 observations. The shortlist
used high 2021 record counts among Ganga-labelled stations, excluding the two
previously diagnosed unsuitable ROIs (Trighat and Assighat). Four locations passed;
the three with higher March water fractions were retained. The fourth passing
location, Rasoolabad, is retained in the screen audit. Screening files are
`hybrid_screen_input_001` and `hybrid_screen_extraction_001` under `data/interim`.

This is a convenience sample designed to demonstrate feasibility, not representative
coverage. No BOD or coliform magnitude was used to choose stations or dates.
For January, March, May, July, September and November 2021, the ground observation
nearest the 15th within that month was selected; earlier date/event ID breaks ties.
All measured parameters from each selected event are retained. Dates were fixed
before the multi-date image extraction, including dates that later failed.

The existing extractor queries `COPERNICUS/S2_SR_HARMONIZED` at original coordinates:

- A 60 m radius, common B5 20 m grid, ten bands scaled by 0.0001.
- SCL class 6 for water; classes 3/8/9/10 for shadow/cloud.
- At least 90% scene coverage, at most 20% cloud/shadow, at least 20% water,
  and at least three pixels valid in all selected bands.
- Unweighted pixel-centre counts and means; no relaxed thresholds for this run.
- Nearest eligible image within +/-3 days preferred; +/-7 days maximum.
- Calendar dates use the provisional Asia/Kolkata ground convention; actual image
  timestamps retain UTC. Future images are explicitly allowed for retrospective pairing.

See the [official collection description](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
for band scaling and SCL definitions. The output is a tabular representation of
satellite image pixels paired with chemistry; raw image patches are not training
inputs in this functionality.

## Reproduce from the correct repository

The working copy is:

```powershell
cd "C:\Users\Lakshyajeet\Desktop\clone of dbs project\River-Water-Quality-Prediction"
```

To rebuild the combined table from the already downloaded features, no network
or authentication is needed (choose an unused output directory):

```powershell
.\.venv-check\Scripts\python.exe scripts/hybrid_demo.py export --prepared data/interim/hybrid_demo_input_001 --extracted data/interim/hybrid_demo_extraction_001 --output outputs/hybrid_demo_rebuilt
```

To reproduce the full extraction into new directories:

```powershell
.\.venv-check\Scripts\python.exe scripts/hybrid_demo.py prepare --stations cpcb-0ecd9698ec1a2eb1 cpcb-1c6a3feabae131a4 cpcb-baa51b840df3bdc2 --output data/interim/hybrid_demo_input_002
.\.venv-check\Scripts\python.exe scripts/extract_sentinel_pilot.py data/interim/hybrid_demo_input_002/ground.csv data/interim/hybrid_demo_input_002/ground.provenance.json --config data/interim/hybrid_demo_input_002/extraction.yaml --output data/interim/hybrid_demo_extraction_002
.\.venv-check\Scripts\python.exe scripts/hybrid_demo.py export --prepared data/interim/hybrid_demo_input_002 --extracted data/interim/hybrid_demo_extraction_002 --output outputs/hybrid_demo_002
```

Preparation defaults to the existing `data/interim/cpcb_preparation_001` ground
data and provenance. A fresh checkout needs those local files regenerated using
`scripts/prepare_cpcb.py` and the original CSVs (see cpcb_pilot.md). Install
`requirements-earthengine.txt` and configure the documented Earth Engine project
for network extraction; the offline export only requires the core package.
All source and output datasets are ignored by Git. Existing outputs are protected
against overwriting. The exporter checks input hashes against the extraction run.

## Scope and limitations

This completes the small-station demonstration of **functionality 1**. It does not
complete fusion, feature selection, explainability, temporal attention, physics
constraints, or domain adaptation. The separate CPCB-only BOD MVP remains available.

Supplied coordinates, original source URL/license, sampling-time convention and
source QC remain unverified. The March location images include bridge/barrage
crossings near ROIs, so SCL-labelled water pixels can still contain structures or
bank mixtures at 20 m resolution. Other dates have automated mask checks, not an
independent manual annotation. The dataset is sufficient to demonstrate extraction,
filtering and joining; it is not proof of pure-water spectroscopy, model accuracy,
or cross-region generalization. The spatial centre distance is zero because the
image ROI is centred on the supplied ground coordinate; actual pixels span 60 m.

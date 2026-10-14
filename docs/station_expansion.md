# More stations and real satellite matches

Added **30 real Sentinel-2/CPCB matches at four additional stations**. The preserved expanded dataset contains **130 observations across seven stations**, compared with the original 100 across three. Existing datasets, models, WQI rules, assumptions and categories remain unchanged.

## New observations

| Additional station | Latitude | Longitude | Accepted matches |
|---|---:|---:|---:|
| Ganga at Garhmukteshwar | 28.760620 | 78.143766 | 10 |
| Ganga downstream Allahabad, Sangam | 25.419206 | 81.900522 | 7 |
| Ganga at Narora | 28.189700 | 78.395878 | 10 |
| Ganga upstream Vindhyachal, Mirzapur | 25.244031 | 82.418272 | 3 |

New observation dates span 2020-08-31 to 2021-12-21; measured BOD ranges from 1.1 to 2.8 mg/L. Every accepted event has measured BOD/fecal concentration, exact-matched CPCB pH/DO, and ten actual Sentinel-2 spectral means. No synthetic records or assumed BOD labels were added.

## Selection and quality handling

Used recorded CPCB coordinates and dates. Candidates had complete biological and chemical measurements, matched exactly by station name, date, latitude and longitude. Tried newest dates first, with no ranking or filtering by BOD magnitude. Stations were chosen as additional Ganga locations with available measurements; they are not a random representative sample.

Kept the prior extraction settings: harmonized Sentinel-2 surface reflectance, 60 m ROI, 20 m grid, SCL water-class pixels, cloud/shadow exclusion, cloud fraction <=0.20, water fraction >=0.20, coverage >=0.90 and at least three valid water pixels. Primary date window is +/-3 days with fallback to +/-7. Closest eligible image is selected deterministically. These are operational pilot thresholds, not validated river-wide scientific guarantees.

The completed bounded run tried 86 station/date events: 30 passed and 56 had no usable scene. Before this run, an initial Trighat/Ghazipur candidate had 28 failed attempts; its 60 m ROI consistently classified as land. That failed audit is retained separately. Coordinates were not silently shifted and thresholds were not loosened to turn failed candidates into accepted matches.

The target was 30 accepted matches, at most ten per candidate station. Sangam contributed seven before its per-station attempt cap; Mirzapur contributed the final three. Thus four stations were added, rather than forcing exactly three equal groups. The two unused backup candidates were not needed.

## Evidence and preserved data

`outputs/new_station_matches_v2_20261014/` contains:

- `paired.csv`: separate 130-event dataset; original 100 rows/values are preserved.
- `new_paired.csv`: only the 30 new events, including pH/DO.
- `chemical_audit.csv`: exact chemical records, timestamps, source URLs, row numbers and hashes.
- `new_ground.csv`: canonical biological source records for the accepted events.
- `source_ground_provenance.json`, `source_ground_preparation.json`: original preparation/source metadata.
- `satellite_features.csv`: extracted band values and source image metadata.
- `event_outcomes.csv`, `extraction_audit.csv`, `planned_events.csv`: attempts, image-quality checks, rejection reasons and preplanned candidates.
- `rejected_trighat_event_outcomes.csv`, `rejected_trighat_extraction_audit.csv`: initial all-land candidate audit.
- `run.json`: source/settings hashes, station counts and selection limitations.

All these artifacts are preserved in [the shared demo bundle](../demo_bundle/files/outputs/new_station_matches_v2_20261014/). Live extraction checkpoints remain local and ignored.

Verification: 130 unique event IDs, 30 genuinely new-station events, all original values preserved to numeric tolerance, finite required inputs, exact chemical matches and all accepted quality thresholds checked. Original saved-model hashes remain unchanged.

## Reproduce and use

Restore the bundle to inspect/train offline without Earth Engine:

```powershell
.\.venv-check\Scripts\python.exe scripts/restore_demo_bundle.py
```

Live re-extraction requires the original prepared biological source, downloaded chemical exports, authenticated Earth Engine project, and network access:

```powershell
.\.venv-check\Scripts\python.exe scripts/expand_station_matches.py --output outputs/station_expansion_reproduction
```

The script checkpoints each attempt and rejects changed resume inputs/settings. Future catalog processing may alter accepted scenes; the preserved snapshot is the evidence for this run. No full raw exports, environment or credentials are included in the bundle.

The [unseen-station report](unseen_station_evaluation.md) evaluates both the original and expanded cohorts and the unchanged saved models on the new stations. The old chronological 80/20 results remain historical results for the original 100 observations, not for this expanded dataset.

More observations broaden the demonstration; they do not guarantee better prediction. SCL classification, near-bank/bridge pixels, coordinate/timezone assumptions and laboratory QC remain limitations. All locations are still on the Ganga in Uttar Pradesh; this is not different-river or foreign validation. The notebook has not yet been updated with these artifacts.

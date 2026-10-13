# Exploratory India-to-US comparison and prediction explanations

The saved India-trained fusion model has been applied to four real USGS sampling
pairs across three Georgia river stations. All four have real matched Sentinel-2
features. The model was not retrained, scaled, tuned or recalibrated using US data.
Each prediction has satellite/ground contributions calculated by integrated gradients.

The comparison runs successfully, but the frozen model transfers poorly on these
four cases. That is a result to report, not evidence of cross-region robustness.

## US source and compatibility

Public [Water Quality Portal web services](https://www.waterqualitydata.us/webservices_documentation/)
were queried with provider NWIS, river/stream sites, water samples and dates
2017-01-01 through 2023-12-31. The legacy service's documented USGS-update limitation
means this snapshot should not be treated as a complete current inventory. Local
raw exports in `data/raw/us_search_001` contain:

- 7,449 BOD result rows, with parameter codes, methods, units, qualifiers and status.
- 15,038 fecal-coliform rows, preserving MPN and colony-count units separately.
- Metadata for 156 BOD stations, including coordinate datum and location uncertainty.

Each file has a `.download.json` sidecar with the exact request URL, retrieval UTC,
byte count, SHA256 and content type. These are USGS-origin NWIS records, not records
from another agency relabelled as USGS. Originals are preserved and ignored by Git.

The adapter keeps only routine water samples marked Accepted, Actual, finite,
nonnegative and unqualified/uncensored. BOD must be USGS code 00310 in mg/l (the
reported five-day/20C test); fecal coliform must be MPN/100 ml. Numeric values are
unchanged; only spelling is normalized to mg/L and MPN/100mL. Colony counts are
not converted to MPN, E. coli is not substituted, and nondetects are not imputed.
Exclusions are saved with raw values, units and source activities.

BOD and fecal samples have separate activity IDs. The demonstrated pairs were
collected five minutes apart, at the same organization/station/calendar date and
source time zone (EST or EDT). This explicit near-time pairing rule allows at most
five minutes, uses a unique nearest sample, and rejects ambiguities/reuse. It is
not a claim that both results came from one bottle or one activity. Forty-five
compatible pairs survive the screening/pairing rules across the retrieved snapshot.

The bounded demonstration uses all qualifying dates from 2019 onward for the
three named stations below: four pairs total. Stations were chosen for river
context and compatible measurements, not a presumed clean-water label or favourable
model errors. No case was removed after examining its prediction.

| USGS station | River/location | Ground date | Image date | Date offset |
|---|---|---|---|---:|
| 02212940 | Ocmulgee near Macon | 2019-03-18 | 2019-03-23 | +5 days |
| 02198920 | Savannah at Port Wentworth | 2019-03-20 | 2019-03-20 | 0 |
| 02198920 | Savannah at Port Wentworth | 2020-04-21 | 2020-04-18 | -3 days |
| 02226010 | Altamaha near Gardi | 2021-01-21 | 2021-01-23 | +2 days |

Savannah is explicitly classified by the source as a tidal stream. The US cases
may predate the 2021 India training period: this tests geographic transfer of a
frozen estimator retrospectively, not future forecasting.

## Coordinates and image checks

Station metadata reports NAD83. Coordinates were transformed from EPSG:4269 to
EPSG:4326 with pyproj, disallowing ballpark operations. The selected operation is
NAD83 to WGS84 (1), with reported operation accuracy 4 m; original coordinates and
source horizontal uncertainty are retained. The operation happens to leave these
displayed latitude/longitude values unchanged. Initial registry metadata reported
an unspecified deferred operation; `stations_transform_review.csv` records the
actual operation, verified to produce identical prepared ground data.

Extraction uses the same 60 m ROI, common 20 m grid, ten bands, scaling, cloud,
water and coverage thresholds as the Indian demonstration. Local calendar matching
uses America/New_York and image times retain UTC. All four events passed; candidate
audits contain two cloud rejections and one water-fraction rejection before
selecting the nearest eligible scenes. Ocmulgee uses the seven-day fallback.

Diagnostics in `data/interim/us_location_review_002` include selected-scene RGB,
SCL and near-infrared PNGs with date-specific filenames. Yellow is the 60 m circle,
red is the supplied station coordinate. Near-infrared composites use B8/B4/B3:
vegetation appears red and the nearby water channels dark. The Ocmulgee and
Altamaha points are near banks; only 7 and 11 common-valid SCL water pixels were
used, respectively, not the land pixels filling the rest of their circles.
Savannah uses 27 pixels at each date and includes a bridge near the station.
These are visual checks, not independent verification of physical sample locations
or proof that every extracted pixel is pure water. Exact masks/scene IDs are retained.

## Frozen model results

Input scaling and weights come exclusively from the original Indian checkpoint
`outputs/fusion_demo_001/model.pt`. Its SHA256 is checked before and after evaluation.
The original Indian input hashes must agree with the checkpoint; no optimizer is
invoked. The mean comparison uses only the six original Indian training targets.

| Station/date | Actual BOD | Frozen fusion estimate | Absolute error |
|---|---:|---:|---:|
| Ocmulgee, 2019-03-18 | 9.0 | 0.5473 | 8.4527 |
| Savannah, 2019-03-20 | 2.0 | 0.7348 | 1.2652 |
| Savannah, 2020-04-21 | 3.0 | 1.2043 | 1.7957 |
| Altamaha, 2021-01-21 | 3.0 | 0.6086 | 2.3914 |

All values are mg/L. Frozen fusion MAE is **3.4762 mg/L**, RMSE **4.5275 mg/L**;
the India training-mean reference MAE is **2.2250 mg/L**. Negative R2 values are
retained in metrics.csv. The model is worse than this simple reference here.

Three fecal inputs are below the Indian training range. Seven to ten spectral
features per US event fall outside their Indian training ranges, and Ocmulgee's
9 mg/L BOD exceeds the Indian training target range. These checks identify
extrapolation; they do not establish why every error occurred. Differences in
river conditions, dates, methods and pixel mixtures all remain possible factors.

US methods are documented, but the supplied CPCB laboratory methods remain
unverified. Identical parameter labels/units do not prove identical laboratory
definitions. This is therefore an exploratory method-compatible-by-label/unit
comparison, not a validated scientific inter-laboratory benchmark.

## Explainability

[Integrated gradients](https://arxiv.org/abs/1703.01365) computes input contributions
along a path from a reference to the observation. The reference here is zero in
standardized model coordinates: mean Indian training reflectance and mean
log1p fecal coliform. There are 512 midpoint integration steps.

Each output records contributions for every band and fecal coliform, plus their
satellite and ground totals, all in predicted BOD mg/L. Positive contributions
raise the estimate relative to the reference; negative contributions lower it.
Reference prediction + satellite contribution + ground contribution approximately
equals the prediction; the maximum residual here is below 0.001 mg/L.

Example: for Savannah on 2019-03-20, the model's reference is approximately
2.3926 mg/L. Satellite inputs contribute -0.0813 mg/L and the ground input
-1.5766 mg/L, producing approximately 0.7348 mg/L. This explains the model's
calculation, not the true cause of pollution. Signed group contributions can
cancel; the larger absolute group contribution is only a reference-dependent
summary. The method is not SHAP and does not complete the separate SHAP requirement.

## Teacher demonstration and reproducibility

Run the offline frozen-model evaluation/explanation into a fresh directory:

```powershell
cd "C:\Users\Lakshyajeet\Desktop\clone of dbs project\River-Water-Quality-Prediction"
.\.venv-check\Scripts\python.exe scripts/evaluate_crossregion.py --output outputs/crossregion_teacher_check
```

Defaults use the saved Indian fusion model, `outputs/hybrid_demo_001` as its
training reference, and `outputs/us_matching_001` as the US input. No network
queries or retraining are needed for this command. Output files:

- `predictions_explanations.csv`: all actual/predicted values, errors, range checks,
  per-feature attributions and satellite/ground contribution totals.
- `metrics.csv`: frozen fusion and Indian training-mean errors.
- `us_paired.csv`: measured chemistry, source dates, scene IDs and ten image bands.
- `report.json`: model/source hashes, method, limits and numerical completeness.
- `example_inputs.csv`: required model inputs without BOD answers.

To reproduce from downloaded WQP exports (all output paths must be fresh):

```powershell
.\.venv-check\Scripts\python.exe scripts/prepare_us_demo.py --output data/interim/us_demo_input_002
.\.venv-check\Scripts\python.exe scripts/extract_sentinel_pilot.py data/interim/us_demo_input_002/ground.csv data/interim/us_demo_input_002/ground.provenance.json --ground-kind usgs --config data/interim/us_demo_input_002/extraction.yaml --output data/interim/us_demo_extraction_002
.\.venv-check\Scripts\python.exe scripts/match_datasets.py data/interim/us_demo_input_002/ground.csv data/interim/us_demo_input_002/ground.provenance.json data/interim/us_demo_extraction_002/features.csv data/interim/us_demo_extraction_002/features.provenance.json --ground-kind usgs --config data/interim/us_demo_input_002/matching.yaml --output outputs/us_matching_002
.\.venv-check\Scripts\python.exe scripts/evaluate_crossregion.py --us-data outputs/us_matching_002 --output outputs/crossregion_demo_002
```

`scripts/download_wqp.py` reproduces the raw queries (bounded to 25 MiB per request).
Use `--provider NWIS`, BOD's default characteristic or `--characteristic "Fecal Coliform"`,
and `--service Station` for station metadata. Preserve original filenames expected
by the preparation CLI. Install `requirements-crossregion.txt` in an environment
with CPU PyTorch already installed; see fusion_demo.md for CPU setup.
All downloaded data and generated artifacts are ignored by Git.

## Scope

This demonstrates cross-regional execution and prediction explanations at small
proof-of-concept scale. It does not demonstrate robust US accuracy, a clean/dirty
classification, domain adaptation, or scientific method equivalence. All four
failures remain visible. A larger independent dataset and verified laboratory
definitions are required before scientific robustness claims.

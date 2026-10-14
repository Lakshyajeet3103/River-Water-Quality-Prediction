# Project Status

Selected satellite model (2026-10-14): user finalised the ratios-only alpha=1 checkpoint, now recorded in `configs/working_satellite_model.json` and used by `scripts/predict_satellite.py`. Same familiar-station test MAE 0.5911 mg/L, RMSE 0.7343, R2 0.4907. Original checkpoints and historical experiment/notebook/WQI results remain preserved. Selection followed exploratory repeated test inspection; this checkpoint has no established unseen-station error. See [current working model](docs/working_model.md).

Seen-station refinements (2026-10-14): original 80/20 cohort retained. Predeclared ratios-only trial MAE 0.5911 vs original 0.8225 mg/L. Raw-band alpha validation chose original alpha=1; joint validation-selected ratios/alpha=0.01 had MAE 0.8758. Three-day filtering leaves 57 training/19 test events: date-only retraining MAE 0.9001 vs original 0.7397 on the same 19 cases. All versions archived; none promoted based on repeated test inspection. See [full comparisons](docs/satellite_seen_refinements.md). Main models/WQI categories unchanged.

Familiar-station satellite selection (2026-10-14): unchanged original 80/20 split, training-only internal 60/20 validation selects visible B2/B3/B4. Test MAE 1.0961 vs original ten-band model 0.8225 mg/L; selection worsened this test and is not promoted. All original model/data hashes preserved. See [seen-station report](docs/satellite_band_selection_seen.md).

Satellite-only band selection (2026-10-14): fixed 17 subsets selected using inner station holdouts, with outer held-out stations excluded and shared sensing acquisitions purged at both levels. Same 130 outer cases: selected-subset MAE 1.3315 vs all-band 1.7282 mg/L (22.96% reduction); training-mean baseline 1.2435 remains better. Seven experimental checkpoints are preserved separately. No main-model or WQI changes. See [protocol/results](docs/satellite_band_selection.md).

Station expansion/transfer (2026-10-14): added 30 real matches at Garhmukteshwar (10), Sangam (7), Narora (10) and Mirzapur upstream (3). Separate expanded snapshot has 130 events/seven stations. Original three-station and expanded seven-station holdouts are complete with shared-acquisition purging. Expanded pooled MAE: ground-only 0.898, combined 1.590, satellite-only 1.728 mg/L; transfer remains limited. Frozen saved models also tested on the 30 new cases. Existing main models and WQI categories are unchanged. Reports: [data](docs/station_expansion.md), [evaluation](docs/unseen_station_evaluation.md). The 208-file bundle verifies; notebook integration is pending.

Ground-only comparison completed (2026-10-14): trained a separate fixed-setting MLP using fecal coliform, pH and DO on the saved 80 training cases. On identical 20 test cases, MAE is 0.2484 mg/L versus 0.3173 combined and 0.8225 satellite-only. Ground-only beats the all-input combined model overall; no model was replaced or tuned to change this result. New artifacts are included in the verified 151-file bundle. See [comparison report](docs/ground_only_comparison.md). Notebook integration remains pending.

WQI extension (2026-10-14): method selected, 100-event compatibility audit completed, and an explicitly assumption-based downstream calculator implemented. Water temperature is missing in the inspected CPCB exports. Evaluation on the saved 20 test cases is complete: scenario WQI MAE 1.094; category agreement 80/90/95% at assumed 20/25/30 C, versus 90% for a training-mean BOD baseline. These scores share measured pH/DO/fecal values and cannot demonstrate independent WQI prediction. Models remain unchanged; notebook integration is pending. See [WQI evaluation](docs/wqi_evaluation.md).

## Current demonstration status — 2026-10-14

An executed [classroom demonstration notebook](notebooks/river_water_quality_demo.ipynb)
now presents matching, predictions, fair comparisons, feature ranking, actual SHAP,
gate weights and experimental losses/results. All 10 code cells ran without errors;
seven charts are embedded. It reuses saved outputs without retraining or downloads.

The active scope is a small classroom proof of concept. Earlier status entries
below describe historical runs and are not the latest dataset/model inventory.

There are 100 real India pairs, enriched with exact CPCB pH and dissolved-oxygen
matches. Existing satellite-only and combined scikit-learn models are preserved.
New comparison models share identical 80/20 assignments: satellite-only test MAE
0.8225 mg/L, satellite + fecal 0.3268, all-chemical combined 0.3173, and a selected
B8/B8A/B11 + fecal model 0.1852. Feature choice uses only an inner 60/20 split
inside the outer training set, then refits on all 80; the outer test is not used
for selection. Actual Kernel SHAP explains three all-chemical-model test cases.

These are small, previously examined same-river examples, not evidence of broad
generalization. No comparison model automatically replaces the working model.
See [the latest experiment guide](docs/straightforward_demonstrations.md).
Temporal gating, a nonnegative-BOD training penalty, and station-adversarial
gradient reversal are now implemented in a separate 130-parameter PyTorch
experiment. Five variants use the same 80/20 data: baseline MAE 0.3211 mg/L,
gate-only 0.2799, constraint-only 0.3221, adversarial-only 0.3252 and all-on
0.2827. The constraint is not a differential-equation PINN, and station
adversarial regularization is not proven foreign-river adaptation. The station
classifier still distinguishes stations strongly. See
[the experimental-network guide](docs/experimental_network.md).
The original working models remain preserved; US evaluation is archived.

## Friend-testing snapshot (2026-10-13)

Branch `friend-testing-data-2026-10-13` includes the current feature-selection and
US-comparison code plus a reviewed 48-file, 238,885-byte demo bundle. Original
large raw exports and credentials are excluded. The deliberate shared snapshot
contains real small paired datasets, saved checkpoints, cached extraction data,
provenance and results; source/method limitations remain in the bundle README.

Verification: all 232 tests passed. Restoring into a fresh directory reproduced
the 14-row Indian hybridization, saved fusion predictions and four-case frozen
US comparison. See `demo_bundle/README.md` for setup and commands.

## Latest: exploratory US evaluation and explanations (2026-10-13)

Downloaded USGS-origin WQP BOD, fecal-coliform and station exports with request
metadata/hashes. Screened accepted unqualified actual results, requiring BOD code
00310 and fecal MPN units. Forty-five near-time pairs survived; a bounded case
study used four pairs from Ocmulgee, Savannah and Altamaha stations. Separate
samples are five minutes apart; original activity IDs and methods are retained.

All four matched real Sentinel-2 scenes. The frozen India checkpoint produced
US MAE 3.4762 mg/L, worse than the Indian training-mean reference (2.2250 mg/L).
No US fitting or selection occurred and no case was removed after its prediction.
Integrated-gradients explanations separate satellite/ground contributions; this
is not SHAP or domain adaptation. See `docs/crossregion_demo.md` and local
`outputs/crossregion_demo_001/predictions_explanations.csv`.

All 229 tests passed; pip check passes. Checkpoint hash is unchanged and saved
inference reproduces evaluation. Attribution completeness error is below 0.001
mg/L. Source-method comparability, near-time pairing and mixed-pixel limitations
are explicit; poor transfer is not described as robustness.

## Latest: available-input feature selection (2026-10-13)

Implemented and executed training-only absolute-correlation ranking and a
six-candidate validation search: top 3/5/10 satellite bands with fecal coliform
off/on. The winner retains all ten bands and fecal coliform (validation MAE
0.1750 mg/L), so the three-case test result matches the original fusion run.
No reduction or accuracy gain was forced or claimed. The selected-input model
and auditable tables are in `outputs/feature_selection_001`.

See `docs/feature_selection.md`. Spectral selection and inclusion of the sole
available ground predictor are demonstrated; choosing among several chemical
parameters still requires additional overlapping measurements. Existing test
events are reused retrospectively and are not a fresh final evaluation.

Verification: all 214 tests passed. Selected-checkpoint inference reproduces the
saved evaluation; excluded inputs cannot affect predictions, and altered test
answers cannot affect feature rankings or candidate selection. Nothing committed.

## Latest: report functionality 2, fusion proof of concept (2026-10-13)

Implemented and executed a small two-branch PyTorch model: ten spectral band means
through one encoder, log1p fecal coliform through another, concatenation and a
shared BOD prediction head. Target BOD is excluded from inputs. Existing real
paired data provide 6 train, 5 validation and 3 test events with chronological
boundaries and shared-scene guards. Preprocessing fits training rows only.

Saved model: `outputs/fusion_demo_001/model.pt`. Validation-selected checkpoint
epoch 454; test MAE 0.5864 mg/L, training-mean reference 0.8500 mg/L. Three test
events are insufficient for an accuracy/generalization claim. One Anoopshahar
prediction misses by 1.6119 mg/L; full per-event results are retained.
See `docs/fusion_demo.md`. Report functionalities 1 and 2 are demonstrated at
small proof-of-concept scope; the other six remain outstanding.

Verification: all 205 tests passed; installed dependencies pass pip check. Loaded
checkpoint predictions reproduce evaluation outputs using inputs without BOD.
Both encoders and the fusion head have learned weight updates. No commit or push.

## Latest: report functionality 1, small-station demonstration (2026-10-13)

The user requested the first report functionality on a few stations. Real
Sentinel-2 extraction/matching is complete for 18 planned dates across three
stations: Anoopshahar upstream 4/6, Varanasi downstream 6/6, Bijnor barrage 4/6.
The 14 paired records contain BOD, fecal coliform, ten bands, timestamps and quality
metadata. Four events were rejected by cloud checks; no query errors occurred.
See `docs/hybrid_demo.md` and local `outputs/hybrid_demo_001/paired.csv`.

Verification: all 196 tests passed. The offline rebuild produced the identical
paired table; all 14 events passed configured limits and retained original CPCB
values. Original raw CSV hashes are unchanged. Nothing was committed or pushed.

Functionality 1 is demonstrated at this bounded feasibility scope. The other seven
report functionalities are still outstanding. Source metadata and pixel-mixture
limitations are explicitly documented; no prediction or generalization claim is
made from these 14 pairs. The CPCB-only MVP below is retained separately.

## Active scope: CPCB-only MVP (2026-10-13)

The user reduced scope to a bare-bones MVP and explicitly accepted CPCB-only
prediction. Implemented `scripts/mvp.py train` and `predict`: BOD forecasting from
strictly earlier station history, validation-selected simple model, saved artifact,
CSV output, and audited chronological evaluation on actual supplied data.

Local run `outputs/mvp`: 1,672 training, 921 validation, 1,036 test examples;
101 training stations. Last-reading persistence beat Random Forest on validation
MAE and is the saved predictor. Test MAE 1.1425 mg/L, RMSE 3.4538 mg/L. See
docs/mvp.md for scope, commands, source limitations, and the historical example.
Advanced phases below are deferred; they are not required for this MVP.

The full suite passed 188 tests; after adding the end-to-end CLI test, all 16
focused MVP tests passed. Saved-model inference reproduced the held-out example.
Software verification is separate from real-data accuracy.

## Current Phase

**Phase 3 — Satellite-only & Ground-station-only Baselines**

Status: Baseline software implemented and tested; review pending. Real data,
target choice, and research evaluation remain outstanding.

Next: **Verify real inputs and establish baseline results**, then Phase 4 after review.

## Overall Progress

- [x] Project specification and scope
- [x] Research-paper direction defined
- [x] Phase 0 — Repository scaffolding
- [x] Phase 1 — Data interfaces and software tests; approved for progression
- [x] Phase 2 — Matching software tested and approved for progression
- [ ] Phase 3 — Baseline software implemented; review and real-data experiments pending
- [ ] Phase 4 — Dual-branch multimodal fusion
- [ ] Phase 5 — Dynamic temporal attention gating
- [ ] Phase 6 — Spectral & chemical feature selection
- [ ] Phase 7 — Physics-informed constraints
- [ ] Phase 8 — Domain-adversarial learning & cross-regional evaluation
- [ ] Phase 9 — SHAP-based explainability
- [ ] Phase 10 — Ablations, final evaluation & paper preparation

## Current Objective

Update: Real user-supplied biological CSVs are now locally prepared with full
cell exclusions. A three-event Earth Engine feasibility pilot has been run.
The corrected pilot matched one event (10 bands, 22 common-valid water pixels);
two events had no usable scene. All 173 software tests pass.
Live imagery diagnosis found a land ROI at Ghazipur and a mostly-bank ROI at
Assighat. Pixel validation and cross-fold Sentinel-2 tile-alias guards are fixed.
See docs/pilot_diagnosis.md for evidence and the bounded next-step handoff.
See docs/cpcb_pilot.md; source/timezone/CRS verification and sufficient paired
coverage remain outstanding. No baseline research training has occurred.

Verify regression baseline preparation, station/time splits, train-only scaling,
validation-only selection, held-out evaluation, and traceable artifacts. Actual
provider inputs and target selection remain unresolved. All fits so far use labelled
software fixtures; no research metrics or trained research model are supplied.

## Last Completed

Phase 2 approved for progression. Phase 3 tooling passes all 137 software tests; see
docs/baselines.md for supported workflows and research prerequisites.

## Next Action

Review Phase 3 software, inspect the real CPCB file, select a justified target,
and obtain verified matched features before conducting research experiments.

## Development Rule

**Inspect → Plan → Implement → Test → Review → Approve → Next phase**

Do not begin the next phase until the current phase has been reviewed and approved.

## Research Guardrails

- Never fabricate CPCB, Sentinel-2, or USGS observations.
- Keep synthetic fixtures separate from real experimental data.
- Prevent spatial and temporal data leakage.
- Record seeds, configurations, preprocessing, and splits for experiments.
- Do not claim novelty or patentability without documented prior-work verification.
- Document the scientific justification/source for every physics-informed constraint.

## Reference Documents

- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)
- [Research decisions](docs/decisions.md)
- [Dataset tracking](docs/datasets.md)
- [Prior work](docs/prior_work.md)

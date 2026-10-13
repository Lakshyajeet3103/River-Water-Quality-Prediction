# Project Status

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

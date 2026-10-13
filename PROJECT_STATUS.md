# Project Status

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

## Exploratory experiment audit update — 2026-10-14

Local restored artifacts document exploratory India and US experiments.
Treat them as preliminary evidence, not final validation.

**India fusion pilot:** 14 paired events; chronological split of 6 train,
5 validation, and 3 test events. Test MAE 0.5864 mg/L, RMSE 0.9329 mg/L,
and R² 0.0135 (n=3). This test set is too small to support a reliable
generalization or superiority claim.

**Feature selection:** the selected top-10-band-plus-ground configuration
produced the same reported metrics as the all-feature reference. No
improvement was demonstrated.

**US transfer pilot:** evaluation on four US events produced MAE 3.4762
mg/L and RMSE 4.5275 mg/L; the India training-mean baseline had MAE
2.2250 mg/L and RMSE 3.5394 mg/L. This indicates poor transfer in this
tiny exploratory sample, not a broad conclusion about US rivers.

**Outstanding checks:** verify source provenance, units, timestamps/time
zones, station coordinates, measurement-method comparability, satellite
water-pixel quality, and possible shared-scene leakage. Reproduce results
from committed code and configuration, then run a larger independent
evaluation. Do not describe simulator/pilot outputs as field validation,
claim water safety, or mark later roadmap phases complete based solely
on these artifacts.

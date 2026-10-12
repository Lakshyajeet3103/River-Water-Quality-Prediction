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

# Project Status

## Current Phase

**Phase 1 — Data Interfaces & Synthetic Test Fixtures**

Status: 🟢 Complete

Next: **Phase 2 — Satellite–Ground Spatiotemporal Hybridization**

## Overall Progress

- [x] Project specification and scope
- [x] Research-paper direction defined
- [x] Phase 0 — Repository scaffolding
- [x] Phase 1 — Data interfaces & synthetic test fixtures
- [ ] Phase 2 — Satellite–ground spatiotemporal hybridization
- [ ] Phase 3 — Satellite-only & ground-station-only baselines
- [ ] Phase 4 — Dual-branch multimodal fusion
- [ ] Phase 5 — Dynamic temporal attention gating
- [ ] Phase 6 — Spectral & chemical feature selection
- [ ] Phase 7 — Physics-informed constraints
- [ ] Phase 8 — Domain-adversarial learning & cross-regional evaluation
- [ ] Phase 9 — SHAP-based explainability
- [ ] Phase 10 — Ablations, final evaluation & paper preparation

## Current Objective

Implement real-data-ready contracts without introducing real experimental observations. Phase 1 standardizes the columns expected from Sentinel-2, CPCB, and USGS sources and provides deterministic synthetic fixtures for software tests.

## Last Completed

Phase 1 data interfaces: canonical source schemas, an abstract data-source adapter contract, schema validation, synthetic test fixtures, and automated tests.

## Next Action

Begin Phase 2: spatial and temporal alignment of satellite observations with CPCB measurements, while preparing the USGS pathway for cross-regional evaluation.

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
- [Phase 1 interfaces](docs/phase1_data_interfaces.md)
- [Prior work](docs/prior_work.md)

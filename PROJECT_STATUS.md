# Project Status

## Current Phase

**Phase 0 — Repository Scaffolding**

Status: 🟢 Complete

Next: **Phase 1 — Data Interfaces & Synthetic Test Fixtures**

## Overall Progress

- [x] Project specification and scope
- [x] Research-paper direction defined
- [x] Phase 0 — Repository scaffolding
- [ ] Phase 1 — Data interfaces & synthetic test fixtures
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

Build standardized, testable interfaces for Sentinel-2, CPCB, and USGS data. No real experimental data or ML models should be introduced until the appropriate phase.

## Last Completed

Phase 0 repository infrastructure: project structure, Python configuration, reproducibility utilities, configuration system, documentation scaffolds, tests, CI, and data/artifact ignore rules.

## Next Action

Begin Phase 1: data interfaces and synthetic test fixtures.

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

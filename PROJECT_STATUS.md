# Project Status

## Current Phase

**Phase 0 — Repository Scaffolding**

Status: 🟢 Complete (pending PR merge)

PR: #1 — Phase 0: Repository scaffolding

Next: **Phase 1 — Data Interfaces & Synthetic Test Fixtures**

---

## Overall Progress

- [x] Project specification and scope
- [x] Research-paper direction defined
- [x] Phase 0 repository scaffolding
- [ ] Phase 1 — Data interfaces and test fixtures
- [ ] Phase 2 — Satellite–ground spatiotemporal hybridization
- [ ] Phase 3 — Satellite-only and ground-station-only baselines
- [ ] Phase 4 — Dual-branch multimodal fusion
- [ ] Phase 5 — Dynamic temporal attention gating
- [ ] Phase 6 — Spectral and chemical feature selection
- [ ] Phase 7 — Physics-informed constraints
- [ ] Phase 8 — Domain-adversarial learning + cross-regional evaluation
- [ ] Phase 9 — SHAP explainability
- [ ] Phase 10 — Ablation studies, final evaluation, and paper preparation

---

## Current Objective

Establish standardized, testable interfaces for Sentinel-2, CPCB, and USGS data without introducing real experimental data or ML models yet.

## Last Completed

Phase 0 repository infrastructure:

- Project structure
- Python configuration
- Reproducibility utilities
- Configuration system
- Documentation scaffolds
- Test harness
- GitHub Actions workflow
- Data/artifact ignore rules

## Next Action

Merge PR #1 after review, then begin Phase 1.

## Development Rule

Work one phase at a time:

**Inspect → Plan → Implement → Test → Review → Approve → Next phase**

Do not begin the next phase until the current phase has been reviewed and approved.

## Research Guardrails

- Do not fabricate real CPCB, Sentinel-2, or USGS observations.
- Keep synthetic fixtures separate from real experimental data.
- Prevent spatial and temporal data leakage.
- Record seeds, configurations, preprocessing, and splits for experiments.
- Do not claim novelty or patentability without documented prior-work verification.
- Every scientific constraint must have a documented justification/source.

## Important Decisions

See [`docs/decisions.md`](docs/decisions.md).

## Dataset Tracking

See [`docs/datasets.md`](docs/datasets.md).

## Prior Work / Novelty Tracking

See [`docs/prior_work.md`](docs/prior_work.md).

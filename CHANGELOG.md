# Changelog

Important project milestones and repository-level changes.

## 2026-09-01

### Phase 1 — Data Interfaces & Synthetic Test Fixtures

- Added canonical schemas for Sentinel-2, CPCB, and USGS source data.
- Added a shared abstract data-source adapter contract and schema validation.
- Added deterministic synthetic fixtures for software/unit testing only.
- Added automated tests covering all three schemas and the adapter contract.
- Added Phase 1 interface documentation.
- Updated project status and roadmap to mark Phase 1 complete and Phase 2 next.

## 2026-08-27

### Phase 0 — Repository Scaffolding

- Added reproducible Python project configuration.
- Added repository structure for data, source code, experiments, notebooks, tests, configs, and documentation.
- Added configuration and seed utilities.
- Added dataset and prior-work documentation scaffolds.
- Added initial tests and GitHub Actions workflow.
- Added data/artifact ignore rules.

### Project Tracking

- Added `PROJECT_STATUS.md` as the current-state dashboard.
- Added `ROADMAP.md` for phase-level progress.
- Added `docs/decisions.md` for research and technical decisions.

## Planned

- Phase 2: satellite–ground spatiotemporal hybridization.
- Phase 3+: modeling, multimodal fusion, explainability, domain adaptation, cross-regional evaluation, and research-paper preparation.

# Changelog

Important project milestones and repository-level changes.

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

## 2026-10-12 — Phase 0 MVP foundation improvements

- Preserved the existing package, YAML configuration, and generic loader.
- Added validated experiment settings and an explicitly unresolved target definition.
- Added non-mutating table validation with explicit duplicate handling.
- Extended seeding to NumPy; added configuration, validation, and seed tests.
- Added lightweight installation/environment checks and Windows setup instructions.
- Added ignored interim-data storage; no datasets, models, or research metrics added.

## Next planned work

### 2026-10-12 — Phase 1 data interfaces (review pending)

- Added internal CPCB/USGS ground-measurement and Sentinel-2 feature contracts.
- Added local CSV/provenance ingestion with explicit mappings and source hashes.
- Added strict unit, coordinate, timestamp, duplicate, and metadata checks.
- Added a local validation CLI and clearly labelled synthetic software tests.
- Documented contracts and unverified real-source compatibility; no new dependencies.
- Preserved Phase 0 changes; no real datasets, matching, models, or metrics added.

### Subsequent phases

### 2026-10-12 — Phase 2 local matching (review pending)

- Added station/spatial/calendar-date matching with explicit quality/band settings.
- Added signed offsets, primary/fallback tiers, deterministic selection, and
  retrospective/prior-dates-only modes.
- Added per-ground-row exclusions, candidate audits, and separate relational outputs.
- Added a non-overwriting CLI with configuration, provenance, hashes, and versions.
- Added synthetic matching/CLI tests; all 106 software tests pass.
- Declared existing tzdata explicitly for Windows timezone support.
- No real inputs, imagery extraction, models, or scientific metrics added.

### Remaining phases

### 2026-10-12 — Phase 3 regression baseline tooling (research pending)

- Added shared event cohorts, explicit target/predictor units, and missingness logs.
- Added station holdouts and buffered chronological splits with shared-scene guards.
- Added mean/Ridge/Random Forest models, train-only scaling, validation selection,
  and held-out metrics/predictions.
- Added non-overwriting training CLI and labelled, traceable model/report artifacts.
- Added optional scikit-learn/joblib dependencies and ML-enabled CI.
- Added labelled synthetic software tests; no real dataset, target, WQI formula,
  research metrics, or trained research model supplied.
- Verified all 137 tests, CLI help, dependency consistency, and environment checks.

### Research next steps

### 2026-10-13 — Real CPCB preparation and authorized extraction pilot

- Preserved original CSVs; prepared 9,916 numeric measurements with cell-level audits.
- Added internal IDs, source hashes, station registry, and explicit unresolved provenance.
- Added a bounded three-event Sentinel-2 extractor with SCL quality masks and scaling.
- Corrected default edge weighting to common unweighted count/mean reductions;
  initial pilot_001 outputs are retained but superseded by corrected pilot_002 runs.
- Added synthetic adapter/quality regression tests; no research training or metrics.

- Phase 1 review and actual provider-source compatibility verification.
- Phase 2: satellite–ground spatiotemporal hybridization.
- Phase 3+: modeling, multimodal fusion, explainability, domain adaptation, cross-regional evaluation, and research-paper preparation.

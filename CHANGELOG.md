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


### 2026-10-14 — Advanced college-demo features

- Added a runnable synthetic-data DANN experiment with a gradient-reversal layer, regression head, and domain classifier.
- Added learned temporal gating for satellite and ground branches with observation-age features.
- Added dynamic INT8 quantization and CPU latency/output-delta benchmarking for supported Linear layers.
- Added comparison CSVs, quantization benchmark, plot, JSON report, and saved demo state dictionaries.
- Added optional `advanced-demo` dependencies and a demo guide; all synthetic results are explicitly labelled as synthetic.
- Added smoke tests for synthetic data, DANN outputs/gradient reversal, and temporal gate outputs.
- Verified 179 tests passed. The recorded synthetic run did not show DANN or temporal gating outperforming their baselines; INT8 quantization increased state-dict size and latency for this small model. No real-data performance improvement is claimed.

## 2026-10-14 — Advanced feature demos

- Added synthetic-data DANN and temporal attention-gating demonstrations.
- Added a dynamic INT8 quantization benchmark and saved demo artifacts.
- Added tests and documentation for running and interpreting the demos.
- Documented that recorded synthetic results do not establish real-world
  or cross-region performance; the tiny INT8 model was larger and slower
  than its FP32 counterpart in the recorded benchmark.

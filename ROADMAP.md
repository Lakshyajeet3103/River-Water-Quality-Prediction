# Project Roadmap

This roadmap tracks the project from infrastructure setup through the research-paper-quality evaluation.

| Phase | Milestone | Status |
|---|---|---|
| 0 | Repository scaffolding | 🟢 Complete |
| 1 | Data interfaces & synthetic test fixtures | Software-tested; approved for progression |
| 2 | Satellite–ground spatiotemporal hybridization | Software-tested; approved for progression |
| 3 | Satellite-only & ground-station-only baselines | Software implemented; research runs/review pending |
| 4 | Dual-branch multimodal fusion | ⚪ Planned |
| 5 | Dynamic temporal attention gating | ⚪ Planned |
| 6 | Spectral & chemical feature selection | ⚪ Planned |
| 7 | Physics-informed constraints | ⚪ Planned |
| 8 | Domain-adversarial learning & cross-regional evaluation | ⚪ Planned |
| 9 | SHAP-based explainability | ⚪ Planned |
| 10 | Ablations, final evaluation & paper preparation | ⚪ Planned |

## Phase Details

### Phase 0 — Repository Scaffolding
Reproducible project infrastructure, configuration, documentation, testing, CI, and repository structure.

### Phase 1 — Data Interfaces & Synthetic Test Fixtures
Standardized interfaces for Sentinel-2, CPCB, and USGS data, plus synthetic fixtures strictly for software/unit testing.

Implemented internal long-format contracts, local CSV adapters, explicit source
mapping and provenance, and strict validation. Real provider export compatibility
is not verified. See `docs/data_contracts.md`; no data downloads or matching yet.

### Phase 2 — Satellite–Ground Hybridization
Spatial and temporal alignment of satellite observations with CPCB measurements and preparation of the USGS pathway for cross-regional evaluation.

Implemented local feature-table matching, configurable quality rules, calendar-day
windows/offsets, candidate/exclusion audits, and traceable outputs. No imagery
downloads/extraction or real-source verification. See `docs/matching.md`.

### Phase 3 — Baselines
Satellite-only and ground-station-only prediction models.

Implemented optional mean/Ridge/Random Forest regression tooling with explicit
targets/units, shared cohorts, station or buffered date splits, train-only scaling,
validation selection, held-out testing, and traceable artifacts. Current tests are
software fixtures only; real target/data/evaluation remain outstanding. See
`docs/baselines.md`.

### Phase 4 — Dual-Branch Fusion
Satellite branch + chemical/tabular branch + basic learned multimodal fusion.

### Phase 5 — Dynamic Temporal Attention Gating
Dynamically weight modalities when observation timestamps differ.

### Phase 6 — Feature Selection
Identify important satellite spectral and ground chemical features.

### Phase 7 — Physics-Informed Learning
Add scientifically justified water-quality/river constraints and measure their effect through ablation.

### Phase 8 — Domain Adaptation & Cross-Regional Evaluation
Add domain-adversarial learning and evaluate an India-trained model on the selected foreign USGS river dataset.

### Phase 9 — Explainability
Add SHAP-based model and feature explanations, including modality contribution where appropriate.

### Phase 10 — Final Research Evaluation
Run baselines and ablations, generate figures/tables, document limitations, and prepare research-paper-quality results.

## Completion Criteria

A phase is complete only when its implementation is tested, documentation is updated, reproducibility requirements are satisfied, and the phase has been reviewed and approved.

## Exploratory real-data experiments — audit update (2026-10-14)

The local experiment archive contains exploratory India CPCB/Sentinel-2
matching and fusion runs, feature-selection experiments, and a small
US cross-region evaluation. These artifacts are evidence of prototype
experimentation, not proof of research-grade validation.

- India pilot: 14 paired events from 18 attempted events; the sample is
  too small to establish generalization.
- Fusion evaluation: chronological split with 6 training, 5 validation,
  and 3 test samples. Test MAE was 0.5864 mg/L (n=3); the small test set
  does not establish superiority or reliability.
- Feature selection: the selected configuration did not demonstrate
  improvement over the all-feature reference.
- Cross-region evaluation: frozen India model MAE was 3.4762 mg/L on
  four US events, worse than the reported training-mean baseline MAE
  of 2.2250 mg/L.
- Source metadata, station coordinates, measurement-method comparability,
  water-pixel validity, and possible shared-scene leakage require further
  verification.

These results do not complete the research-validation phases. Reproduce
the runs from versioned code/configuration and verified inputs before
claiming completion. The restored-demo artifact archive is separate from
the Git repository until deliberately integrated.

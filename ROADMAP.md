# Project Roadmap

This roadmap tracks the project from infrastructure setup through the research-paper-quality evaluation.

| Phase | Milestone | Status |
|---|---|---|
| 0 | Repository scaffolding | 🟢 Complete |
| 1 | Data interfaces & synthetic test fixtures | 🟢 Complete |
| 2 | Satellite–ground spatiotemporal hybridization | 🔵 Next |
| 3 | Satellite-only & ground-station-only baselines | ⚪ Planned |
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
Standardized interfaces for Sentinel-2, CPCB, and USGS data, plus synthetic fixtures strictly for software/unit testing. **Complete.**

### Phase 2 — Satellite–Ground Hybridization
Spatial and temporal alignment of satellite observations with CPCB measurements and preparation of the USGS pathway for cross-regional evaluation.

### Phase 3 — Baselines
Satellite-only and ground-station-only prediction models.

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

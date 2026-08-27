# Project Roadmap

This roadmap tracks the planned implementation from repository setup to the final research-paper-quality evaluation.

| Phase | Milestone | Status |
|---|---|---|
| 0 | Repository scaffolding | 🟢 Complete (pending merge) |
| 1 | Data interfaces & synthetic test fixtures | 🔵 Next |
| 2 | Satellite–ground spatiotemporal hybridization | ⚪ Planned |
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
Set up reproducible project infrastructure, configuration, documentation, testing, and repository structure.

### Phase 1 — Data Interfaces & Synthetic Test Fixtures
Define standardized interfaces for Sentinel-2, CPCB, and USGS data. Add synthetic fixtures strictly for software/unit testing.

### Phase 2 — Satellite–Ground Hybridization
Implement spatial and temporal alignment of satellite observations and CPCB measurements, with the USGS pathway documented for cross-regional evaluation.

### Phase 3 — Baselines
Implement and evaluate satellite-only and ground-station-only prediction models.

### Phase 4 — Dual-Branch Fusion
Build the satellite branch, chemical/tabular branch, and basic learned fusion model.

### Phase 5 — Dynamic Temporal Attention Gating
Allow the model to dynamically weight modalities when observation timestamps differ.

### Phase 6 — Feature Selection
Analyze and select important spectral and chemical features.

### Phase 7 — Physics-Informed Learning
Add scientifically justified water-quality/river constraints and evaluate their effect through ablation.

### Phase 8 — Domain Adaptation & Cross-Regional Evaluation
Add domain-adversarial learning and evaluate an India-trained model on the selected foreign USGS river dataset.

### Phase 9 — Explainability
Add SHAP-based explanations for model and feature contributions, including modality-level analysis where supported.

### Phase 10 — Final Research Evaluation
Run the complete baseline/ablation suite, produce figures and tables, document methodology and limitations, and prepare research-paper-quality results.

## Completion Criteria

A phase is considered complete only when:

- Implementation for that phase is finished.
- Relevant tests pass.
- Documentation is updated.
- Reproducibility requirements are satisfied.
- Results/limitations are recorded where applicable.
- The phase has been reviewed and explicitly approved before moving forward.

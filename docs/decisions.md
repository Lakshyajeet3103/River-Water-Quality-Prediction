# Research Decisions

Record important technical and research decisions, including rationale and status.

## Decision 001 — Incremental implementation

**Date:** 2026-08-27  
**Decision:** Develop the project one milestone at a time rather than implementing the complete architecture at once.  
**Reason:** Enables controlled experimentation, testing, review, and ablation of individual contributions.  
**Status:** Active

## Decision 002 — Separate real and synthetic data

**Date:** 2026-08-27  
**Decision:** Synthetic datasets may be used for software/unit testing only and must never be presented as experimental results.  
**Reason:** Prevents accidental fabrication or confusion between implementation testing and scientific evaluation.  
**Status:** Active

## Decision 003 — Research claims require evidence

**Date:** 2026-08-27  
**Decision:** Novelty and patentability claims must be supported by documented prior-work analysis.  
**Reason:** The project is intended to support a research paper and therefore requires defensible claims.  
**Status:** Active

## Decision 004 — Reproducible experiments

**Date:** 2026-08-27  
**Decision:** Experiments will record random seeds, configurations, preprocessing, data splits, and relevant model settings.  
**Reason:** Results need to be reproducible and suitable for research evaluation.  
**Status:** Active

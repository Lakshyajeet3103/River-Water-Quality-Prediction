# River Water Quality Prediction — Roadmap

## 1. Objective

Develop and evaluate a reproducible workflow for estimating river water quality using satellite-derived observations and measured water-quality data.

The project aims to investigate multimodal fusion while maintaining transparent data processing, leakage-resistant evaluation, reproducible experiments, and honest reporting of uncertainty.

## 2. Phase Overview

| Phase | Area | Status |
|---|---|---|
| 1 | Project setup and data workflow | Implemented components; continue verification |
| 2 | Satellite and water-quality data preparation | Partial; further validation required |
| 3 | Paired datasets and experimental evaluation | Initial pilot available; expansion required |
| 4 | Multimodal fusion | Initial scikit-learn baseline implemented; advanced architecture planned |
| 5 | Robust validation and generalization | In progress |
| 6 | Research reporting and deployment assessment | Pending further evidence |

A phase marked implemented or partial does not imply that the entire research objective has been completed or validated.

## 3. Phase 1 — Project Setup and Data Workflow

### Completed
- Established the repository structure and project documentation.
- Added data-processing and experiment workflow components.
- Added automated tests and reproducibility instructions.
- Recorded experiment outputs and methodological limitations.

### Remaining
- Continue testing after changes.
- Verify that instructions work in a clean environment.
- Keep the project status, experiment audit, and README consistent with actual implementation.

## 4. Phase 2 — Satellite and Water-Quality Data Preparation

### Current direction
Prepare satellite-derived features and corresponding in-situ water-quality measurements for paired experiments.

The current fusion baseline uses ten Sentinel-2 bands and measured fecal coliform as input features, with BOD as the target.

### Remaining work
- Verify station coordinates and reference-data provenance.
- Check satellite acquisition dates against sampling dates.
- Document cloud filtering, spatial extraction, and temporal matching rules.
- Establish quality checks for missing, invalid, or inconsistent measurements.
- Record failed downloads and unavailable reference observations.
- Avoid describing an observation as validated when matching or source quality remains uncertain.

## 5. Phase 3 — Paired Dataset and Evaluation

### Current status
An initial paired pilot contains 14 events from three Ganga River stations.

The current chronological split contains:
- 6 training events
- 5 validation events
- 3 test events

### Remaining work
- Expand the number of paired events and stations.
- Preserve independent training, validation, and test partitions.
- Prevent repeated measurements or closely related observations from creating leakage across splits.
- Document the rationale for split boundaries and matching tolerances.
- Report sample counts alongside all performance metrics.
- Where possible, add station-held-out and time-held-out evaluation.

## 6. Phase 4 — Multimodal Fusion

### Completed baseline
An initial scikit-learn MLP fusion baseline is implemented. It combines ten Sentinel-2 bands with measured fecal coliform to estimate BOD.

The baseline saves metrics, predictions, split assignments, run metadata, and a serialized model artifact.

### Preliminary baseline results

| Model | Split | MAE (mg/L) | RMSE (mg/L) | R² |
|---|---|---:|---:|---:|
| Scikit-learn MLP fusion | Train | 0.186381 | 0.240811 | 0.954784 |
| Scikit-learn MLP fusion | Validation | 0.409292 | 0.477922 | 0.816154 |
| Scikit-learn MLP fusion | Test | 0.687921 | 0.776960 | 0.315743 |
| Training-mean baseline | Test | 0.850000 | 0.981071 | -0.090995 |

The test set contains only three events. These results are exploratory and do not establish reliable performance on unseen rivers or future sampling periods.

A separate exploratory model reported test MAE of 0.5864 mg/L, RMSE of 0.9329 mg/L, and R² of 0.0135. That result belongs to a different experiment and must remain separate unless a controlled comparison uses consistent data and splits.

### Planned work
- Establish a consistent comparison protocol across models.
- Investigate an advanced dual-branch architecture for satellite and tabular inputs.
- Compare fusion against simple baselines and appropriate single-modality alternatives.
- Tune models using training and validation data only.
- Keep the test set isolated until final evaluation.
- Report uncertainty and per-station results when sample sizes permit.

## 7. Phase 5 — Robust Validation

### Priorities
1. Expand temporal and geographic coverage.
2. Evaluate on stations or dates excluded from model development.
3. Check for leakage through preprocessing, feature construction, or split design.
4. Compare with simple baselines using identical evaluation data.
5. Examine errors by station, season, target range, and data quality.
6. Report sample sizes and limitations alongside every metric.
7. Test whether performance remains useful when required input measurements are unavailable.

The current experiment estimates BOD using measured fecal coliform as an input. A satellite-only model or a future-time forecasting model would be a different task and must be evaluated separately.

## 8. Phase 6 — Research Reporting and Deployment

### Planned deliverables
- A reproducible methods description.
- Documented dataset sources and matching procedures.
- Versioned experiment configurations and outputs.
- Consistent baseline and fusion comparisons.
- Transparent limitations and uncertainty analysis.
- A final report that distinguishes implemented functionality from planned work.

Deployment should be considered only after sufficient data and independent validation demonstrate that the system is appropriate for its intended use.

## 9. Reproduction

Run the current scikit-learn fusion baseline from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path

.\.venv\Scripts\python.exe scripts\train_fusion.py `
  --paired artifacts\restored-demo\outputs\hybrid_demo_001\paired.csv `
  --splits artifacts\restored-demo\outputs\fusion_demo_001\split_assignments.csv `
  --output artifacts\restored-demo\outputs\fusion_sklearn_001
```

Run the automated test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The most recently recorded test run passed 176 tests in 12.28 seconds. Re-run the suite after modifications to confirm the current state.

## 10. Exit Criteria

The project can progress toward research validation when:
- Paired observations have sufficient quality and coverage.
- Satellite and in-situ measurements are matched using documented rules.
- Model comparisons use consistent, leakage-resistant splits.
- Results are reproduced from documented commands.
- Independent evaluation supports the intended use.
- Claims remain proportional to the evidence.

**Overall status:** Initial fusion baseline implemented. Dataset expansion, advanced multimodal fusion, and comprehensive validation remain outstanding.

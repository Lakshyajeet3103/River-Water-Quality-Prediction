# Project Status

Last verified locally: 2026-10-14

## Implemented

- Data ingestion, preparation, validation, and satellite-ground matching workflows.
- Sentinel-2 feature extraction pilot configurations and cloud-threshold comparison.
- Scikit-learn MLP fusion baseline using ten Sentinel-2 bands and measured
  fecal coliform to estimate ground-measured BOD.
- Saved train/validation/test assignments and frozen cross-region evaluation
  artifacts with Integrated Gradients explanations.
- SHAP KernelExplainer workflow for the saved scikit-learn fusion model.

## SHAP demo

Entry point: `scripts/explain_shap_crossregion.py`

Run from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe .\scripts\explain_shap_crossregion.py
```

Output directory:
`artifacts/restored-demo/outputs/shap_crossregion_001/`

The current run explains four US observations using six India training
observations as the SHAP background. It generates feature contributions,
global feature importance, two plots, prediction/additivity checks, and
a JSON report. The saved model is not retrained.

## Latest verification

- SHAP version used: 0.53.0.
- Latest full test run: 176 passed in 12.95 seconds.
- SHAP contribution reconstruction matched the saved-model predictions
  to floating-point precision in the four-observation run.

These results describe the latest local run; rerun tests and the demo after
subsequent code or dependency changes.

## Known limitations

- Only 14 paired India events are available in the documented fusion pilot,
  with 6 train, 5 validation, and 3 test observations.
- The current US SHAP evaluation contains only four observations.
- The model requires ground fecal-coliform measurements at inference time.
- Cross-region generalization and operational accuracy are not established.
- SHAP feature importance is model attribution, not causal evidence.

## Planned advanced features

1. Physics-Informed Neural Network (PINN).
2. Dynamic Temporal Attention Gating.
3. Domain-Adversarial Neural Network (DANN).

These are pending implementation and verification. They must not be described
as completed until code, reproducible outputs, and tests demonstrate them.

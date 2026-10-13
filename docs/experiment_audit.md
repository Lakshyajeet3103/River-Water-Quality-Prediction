# Exploratory Experiment Audit

## Purpose

This document records the scope and limitations of exploratory experiments
found in the separate local `restored-demo` artifact bundle. It does not
certify that these experiments are reproducible from the current Git checkout.

## India fusion demonstration

- 14 paired events.
- Chronological split: 6 training, 5 validation and 3 test events.
- Target: same-event BOD estimation, not future BOD forecasting.
- Inputs: ten Sentinel-2 bands and measured fecal coliform.
- Test MAE: 0.5864 mg/L.
- Test RMSE: 0.9329 mg/L.
- Test R²: 0.0135.

The test contains only three events. The result is a feasibility demonstration,
not reliable evidence of generalization or superiority over alternatives.
Because measured fecal coliform is an input, this model cannot be used as-is
where that ground measurement is unavailable.

## Feature-selection demonstration

The saved report selects all ten satellite bands plus fecal coliform.
Selected-feature and all-feature reference metrics are identical in the
saved report. Therefore, this run does not demonstrate an accuracy improvement
from feature selection. The three-event test set was also used in the earlier
fusion demonstration, so it is not a fresh independent final test.

## India-to-US exploratory transfer

The saved report evaluates a frozen India-trained model on four US events:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Frozen India fusion model | 3.4762 | 4.5275 | -1.6664 |
| India training-mean reference | 2.2250 | 3.5394 | -0.6296 |

The frozen model performs worse than the training-mean reference on these
four events. The sample is too small to support broad conclusions about
US rivers or geographic transfer. No US-labelled recalibration is reported.

The saved explanation method is integrated gradients, not SHAP. These
attributions are model explanations relative to the stated reference inputs;
they are not causal effects or water-safety conclusions.

## Status interpretation

These outputs show that exploratory prototypes exist. They do not mean that
the corresponding research phases are complete. Reproduction, data-quality
review, stronger baselines, leakage checks, larger evaluation cohorts and
independent final testing remain necessary.

The reports and model files are not copied into this repository by this
documentation change.

## Reproducible scikit-learn fusion baseline

A scikit-learn MLP baseline was trained using the paired-event CSV and the
saved chronological split assignments.

- Output directory: `artifacts/restored-demo/outputs/fusion_sklearn_001`
- Model: `fusion_model.joblib`
- Predictions: `test_predictions.csv`
- Metrics: `metrics.csv`
- Run metadata: `run.json`
- Split assignments: `split_assignments.csv`

| Model | Split | n | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Scikit-learn MLP fusion | Train | 6 | 0.186381 | 0.240811 | 0.954784 |
| Scikit-learn MLP fusion | Validation | 5 | 0.409292 | 0.477922 | 0.816154 |
| Scikit-learn MLP fusion | Test | 3 | 0.687921 | 0.776960 | 0.315743 |
| Training-mean baseline | Test | 3 | 0.850000 | 0.981071 | -0.090995 |

The scikit-learn baseline has lower test MAE and RMSE than the training-mean
reference on these three test observations. The sample is too small to establish
reliable generalization or broad model superiority. The training metrics are
not evidence of independent test performance.

This is same-event BOD estimation using satellite bands and measured fecal
coliform, not future BOD forecasting. The model requires fecal-coliform
measurements at inference time. These metrics belong to this scikit-learn run
and must not be conflated with the earlier exploratory model's results.

The full repository test suite passed: 176 tests in 12.51 seconds. This
validates the tested software behavior, not the scientific validity of the
model or its performance on new rivers.

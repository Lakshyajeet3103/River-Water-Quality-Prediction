# Previously untested saved models: unseen Indian stations

Evaluated the frozen 80-row-trained ratio MLP, selected four-input MLP and all
five experimental two-branch networks. No retraining, scaling updates or
test-based feature/checkpoint selection took place. Every checkpoint was
verified against its saved SHA256 before and after inference.

The new dataset contains 30 real matched observations from four stations absent
from the original training data: Garhmukteshwar, Sangam, Narora and Mirzapur.
Four observations share a sensing acquisition with training data. The primary
comparison excludes those four, leaving **26 observations**. All 30 results
are also retained. This is frozen-model station transfer, not rotating station
cross-validation or future forecasting. All locations are on the Ganga in UP.

## Primary results: acquisition-independent new-station cases

All model outputs are BOD in mg/L. Percentage error is MAPE:
`100 * mean(abs(predicted BOD - measured BOD) / measured BOD)`.
It is not a success rate or guaranteed margin of error.

| Model | Inputs | Familiar test n | Familiar MAE | Familiar MAPE | New-station test n | New-station MAE | New-station MAPE |
|---|---|---:|---:|---:|---:|---:|---:|
| Satellite ratio MLP | 10 bands + 5 computed ratios | 20 | 0.5911 | 35.67% | 26 | 0.9531 | 51.18% |
| Selected four-input MLP | B8, B8A, B11 + fecal | 20 | 0.1852 | 9.37% | 26 | 0.7548 | 33.67% |
| Two-branch: switches off | 10 bands + fecal, pH, DO | 20 | 0.3211 | 17.95% | 26 | 1.1163 | 57.24% |
| Two-branch: gate only | Same + date gap | 20 | 0.2799 | 16.39% | 26 | **0.6925** | **33.26%** |
| Two-branch: constraint only | 10 bands + fecal, pH, DO | 20 | 0.3221 | 18.00% | 26 | 1.1176 | 57.32% |
| Two-branch: station adversarial only | 10 bands + fecal, pH, DO | 20 | 0.3252 | 18.38% | 26 | 1.0971 | 56.28% |
| Two-branch: all components | Same + date gap | 20 | 0.2827 | 16.49% | 26 | 0.6937 | 33.35% |
| Training-mean reference | No per-event inputs | 20 | 1.0054 | 58.07% | 26 | **0.5197** | 33.46% |

All models have worse MAE than the constant training-mean reference on these
26 cases. Gate-only has the best model MAE; its MAPE is slightly below the
reference's MAPE, but its MAE/RMSE are worse. A lower percentage error alone
does not establish better performance across metrics. Gate-only reduces
new-station MAE about 38% versus the switches-off network. Constraint-only
does not help; station-adversarial alone gives only a small reduction.

The selected-feature model's low familiar-station error does not carry over to
new stations. These new cases have a narrow BOD range, and all model R2 scores
are negative. This is evidence of limited transfer, not robust monitoring.

## Secondary results: all 30 new-station observations

| Model | Test n | MAE, mg/L | MAPE |
|---|---:|---:|---:|
| Satellite ratio MLP | 30 | 0.9047 | 47.94% |
| Selected four-input MLP | 30 | 0.7302 | 32.95% |
| Two-branch: switches off | 30 | 1.1504 | 59.32% |
| Gate only | 30 | 0.7416 | 36.28% |
| Constraint only | 30 | 1.1518 | 59.41% |
| Station adversarial only | 30 | 1.1321 | 58.40% |
| All components | 30 | 0.7429 | 36.38% |
| Training-mean reference | 30 | 0.5172 | 33.64% |

The four acquisition-overlap cases are identified in `cohort_audit.csv`.
No observations were excluded based on error, BOD magnitude or model performance.
The saved models are frozen, so shared acquisitions were removed from test
rather than changing their training data. This stricter 26-case score is not
directly comparable to earlier models evaluated on the full 30-case set.

## Reproduction and artifacts

Code and artifacts are in the isolated workspace
`C:\Users\Lakshyajeet\Desktop\clone of dbs project\unseen-model-evaluation`,
on local branch `unseen-model-evaluation-2026-10-14`, based on the comprehensive
`bod-categories-crossregion-2026-10-14` branch. The frontend workspace stays on
main with its existing local changes preserved. This evaluation is not committed
or pushed automatically.

Using the existing virtual environment, run from that isolated workspace:

```powershell
& 'C:\Users\Lakshyajeet\Desktop\clone of dbs project\River-Water-Quality-Prediction\.venv-check\Scripts\python.exe' scripts/restore_demo_bundle.py
& 'C:\Users\Lakshyajeet\Desktop\clone of dbs project\River-Water-Quality-Prediction\.venv-check\Scripts\python.exe' scripts/evaluate_saved_models_unseen.py
```

Report artifacts: `docs/saved_models_unseen_results/` contains pooled MAE/RMSE/R2/
MAPE, station-level scores, each prediction, cohort audit, actual matched inputs,
comparison chart and hashed provenance. Familiar-station predictions reproduce
their existing saved MAEs. Inference explicitly excludes ground BOD and station
ID columns. Source QC, coordinates, laboratory methods, matching dates and
water-pixel assumptions retain the prior dataset limitations.

# Familiar-station satellite refinements

Completed the requested ratio, regularisation and tighter-date experiments on the original three-station dataset. The saved 80 training / 20 later test observations remain unchanged; existing datasets, main models and WQI calculations/categories are preserved.

## Same original 20 test observations

| Predeclared trial | Training observations | MAE, mg/L | RMSE, mg/L | R2 |
|---|---:|---:|---:|---:|
| Original ten-band model | 80 | 0.8225 | 1.0669 | -0.0750 |
| Ten bands plus five ratios/contrasts, original alpha=1 | 80 | 0.5911 | 0.7343 | 0.4907 |
| Regularisation-only, internally selected alpha=1 | 80 | 0.8225 | 1.0669 | -0.0750 |
| Jointly validation-selected ratios plus alpha=0.01 | 80 | 0.8758 | 1.2836 | -0.5561 |
| Training mean BOD | 80 | 1.0054 | 1.0440 | -0.0293 |

The ratios-only trial reduced average error by about 28% versus the original model on these same 20 cases. However, the joint pipeline selected using internal validation performed worse on the later test set. The ratios-only version is reported as a predeclared trial, not a new independently validated winner selected after seeing these results. It is not automatically promoted to replace the main model.

Regularisation-only internal validation chose alpha=1, retaining the original value. Stronger or weaker penalties were not silently selected based on outer test scores.

## Three-day matching: fair comparison on the same 19 cases

Filtering absolute sample/image date gaps to <=3 days leaves 57 of 80 training observations and 19 of 20 test observations. The original internal 60/20 training split becomes 46 fitting / 11 validation cases. No new imagery is queried or sampling dates shifted.

| Trial on identical 19 eligible test observations | Training observations | MAE, mg/L | RMSE, mg/L |
|---|---:|---:|---:|
| Original model trained with up-to-seven-day matches | 80 | 0.7397 | 0.9466 |
| Date-only change: raw bands, alpha=1, three-day training subset | 57 | 0.9001 | 1.2255 |
| All-window ratios-only trial, alpha=1 | 80 | 0.5660 | 0.7124 |
| Three-day jointly selected ratios/alpha=0.1 | 57 | 0.9573 | 1.3272 |

Tighter dates did not help this retraining experiment. Compare the first two rows to isolate the training-date filtering change on identical test events. Do not compare 0.9001 on 19 filtered cases with 0.8225 on the original 20 as if the test sets were equal. Merely excluding the seventh-day test case changes the evaluation population and is not itself a model improvement.

The experiment cannot distinguish every cause: tighter alignment also reduces training sample size and changes its composition. All three-day variants remain archived even though they performed worse.

## Inputs and tuning rules

Raw satellite surface-reflectance values produce these five deterministic additional features:

| Feature | Calculation |
|---|---|
| `ratio_B3_B4` | B3 / B4 |
| `ratio_B8_B4` | B8 / B4 |
| `ratio_B5_B4` | B5 / B4 |
| `contrast_B3_B8` | (B3 - B8) / (B3 + B8) |
| `contrast_B4_B3` | (B4 - B3) / (B4 + B3) |

These are numeric features, not direct BOD measurements. Nonfinite bands or denominators with absolute magnitude <1e-6 are rejected, not filled or silently clipped. No observations in this cohort failed these checks. Ground chemical measurements and target BOD are not used to calculate features.

Used a fixed alpha grid: 0.01, 0.1, 1, 10, 100. Larger alpha penalises large weights more strongly. Kept the existing model family/hidden layout: MLPRegressor, eight ReLU hidden neurons, L-BFGS, max_iter=3000, seed=42. Adding ratios increases the input-layer parameter count; the hidden architecture is unchanged.

Both raw-band and ratio input modes are evaluated inside the saved chronological training-only split. Minimum internal validation MAE chooses the pipeline; ties prefer fewer inputs, then smaller alpha and input-mode name. Date-filtered choices use only the filtered fitting/validation subsets. Test observations never enter fitting, scaling or candidate selection. Input/target scalers fit only the corresponding training subset, and nonnegative output clipping is retained.

Because these 20 test observations have been inspected repeatedly, all results are exploratory internal comparisons. Improvements need a fresh independent test before any broader accuracy claim. This does not establish improved unseen-station, foreign-river or overall-WQI prediction.

## Files, inference and reproduction

Outputs: `outputs/satellite_seen_refinements_20261014/`, preserved in [the demo bundle](../demo_bundle/files/outputs/satellite_seen_refinements_20261014/). Five separate checkpoints, both candidate-validation tables, date eligibility for every observation, test predictions, metrics, charts and provenance are included.

To use the ratios-only experimental checkpoint with raw satellite inputs:

```python
import joblib
import pandas as pd
from river_water_quality.training.satellite_refinements import predict

model = joblib.load('outputs/satellite_seen_refinements_20261014/ratios_only_alpha1.joblib')
inputs = pd.read_csv('outputs/satellite_only_80_20_20261014/example_inputs.csv')
predicted_bod_mg_l = predict(model, inputs)
```

This wrapper computes the ratio columns before applying saved scaling. Do not pass raw ten-band arrays directly to the fifteen-input estimator or insert ratio columns into the original saved model.

Reproduce from the repository root with the existing demonstration dependencies:

```powershell
.\.venv-check\Scripts\python.exe scripts/restore_demo_bundle.py
.\.venv-check\Scripts\python.exe scripts/refine_satellite_seen.py --output outputs/seen_refinements_reproduction
```

Choose a fresh output directory. No network, Earth Engine login, GPU or additional raw data are needed. Original-model prediction equivalence, unchanged source hashes and all new checkpoint reloads were checked. Nine relevant ratio/selection tests passed, including arithmetic, near-zero rejection, target-free prediction, training-only scaling and validation isolation. Charts were visually checked. Notebook integration remains pending.

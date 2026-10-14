# Satellite band selection at familiar stations

Tested the same 17 satellite-only candidate subsets on the original 100-event, three-station cohort. Kept the saved 80 training / 20 later test assignments and existing model type/settings. This is separate from the 130-event unseen-station study.

## Results on the same 20 test observations

| Approach | Inputs | MAE, mg/L | RMSE, mg/L | R2 |
|---|---|---:|---:|---:|
| Training-selected visible bands | B2, B3, B4 | 1.0961 | 1.3767 | -0.7899 |
| Original saved satellite-only model | All ten bands | 0.8225 | 1.0669 | -0.0750 |
| Training mean BOD | Constant | 1.0054 | 1.0440 | -0.0293 |

**Band selection did not improve this familiar-station test.** The selected subset has higher MAE and RMSE than the original model. The original saved predictor remains unchanged and is the better of these two satellite models on this cohort. Do not claim the unseen-station improvement also applies at familiar stations.

## Selection procedure

Used the already saved internal chronological split of the 80 outer training observations: 60 for candidate fitting and 20 for candidate validation. Satellite acquisition groups and sample/image date intervals are checked for separation at both inner and outer boundaries. All validation/test stations occur in their corresponding training data.

Each candidate uses only its satellite columns. BOD is solely the training/validation target; ground pH, oxygen and fecal concentration are not predictors. The same MLP settings are retained: eight ReLU hidden neurons, L-BFGS, alpha 1, max_iter 3000, seed 42; scalers fit only candidate-fitting rows.

Minimum internal-validation MAE chooses the subset; fewer inputs and candidate name break ties. The selected visible-band model is refitted on all 80 training rows, and only then scored on the original 20 test observations. No outer-test predictions for alternative subsets are searched to replace that choice. A choice favoured by a small internal validation set can still perform worse on later data.

The separate selected checkpoint converged in 77 iterations without captured warnings. Reload reproduced predictions. The original baseline checkpoint reproduced its previously saved test predictions to numerical tolerance, and all protected source/model hashes remained unchanged.

The test examples were examined in earlier demonstrations; this is exploratory internal evaluation, not a fresh independent performance estimate. Twenty test cases from three known stations are limited evidence. No regularisation search, new band ratios, date-window change, main-model replacement or WQI/category change was made.

## Files and reproduction

Saved in `outputs/satellite_band_selection_seen_20261014/` and [the demo bundle](../demo_bundle/files/outputs/satellite_band_selection_seen_20261014/): selected model, candidate validation scores, inner/outer assignments, identical-test predictions, overall/station metrics, satellite-only example inputs, comparison chart and provenance hashes.

```powershell
.\.venv-check\Scripts\python.exe scripts/restore_demo_bundle.py
.\.venv-check\Scripts\python.exe scripts/select_satellite_bands_seen.py --output outputs/seen_band_selection_reproduction
```

Run from the repository root with the existing demonstration dependencies. Output must be a fresh directory. No downloads, Earth Engine login or GPU are required. Eight relevant selection/preprocessing tests passed, including tests that reject outer IDs in selection assignments and shared acquisitions. The chart was visually checked. Notebook integration is pending.

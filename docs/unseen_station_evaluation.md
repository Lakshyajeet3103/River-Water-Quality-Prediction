# Unseen-station testing

Implemented leave-one-station-out evaluation using the existing friend's model type and fixed settings. Each test station is completely absent from training in its fold. Three separate model variants are compared: satellite-only, ground-only and full combined. No main saved model is replaced; no WQI formula, category or assumption is changed.

## Expanded 130 observations: seven stations

The [expanded real-match dataset](station_expansion.md) adds 30 observations at four new stations. The same protocol now rotates through seven folds, training on six stations and testing the seventh, with shared-acquisition training rows purged. Every one of the 130 events is tested once per model variant.

| Model | Pooled test observations | MAE, mg/L | RMSE, mg/L | R2 | Station-macro MAE |
|---|---:|---:|---:|---:|---:|
| Satellite-only | 130 | 1.7282 | 1.9335 | -2.4710 | 1.2346 |
| Ground-only | 130 | 0.8981 | 1.0833 | -0.0896 | 0.7414 |
| Combined | 130 | 1.5900 | 1.8483 | -2.1721 | 1.1408 |
| Training-fold mean BOD | 130 | 1.2435 | 1.4095 | -0.8446 | 0.8702 |

Ground-only has the lowest MAE here, but its pooled R2 is still negative. The expanded sample and training composition differ from the original study below, so their metric differences do not isolate the effect of adding data. This remains an exploratory transfer test, not proven reliable station generalisation. Some new station test sets contain only three or seven examples.

All 21 expanded-fold model fits completed without captured convergence warnings. They are saved separately under `outputs/unseen_stations_expanded_20261014/`; the previous checkpoints are not replaced.

## Frozen saved models: 30 genuinely new-station examples

Separately, tested the existing saved 80/20 models without retraining on all 30 new examples. Their stations and observation IDs are absent from the original 100-event cohort. This avoids confusing the main saved model with the newly trained holdout-fold models.

| Frozen model | New test n | MAE, mg/L | RMSE, mg/L |
|---|---:|---:|---:|
| Satellite-only main predictor | 30 | 0.9205 | 1.1406 |
| Ground-only baseline | 30 | 0.4850 | 0.6564 |
| Full combined comparison model | 30 | 0.8271 | 0.9627 |
| Original training mean BOD | 30 | 0.5172 | 0.6291 |

Ground-only slightly beats the mean on MAE, but has worse RMSE. All frozen-model R2 values are negative. There is no strong accuracy claim or automatic replacement/promotion. Frozen predictions and their model hashes are saved in `frozen_models_new_station_predictions.csv` and `frozen_models_new_station_metrics.csv` in the expanded output directory.

## Original 100 observations: three stations

The original three-station dataset is preserved. Each fold holds out one whole station and trains on the remaining two, after removing training observations that share a Sentinel-2 sensing acquisition with any test observation. Each observation receives exactly one prediction from each model trained without its station.

| Held-out station | Training rows after purge | Test rows | Purged training rows | Combined MAE, mg/L |
|---|---:|---:|---:|---:|
| Anupshahar | 57 | 38 | 5 | 1.1201 |
| Varanasi, Malviya Bridge | 63 | 37 | 0 | 1.8518 |
| Bijnor barrage | 70 | 25 | 5 | 1.5396 |

| Model | Pooled test observations | MAE, mg/L | RMSE, mg/L | R2 |
|---|---:|---:|---:|---:|
| Satellite-only | 100 | 2.0849 | 2.2176 | -2.8026 |
| Ground-only | 100 | 1.8990 | 2.1492 | -2.5716 |
| Combined | 100 | 1.4957 | 1.6428 | -1.0868 |
| Mean BOD from each training fold | 100 | 1.6547 | 1.7492 | -1.3660 |

This is a harder evaluation than predicting later dates at known stations. Combined has a lower pooled MAE than the training-mean baseline, but errors remain large, and negative R2 indicates weak transfer. The result does not support reliable prediction at an arbitrary new river location. It documents a limitation rather than concealing it.

Pooled metrics weight stations by their observation counts. `station_macro_mae` additionally weights each station equally. Fold-specific R2 can be extremely negative for narrow BOD ranges and should be interpreted alongside absolute errors and sample counts.

## Protocol

All variants use eight ReLU hidden neurons, L-BFGS, alpha 1, seed 42, maximum 3000 iterations. Input and target scalers fit only training rows. Ground fecal concentration is log1p-transformed; predictions use the existing nonnegative clipping rule. BOD labels, station IDs and test-station rows never enter feature preprocessing/training.

Shared acquisition aliases are grouped by Sentinel-2 sensing time, conservatively purging overlapping training groups. This is retrospective station transfer: training dates may overlap test dates, so it is not a future forecast or a combined station-and-future holdout. All stations are on the Ganga; this is not a foreign-river evaluation.

No hyperparameter or feature selection uses held-out-station results. Fold convergence warnings and model hashes are saved. Each fold checkpoint is reloaded and checked against its predictions. All old models and source data are hash-checked after evaluation.

## Run

```powershell
.\.venv-check\Scripts\python.exe scripts/restore_demo_bundle.py
.\.venv-check\Scripts\python.exe scripts/evaluate_unseen_stations.py --output outputs/unseen_station_reproduction
```

Default input is the saved enriched 100-event dataset. The script accepts `--paired` for a separately preserved expanded dataset with its provenance/run hash. Output must be a fresh directory.

Saved artifacts contain fold assignments (including purged rows), per-observation predictions, station/pooled metrics, nine fold checkpoints for the original dataset, a chart and hash/provenance report. Results are in `outputs/unseen_stations_original_20261014/`.

Tests check complete station isolation, invariance of training selection to test answers, purging tile/processing aliases, and refusal of duplicate IDs or insufficient data after purging. This step leaves WQI and the existing demonstration notebook unchanged.

Expanded reproduction after restoring the bundle:

```powershell
.\.venv-check\Scripts\python.exe scripts/evaluate_unseen_stations.py --paired outputs/new_station_matches_v2_20261014/paired.csv --output outputs/unseen_expanded_reproduction
```

Both complete studies, checkpoints and charts are preserved in the demo bundle. All 208 bundle files verify. Full suite: 281 tests passed; the actual expanded-data run additionally verified every checkpoint reload and preserved-source hash. Charts were visually checked.

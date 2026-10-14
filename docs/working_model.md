# Satellite-only BOD model documentation

## Purpose and status

**Current selected version (2026-10-14):** the user selected the ten-band plus five-ratio/contrast model with the original alpha=1 settings. The checkpoint is `outputs/satellite_seen_refinements_20261014/ratios_only_alpha1.joblib`; `configs/working_satellite_model.json` records its SHA256 and status. Its familiar-station test MAE is **0.5911 mg/L**, RMSE **0.7343 mg/L**, R2 **0.4907**, on the original same 20 test cases. Eight hidden ReLU neurons are retained; there are now 15 model inputs and 137 network weights/biases. It converged in 319 iterations.

Supply the same ten raw scaled bands; the prediction wrapper calculates five additional features. No ground measurements are required. Use:

```powershell
.\.venv-check\Scripts\python.exe scripts/predict_satellite.py --input outputs/satellite_only_80_20_20261014/example_inputs.csv
```

Or add `--output outputs/selected_satellite_predictions.csv` to save a new CSV. Existing output files are not overwritten. See [ratio definitions, preprocessing and all trials](satellite_seen_refinements.md). The original ten-band model is preserved; the remaining sections below document that historical version. Existing notebook, WQI and earlier experiment results still refer to their original checkpoints until explicitly updated.

This selection follows user review of repeatedly examined test cases, not independent validation. The 0.5911 mg/L score is an average absolute error, not a guaranteed plus/minus interval. Unseen-station performance for this ratio checkpoint is not established; do not assign the separate band-selection study's 1.3315 mg/L result to it.

Demonstrate estimation of a measured water-quality parameter from real matched satellite and ground observations. The current model predicts BOD (mg/L) using satellite features alone. CPCB BOD provides training labels, not prediction-time inputs. Fecal coliform remains in the archived paired table but is never passed to this model. This is a small demonstration, not a validated monitoring service.

The saved model is `outputs/satellite_only_80_20_20261014/model.joblib`, restored from the byte-preserved Git bundle. The earlier combined model remains separately saved and unchanged. `working-model` is a Git-valid branch name; spaces are not allowed in branch names.

## Input and output contract

Each input row needs these finite numeric columns, in scaled surface-reflectance units:

```text
satellite_B2, satellite_B3, satellite_B4, satellite_B5, satellite_B6,
satellite_B7, satellite_B8, satellite_B8A, satellite_B11, satellite_B12
```

An optional `observation_id` identifies the output. Column order in a CSV may differ; prediction explicitly selects the saved feature order. Missing required columns or non-finite values fail rather than being filled. Extra ground or target columns are ignored by prediction. Do not supply raw satellite digital numbers or apply feature standardization yourself.

`predict(model, frame)` in `scripts/train_satellite_only.py` returns one estimated BOD value per input row, in the same order. It applies the saved input scaler, estimator and inverse target scaler, then clips negative outputs to zero. Zero is a clipped estimate, not confirmation of zero measured BOD.

```python
import joblib
import pandas as pd
from scripts.train_satellite_only import predict

model = joblib.load("outputs/satellite_only_80_20_20261014/model.joblib")
inputs = pd.read_csv("outputs/satellite_only_80_20_20261014/example_inputs.csv")
predictions = predict(model, inputs)
```

Run from the repository root after installation/restoration. The serialized dictionary contains `model`, `x_scaler`, `y_scaler`, `features`, `target`, `seed` and `satellite_only`. Use this repository's saved artifact with the documented environment.

## Ground and satellite preparation

An enriched version of the same 100 observations adds `ground_ph` and `ground_dissolved_oxygen` for future experiments. See [the chemical dataset documentation](chemical_dataset.md). These columns are not inputs to either current saved model; this preparation does not alter their predictions.

The 100-event dataset spans 2018-11-25 through 2021-12-28. Stations are Ganga U/S Anupshahar (38 events), Ganga at Varanasi D/S/Malviya Bridge (37), and Madhya Ganga Barrage/Bijnor (25). Original CPCB-labelled biological exports were prepared into dated BOD and fecal-coliform observations; only complete candidate events were used in this expansion. Satellite-only training does not use fecal values, but selection of the initial cohort required their availability.

Satellite source: `COPERNICUS/S2_SR_HARMONIZED` through the existing Earth Engine project. Extraction scales digital numbers by 0.0001 and averages common-valid SCL-water pixels within a 60 m buffer on a common B5 20 m grid, using pixel centres and default nearest resampling.

Candidate selection uses the nearest usable acquisition, primary window ±3 calendar days and fallback up to ±7 days, in Asia/Kolkata. Settings require cloud/shadow fraction <= 0.2, water fraction >= 0.2, coverage >= 0.9, and at least 3 valid water pixels. SCL class 6 is water; classes 3/8/9/10 are masked cloud/shadow. Failed candidates are logged, not replaced with fabricated features. Pixel averages are inputs; this model does not consume raw image patches.

The dataset was grown from 14 to 64 and then 100 events. Across the two expansions, 116 additional candidates were attempted: 86 passed and 30 had no usable scene. Original data, model versions, candidate audits and source hashes remain available in the bundle. Selection was newest-first round robin over the three stations, based on complete parameters and image availability, not BOD magnitude.

## Training and evaluation

Rows are sorted by sample date, station and observation ID. Oldest 80 train; newest 20 test. Training sampling/acquisition intervals end 2021-07-15; test intervals begin 2021-09-12. The trainer rejects shared Sentinel-2 acquisition groups or overlapping date intervals across this boundary. Both standard scalers are fitted on training data only.

The friend's MLP design is retained: one 8-neuron ReLU hidden layer, L-BFGS solver, alpha=1.0, maximum 3000 iterations, seed=42. There are ten inputs instead of eleven, hence 97 trainable network weights/biases. Training converged in 89 iterations. No validation set or hyperparameter search is used in this requested 80/20 run.

| Method | n | MAE (mg/L) | RMSE (mg/L) | R² |
|---|---:|---:|---:|---:|
| Satellite-only, training | 80 | 0.5148 | 0.6144 | 0.7195 |
| Satellite-only, testing | 20 | 0.8225 | 1.0669 | -0.0750 |
| Training-mean baseline, testing | 20 | 1.0054 | 1.0440 | -0.0293 |

Lower MAE indicates lower average absolute error on this test. Higher RMSE and negative R² mean the model does not beat simple references consistently. Scores are not classification accuracy percentages. The older combined model was trained/tested on different rows; comparing its published score directly does not isolate the value of ground inputs.

Saved-model reload reproduced predictions, and prediction from satellite-only CSVs was checked. Train-only scaler statistics and acquisition separation were verified. The repository suite passed 232 tests before this documentation update; those tests validate software behavior, not scientific accuracy.

## Limitations and new locations

The ground source URL/license, laboratory quality flags, coordinate reference and displayed sampling-time conventions remain provisional as recorded in the source manifests. SCL classification, bank/bridge mixtures and sampling-position uncertainty remain possible. All source assumptions are retained rather than upgraded by the model run.

This is retrospective estimation at known stations. Several test observations were previously examined during earlier model experiments, so the test is not an untouched final evaluation. A 100-row convenience cohort does not demonstrate broad river coverage, unseen-station performance or future forecasting.

For new coordinates, supply a date, extract the same ten features using the established image processing, and require usable water pixels. Never interpret clouds or missing imagery as zero BOD. An estimate elsewhere is experimental; there is no automatic coordinate-to-BOD application in this branch. The current saved model uses no US observations, and the earlier US evaluation is not evidence of this satellite-only model's performance.

## Preservation and reproducibility

`demo_bundle/manifest.json` records exact hashes for archived data, models and reports; restore with `scripts/restore_demo_bundle.py`. Restoration refuses to overwrite modified files. The CPU dependency versions used for the saved model are pinned in `requirements-working-model.txt`. Use standard Python 3.12 or 3.13: the pinned NumPy version requires Python >=3.12, and the project currently supports Python <3.14.

The current model, prior combined model and 100-row data are separate artifacts. Rerun training with `--output` pointing to a fresh directory. No retraining is needed to run saved predictions. No CUDA or PyTorch dependency is required for this scikit-learn demonstration.

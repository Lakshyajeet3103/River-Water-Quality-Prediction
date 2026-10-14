# River Water Quality Prediction — working model

**Start the presentation here:** [demonstration notebook](notebooks/river_water_quality_demo.ipynb), with executed tables, predictions and seven embedded charts. See [opening and running instructions](notebooks/README.md). It runs offline without retraining.

A small proof of concept that estimates **BOD in mg/L from Sentinel-2 satellite readings**. BOD measures oxygen used while microorganisms break down organic matter. The output is an estimate, not a laboratory measurement or drinking-water safety rating.

**New presentation demonstrations:** [SHAP for all three recommended models and
a simple oxygen-balance PINN](docs/oxygen_pinn_and_three_model_shap.md). The PINN
is an optional equation-based simulator after BOD prediction; its assumed oxygen
trajectories do not establish real river forecasts or change BOD-model accuracy.

**BOD-led categories:** [run the Best / Good / Bad demonstration](docs/bod_led_category.md).
It combines satellite-predicted BOD (70% weight) with training-only station-average
pH, oxygen and fecal coliform. The custom categories agree with the measured-input
reference on 13/20 saved examples (65%); they are not official WQI or safety labels.

**Cross-region explainability:** [four real US matches and signal contributions](docs/crossregion_explainable.md)
evaluate frozen India-trained models. The selected satellite model's US MAE is
2.653 mg/L; the combined model has satellite/ground Shapley explanations. Results
show limited transfer, rather than demonstrated foreign-river robustness.

**Selected satellite model:** ten bands plus five internally calculated ratios/contrasts, original alpha=1 and eight hidden neurons. Familiar-station average error is **0.591 mg/L** on the original same 20 test cases. Selection is recorded in `configs/working_satellite_model.json`; the original checkpoint and historical notebook/experiment results are preserved. See [current model instructions](docs/working_model.md).

This branch, `working-model`, preserves the trained satellite-only model, its 100 matched Indian observations, test predictions, and the earlier ground-plus-satellite model. The current model needs **no fecal-coliform reading** at prediction time.

**Prepared chemical dataset:** all 100 events now have matching CPCB pH and dissolved-oxygen readings in `outputs/india_100_chemical_dataset_20261014/paired.csv`. This separate dataset is ready for the next feature-selection/fusion experiments; saved models retain their existing inputs. See [dataset sources, units and audit](docs/chemical_dataset.md).

**New demonstrations:** a fair comparison uses identical 80/20 splits for satellite-only and combined models, training-only feature selection, and actual SHAP explanations. Average test errors are 0.82 mg/L (satellite only), 0.32 (all ground readings added), and 0.19 (selected B8/B8A/B11 + fecal coliform). These exploratory models are saved separately; the main predictor remains unchanged. See [results, charts and reproduction instructions](docs/straightforward_demonstrations.md).

**Switchable experimental network:** a separate small two-branch PyTorch model now demonstrates a date-gap gate, nonnegative-BOD training penalty, and station-adversarial gradient reversal. Five on/off variants are saved with losses, weights and predictions. The gate helped this sample; the other individual additions did not improve accuracy. See [mechanisms, switches, charts and limitations](docs/experimental_network.md). The working scikit-learn models remain preserved.

## Quick start: Windows PowerShell

**Ratios/settings/dates tested at familiar stations:** [refinement report](docs/satellite_seen_refinements.md) preserves all trials. Adding five ratios/contrasts with the original settings reduced test MAE from 0.823 to 0.591 mg/L; regularisation-only kept alpha=1, while joint tuning and three-day retraining performed worse. Same-test comparisons and coverage are explicit. No main model or WQI replacement.

**Familiar-station band test completed:** [same-20-example comparison](docs/satellite_band_selection_seen.md) selected B2/B3/B4 using training-only chronological validation. Test MAE increased to 1.096 mg/L versus the saved ten-band model's 0.823. The selected checkpoint is preserved as an experiment; the original model remains unchanged.

**Satellite-band selection completed:** [nested station-validation experiment](docs/satellite_band_selection.md) keeps the model settings fixed while selecting fewer bands inside training data. On the same 130 unseen-station cases, MAE falls from 1.728 to 1.331 mg/L (23% reduction), but a training-mean baseline remains better at 1.244. Selected checkpoints are separate; the original predictor and WQI categories remain unchanged.

**Broader station demonstration completed:** [30 new real matches](docs/station_expansion.md) expand the saved dataset to 130 observations at seven stations. [Unseen-station evaluation](docs/unseen_station_evaluation.md) holds each station out, purges shared satellite acquisitions, and also tests unchanged saved models on the new stations. Results show limited transfer; existing main models and WQI categories are preserved. Notebook integration is pending.

**Ground-only baseline completed:** the [same-test comparison](docs/ground_only_comparison.md) scores satellite-only, ground-only and combined models on identical 20 examples. Ground-only MAE is 0.248 mg/L, combined 0.317 and satellite-only 0.823. The new checkpoint and results are preserved separately; the main model is unchanged.

**Optional WQI calculation:** [calculator and commands](docs/wqi_calculator.md) combine predicted BOD with measured station parameters. Missing water temperature means current results are explicitly labelled assumption-based scenarios; model training is unchanged. The [20-example comparison](docs/wqi_evaluation.md) is complete, including a simple baseline and temperature sensitivity. WQI results are not yet integrated into the notebook.

Run these commands from the repository root, using standard Python 3.12 or 3.13:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-working-model.txt
.\.venv\Scripts\python.exe scripts/restore_demo_bundle.py
```

If Python 3.12 is not installed, replace `py -3.12` with your installed supported Python. If you already use `.venv-check`, substitute `.venv-check` for `.venv` in the commands. Neither activation, CUDA, PyTorch nor Earth Engine authentication is needed for the saved offline demonstration.

Predict BOD for the 20 saved satellite-input examples:

```powershell
.\.venv\Scripts\python.exe scripts/predict_satellite.py --input outputs/satellite_only_80_20_20261014/example_inputs.csv
```

The script restoring the bundle checks file hashes and refuses to overwrite a different existing file. The saved input examples contain only identifiers and satellite readings—no BOD answers or fecal-coliform measurements.

## Model and data

- **Model:** scikit-learn `MLPRegressor`, one hidden layer of 8 ReLU neurons, using the friend's model design.
- **Inputs:** ten scaled Sentinel-2 surface-reflectance bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11 and B12; the selected model's wrapper calculates five ratios/contrasts internally.
- **Output:** estimated BOD, mg/L; negative model outputs are clipped to zero.
- **Data:** 100 real satellite–CPCB matched events at three Ganga stations: Anupshahar (38), Varanasi (37), and Bijnor (25).
- **Training/testing:** oldest 80 events train; newest 20 test. Sampling/acquisition intervals and satellite acquisition groups do not overlap between folds. Input and target scaling are fitted on training rows only.
- **Hardware:** CPU. Saved with scikit-learn 1.9.1; the setup file pins the main training dependencies.

## Results

| Method | Test cases | Average absolute error, mg/L | RMSE, mg/L | R² |
|---|---:|---:|---:|---:|
| Selected bands plus ratios, alpha=1 | 20 | 0.5911 | 0.7343 | 0.4907 |
| Historical ten-band satellite-only model | 20 | 0.8225 | 1.0669 | -0.0750 |
| Predict the training average | 20 | 1.0054 | 1.0440 | -0.0293 |

The selected ratios-only trial has lower MAE/RMSE than these references on this familiar-station sample. These pilot test observations have been examined repeatedly; they are not a fresh independent final test, and this is not evidence of reliable monitoring or unseen-station transfer. Original benchmarks remain historical results.

## Where to find everything

After restoration:

| File/folder | Contents |
|---|---|
| `configs/working_satellite_model.json` | Current selected checkpoint, hash and limitations |
| `outputs/satellite_seen_refinements_20261014/ratios_only_alpha1.joblib` | Selected bands-plus-ratios estimator and fitted scalers |
| `outputs/satellite_only_80_20_20261014/model.joblib` | Preserved historical ten-band estimator and fitted scalers |
| `outputs/satellite_only_80_20_20261014/example_inputs.csv` | Twenty satellite-only prediction inputs |
| `outputs/satellite_only_80_20_20261014/test_predictions.csv` | Actual BOD, predicted BOD and errors |
| `outputs/satellite_only_80_20_20261014/run.json` | Settings, hashes, versions and limitations |
| `outputs/matches_expansion_100_20261014/paired.csv` | Full 100-event dataset |
| `outputs/fusion_sklearn_expanded_20261014/fusion_model.joblib` | Preserved previous model requiring fecal coliform |
| `outputs/india_additional36_test_20261014/` | Preserved previous-model test on 36 additional events |

The committed copies live under `demo_bundle/files/`; `demo_bundle/manifest.json` records their SHA256 hashes. Original bulk exports and credentials are not included.

## Reproduce training

Use a new output directory to preserve the saved model:

```powershell
.\.venv\Scripts\python.exe scripts/train_satellite_only.py --output outputs/satellite_only_reproduction
```

This uses the same 100-row dataset, chronological 80/20 split, model settings and seed. It refuses to replace an existing output directory. Floating-point results can vary across environments.

## Can I supply river coordinates?

Coordinates alone are not model inputs. A new estimate also needs a date and an appropriate satellite image, from which the ten band values must be extracted using the same processing. The existing Earth Engine extractor requires separate setup and rejects images that fail cloud/water-pixel checks. This branch provides the saved-feature prediction demonstration; it does not provide a one-command arbitrary-coordinate prediction application.

Predictions outside the three tested station areas are experimental. BOD is indirectly inferred from satellite signals. Do not use these estimates as measurements or safety decisions.

Read the [model documentation](docs/working_model.md) for extraction rules, inputs, assumptions and verification. Earlier demonstrations remain documented in the [previous README](docs/previous_demo_readme.md) and [bundle guide](demo_bundle/README.md).

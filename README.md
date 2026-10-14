# River Water Quality Prediction — working model

**Start the presentation here:** [demonstration notebook](notebooks/river_water_quality_demo.ipynb), with executed tables, predictions and seven embedded charts. See [opening and running instructions](notebooks/README.md). It runs offline without retraining.

A small proof of concept that estimates **BOD in mg/L from Sentinel-2 satellite readings**. BOD measures oxygen used while microorganisms break down organic matter. The output is an estimate, not a laboratory measurement or drinking-water safety rating.

This branch, `working-model`, preserves the trained satellite-only model, its 100 matched Indian observations, test predictions, and the earlier ground-plus-satellite model. The current model needs **no fecal-coliform reading** at prediction time.

**Prepared chemical dataset:** all 100 events now have matching CPCB pH and dissolved-oxygen readings in `outputs/india_100_chemical_dataset_20261014/paired.csv`. This separate dataset is ready for the next feature-selection/fusion experiments; saved models retain their existing inputs. See [dataset sources, units and audit](docs/chemical_dataset.md).

**New demonstrations:** a fair comparison uses identical 80/20 splits for satellite-only and combined models, training-only feature selection, and actual SHAP explanations. Average test errors are 0.82 mg/L (satellite only), 0.32 (all ground readings added), and 0.19 (selected B8/B8A/B11 + fecal coliform). These exploratory models are saved separately; the main predictor remains unchanged. See [results, charts and reproduction instructions](docs/straightforward_demonstrations.md).

**Switchable experimental network:** a separate small two-branch PyTorch model now demonstrates a date-gap gate, nonnegative-BOD training penalty, and station-adversarial gradient reversal. Five on/off variants are saved with losses, weights and predictions. The gate helped this sample; the other individual additions did not improve accuracy. See [mechanisms, switches, charts and limitations](docs/experimental_network.md). The working scikit-learn models remain preserved.

## Quick start: Windows PowerShell

Run these commands from the repository root, using standard Python 3.12 or 3.13:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-working-model.txt
.\.venv\Scripts\python.exe scripts/restore_demo_bundle.py
```

If Python 3.12 is not installed, replace `py -3.12` with your installed supported Python. If you already use `.venv-check`, substitute `.venv-check` for `.venv` in the commands. Neither activation, CUDA, PyTorch nor Earth Engine authentication is needed for the saved offline demonstration.

Predict BOD for the 20 saved satellite-input examples:

```powershell
.\.venv\Scripts\python.exe -c "import joblib,pandas as pd; from scripts.train_satellite_only import predict; m=joblib.load('outputs/satellite_only_80_20_20261014/model.joblib'); x=pd.read_csv('outputs/satellite_only_80_20_20261014/example_inputs.csv'); print(pd.DataFrame({'observation_id':x.observation_id,'predicted_bod_mg_l':predict(m,x)}).to_string(index=False))"
```

The script restoring the bundle checks file hashes and refuses to overwrite a different existing file. The saved input examples contain only identifiers and satellite readings—no BOD answers or fecal-coliform measurements.

## Model and data

- **Model:** scikit-learn `MLPRegressor`, one hidden layer of 8 ReLU neurons, using the friend's model design.
- **Inputs:** ten scaled Sentinel-2 surface-reflectance bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11 and B12.
- **Output:** estimated BOD, mg/L; negative model outputs are clipped to zero.
- **Data:** 100 real satellite–CPCB matched events at three Ganga stations: Anupshahar (38), Varanasi (37), and Bijnor (25).
- **Training/testing:** oldest 80 events train; newest 20 test. Sampling/acquisition intervals and satellite acquisition groups do not overlap between folds. Input and target scaling are fitted on training rows only.
- **Hardware:** CPU. Saved with scikit-learn 1.9.1; the setup file pins the main training dependencies.

## Results

| Method | Test cases | Average absolute error, mg/L | RMSE, mg/L | R² |
|---|---:|---:|---:|---:|
| Satellite-only model | 20 | 0.8225 | 1.0669 | -0.0750 |
| Predict the training average | 20 | 1.0054 | 1.0440 | -0.0293 |

Average absolute error is lower than the simple reference, but RMSE is slightly worse and R² is negative. This is a working demonstration with mixed accuracy, not evidence of reliable monitoring. These pilot test observations have been examined in earlier runs; they are not a fresh independent final test. The older combined model used different folds, so its score is not a fair direct comparison.

## Where to find everything

After restoration:

| File/folder | Contents |
|---|---|
| `outputs/satellite_only_80_20_20261014/model.joblib` | Current satellite-only estimator and fitted scalers |
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

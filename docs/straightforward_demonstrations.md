# Fair comparisons, feature selection and SHAP

These demonstrations use the same enriched 100 Indian observations and the already saved 80-training/20-testing assignments. The existing models are preserved; four comparison models are saved separately under `outputs/straightforward_demos_20261014/`.

## Run and inspect

From the repository root, using standard Python 3.12 or 3.13:

```powershell
.\.venv-check\Scripts\python.exe -m pip install -r requirements-demonstrations.txt
.\.venv-check\Scripts\python.exe scripts/restore_demo_bundle.py
```

Open `model_comparison.png`, `feature_ranking.png`, and `shap_examples.png` in the output folder for the visual demonstration. All predictions, errors, ranking values and explanations are also saved as CSVs.

To reproduce without replacing existing results:

```powershell
.\.venv-check\Scripts\python.exe scripts/run_straightforward_demos.py --output outputs/straightforward_reproduction
```

Replace `.venv-check` with your environment name when necessary. No satellite downloads or Earth Engine login are required for this offline experiment. CPU is sufficient. The script refuses to overwrite an existing output directory.

## Fair comparison

All models use one hidden layer of eight ReLU neurons, L-BFGS, alpha=1.0, maximum 3000 iterations, and seed=42. Models share the same events, BOD target, training-only scaler fitting, and zero-clipping rule. Input counts differ, so parameter counts differ; identical settings do not mean identical network sizes. Fecal coliform is log1p-transformed; pH and DO retain their original numeric values before standardization.

| Model | Inputs | Test n | MAE (mg/L) | RMSE (mg/L) | R² |
|---|---|---:|---:|---:|---:|
| Satellite only | Ten bands | 20 | 0.8225 | 1.0669 | -0.0750 |
| Satellite + fecal | Ten bands + fecal coliform | 20 | 0.3268 | 0.4660 | 0.7949 |
| Combined, all chemicals | Ten bands + fecal coliform + pH + DO | 20 | 0.3173 | 0.4336 | 0.8224 |
| Selected features | B8, B8A, B11 + fecal coliform | 20 | 0.1852 | 0.2368 | 0.9470 |
| Training-average reference | One constant prediction | 20 | 1.0054 | 1.0440 | -0.0293 |

The ground readings helped on this sample. Adding pH and DO gives only a small improvement beyond fecal coliform. These results are a fairer comparison than the earlier runs, which used different test sets. The satellite-only reference reproduces the saved working model to numerical tolerance.

## Feature selection

Only the outer 80 training events are passed to selection. They are split chronologically into 60 selection-training and 20 selection-validation observations, with sampling/acquisition intervals and scene groups checked for separation.

Absolute Pearson correlation with BOD ranks satellite and ground inputs separately using the inner 60 rows. Twelve candidate models compare the top 3/5/10 satellite bands with the top 0/1/2/3 ground parameters. Minimum validation MAE selects the winner; fewer inputs and candidate name break ties. The selected inputs are then refitted on all 80 outer training observations before scoring the same 20 test cases. No outer test answers enter ranking, candidate training or model selection.

The selected model retains `satellite_B8`, `satellite_B8A`, `satellite_B11`, and `ground_fecal_coliform`. The leading correlation-ranked ground features are fecal coliform, pH and DO, in that order. Correlation ranking is a simple feature-selection heuristic, not proof of causal importance.

`feature_candidates.csv` contains validation-only scores. `feature_ranking.csv` contains scores and selected flags. `selection_split_assignments.csv` makes the inner split auditable.

## Actual SHAP explanations

The all-chemical combined model is explained using [SHAP KernelExplainer](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html), not the older integrated-gradients routine. The background is a seed-42 sample of 20 training events. The earliest test example per station is selected by date before explanation; three examples are explained with 512 coalition samples, identity link and no L1 feature sparsification (`l1_reg=0`).

SHAP explains the full prediction wrapper, including fecal transformation, fitted scaling, inverse target scaling and clipping. Contributions therefore use mg/L. For each case:

```text
training-background reference + all feature contributions = predicted BOD
```

Red bars increase the estimate relative to the background; blue bars decrease it. `shap_explanations.csv` includes individual contributions and their satellite/ground sums. The saved reconstruction residual was zero to floating-point precision; the script requires residual <= 0.000001 mg/L.

These are approximate, reference-dependent model explanations. Correlated satellite bands and ground variables limit causal interpretation; the contributions do not establish that changing a feature would physically change BOD. Group sums may cancel positive and negative contributions. SHAP is run after model fitting and is not used for test-driven selection.

## Saved outputs and next use

- Four `.joblib` files: satellite-only, satellite-plus-fecal, all-chemical combined, and selected-feature models.
- `metrics.csv` and `test_predictions.csv`: identical-test comparison and all 20 predictions per model.
- Outer and inner split assignments; feature rankings and candidate scores.
- SHAP background IDs/inputs, explained inputs, contribution CSV and charts.
- `run.json`: hashes, versions, model settings, chosen features, warnings and limitations.

To use a saved comparison model in Python:

```python
import joblib
import pandas as pd
from river_water_quality.training.sklearn_demonstrations import predict

model = joblib.load("outputs/straightforward_demos_20261014/selected_features.joblib")
rows = pd.read_csv("outputs/india_100_chemical_dataset_20261014/paired.csv")
print(predict(model, rows))
```

The combined model needs all three ground inputs; the selected model needs fecal coliform. They are not coordinate-only predictors. BOD values in a CSV are ignored by prediction and used only for evaluation. The satellite-only model remains the current saved coordinate/image-based proof-of-concept predictor.

No model was automatically promoted based on these test scores. The test observations have previously been examined, all three sites are on the Ganga, and the sample is small. This demonstrates the approach, not unseen-river reliability, operational forecasting, or water safety. The existing source/time/coordinate/laboratory and water-pixel assumptions still apply.

Verification: 12 focused tests passed, including preparation/bundle checks, no target use at prediction, training-only scaling, rejection of shared acquisition aliases across folds, and ranking independence from inner-validation answers. Saved models reloaded correctly, original model hashes remained unchanged, SHAP contributions reconstructed predictions, and `pip check` reported no broken requirements.

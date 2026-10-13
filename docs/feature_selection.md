# Feature-selection proof of concept

The feature-selection workflow is implemented and executed on the existing real
paired dataset. It ranks the ten satellite bands and checks whether the available
ground predictor, fecal coliform, helps estimate BOD. The BOD answer is excluded
from input features. No new data or synthetic research observations were added.

## How selection works

1. Use the existing chronological split: six training events, five validation
   events and three November test events, with the same seven-day boundary gap.
2. Calculate each feature's absolute Pearson correlation with BOD using **only
   the six training events**. Fecal coliform is log1p-transformed as in the model.
   Constant inputs receive score zero; ties use feature names for stable ordering.
3. Take the top 3, top 5, or all 10 ranked satellite bands. For each set, fit one
   model with fecal coliform and one without it: six predeclared candidates.
4. Fit scaling and neural-network weights on training only, and choose checkpoint
   epochs and the feature set using validation MAE only. Exact score ties favour
   fewer inputs, then the candidate name.
5. Score only the chosen set and the declared all-input reference on test.

Training-only univariate ranking is a small baseline for selection, consistent
with the approach described in [scikit-learn's feature-selection documentation](https://scikit-learn.org/stable/modules/feature_selection.html).
Correlation is association, not causal importance. Many spectral bands are
correlated with one another, so these scores do not prove independent usefulness.

The existing network dimensions are retained for compatibility. Excluded inputs
are set to zero **after standardization** in both training and inference. Those
values cannot affect the prediction; the trained checkpoint stores its selection.
This removes inputs from computation but does not prune network weights. When
ground input is excluded, that candidate is a satellite-only ablation with a
constant ground path, not a claim of active two-modality fusion.

## Actual result

| Satellite inputs | Fecal coliform input | Validation MAE (mg/L) |
|---|---|---:|
| Top 3: B8A, B11, B12 | No | 1.1255 |
| Top 3: B8A, B11, B12 | Yes | 0.2777 |
| Top 5: B7, B8, B8A, B11, B12 | No | 1.0395 |
| Top 5: B7, B8, B8A, B11, B12 | Yes | 0.1860 |
| All 10 bands | No | 1.2351 |
| **All 10 bands** | **Yes** | **0.1750** |

The automatic choice retained **all ten bands and fecal coliform**. A selection
procedure can correctly retain every available feature when reduced candidates
perform worse. No subset was forced after viewing results. The saved model is
equivalent to the original all-input fusion run; no accuracy improvement is claimed.
Its three-case retrospective test MAE is 0.5864 mg/L, the same as the reference.

The training ranking puts fecal coliform first (absolute correlation 0.9938),
followed by B11 (0.5123), B12 (0.4864), and B8A (0.4647). This ranking is based
on only six examples and may reflect station differences. It is not SHAP,
per-prediction explanation, or proof of a physical relationship.

## What a teacher can inspect

In `outputs/feature_selection_001`:

- `feature_ranking.csv`: each available feature, training correlation, score and
  whether it was selected.
- `validation_candidates.csv`: six candidate sets, validation errors, selected
  epochs and exactly one winner.
- `metrics.csv` and `test_predictions.csv`: chosen-model/reference comparisons.
- `model.pt`: weights, training-only preprocessing, and the saved input mask.
- `example_inputs.csv`: required selected inputs, without BOD answers.
- `report.json`: method, splits, original source report, hashes and limitations.

Run predictions from the saved selected-input model:

```powershell
cd "C:\Users\Lakshyajeet\Desktop\clone of dbs project\River-Water-Quality-Prediction"
.\.venv-check\Scripts\python.exe scripts/fusion.py predict --model outputs/feature_selection_001/model.pt --input outputs/feature_selection_001/example_inputs.csv --output outputs/feature_selection_001/my_predictions.csv
```

To reproduce the search into a new directory:

```powershell
.\.venv-check\Scripts\python.exe scripts/select_features.py --output outputs/feature_selection_002
```

Use a new output name when rerunning. The same environment and dependencies as
the fusion demonstration are required; see `fusion_demo.md`. Data/model outputs
are local and ignored by Git. Legacy fusion checkpoints remain supported and
default to all inputs. A masked model requires only its selected input columns
when used through the prediction CLI; excluded columns are not needed.

Verification: all 214 tests passed, including training-only ranking, selection
invariance to changed test answers, excluded-input invariance, mask validation,
legacy checkpoint compatibility and selected-column CLI prediction. The saved
selected-input model reproduces its evaluation output.

## Honest scope

Spectral selection and ground-input inclusion are demonstrated for the available
inputs. There is only **one** ground predictor in this paired dataset, so this
does not demonstrate choosing among pH, dissolved oxygen, COD or other chemical
parameters. That needs overlapping real measurements of those parameters.

The dataset is tiny and the ranking/search may overfit. Its three test events
were already examined in the earlier fusion demonstration, so this is a
retrospective comparison, not a new untouched final evaluation. The code keeps
their values out of ranking, fitting and selection; credible accuracy claims
need new data and a fresh evaluation. Source/QC/time conventions and possible
bridge/bank pixel mixtures retain their previous limitations.

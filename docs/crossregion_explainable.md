# Cross-region explainable evaluation

This demonstration evaluates frozen India-trained models on the four saved real
USGS/Sentinel-2 matches. It adds satellite-versus-ground explanations for a
compatible combined model. Neither model is retrained, calibrated or selected
using US observations.

## Run and inspect

```powershell
.\.venv-check\Scripts\python.exe scripts/restore_demo_bundle.py
.\.venv-check\Scripts\python.exe scripts/evaluate_crossregion_explainable.py
```

Open [the comparison/explanation chart](crossregion_explainable_results/crossregion_explanation.png),
[prediction table](crossregion_explainable_results/test_predictions.csv) and
[modality explanations](crossregion_explainable_results/modality_explanations.csv).
The report also saves training background rows, foreign input rows and source/model
hashes. Existing models and category calculations are preserved.

## Results

Both models were trained on the same 80 India observations. Both are evaluated on
the same 20 India test observations and, separately, the same four US observations.

| Frozen model | India MAE, mg/L (20 cases) | US MAE, mg/L (4 cases) |
|---|---:|---:|
| Selected satellite bands plus ratios | 0.5911 | 2.6533 |
| Satellite plus measured fecal coliform | 0.3268 | 3.8415 |
| India training-mean BOD reference | 1.0054 | 2.0994 |

Both trained models perform worse than the simple training-mean reference in the
US sample. This demonstrates a cross-region test and exposes limited transfer;
it does not demonstrate successful cross-country robustness. Four cases at
three stations are insufficient for broad conclusions. Their measured BOD values
are 2, 3, 9 and 3 mg/L; the rivers are not assumed to be clean.

## What the explanation means

The main satellite model has only satellite inputs. Its predictions cannot have
a ground-input contribution. The saved satellite-plus-fecal model accepts ten
satellite bands and a measured fecal count, so it supports modality comparison.
US pH and dissolved oxygen are absent; the all-chemistry model and the historical
station-average category layer cannot be applied to these US stations.

For each of four US cases and one India test case per station, we calculate exact
two-group Shapley contributions: one player is all satellite bands, the other is
fecal coliform. The reference is a fixed seed-42 sample of 20 India training rows.
We evaluate neither group present, satellite only, ground only, and both present.
Each group's contribution averages its incremental prediction change before and
after the other group is included. This allocates their interaction equally.

`background prediction + satellite contribution + ground contribution = prediction`

Positive contributions raise estimated BOD; negative contributions lower it.
The larger absolute group contribution is labelled the dominant signal for that
example. These are explanatory attributions, not percentages of accuracy, causal
effects or learned attention weights. They depend on the training reference, and
replacing correlated inputs can create unusual input combinations. They differ
from summing an individual-feature SHAP explanation because each modality is
treated as a single player.

## Suitable wording for the project report

“We implemented cross-regional explainable evaluation using four real USGS
observations paired with Sentinel-2 readings. Frozen India-trained satellite-only
and combined models were evaluated against an India training-mean baseline.
Grouped Shapley explanations quantified satellite and ground-input contributions.
Performance declined on the foreign sample, indicating limited transfer in this
small proof of concept.”

The India test set has been repeatedly examined. Ground sampling methods, units,
matching tolerances and water-pixel assumptions need additional verification
before a larger scientific comparison. This report evaluates BOD, not a complete
foreign WQI or safe-water classification.

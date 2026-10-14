# BOD-led water-quality category

This custom classroom demonstration returns **Best, Good or Bad** for a known
station. It uses the saved satellite ratio model to estimate BOD, then combines
that estimate with historical station-average pH, dissolved oxygen and fecal
coliform. These averages use only the original 80 training observations. The 20
test observations never contribute to them. The saved BOD model is not retrained.

## Run

From the repository root, after installing the existing project dependencies:

```powershell
.\.venv-check\Scripts\python.exe scripts/restore_demo_bundle.py
.\.venv-check\Scripts\python.exe scripts/bod_led_category.py --input docs/bod_led_category_results/example_inputs.csv
```

The terminal displays the category and a boundary-sensitivity flag. Supply your
own CSV with `station_id` and the ten scaled surface-reflectance columns
`satellite_B2`, `satellite_B3`, `satellite_B4`, `satellite_B5`, `satellite_B6`,
`satellite_B7`, `satellite_B8`, `satellite_B8A`, `satellite_B11`, `satellite_B12`.
`observation_id` is optional. No current chemical readings or BOD answers are
required. Coordinates alone are insufficient: satellite extraction happens
upstream. Use dates after the station's recorded `training_end_date`.

Add `--output outputs/my_categories.csv` to save details; existing files are
not overwritten. Unknown stations are rejected because they lack a historical
chemistry profile. This version covers the original three stations only.

Reproduce the evaluation and station profiles:

```powershell
.\.venv-check\Scripts\python.exe scripts/bod_led_category.py --demo
```

## Fixed custom formula

The existing documented [four-parameter equations](wqi_method.md) convert each
input to a sub-score. For this custom index only, sub-scores are bounded to
0–100 to handle rounded equation endpoints. The aggregate is:

`0.70 * BOD_subscore + 0.10 * pH_subscore + 0.10 * DO_subscore + 0.10 * fecal_subscore`

Best means score >=80; Good means 60 <= score <80; Bad means score <60.
These weights and thresholds are project choices fixed before this evaluation,
not official WQI weights or independently validated class boundaries. A higher
score is better. Historical chemistry uses arithmetic means, including fecal
coliform; spikes can therefore influence the fecal average.

Oxygen conversion assumes 25 C, 1 atmosphere and zero salinity because measured
water temperature is unavailable. The same assumptions apply to the reference.
“Best” means the highest category of this custom index; it does not establish
safe drinking, swimming, or absence of contamination. High BOD weight can hide
important changes in other pollutants, and historical chemistry is not current
chemistry. These limitations also apply when only category names are displayed.

The boundary flag checks whether changing predicted BOD by +/-0.591148 mg/L
changes its category. This historical mean error is a sensitivity scale, not a
confidence interval or a guaranteed error bound.

## Evaluation

The reference category uses each held-out observation's actual BOD, pH, oxygen
and fecal measurement with this same custom formula. The prediction uses only
satellite-predicted BOD and training station averages. This tests both BOD error
and the loss of information caused by using historical averages.

On the existing 20 examples, **13 categories agree (65%)**. The mean-training-BOD
baseline agrees on 7/20 (35%). Ten predictions are boundary-sensitive. Reference
categories contain 13 Best and 7 Good examples, with **no Bad examples**, so Bad
classification performance cannot be established. A predictor always saying
Best would also agree on 13/20; 65% alone does not demonstrate a strong classifier.

These examples have been examined repeatedly during model development. Results
are exploratory agreement with a custom reference, not independent water-quality
classification accuracy. Full predictions, confusion table, baseline results,
training profiles and source/model hashes are in `bod_led_category_results/`.
The previous WQI calculator and archived notebook retain their original method.

# CPCB-only BOD MVP

The MVP estimates a future BOD reading in mg/L at a known monitoring station from
recent observed BOD history. BOD is the measured column explicitly labelled
`Biochemical Oxygen Demand (mg/L)` in the supplied files. It does not calculate WQI,
use satellite images, or classify water safety. The user explicitly accepted this
smaller scope. No UI, server, SHAP, or advanced models are required.

## What was run

The two original CSVs were parsed with the existing audited adapter. There are
4,538 numeric BOD readings, covering 2018-2021 despite broader filename ranges.
No values were imputed, no extreme positive readings were clipped, and the original
files were preserved. Other chemical parameters are not predictors in this MVP.

Each example uses the prior three readings from the same station, the days until
the forecast date, and that date's month. The three readings must fall within
180 days of the forecast, and the latest must be 1-90 days earlier. These are
explicit operational bounds for this prototype, not scientifically established
forecast horizons. Current/future measurements never enter a forecast's features.

The 3,629 retained examples are split by target date:

| Period | Purpose | Examples |
|---|---|---:|
| 2018-2019 | Fit models | 1,672 |
| 2020 | Select lowest-MAE model | 921 |
| 2021 | Held-out evaluation | 1,036 |

There are 101 stations represented in training. Excluded from forecasting: 424
readings with fewer than three predecessors, 200 with history beyond 180 days,
145 with the latest reading beyond 90 days, and 140 examples from stations absent
from training. Every exclusion is saved. Source-cell exclusions for all four
parameters are separately retained by the adapter.

This is rolling one-step evaluation: after a test reading has actually been
observed, it can be history for the next forecast. Model parameters remain fitted
on training data only. This does not claim to forecast all of 2021 at the end of
2020. It evaluates later dates at familiar stations, not unseen-station performance.
The fit/selection separation follows [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

## Results

| Predictor | 2020 validation MAE (mg/L) | 2021 test MAE (mg/L) |
|---|---:|---:|
| Training mean | 9.6251 | 9.9922 |
| Last observed reading | **1.5049** | **1.1425** |
| Random Forest | 1.7868 | Not evaluated: not selected |

The saved predictor is **last observed reading**. Random Forest did not improve
validation accuracy. We did not tune after seeing test results. The selected
predictor has test RMSE 3.4538 mg/L and R2 0.9356. Pooled R2 can reflect differences
between stations and does not prove detection of pollution changes. MAE is a mean
error, not an uncertainty interval or a guarantee for an individual prediction.

The working historical example is Ganga at Varanasi downstream (Malviya Bridge),
station `cpcb-1c6a3feabae131a4`. Earlier BOD readings are 3.5, 3.8, and 3.7 mg/L;
the prediction for 2021-01-01 is **3.7 mg/L**, 15 days after the latest reading.
That station name comes from the supplied file, not independent location verification.

## Commands

Use the existing `.venv-check` interpreter from the repository root. For a fresh
checkout create a standard Python virtual environment and install
`requirements-ml.txt`; source CSVs are local and are not committed.

Train and evaluate into a new directory:

```powershell
.\.venv-check\Scripts\python.exe scripts/mvp.py train --input data/raw/cpcb/swq_manual_biological_parameters_cpcb_up_1961_2020.csv data/raw/cpcb/swq_biological_parameter_manual_cpcb_up_2021_2025.csv --output outputs/my_mvp
```

Use the saved model and generated historical example (the training command prints
its date; for these files it is 2021-01-01):

```powershell
.\.venv-check\Scripts\python.exe scripts/mvp.py predict --model outputs/my_mvp/model.joblib --history outputs/my_mvp/example_history.csv --date 2021-01-01 --output outputs/my_mvp/prediction.csv
```

The existing `outputs/mvp` run is already trained and its example prediction is
saved as `example_prediction.csv`. Choose a fresh filename to run another prediction.

For actual use, make a history CSV with columns `station_id,sample_date,bod_mg_l`.
Dates must be YYYY-MM-DD and values nonnegative finite mg/L. Include at least
three recent readings per station; the station must be represented in training.
Keep its internal station ID consistent with the saved model. Forecast dates must
be after the validation cutoff and after every supplied history date. Unsupported
stations, duplicate station/date rows, stale history, and malformed values fail
with an explanation. A batch containing invalid history fails as a whole.

The raw files end in 2021; using them alone for a 2026 prediction correctly fails
the stale-history check. New measurements are needed for current predictions.
Load only your own saved joblib artifact.

## Saved output

`model.joblib` stores the selected predictor and trained-station list.
`report.json` records raw-file hashes, model hash, software versions, settings,
selection, limitations and metrics. `metrics.csv` and `test_predictions.csv`
contain evaluation results. `examples.csv` holds feature values and split labels.
`history.csv`, `stations.csv`, `source_exclusions.csv`, and
`forecast_exclusions.csv` preserve inputs and omissions. The generated
`example_history.csv` provides a runnable held-out example.

Original URL, license, laboratory method, source QC and sampling-time semantics
remain unverified. These are provisional results on user-supplied files. Source
verification and evaluation on fresh measurements are needed before operational
reliance. This working MVP does not establish drinking-water safety.

# Phase 3: regression baseline workflow

Phase 3 implements a training/evaluation workflow, not completed scientific
experiments. Real CPCB files and a one-event satellite feasibility match are now
available (see pilot_diagnosis.md), but no research target has been chosen, and no
WQI formula or classification thresholds are invented.
All current model fits are temporary, labelled software tests.

## What is implemented

- A training-mean reference, Ridge regression, and Random Forest regression.
- Satellite-only and ground-only models on the same complete-case cohort.
- Explicit target/predictor/unit declarations and long-to-wide feature preparation.
- Station holdouts or buffered chronological train/validation/test partitions.
- Training-only scaling, validation-only selection, and held-out test predictions.
- MAE, RMSE, and R2, with undefined constant-target R2 left blank.
- Traceable reports, exclusions, saved models, and refusal to overwrite a run.

The ground-only baseline estimates one parameter from other measurements of the
same event. It does not demonstrate prediction at stations with no ground data.
The satellite-only baseline uses only the declared bands; station IDs, coordinates,
dates, quality flags, and the target are not numeric predictors. No fusion, neural
network, feature search, SHAP, or domain adaptation is implemented in this phase.

## Install and test

From the repository root using the existing Windows environment:

```powershell
.\.venv-check\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv-check\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv-check\Scripts\python.exe scripts/train_baselines.py --help
```

For a new environment follow the README virtual-environment steps and install
`requirements-ml.txt` instead of the foundation-only requirements. The optional
`ml` extra adds scikit-learn and joblib; their dependencies include SciPy. No deep
learning framework is required. Foundation-only installations skip Phase 3 ML
tests; CI installs both `test` and `ml` extras to exercise all phases.

## Before a research run

1. Obtain the real CPCB file and review its columns, dates, station coordinates,
   parameters, units, measurement methods, quality flags, and coverage. A WQI file
   is not needed to regress an available measured parameter.
2. Choose a measured target based on actual coverage and scientific suitability.
   Set `target.task: regression`, `target.column` to its exact parameter code,
   and `target.definition_source` to the reviewed definition/reference. Here
   `column` identifies a long-format ground parameter, not the generic value column.
   References must be reviewed manually; the code checks presence, not validity.
3. Set `baseline.target_unit`, `ground_units` and `satellite_units` dictionaries
   mapping exact feature codes to verified unit strings. All units must agree with
   both the input rows and original source provenance. Do not include the target,
   aliases of it, or parameters used to derive it among ground predictors. Derived
   leakage cannot be inferred automatically from names; review the source definition.
4. Use the Phase 1/2 CLI tools to produce a verified local matching run. Inspect
   quality settings, exclusions, dates, products, and source provenance before
   interpreting any fit. The separate bounded Earth Engine pilot is documented
   in cpcb_pilot.md and pilot_diagnosis.md.
5. Choose explicit folds and `baseline.data_purpose: research` only for real,
   reviewed inputs. `software_test` labels every metric/prediction/model artifact
   as software testing, not scientific evidence. Purpose is a user declaration,
   not proof that data is real. Never relabel generated fixtures as research data.

If a future WQI target is selected, it must be supplied externally with a reviewed
formula, provenance, and justified units. This phase does not calculate WQI. Inputs
used to calculate that target must not become circular predictors.

## Split policies

`baseline.split.mode: station_holdout` requires non-empty disjoint lists named
`train_stations`, `validation_stations`, and `test_stations`. IDs remain text,
including leading zeros. Events at unassigned stations are logged as excluded.
Station holdout does not itself establish river/region/country generalization;
station aliases, nearby locations, and spatial dependence need source-specific review.

`baseline.split.mode: chronological` requires `train_end_date`,
`validation_end_date`, and a nonnegative `gap_days`. Dates are calendar dates in
the matching run's configured timezone. For every event consider the interval
covering both its ground sample date and satellite local date:

- Train: the entire interval ends on/before train_end_date.
- Validation: the interval starts strictly after train_end_date + gap_days and
  ends on/before validation_end_date.
- Test: the interval starts strictly after validation_end_date + gap_days.
- Otherwise: exclude and log temporal_boundary_or_gap.

Every retained fold needs at least two complete events. This is a software minimum,
not a claim of adequate statistical power. Replicates on a station/date stay in
the same chronological fold. Scene IDs cannot cross retained folds in either
policy. Recognized Sentinel-2 catalog IDs also share a group when their sensing
timestamps match, across tile IDs and processing versions (bare or full asset
paths). This blocks overlapping-tile aliases that exact scene IDs miss. Other
providers retain exact-ID checks and need provider-specific acquisition review.
This conservative guard can reject station holdouts where a scene covers
several stations; revise the evaluation grouping, not the scene IDs. Wider seasonal,
event, catchment, and spatial correlations still need scientific evaluation design.
There is no random row-level split option.

Chronological evaluation is not necessarily prospective prediction: retrospective
matching permits future imagery within an event interval, and same-event ground
predictors may not be available at deployment. Review the prediction time, modality
availability, and matching direction before making forecasting claims.

## Fit and evaluate

Set the target/baseline settings in `configs/base.yaml` or a separate study config.
The scaffold deliberately leaves target, purpose, predictors, and split choices
unset. Fixed algorithm defaults are Ridge alpha 1, 100 forest trees, and minimum
leaf size 2; these are starting software settings, not validated optimal values.

```powershell
.\.venv-check\Scripts\python.exe scripts/train_baselines.py data/interim/matching_run_001 --config configs/base.yaml --output outputs/baseline_run_001
```

Paths are examples; no such real inputs are included. The command requires
`matches.csv`, `ground.csv`, `satellite.csv`, and `run.json` from Phase 2. It checks
link/metadata consistency, quality flags, declared units, and source metadata/hashes.
New hashes fingerprint the actual matching files read for this baseline run;
original raw source hashes/provenance are retained in the report. Hashes provide
traceability, not independent proof of scientific authenticity.

Events lacking the target or any configured predictor are logged in
cohort_exclusions.csv. Both models retain the same events; no imputation or silent
missing-row removal occurs. Complete-case selection can bias coverage and must be
reported. If no complete events or insufficient fold sizes remain, the command
fails rather than manufacturing observations or relaxing rules.

Ridge's StandardScaler is fitted in a pipeline on training data only. The forest
uses a fixed random seed and one worker. Both candidates are fitted on training
only and scored on validation MAE. The lower-MAE candidate per modality is selected
(alphabetical name breaks exact ties). Only those selected candidates and the mean
reference are scored on test; no validation refit occurs. Do not revise parameters
after inspecting test performance and then report that same test as untouched.
These practices follow [scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

## Output artifacts

| File | Contents |
|---|---|
| cohort_exclusions.csv | Events missing targets or required predictors |
| split_assignments.csv | Every retained-cohort event, fold/exclusion reason, dates, station, scene |
| metrics.csv | Validation candidates and selected-model/reference test metrics, counts, purpose labels |
| test_predictions.csv | Actual/predicted test targets by event and modality, purpose labels |
| models.joblib | Selected fitted estimators, predictor order, target/unit, purpose label |
| run.json | Configuration, target, seed, feature lists, selection policy, input hashes, source report, versions |

Output must be a new directory. Computation completes before persistence; a write
failure can still leave a partial run, so inspect it and use a new path. Generated
outputs and joblib artifacts are ignored by Git. Load only your own reviewed model
artifacts; joblib uses Python object serialization. The saved feature order must be
used for later inference.

## Next action

Review the Phase 3 software and inspect your real CPCB input. Select a justified
target, acquire corresponding features, validate/match them, and establish genuine
baseline results. Phase 4 fusion should be assessed against those results after
review, rather than treating software-test fits as research progress.

# Assumption-based WQI calculator (step 3)

The existing BOD model is preserved. A downstream calculation combines predicted BOD with measured pH, dissolved oxygen and fecal coliform, using the [selected four-parameter index](wqi_method.md).

The [input audit](wqi_input_audit/README.md) found no matched measured water temperatures. Therefore the CLI requires explicit assumed temperature, atmospheric pressure and salinity. No default conditions are silently substituted. Outputs say **assumption-based WQI scenario**. The assumptions describe a classroom scenario, not observed conditions at the river.

## Run the saved single-observation example

From the repository root, restore the existing model/data if necessary:

```powershell
.\.venv-check\Scripts\python.exe scripts/restore_demo_bundle.py
.\.venv-check\Scripts\python.exe scripts/calculate_wqi.py --demo --assumed-temperatures 20 25 30 --assumed-pressure-atm 1 --assumed-salinity-ppt 0 --output outputs/wqi_calculator_demo
```

Choose a fresh output directory for each run. The command selects one existing test observation and runs the unchanged saved satellite-only model. It does not train, calculate reference WQI, evaluate accuracy or alter the notebook.

Preserved demonstration files: [inputs](wqi_calculator_example/inputs.csv), [scenario results](wqi_calculator_example/wqi_scenarios.csv), [provenance](wqi_calculator_example/provenance.json).

Example: Varanasi, 2021-09-15. Predicted BOD 2.21836 mg/L; measured pH 7.69, DO 6.2 mg/L, fecal 9,200 MPN/100 mL. All scenarios use assumed pressure 1 atmosphere and salinity 0 parts per thousand.

| Assumed water temperature | Estimated scenario WQI | Index category |
|---|---:|---|
| 20 C | 58.4303 | Good |
| 25 C | 60.9114 | Good |
| 30 C | 63.4483 | Good-Excellent |

The category changes with the assumption. This is a sensitivity illustration, not an uncertainty interval, proof of accuracy or evidence that the river was safe. If showing only the 25 C result, show the other two scenarios alongside it rather than selecting the most favourable category.

## Your own input CSV

Required columns:

- `predicted_bod_mg_l`: output from your saved BOD model.
- `ground_ph`: measured, dimensionless.
- `ground_dissolved_oxygen`: measured mg/L.
- `ground_fecal_coliform`: reported MPN/100 mL estimate, used as concentration for this demonstration.

Optional identifiers: `observation_id`, `station_name`, `sample_date`.

```powershell
.\.venv-check\Scripts\python.exe scripts/calculate_wqi.py --input my_predictions_with_ground.csv --assumed-temperatures 20 25 30 --assumed-pressure-atm 1 --assumed-salinity-ppt 0 --output outputs/my_wqi_scenarios
```

Outputs include saved inputs, scores/categories, all four sub-indices and weighted contributions, oxygen solubility/saturation, assumptions, per-row status/message and provenance. `invalid_input` means no score or category. `formula_warning` retains rounded-source-equation anomalies for inspection. Categories use unrounded scores. DO saturation beyond 140% is unsupported by the selected index and is not silently capped; this can occur in warm scenarios for some cohort records.

## Oxygen method and measured-condition support

Implemented Benson-Krause equilibrium oxygen equations 7-11 from [USGS memorandum QW 2011.03](https://water.usgs.gov/admin/memo/QW/qw11.03.pdf). Supported temperature 0-40 C, pressure 0.5-1.1 atmosphere and salinity 0-40 parts per thousand. Percent saturation equals measured DO divided by calculated equilibrium oxygen, multiplied by 100. No online calls are required to use the calculator.

Library `river_water_quality.wqi.calculate_wqi` accepts supplied DO percentage saturation directly. It can therefore use measured saturation when available. `oxygen_percent_saturation` can use supplied measured temperature/pressure/salinity, but the caller must record their provenance. The explicitly named `calculate_scenario_wqi` wrapper and CLI always label conditions as assumed.

Source tables and boundary conventions are recorded in the method document. MPN concentration is an estimate, not an exact colony count. This implementation is not a full nine-parameter NSF WQI, a drinking-water compliance decision or a model independently predicting WQI from satellite imagery alone.

## Verification and remaining steps

Focused tests check independent rounded oxygen reference points, physical pressure/salinity effects, a hand-calculated WQI, BOD substitution effects, category boundaries, missing/invalid inputs and visible source anomalies. Existing model/data hashes are checked for the saved demo.

Step 4 is complete: [20-example scenario comparison](wqi_evaluation.md), including a training-mean BOD baseline and explicit assumptions. Step 5 remains: add those results to the notebook.

# River Water Quality Prediction: bare-bones MVP

**Dual-branch fusion is now runnable:** a small PyTorch model combines ten
satellite band features with fecal coliform to estimate BOD. The current real-data
proof of concept has 6 training, 5 validation and 3 test events. See
[fusion architecture, results and prediction command](docs/fusion_demo.md).
It demonstrates the second report functionality; the small test set does not
establish reliable accuracy. The CPCB-only forecasting MVP below is separate.

**Satellite-ground functionality is now demonstrated on three stations:** 18
sampling events attempted, 14 paired successfully with ten Sentinel-2 bands,
four rejected by cloud checks. Open `outputs/hybrid_demo_001/paired.csv` locally.
See [hybridization demonstration and reproducible commands](docs/hybrid_demo.md).
This adds the first report functionality alongside the CPCB-only MVP below.

The current MVP estimates **BOD (mg/L) at a known monitoring station on a future
date**, using that station's recent CPCB BOD readings. It runs locally with two
commands: train and predict. No Earth Engine account is needed for this MVP.

Training compares Random Forest with the last reading and a training-mean
reference. The supplied data selected **last reading** on validation error:
Random Forest did not improve on that simple forecast. The held-out 2021 result
is **MAE 1.14 mg/L over 1,036 eligible readings**. This evaluates known stations
with recent observed history; it does not demonstrate prediction at unmonitored
locations or present-day accuracy.

## Run the MVP

From the repository root in PowerShell, using the existing environment:

```powershell
.\.venv-check\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv-check\Scripts\python.exe scripts/mvp.py train --input data/raw/cpcb/swq_manual_biological_parameters_cpcb_up_1961_2020.csv data/raw/cpcb/swq_biological_parameter_manual_cpcb_up_2021_2025.csv --output outputs/my_mvp
.\.venv-check\Scripts\python.exe scripts/mvp.py predict --model outputs/my_mvp/model.joblib --history outputs/my_mvp/example_history.csv --date 2021-01-01 --output outputs/my_mvp/prediction.csv
```

The example date above is for the current supplied files. For different inputs,
use the example date printed after training. The generated example is a historical
held-out prediction, not a forecast for today. Choose new output paths on reruns.
An already trained local run is available in `outputs/mvp`; its model, reports,
and example inputs are ignored by Git along with source datasets.

For your own predictions, provide a CSV with `station_id,sample_date,bod_mg_l`.
Use station IDs from the saved model's training stations (listed in report/model
artifacts and represented in examples.csv); keep each station's latest three
readings within 180 days of the forecast and its latest reading within 1-90 days.
All history dates must precede the forecast. The 2021 files are too old to predict
today without fresh measurements. See [MVP usage and results](docs/mvp.md).

## Scope

Current implemented scope includes CPCB-only data preparation, chronological
evaluation, a saved forecast model, CSV predictions, the small real-data
satellite-ground hybridization demonstration, and the two-branch fusion model.
Attention, SHAP, physics constraints and other advanced research features
remain outstanding. The older research plan below records prior direction and is not
the current MVP specification.

## Archived research plan and foundation documentation

This project develops a multimodal AI system for **river water quality prediction** by combining **Sentinel-2 satellite imagery** with **ground-station water-quality measurements**. The system is designed to estimate water quality in river regions where regular physical monitoring stations may be unavailable.

The project focuses on **multimodal fusion, explainable AI, physics-informed learning, and cross-regional generalization**.

## Phase 0 MVP foundation: setup and usage

The foundation provides configuration loading/validation, non-mutating
observation-table validation, and Python/NumPy seeding. Phase 1 adds local CSV
ingestion with provenance (see below); no real datasets have been supplied or
verified. No research models have been trained. Phase 3 below provides optional
baseline training tooling; no WQI calculation or complete research pipeline exists.
The architecture and functionalities described below are planned research work.

### Prerequisites and Windows PowerShell setup

Use standard CPython 3.10–3.13 with pip, preferably Python 3.12. Run from the
repository root. Select that interpreter in VS Code before creating an environment;
an MSYS2/MinGW Python may not support the standard Windows dependency wheels.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
python scripts/validate_environment.py
```

Activation is optional: use `.\.venv\Scripts\python.exe` in place of `python`
if PowerShell prevents activation. `requirements.txt` installs the existing package
and its test extra; dependency versions remain centralized in `pyproject.toml`.
Runtime dependencies are NumPy (seeding), pandas (table validation), and PyYAML
(configuration). pytest is for software tests. No deep-learning frameworks are needed.

### Configuration and reproducibility

`configs/base.yaml` is the central configuration. Use
`river_water_quality.utils.config.load_config(path)` for validated project settings;
`load_yaml(path)` remains available for generic YAML. Relative dataset paths are
interpreted from the repository root by future consumers. Validation does not create
directories or require datasets to exist. `target.column` and
`target.definition_source` remain null until a documented scientific definition is
selected. Setting a target column requires a source reference; this check does not
verify the scientific validity of that source. No WQI formula or class thresholds
are supplied.

Call `river_water_quality.utils.seed.set_seed(seed)` for Python and NumPy's legacy
generator. Pass the seed explicitly to `numpy.random.default_rng(seed)` when using
that API. The deterministic flag records intent; hardware/framework determinism is
not implemented. Future experiment runs must save the actual configuration, seed,
dependency versions, dataset provenance, preprocessing decisions, and station/time/
region split definitions alongside outputs. Never fit preprocessing on held-out data.

### Data layout and adding real datasets

```text
data/
  raw/           # Original CPCB, Sentinel-2, or USGS files; preserve unchanged
  interim/       # Intermediate cleaning/alignment outputs
  processed/     # Validated derived tables for later experiments
  external/      # External/reference sources
```

Place legitimately obtained data in the appropriate ignored directory manually;
the project does not download data. Record its source URL, access date, license,
station identifiers, coordinate system, timestamps/timezone, units, quality flags,
and missing-value conventions in `docs/datasets.md`. Document any conversion or
preprocessing assumptions before generating derived files. Do not put credentials
in committed configuration or commit observations/generated artifacts.

`river_water_quality.data.validation.validate_observations` accepts a pandas table
and explicit `required_columns`, `column_types` (numeric/string/datetime/boolean),
and `observation_key`. Required/key fields cannot be null or blank. Empty tables,
duplicate column names, and invalid types raise errors. Duplicate observations are
rejected by default; `duplicate_policy="allow"` explicitly permits them without
removing rows. Validation never fills, converts, or silently drops observations.
Choose keys from the real schema; no universal station/date schema is assumed yet.
Scientific ranges, units, infinity checks, and source-specific quality rules belong
to the next phase. Test fixtures are synthetic software inputs, not research evidence.

Phase 1 implements the internal contracts described below; actual source-file
compatibility still requires verification. Subsequent phases cover matching, baselines,
fusion, and independent evaluation of advanced components as tracked in `ROADMAP.md`.

## Phase 1: local data interfaces

Implemented: internal long-format ground-measurement and satellite-feature schemas,
local UTF-8 CSV ingestion with explicit column mappings, mandatory JSON provenance,
unit consistency, strict numeric conversion, coordinate/date/quality-metadata checks,
duplicate rejection, and source/manifest hashes. See
[data contracts and exact validation commands](docs/data_contracts.md).

Run `python scripts/validate_dataset.py --help` for the local validation CLI.
All fixtures live in software tests and are labelled synthetic. No real data is
included; provider-specific mappings and scientific suitability are still unverified.
No WQI formulas or scientific research results are supplied.

## Phase 2: local satellite–ground matching

Implemented exact-station matching with coordinate-distance checks, timezone-aware
calendar-day offsets, configurable primary/fallback windows, cloud/water/band and
ground quality requirements, deterministic selection, and exclusion/candidate logs.
A CLI writes separate matched tables and a configuration/provenance/hash report
to a new directory. See [matching rules and commands](docs/matching.md).

Set the `matching` configuration explicitly; nulls/empty lists prevent unreviewed
thresholds. This consumes existing feature CSVs, not imagery. Retrospective matching
allows future dates; prior-dates-only mode excludes same-day scenes with unknown
sampling times. Shared scenes require grouping in future splits. Real inputs remain
unverified; no trained research models or experimental results are provided.

## Phase 3: regression baselines

Implemented mean-reference, Ridge, and Random Forest regressors for ground-only
and satellite-only inputs, using explicit target/predictor units and a shared
complete-case cohort. Training uses station holdouts or buffered chronological
splits, rejects scenes crossing folds, fits scaling on training only, and selects
models on validation MAE before held-out testing. Reports include exclusions,
predictions, fitted model artifacts, configuration, source hashes, and purpose labels.

```powershell
.\.venv-check\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv-check\Scripts\python.exe -m pytest -p no:cacheprovider
.\.venv-check\Scripts\python.exe scripts/train_baselines.py --help
```

scikit-learn and joblib are optional dependencies; CI includes them. The default
target, predictors, data purpose, and splits remain unresolved, so training refuses
to guess. No real-data model or metrics are supplied. A first measured-parameter
target can be selected after inspecting the actual CPCB columns; WQI files are not
required for that task. See [baseline setup and limitations](docs/baselines.md).

## Planned research workflow

Optional satellite-access setup is documented in [Earth Engine setup](docs/earth_engine.md).
`scripts/test_earth_engine.py` performs a bounded metadata check; interactive login
requires the explicit `--authenticate` flag and a registered Cloud project ID.
The bounded [CPCB preparation and Sentinel-2 pilot](docs/cpcb_pilot.md) is implemented
and has been run against the user-supplied files. It is not a full research dataset
or validated operational extraction pipeline.

**Satellite Imagery + Ground Monitoring Data**
→ Data Hybridization & Alignment
→ Satellite Feature Extraction + Chemical Feature Processing
→ Multimodal Fusion
→ Water Quality Prediction
→ Explainability & Feature Analysis
→ Cross-Regional Evaluation

## Final Functionalities

### 1. Satellite–Ground Station Data Hybridization

Align Sentinel-2 satellite imagery with CPCB ground-station water-quality measurements using spatial and temporal information. USGS water-quality data is additionally incorporated for cross-regional evaluation.

### 2. Dual-Branch Multimodal Fusion Architecture

A dedicated AI branch processes satellite spectral information while another processes ground-station chemical measurements. The two representations are combined using a learned fusion mechanism to produce the final prediction.

### 3. Spectral and Chemical Feature Selection

Automatically identify the most useful satellite spectral bands and ground-measured chemical parameters for predicting river water quality.

### 4. Cross-Regional Explainable Evaluation

Evaluate the model beyond the Indian training region and analyze how well it generalizes to different river systems.

### 5. Dynamic Temporal Attention Gating

Handle differences between the timestamps of satellite observations and ground-station measurements by dynamically determining how much each modality should contribute.

### 6. SHAP-Based Explainability

Use SHAP-based analysis to identify which satellite and chemical features have the greatest influence on individual predictions.

### 7. Physics-Informed Neural Network

Incorporate basic river and water-quality constraints into the learning process to improve the reliability and physical plausibility of predictions.

### 8. Domain-Adversarial Neural Network

Improve generalization between different river systems and regions by reducing domain-specific bias in the learned representations.

## Novel Contributions

The project specifically focuses on three novel aspects:

* **Satellite–Ground–USGS Dataset Hybridization** — a unified pipeline combining Sentinel-2 imagery, CPCB ground measurements, and USGS data for cross-regional analysis.
* **Dual-Branch Multimodal Fusion** — jointly learning from satellite spectral information and ground-based chemical measurements.
* **Cross-Regional India-to-Foreign River Evaluation** — training on Indian river data and evaluating on a foreign river system to study generalization.

## What We Need to Build

1. Collect and preprocess Sentinel-2 satellite imagery.
2. Collect and preprocess CPCB ground-station measurements.
3. Obtain and process the selected USGS river dataset.
4. Perform spatial and temporal matching between satellite and ground observations.
5. Build the satellite-processing branch.
6. Build the chemical/tabular-processing branch.
7. Implement multimodal fusion with temporal attention gating.
8. Implement spectral and chemical feature selection.
9. Add the physics-informed learning component.
10. Add the domain-adversarial component for cross-region generalization.
11. Add SHAP-based explainability.
12. Train and validate the complete model.
13. Compare against satellite-only and ground-station-only baselines.
14. Perform cross-regional evaluation.
15. Analyze accuracy, generalization, modality contribution, and important features.
16. Produce an evaluation notebook and results suitable for a **research paper**.

## Expected Output

The final project will produce:

* A trained multimodal water-quality prediction model
* A complete satellite + ground-station data pipeline
* Cross-regional evaluation results
* Explainable predictions showing influential features/modalities
* Comparative baseline results
* An evaluation/demo notebook
* A **research paper on cross-regional satellite–ground-station fusion for river water-quality estimation**

## Repository Structure

```text
├── configs/                  # Reproducible experiment configurations
├── data/
│   ├── raw/                  # Real raw datasets; not committed
│   ├── processed/            # Derived datasets; not committed
│   └── external/             # External/reference data; not committed
├── docs/                     # Dataset and prior-work documentation
├── experiments/              # Experiment scripts/results metadata
├── notebooks/                # Exploratory analysis and final evaluation notebooks
├── src/
│   ├── data/                 # Dataset interfaces and loaders
│   ├── preprocessing/        # Data cleaning and hybridization
│   ├── models/               # Baselines and multimodal models
│   ├── training/             # Training pipelines
│   ├── evaluation/           # Metrics and evaluation
│   ├── explainability/       # SHAP and related analysis
│   └── river_water_quality/  # Shared Python utilities/package
└── tests/                    # Automated tests
```

## Development Status

### Phase 0 — Repository Scaffolding

- [x] Project package configuration
- [x] Directory structure
- [x] YAML configuration system
- [x] Reproducibility seed utility
- [x] Dataset documentation scaffold
- [x] Prior-work/novelty tracking scaffold
- [x] Initial automated smoke tests
- [x] GitHub Actions test workflow
- [ ] Real dataset integration
- [x] Local data ingestion and matching software
- [x] Regression baseline training/evaluation software
- [ ] Real-data model training and scientific evaluation

Later phases add multimodal fusion, temporal attention, feature selection,
physics-informed learning, domain-adversarial learning, explainability, and
cross-regional evaluation after verified data and baseline experiments.

## Research Integrity

Synthetic data may be used for software/unit testing only. It must never be presented as experimental evidence. Real dataset sources, preprocessing decisions, splits, seeds, hyperparameters, and evaluation procedures will be documented as the project develops.

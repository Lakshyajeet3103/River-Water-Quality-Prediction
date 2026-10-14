# River Water Quality Prediction

**Multimodal River Water-Quality Estimation Using Satellite and Ground-Station Data**

This project investigates the use of satellite-derived spectral information and ground-based water-quality measurements for river water-quality estimation. It combines reproducible data-validation software, satellite–ground matching, regression-baseline tooling, and an exploratory machine-learning fusion experiment.

The repository is a research prototype. Its implemented components, experimental results, limitations, and planned extensions are documented separately to avoid confusing software functionality with scientifically validated performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Current Status](#current-status)
- [Implemented Components](#implemented-components)
- [Fusion Experiment](#fusion-experiment)
- [Reported Results](#reported-results)
- [Installation](#installation)
- [Running the Tests](#running-the-tests)
- [Reproducing the Fusion Experiment](#reproducing-the-fusion-experiment)
- [Data Management](#data-management)
- [Research Workflow](#research-workflow)
- [Planned Extensions](#planned-extensions)
- [Limitations and Research Guardrails](#limitations-and-research-guardrails)
- [Documentation](#documentation)

## Project Overview

River water quality is commonly assessed using laboratory measurements and monitoring stations. Satellite observations may provide additional spatial information, but satellite-derived features and ground measurements differ in their spatial coverage, observation times, units, and measurement characteristics.

This project explores how these information sources can be validated, aligned, and evaluated for water-quality estimation.

The main research directions are:

1. Satellite–ground data alignment.
2. Ground-only and satellite-only regression baselines.
3. Multimodal fusion of spectral and measured water-quality variables.
4. Reproducible evaluation with explicit data splits and saved predictions.
5. Future investigation of feature selection, explainability, physical constraints, and cross-regional generalization.

**Important distinction:** a working data-processing or training pipeline does not, by itself, establish that the method is scientifically validated or suitable for operational water-quality monitoring.

## Current Status

The repository includes tested data interfaces, local CSV ingestion and validation, satellite–ground matching software, regression-baseline tooling, and an exploratory scikit-learn fusion experiment.

| Component | Status |
|---|---|
| Project configuration and validation | Implemented and tested |
| Local data ingestion and provenance | Implemented; provider compatibility requires verification |
| Satellite–ground matching | Implemented for existing feature tables |
| Mean, Ridge, and Random Forest baseline tooling | Implemented |
| Scikit-learn fusion baseline | Implemented; exploratory results available |
| Automated test suite | 176 tests passed in the documented development run |
| Temporal attention | Planned |
| SHAP-based explainability | Planned |
| Physics-informed learning | Planned |
| Domain-adversarial learning | Planned |
| Comprehensive independent cross-region validation | Not established |
| Operational water-quality monitoring system | Not established |

The 176-test result describes the software test run, not the number of real observations or the strength of the scientific evidence.

## Implemented Components

### 1. Configuration and reproducibility

The project provides configuration loading and validation, observation-table validation, and reproducibility utilities for Python and NumPy.

The central configuration is located at `configs/base.yaml`.

The validation layer checks declared schemas and requirements. It does not automatically prove that a target definition, measurement unit, or source is scientifically appropriate.

### 2. Local data interfaces

The project includes interfaces for long-format ground measurements and satellite-feature tables, local CSV ingestion, explicit column mapping, provenance records, source hashes, and validation checks.

These components help make data transformations traceable and reproducible.

Actual compatibility with each external provider's data exports must be checked before using those files in a scientific experiment.

### 3. Satellite–ground matching

The matching workflow operates on existing ground-observation and satellite-feature tables. It supports configurable spatial and temporal matching rules, quality requirements, deterministic candidate selection, and exclusion reporting.

The workflow does not imply that every ground observation has a usable satellite match. Cloud cover, water-pixel availability, station coordinates, and acquisition timing can all limit the number of paired observations.

See [docs/matching.md](docs/matching.md).

### 4. Regression-baseline tooling

The repository includes mean-reference, Ridge, and Random Forest regression workflows for declared ground-only or satellite-only predictors.

The baseline tooling supports explicit target and predictor declarations, train/validation/test splits, training-only preprocessing, validation-based model selection, held-out evaluation, and traceable output artifacts.

See [docs/baselines.md](docs/baselines.md).

### 5. Scikit-learn fusion baseline

The project includes an exploratory multilayer perceptron (MLP) fusion model in `src/river_water_quality/training/fusion.py`.

The entry point is `scripts/train_fusion.py`. Automated coverage is provided in `tests/test_fusion.py`.

This experiment estimates ground-measured biochemical oxygen demand (BOD) using ten Sentinel-2 spectral bands and a measured fecal-coliform input. It is a paired-event estimation task, not a demonstrated future-date forecasting system.

## Fusion Experiment

### Experimental setup

| Property | Description |
|---|---|
| Study region | Three Ganga monitoring stations |
| Paired observations | 14 events |
| Training set | 6 events |
| Validation set | 5 events |
| Test set | 3 events |
| Satellite inputs | Ten Sentinel-2 spectral bands |
| Ground input | Measured fecal coliform |
| Prediction target | Ground-measured BOD |
| Model | Scikit-learn MLP fusion baseline |
| Evaluation metrics | MAE, RMSE, and R² |
| Reference model | Training-mean baseline |

The experiment uses saved split assignments. Scaling is fitted using training data rather than held-out observations, and the fecal-coliform feature is transformed using `log1p`.

### Reported Results

| Model | Split | Events | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Scikit-learn MLP fusion | Train | 6 | 0.1864 | 0.2408 | 0.9548 |
| Scikit-learn MLP fusion | Validation | 5 | 0.4093 | 0.4779 | 0.8162 |
| Scikit-learn MLP fusion | Test | 3 | 0.6879 | 0.7770 | 0.3157 |
| Training-mean reference | Test | 3 | 0.8500 | 0.9811 | -0.0910 |

Metric units follow the declared BOD target units in the paired dataset.

### Interpretation

On the three-event test set, the fusion model has lower MAE and RMSE than the training-mean reference. This is an encouraging preliminary result, but it is insufficient to establish reliable generalization.

The test sample is extremely small, so a single event can substantially change the metrics. The result should be treated as exploratory rather than as evidence of operational accuracy.

The model also requires measured fecal coliform at inference time. Consequently, this experiment does not establish satellite-only estimation of BOD in locations without ground measurements.

The experiment does not demonstrate prediction at unseen stations, future-date forecasting, or reliable transfer to another river system.

## Installation

### Prerequisites

- Windows, Linux, or macOS with a compatible Python installation.
- Python 3.10–3.13; Python 3.12 is a recommended starting point.
- `pip`.
- Git, if cloning the repository.

Use standard CPython on Windows rather than an incompatible MSYS2/MinGW Python environment.

### Windows PowerShell

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project and test dependencies:

```powershell
python -m pip install -r requirements.txt
```

For machine-learning experiments, install the optional machine-learning dependencies:

```powershell
python -m pip install -r requirements-ml.txt
```

If Python is not available as the `python` command, use the full path to your environment's interpreter. In this repository, the existing environment can be invoked directly with:

```powershell
.\.venv\Scripts\python.exe --version
```

If PowerShell blocks virtual-environment activation, activation is optional; call the interpreter directly.

## Running the Tests

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe -m pytest -q
```

The documented development run completed with **176 passing tests**. Run the suite again after modifying code; the output from your current checkout is authoritative.

Validate the environment with:

```powershell
.\.venv\Scripts\python.exe scripts\validate_environment.py
```

## Reproducing the Fusion Experiment

Run the following commands from the repository root in Windows PowerShell.

First, ensure the optional machine-learning dependencies are installed and the paired dataset and split file exist at the specified paths.

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path

.\.venv\Scripts\python.exe scripts\train_fusion.py `
  --paired artifacts\restored-demo\outputs\hybrid_demo_001\paired.csv `
  --splits artifacts\restored-demo\outputs\fusion_demo_001\split_assignments.csv `
  --output artifacts\restored-demo\outputs\fusion_sklearn_001
```

The command writes the experiment outputs to the selected directory, including:

- Evaluation metrics.
- Test predictions.
- Split assignments.
- Run metadata.
- The serialized fitted model.

The committed demonstration outputs include:

- `artifacts/restored-demo/outputs/fusion_sklearn_001/metrics.csv`
- `artifacts/restored-demo/outputs/fusion_sklearn_001/test_predictions.csv`
- `artifacts/restored-demo/outputs/fusion_sklearn_001/split_assignments.csv`
- `artifacts/restored-demo/outputs/fusion_sklearn_001/run.json`

The serialized model is generated by the training command and may be available locally without being committed to the repository.

The experiment is reproducible only when its input dataset, split assignments, dependencies, and preprocessing assumptions are available and consistent with the documented run.

## Data Management

The intended data layout is:

```text
data/
├── raw/          # Original source files; preserve unchanged
├── interim/      # Intermediate cleaning and alignment outputs
├── processed/    # Validated derived tables
└── external/     # External or reference datasets
```

Place only legitimately obtained datasets in the appropriate directories. The repository does not automatically download all required satellite and ground data.

For each real dataset, document:

- Source URL and access date.
- License or usage restrictions.
- Station identifiers and coordinates.
- Coordinate reference system.
- Observation and acquisition timestamps, including timezone.
- Measurement units and quality flags.
- Missing-value conventions.
- Cleaning and transformation decisions.
- File hashes or other provenance identifiers where appropriate.

Do not commit credentials, private tokens, or unreviewed raw datasets.

See [docs/datasets.md](docs/datasets.md) and [docs/data_contracts.md](docs/data_contracts.md).

## Research Workflow

The overall research workflow is:

1. Obtain and document ground-measurement data.
2. Obtain suitable satellite observations or derived spectral features.
3. Validate schemas, units, timestamps, coordinates, and provenance.
4. Match satellite and ground observations using documented spatial and temporal rules.
5. Define a scientifically justified target and predictor set.
6. Create leakage-aware training, validation, and test splits.
7. Establish simple reference models and single-modality baselines.
8. Evaluate fusion models on held-out data.
9. Inspect errors, exclusions, uncertainty, and failure cases.
10. Test generalization across time, stations, and regions.
11. Document limitations and prepare the research report.

A phase should be considered complete only after implementation, testing, documentation, reproducibility checks, and review.

## Planned Extensions

The following are research directions, not claims that every component is currently implemented or validated.

### Temporal attention

Investigate whether learned weighting of different modalities improves estimation when observation times differ.

### Spectral and chemical feature selection

Determine which spectral bands and ground-measured variables contribute useful predictive information. Compare feature-selection results against simple baselines.

### Physics-informed learning

Introduce constraints only when their scientific justification, applicable units, and assumptions are documented. Measure their effects through ablation studies.

### Domain adaptation and cross-regional evaluation

Investigate transfer between river systems and regions. Such evaluation requires verified external data, compatible targets and units, suitable splits, and clear documentation of distribution shifts.

### Explainability

Investigate SHAP or other explainability methods after a model and evaluation protocol are established. Explanations should not be presented as proof of causality.

### Comprehensive evaluation

Expand the sample size, compare with appropriate baselines, repeat evaluations where possible, analyze uncertainty, and report both successful and unsuccessful experiments.

## Limitations and Research Guardrails

- The fusion experiment contains only 14 paired events and three test events.
- The current fusion result is exploratory and does not establish broad generalization.
- Measured fecal coliform is required as an input to the reported fusion model.
- Same-event estimation must not be described as future forecasting.
- The matching workflow operates on feature tables; it does not itself guarantee successful satellite imagery acquisition or extraction.
- Software tests verify implementation behavior, not scientific validity.
- No operational water-quality index (WQI) should be claimed without a documented formula, units, thresholds, and source.
- Planned attention, physics-informed learning, domain-adversarial learning, and explainability features must be independently implemented and evaluated before being described as completed.
- Never fabricate satellite, CPCB, or USGS observations.
- Keep synthetic software fixtures separate from real experimental observations.
- Prevent spatial and temporal data leakage.
- Fit preprocessing on training data only.
- Record data provenance, configuration, seeds, dependencies, and split definitions.
- Report negative or inconclusive results honestly.
- Do not claim novelty or patentability without documented prior-work review.

## Documentation

Further implementation details are available in:

- [Project roadmap](ROADMAP.md)
- [Project status](PROJECT_STATUS.md)
- [Changelog](CHANGELOG.md)
- [Dataset tracking](docs/datasets.md)
- [Data contracts](docs/data_contracts.md)
- [Satellite–ground matching](docs/matching.md)
- [Regression baselines](docs/baselines.md)
- [Experiment audit](docs/experiment_audit.md)
- [Research decisions](docs/decisions.md)
- [Prior work](docs/prior_work.md)

## Development Principle

**Inspect → Plan → Implement → Test → Review → Document**

The goal is not simply to build a complex model. It is to produce a reproducible and scientifically defensible evaluation whose claims are supported by the available data.
## SHAP-Based Cross-Region Explainability

The project now includes a standalone SHAP explainability experiment for
the saved scikit-learn fusion model.

### Method
- Loads the saved model and preprocessing scalers without retraining.
- Uses six India training observations as the SHAP background reference.
- Explains four US observations using SHAP KernelExplainer.
- Explains ten Sentinel-2 spectral bands and the ground fecal-coliform input.
- Produces per-observation SHAP contributions, global feature importance,
  a feature-importance plot, a contribution heatmap, prediction checks,
  and a JSON experiment report.

### Run the SHAP demo

From the repository root in Windows PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
.\.venv\Scripts\python.exe .\scripts\explain_shap_crossregion.py
```

Required packages include SHAP, matplotlib, NumPy, pandas, and joblib.
Install missing packages in the active Python environment if needed.

Outputs are written to:

`artifacts/restored-demo/outputs/shap_crossregion_001/`

The output directory contains the SHAP contribution CSV, feature-importance
CSV, plots, prediction checks, report, and demo README.

### Interpretation and limitations

The current explainability run uses four US evaluation observations and
six India training observations as its background. The feature rankings
are exploratory and may be sensitive to this small sample. SHAP describes
model behavior; it does not establish causal relationships.

The model requires measured fecal coliform at prediction time. This is
not a satellite-only BOD estimation system. Cross-region transfer remains
an exploratory evaluation and should not be presented as established
operational accuracy.

## Current Project Status

Implemented and verified:
- Sentinel-2 and ground-monitoring data preparation and matching workflows.
- Scikit-learn MLP fusion baseline with saved split assignments.
- Frozen cross-region evaluation and Integrated Gradients experiment artifacts.
- SHAP-based explainability for the saved scikit-learn fusion model.
- Cloud-threshold pilot configurations and comparison documentation.
- Regression test suite: 176 tests passed in the latest recorded run.

Planned advanced features:
- Physics-Informed Neural Network (PINN).
- Dynamic Temporal Attention Gating.
- Domain-Adversarial Neural Network (DANN).

These advanced features are planned, not claimed as implemented by this
status entry. The existing pilot datasets are small, and the cross-region
results remain exploratory.

## Advanced Feature Demonstrations

Three additional college-demo workflows are included in
`scripts/demo_advanced_features.py`. They use reproducible synthetic
data and are intended to demonstrate implementation mechanics, not to
claim validated river-water-quality prediction performance.

### Domain-Adversarial Neural Network (DANN)

The DANN demo uses a feature encoder, a BOD-like regression head, and
a domain classifier connected through a gradient-reversal layer.
It compares a source-only multilayer perceptron with a domain-adversarial
model under a synthetic source/target distribution shift.

### Temporal Attention Gating

The temporal-gating demo has separate satellite and ground-data branches.
Observation-age features inform a learned sigmoid gate that combines
the branch representations. It is compared with ordinary concatenation
fusion. This demonstrates learned modality weighting; it does not prove
that the gate improves predictions on real river observations.

### Dynamic INT8 Quantization

The quantization demo benchmarks supported PyTorch Linear layers using
FP32 and dynamic INT8 quantization. It records state-dictionary size,
CPU inference latency, and maximum output difference.

In the recorded run, the FP32 state dictionary was 8,933 bytes and the
INT8 state dictionary was 12,645 bytes. The recorded FP32 latency was
about 0.087 ms per batch, versus about 0.214 ms for INT8. Thus, this
small model did not benefit in size or speed from quantization.

### Run the demonstrations

Install the optional dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,ml,advanced-demo]"
```

Run the full demonstration:

```powershell
.\.venv\Scripts\python.exe scripts\demo_advanced_features.py
```

The script writes results under `artifacts/advanced_demo_001/`,
including comparison metrics, a plot, a quantization benchmark, a JSON
report, and FP32/INT8 model checkpoints.

### Interpretation and limitations

The advanced demonstrations use synthetic data, not CPCB or USGS
observations. Their metrics must not be interpreted as real-world
accuracy or evidence of cross-region generalization. The temporal gate
and DANN did not outperform their respective baselines in the recorded
run. Quantization did not improve the tiny model's size or CPU latency.

The existing SHAP and Integrated Gradients workflows remain separate
explainability approaches for their applicable saved models and data.
They are not evidence that pixel-level satellite explanations have
been implemented.

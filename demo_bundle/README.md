# Friend testing bundle

This branch includes the small real-data proof of concept and saved models, so a
fresh clone can run predictions and evaluations **without Earth Engine access**.
Original full exports, virtual environments and authentication credentials are
not included. The bundle contains 100 Indian pairs and four archived US pairs, cached
extraction/matching data, models, provenance, exclusions and result reports.

## Current India models

Five separate familiar-station ratio/regularisation/date-filter checkpoints and
all comparisons are saved under `outputs/satellite_seen_refinements_20261014`.
The ratios-only trial helped; joint tuning and tighter-date retraining did not.
See [the refinement report](../docs/satellite_seen_refinements.md). Main models are preserved.

The separate familiar-station band-selection experiment is in
`outputs/satellite_band_selection_seen_20261014`; its selected checkpoint performed
worse than the original model and is retained for audit, not promoted. See
[the familiar-station report](../docs/satellite_band_selection_seen.md).

Training-only satellite-band selection and seven separate outer-fold checkpoints
are saved in `outputs/satellite_band_selection_20261014`. See [the selection
experiment](../docs/satellite_band_selection.md). These do not replace the main predictor.

The separate expanded snapshot is in `outputs/new_station_matches_v2_20261014`:
130 observations across seven stations, including 30 new real matches. Original
and expanded station-holdout studies are in `outputs/unseen_stations_original_20261014`
and `outputs/unseen_stations_expanded_20261014`. Fold checkpoints are experimental;
they do not replace the main model. See [data expansion](../docs/station_expansion.md)
and [unseen-station evaluation](../docs/unseen_station_evaluation.md).

The new ground-only checkpoint and identical-test modality comparison are saved in
`outputs/ground_only_comparison_20261014`. This is a separate three-input MLP;
existing models remain unchanged. See [the comparison report](../docs/ground_only_comparison.md).

The separate switchable two-branch experiments are saved under
`outputs/experimental_network_20261014`. The bundle includes five checkpoints,
training losses, branch-weight curves and test predictions. See
[the experiment guide](../docs/experimental_network.md) for optional CPU PyTorch
setup and on/off commands. Loading the archived charts requires no PyTorch setup.

The enriched chemical dataset is in `outputs/india_100_chemical_dataset_20261014`.
Fair model comparisons, training-only feature selection, four separate comparison
models and actual SHAP charts are in `outputs/straightforward_demos_20261014`.
Install `requirements-demonstrations.txt` to reproduce those experiments; see
[the complete demonstration guide](../docs/straightforward_demonstrations.md).
Restoration alone needs no SHAP installation and restores the ready-made charts.

The previous ground-plus-satellite scikit-learn model is preserved in
`outputs/fusion_sklearn_expanded_20261014/fusion_model.joblib`, along with its
36-additional-case evaluation in `outputs/india_additional36_test_20261014`.
The 100-row dataset is `outputs/matches_expansion_100_20261014/paired.csv`.

A separate satellite-only model is saved in
`outputs/satellite_only_80_20_20261014/model.joblib`. It uses the same 8-neuron
scikit-learn architecture but only ten satellite bands. The oldest 80 rows train
it; the newest 20 test it. Acquisition groups do not cross folds, and scaling uses
training data only. Test MAE is 0.8225 mg/L versus 1.0054 mg/L for the training mean;
RMSE is 1.0669 versus 1.0440, so the result is mixed. These are previously examined
pilot observations, not an untouched independent final test. The combined model
used different folds; comparing the two scores is not a controlled comparison.

For only these scikit-learn models, install `requirements-ml.txt`, then run
`python scripts/restore_demo_bundle.py`. To reproduce satellite-only training
without replacing saved results:

```powershell
python scripts/train_satellite_only.py --output outputs/satellite_only_reproduction
```

`example_inputs.csv` contains the 20 satellite-only input rows, with no ground
readings or BOD answers. To predict in Python:

```python
import joblib
import pandas as pd
from scripts.train_satellite_only import predict

model = joblib.load("outputs/satellite_only_80_20_20261014/model.joblib")
inputs = pd.read_csv("outputs/satellite_only_80_20_20261014/example_inputs.csv")
print(predict(model, inputs))
```

## Setup on Windows PowerShell

From the cloned repository root, using standard CPython 3.10-3.13 (3.12 recommended):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-fusion.txt
.\.venv\Scripts\python.exe scripts/restore_demo_bundle.py
```

If Python 3.12 is unavailable, use an installed supported standard Python to
create the environment. `python -m venv .venv` works when python points to it.
All commands below use `.venv`; no activation is necessary.

## Test the functionality

Use fresh output paths on subsequent runs:

```powershell
# Rebuild the 14-row Indian paired table from cached satellite features.
.\.venv\Scripts\python.exe scripts/hybrid_demo.py export --prepared data/interim/hybrid_demo_input_001 --extracted data/interim/hybrid_demo_extraction_001 --output outputs/friend_hybrid_check

# Predict using the saved two-branch model. Input CSV contains no BOD answers.
.\.venv\Scripts\python.exe scripts/fusion.py predict --output outputs/friend_fusion_predictions.csv

# Reproduce feature ranking and the six validation-only feature comparisons.
.\.venv\Scripts\python.exe scripts/select_features.py --output outputs/friend_feature_selection

# Evaluate the unchanged India-trained model on four US pairs and explain predictions.
.\.venv\Scripts\python.exe scripts/evaluate_crossregion.py --output outputs/friend_crossregion_check

# Optional: test the software.
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider
```

Fusion predicts approximately 3.5450, 0.6881 and 1.3024 mg/L for the three saved
Indian test examples. Feature selection retains all ten bands plus fecal coliform
on the current validation set. The US comparison has MAE approximately 3.4762 mg/L,
worse than the Indian training-mean reference (2.2250 mg/L). Expect these failures;
they are preserved rather than hidden. Small floating-point/version differences
may affect retraining; saved-model prediction should closely match saved results.

## Data/model checks

`manifest.json` lists every payload file with size and SHA256. The restore command
verifies hashes first and refuses to overwrite different existing files. It may
be rerun safely when restored files remain unchanged.
Git newline conversion is disabled for payload files so their hashes survive
cloning on Windows, Linux and macOS.

```powershell
.\.venv\Scripts\python.exe scripts/restore_demo_bundle.py --verify-only
```

The `.pt` files store weights and plain metadata and are loaded with
`weights_only=True`. Only load the reviewed checkpoints included with this project.
Restored data remains ignored by Git; the deliberate shared snapshot is tracked
under `demo_bundle/files` with narrowly scoped model-file exceptions.

## Interpretation

These are real observations, not synthetic training data. The Indian measurements
come from user-supplied CPCB-labelled exports; original resource identification,
laboratory methods and reuse terms remain incompletely verified. US records are
USGS-origin WQP data; near-time BOD/fecal samples are distinct activities five
minutes apart. Units are matched but full laboratory-method equivalence is not
established. Sentinel-2 quality masks and mixed-pixel limitations are documented.
Use this small snapshot for peer reproduction/review, not accuracy, safety or
cross-region robustness claims. See docs/hybrid_demo.md, docs/fusion_demo.md,
docs/feature_selection.md and docs/crossregion_demo.md for full scope and sources.

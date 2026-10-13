# Friend testing bundle

This branch includes the small real-data proof of concept and saved models, so a
fresh clone can run predictions and evaluations **without Earth Engine access**.
Original full exports, virtual environments and authentication credentials are
not included. The bundle contains 14 Indian pairs and four US pairs, cached
extraction/matching data, models, provenance, exclusions and result reports.

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

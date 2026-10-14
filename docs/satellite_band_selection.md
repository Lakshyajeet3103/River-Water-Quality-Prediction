# Satellite-only band selection

Tested smaller spectral input subsets with the existing model type and settings. No chemical readings, new band ratios, hyperparameter tuning, date filtering or main-model replacement were added.

## Result on identical unseen-station examples

Separate update: [familiar-station testing](satellite_band_selection_seen.md) uses the original 80/20 chronological split. Selection did not help there: MAE 1.0961 versus the original ten-band model's 0.8225 mg/L. The results below apply only to the unseen-station protocol.

| Approach | Test n | MAE, mg/L | RMSE, mg/L | R2 | Station-macro MAE |
|---|---:|---:|---:|---:|---:|
| Training-selected satellite subset | 130 | 1.3315 | 1.5211 | -1.1484 | 0.9865 |
| All ten satellite bands | 130 | 1.7282 | 1.9335 | -2.4710 | 1.2346 |
| Training-fold mean BOD | 130 | 1.2435 | 1.4095 | -0.8446 | 0.8702 |

The selection procedure reduced pooled MAE by **22.96%** relative to all ten bands on exactly the same 130 observations and station folds. This is a relative reduction in measured average error, not a prediction accuracy percentage or confidence interval.

The simple training-mean baseline still has lower error, and all pooled R2 values remain negative. Satellite-only BOD transfer is therefore still weak. Selection helped six station folds and worsened Narora; this is not a universal improvement.

## How selection avoided using outer test answers

1. Hold out one entire station; purge every remaining training observation sharing a Sentinel-2 sensing acquisition with that station.
2. Within the remaining six training stations, hold out each station in turn for internal validation, again purging shared acquisitions from the internal fitting set.
3. Evaluate the same fixed 17 subsets on those six internal folds. Choose minimum mean internal-station MAE. Ties prefer fewer inputs, then candidate name.
4. Fit the chosen subset on all allowed outer training rows. Only then predict the outer held-out station.
5. Repeat for all seven outer stations. Every observation receives one out-of-station prediction; inner selection never receives outer test observations or their BOD labels.

Every fit retains eight hidden ReLU neurons, L-BFGS, alpha 1, seed 42 and max_iter 3000. Input and target standardisation uses only that fit's training rows. Existing nonnegative clipping is retained. Different input counts imply different parameter counts, even with the same hidden-layer/settings.

Candidates are all ten bands, each of the ten single bands, and six fixed groups: visible, red-edge, NIR, SWIR, visible plus red-edge, and visible plus NIR. These are a small predeclared search, not exhaustive testing of all possible subsets. All 714 internal candidate fits completed without captured training warnings. Seven selected checkpoints were reloaded and verified against their predictions.

## Selected subsets by fold

| Outer test station (never used for its selection) | Training-selected subset |
|---|---|
| Mirzapur upstream | B3 |
| Anupshahar | B3 |
| Varanasi downstream | B11, B12 |
| Sangam | B3 |
| Garhmukteshwar | B3 |
| Bijnor | B2 |
| Narora | B8A |

This does not establish one globally best band. Choices differ because each fold has different training stations. The seven saved checkpoints represent the evaluated selection procedure; none is silently promoted to replace the working model. Do not remove input columns from the existing saved ten-band model, which still needs its original inputs.

These previously examined examples are exploratory internal evaluation, not a fresh independent dataset. Station counts are uneven, including just three Mirzapur observations. Dates can overlap across stations, so this is retrospective transfer rather than forecasting. All stations are on the Ganga; the result does not establish foreign-river performance.

## Saved files and reproduction

Results are under `outputs/satellite_band_selection_20261014/`, with copies in [the demo bundle](../demo_bundle/files/outputs/satellite_band_selection_20261014/):

- Seven `*_selected.joblib` checkpoints.
- `selected_subsets.json`: training-only choices and checkpoint hashes.
- `candidate_scores.csv`, `inner_candidate_metrics.csv`: internal selection scores and training diagnostics.
- `inner_assignments.csv`, `outer_assignments.csv`: station separation and shared-acquisition purges.
- `test_predictions.csv`, `station_metrics.csv`, `pooled_metrics.csv`: identical-example comparisons and explicit baseline.
- `band_selection_comparison.png`: visually checked chart.
- `run.json`: settings, candidate definitions, input/output hashes and limitations.

```powershell
.\.venv-check\Scripts\python.exe scripts/restore_demo_bundle.py
.\.venv-check\Scripts\python.exe scripts/select_satellite_bands.py --output outputs/satellite_selection_reproduction
```

Run from the repository root with `requirements-demonstrations.txt`. Output must be a fresh directory. All data are cached; no Earth Engine login/download or GPU is needed. Existing model/data hashes are verified and preserved.

Tests check internal station/acquisition isolation, satellite-only inputs, selection using internal validation and deterministic tie-breaking. Saved assignments were additionally inspected to confirm outer test IDs are absent from every inner selection fold. WQI rules and existing notebook outputs remain unchanged.

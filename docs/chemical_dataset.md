# Consistent 100-event dataset with chemical readings

The dataset is `outputs/india_100_chemical_dataset_20261014/paired.csv`. It extends the previous 100-event Indian cohort with two exact-match CPCB chemical readings per event. It does not change either saved model or the previous dataset.

## Columns and intended use

| Columns | Unit | Role |
|---|---|---|
| `ground_bod` | mg/L | Training/evaluation target; never an input |
| `satellite_B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12` (each prefixed `satellite_`) | Scaled surface reflectance | Satellite predictors |
| `ground_fecal_coliform` | MPN/100mL | Available ground predictor |
| `ground_ph` | Dimensionless pH | New ground predictor |
| `ground_dissolved_oxygen` | mg/L | New ground predictor |
| `ground_total_coliform` | MPN/100mL | Existing optional column, incomplete; excluded from the default predictor list |
| Identifiers, coordinates, dates, scene ID and image-quality columns | As in the previous paired dataset | Matching, splitting and audit metadata |

All 100 events have finite values for the ten satellite bands, BOD, fecal coliform, pH and dissolved oxygen. The added pH values range from 6.8 to 8.7, and dissolved oxygen from 6.2 to 10.8 mg/L. No new rows were added or removed. Every previous column and observation order is preserved.

The current satellite-only model still needs ten satellite inputs. The saved combined model still needs those ten plus fecal coliform. Neither uses the newly added pH/DO values until a separate model is trained with them. They must not be inserted into an existing saved model's input array.

## Sources and matching

Chemical source: [NWIC/CPCB surface-water manual chemical-parameter dataset](https://nwdp.nwic.gov.in/en/dataset/surface-water-quality-manual-chemical-parameters-cpcb).

The locally downloaded exports are `swq_manual_chemical_parameters_cpcb_up_1961_2020.csv` and `swq_chemical_parameter_manual_cpcb_up_2021_2025.csv`. Their actual records cover 2018–2020 and 2021 respectively; filename years are not assumed to describe actual coverage. Original CSV bytes are verified against their download-sidecar SHA256 hashes before processing. Exact download URLs and retrieval times are saved in `provenance.json`.

Join keys are **station name, displayed calendar date, latitude and longitude**, with a one-to-one match. `Data Acquisition Time` is parsed with `%d-%m-%Y %H:%M`. No coordinate rounding, fuzzy matching, averaging, imputation, date shifts or unit conversions are performed. Duplicate keys, unmatched events, missing/non-finite readings, pH outside 0–14 and negative DO fail explicitly.

Source columns `Potential of Hydrogen (pH)` and `Dissolved oxygen (mg/L)` become `ground_ph` and `ground_dissolved_oxygen`. Source timestamps, raw selected values, file names, hashes and one-based CSV data-record numbers are retained in the audit. Biological/satellite source assumptions remain as documented in the previous dataset.

## Files

- `paired.csv`: the enriched 100-event dataset.
- `chemical_audit.csv`: 200 parameter records, one for each added reading, with source links.
- `matched_chemical_source_records.csv`: the 100 selected chemical source records, including displayed timestamps and original parameter column names.
- `provenance.json`: input/output hashes, download metadata, units, join rule and saved-model hashes.

These files are also preserved in `demo_bundle/files/outputs/india_100_chemical_dataset_20261014/`. Restore them with `python scripts/restore_demo_bundle.py`. This deliberately shared subset contains no credentials or full raw exports.

## Reproduce from original local exports

After restoring the previous 100-row data, place the two original chemical CSVs and their `.download.json` sidecars in `data/raw/cpcb_chemical_check_20261014/`, then run from the repository root:

```powershell
.\.venv-check\Scripts\python.exe scripts/prepare_chemical_dataset.py --output outputs/india_chemical_reproduction
```

Use your actual Python environment path. The script refuses to overwrite an existing output directory. The downloaded raw exports are not required to use the saved enriched dataset; they are needed to reproduce the original-source join.

## Interpretation and next use

The matching proves that readings have the same recorded station, date and coordinates. It does not establish that the biological and chemical measurements came from the same physical bottle. Source sampling-time meaning, timezone, CRS and laboratory QC conventions remain provisional. Basic numeric/plausibility checks are not laboratory validation. Full source reuse terms remain to be confirmed.

For later model comparisons, use the saved chronological 80/20 assignments under `outputs/satellite_only_80_20_20261014/split_assignments.csv`. Fit scalers and perform feature selection only within the training observations. Group shared satellite acquisitions together. This step prepares inputs only; it does not retrain models, compute SHAP or implement the other demonstration components.

Verification: all previous columns/order preserved; 100 exact chemical matches; 200 traceable readings; original raw-file hashes checked; saved models preserved; six focused join/error-case checks passed.

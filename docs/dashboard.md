# Professor demonstration dashboard

## Startup on Windows

From the repository root, using Python 3.13 and Node 22.20 or newer:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.venv\Scripts\python.exe scripts/restore_demo_bundle.py
npm.cmd ci --prefix frontend
```

Start Python in one terminal:

```powershell
.venv\Scripts\python.exe -m uvicorn river_water_quality.api:app --host 127.0.0.1 --port 8000
```

Start React in a second terminal:

```powershell
npm.cmd run dev --prefix frontend -- --host 127.0.0.1
```

Open http://127.0.0.1:5173. API documentation is at http://127.0.0.1:8000/docs. Stop each terminal with Ctrl+C. The Vite proxy keeps requests on the same origin; Earth Engine authentication is never required. OpenStreetMap background tiles need an internet connection; the coordinates and spectral records remain available without tiles.

## Demonstration walkthrough

1. Prediction: choose a station and sampling date; dates marked SHAP have individual explanations. Choose each of the three models and press Run prediction. Saved evaluation results are the default. Choose Run saved checkpoint live to execute Python inference with hash-verified checkpoints and the saved preprocessing. Measured BOD is displayed only as the target for comparison.
2. Accuracy: toggle familiar and unseen cohorts. Inspect the exact saved MAE, MAPE, training-mean baseline, scatter plot and observation table; download the cohort CSV. The model selector updates the chart and table. Experiment variants are listed separately under Architecture.
3. Data & matching: review coordinates, ground and satellite dates, signed date offset, band reflectance, backend-derived ratios and extraction quality. The saved bundle contains no real satellite thumbnails or extraction polygons, so the UI explicitly marks them unavailable. The map is OpenStreetMap, never a substituted satellite image.
4. Category: the existing Python formula uses predicted BOD with weight 70% and training-only station-average pH, DO and fecal coliform with weight 10% each. Best >=80, Good >=60, otherwise Bad. The existing sub-index calculation assumes 25 C, 1 atm and 0 ppt salinity. Current measured chemicals are distinct from the category's historical averages. Unseen stations have no saved training chemistry profile, so category is unavailable rather than silently substituting test-data averages. This is a custom demonstration indicator, not an independently validated full WQI. BOD error is separate from category agreement.
5. SHAP: only the selected model's matching observation is shown; prediction reconstruction is checked. Other observations explicitly say unavailable. Contribution sizes are mg/L, with a local feature chart and satellite/ground/date-gap totals. These explain predictions, not accuracy or causality.
6. Oxygen simulation: run the saved conditional PINN using the selected BOD model and initial measured DO, from zero to five days. Both oxygen and remaining demand are plotted beside mathematical reference solutions. Assumed rates and BOD5 conversion are disclosed. Conditions outside the saved training range are marked extrapolation. Oxygen-exhaustion scenarios are rejected. This is solver validation, not validation against future river measurements.
7. Architecture: the BOD model dropdown displays only the selected model?s diagram. Dedicated diagrams show the selected-feature network (4 inputs ? 8 ReLU hidden neurons ? 1 output), the satellite-ratio network (10 bands + 5 derived features ? saved scaling ? 8 ReLU hidden neurons ? 1 output), and the two-branch fusion model. The first two diagrams include saved preprocessing and output conversion; expandable ratio definitions show the exact backend calculations. The fusion diagram distinguishes satellite and chemical branches, the date-aware learned gate, original negative-BOD constraint, station-adversarial training and separate oxygen PINN. Actual gate weights appear from saved familiar-station outputs when available, or from live two-branch inference. The variant table is read-only and does not pretend to modify checkpoints.

## Data and provenance

The API joins 20 familiar test observations and the strict 26 unseen observations to the bundled paired records by observation ID. It reads `docs/saved_models_unseen_results/{metrics,predictions}.csv` directly. `three_model_support.load()` verifies frozen model and source-data hashes for live inference. Seven SHAP examples per model come from `docs/three_model_shap_results`; PINN inference uses the hash-verified `docs/oxygen_pinn_results/oxygen_pinn.pt`. Training-only category profiles are in `docs/bod_led_category_results/station_profiles.csv`.

Prediction responses include model identity, checkpoint version/hash, input requirements, units, actual inputs and result source. Missing artifacts or measurements produce useful errors. Result state resets on observation/model/source changes. Requests for obsolete data views are aborted, preventing mismatched displays.

## Verification

```powershell
.venv\Scripts\python.exe -m pip install -e ".[test]"
.venv\Scripts\python.exe -m pytest tests/test_dashboard_api.py tests/test_three_model_demo_artifacts.py tests/test_bod_category.py tests/test_oxygen_pinn.py -q
npm.cmd run test --prefix frontend
npm.cmd run lint --prefix frontend
npm.cmd run build --prefix frontend
```

The API tests exercise all seven examples with every BOD checkpoint, compare saved and live results, check exact metric recomputation, SHAP alignment/reconstruction, category calculations, gate weights, missing/invalid selection handling, CSV rows, derived ratios and PINN/reference initial conditions. Browser visual QA requires an available browser automation session.

Four frontend interaction tests use fixture responses generated from the real API artifacts. They verify model switching and result reset, cohort metrics/CSV links, unavailable imagery, PINN/reference chart wiring, and error retry. Charts and maps are mocked in these DOM tests; they do not replace browser visual QA. Regenerate fixtures with `.venv\Scripts\python.exe scripts/export_dashboard_test_fixture.py` after changing API responses or frozen artifacts.

Verified in this workspace: 319 Python tests passed, four frontend interaction tests passed, and frontend lint/build passed. Browser automation exposed no browser, so visual QA is pending. The dashboard API and Vite proxy were checked over HTTP.

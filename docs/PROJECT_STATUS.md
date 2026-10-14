# Project Status — River Water Quality Prediction

Last updated: 2026-10-14

## Implemented and verified in this repository

- CPCB preparation, Sentinel-2 extraction/matching workflow, validation, and pilot configurations.
- Scikit-learn MLP fusion baseline and saved train/validation/test split assignments.
- SHAP KernelExplainer script for the saved fusion model (requires the corresponding local model and paired-data artifacts).
- College-demo DANN workflow with a gradient-reversal layer, regression head, and domain classifier.
- College-demo temporal attention gating with separate satellite and ground branches and observation-age inputs.
- Dynamic INT8 quantization benchmark for supported PyTorch Linear layers.
- Automated tests for the advanced demo components.

## Run the advanced demo

Install the optional dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,ml,advanced-demo]"
```

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\demo_advanced_features.py
```

Quick smoke run:

```powershell
.\.venv\Scripts\python.exe scripts\demo_advanced_features.py --quick
```

Outputs: `artifacts/advanced_demo_001/` (comparison metrics, quantization benchmark, plot, JSON report, and model state dictionaries).

## Important interpretation of results

The new advanced-feature demo generates **synthetic data** with a simulated source/target distribution shift. It does not load CPCB or USGS measurements and does not establish real cross-country transfer. In the recorded full run, source-only MAE/RMSE were 0.343/0.448, while DANN MAE/RMSE were 0.353/0.454. Ordinary concatenation MAE/RMSE were 0.246/0.300, while temporal gating MAE/RMSE were 0.270/0.346. The tiny INT8-quantized model's state dictionary was 12,645 bytes versus 8,933 bytes for FP32, and mean CPU batch latency was about 0.214 ms versus 0.087 ms in this environment. These are valid benchmark outcomes and should be reported honestly; they do not demonstrate a performance gain.

## Existing real-data limitations

- The documented India fusion pilot has only 14 paired events (6 train, 5 validation, 3 test).
- The saved-model US SHAP evaluation explains only four observations, using six India training rows as background.
- The fusion model requires measured ground fecal coliform at inference time.
- Cross-region generalization, operational monitoring accuracy, and satellite-only estimation are not established.
- SHAP is model attribution, not causal evidence.

## Still planned

- Physics-Informed Neural Network (PINN).
- Real-data validation and tuning of DANN and temporal gating using verified, compatible India/US datasets.
- Pixel-level satellite explanations, if required by the final rubric.

Do not label synthetic-demo metrics as measured river-water-quality results.

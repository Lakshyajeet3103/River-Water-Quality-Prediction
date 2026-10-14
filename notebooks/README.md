# Classroom demonstration notebook

Open [river_water_quality_demo.ipynb](river_water_quality_demo.ipynb) in VS Code. It has already been executed and contains saved tables and seven embedded charts. You can present it immediately by scrolling through the sections.

## Rerun

From the repository root:

```powershell
.\.venv-check\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

In the notebook's top-right **Select Kernel** menu, choose **Python Environments** and the project's `.venv-check\Scripts\python.exe`, then click **Run All**. Substitute `.venv` if that is your environment. Use standard Python 3.12 or 3.13 for the pinned dependencies.

The first cell verifies/restores the saved bundle without overwriting modified files. The notebook runs actual predictions with the saved scikit-learn model and displays archived comparison/experimental outputs. It does not retrain or download satellite data. Earth Engine login, CUDA and PyTorch installation are not required to run this notebook.

## Suggested presentation

1. Show the 100 real station/date matches and their satellite quality checks.
2. Run the satellite-only prediction cell: no fecal reading or BOD answer is supplied.
3. Compare satellite-only and combined models on identical test examples.
4. Show selected inputs and explain that selection used training data only.
5. Explain one prediction using the SHAP chart.
6. Show date-gap weights, the training constraint and station-classifier losses.
7. Discuss the mixed results: some components helped; others ran without improving accuracy.

The notebook distinguishes BOD estimation from overall water safety, known-station testing from broad transfer, and simple constraints from a full physics-informed model. Existing models remain preserved.

Verification: all 10 code cells ran without errors; seven PNG outputs are embedded. Assertions check dataset hashes, saved predictions, SHAP reconstruction, normalized weights and unchanged original models.

`scripts/build_demo_notebook.py` regenerates the notebook source and clears outputs. Use it only when intentionally rebuilding; the ready-to-present version already includes executed outputs.

# Computational project submission notebook

Open [river_water_quality_demo.ipynb](river_water_quality_demo.ipynb) in VS Code. The executed notebook is organized as a formal academic submission with a title block, abstract, research objectives, methodology, numbered figures and tables, discussion, references, and reproducibility checks.

## Run it

From the repository root, using Python 3.13:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

Choose **Select Kernel -> Python Environments -> .venv\Scripts\python.exe**, then **Run All**. CPU PyTorch is now required because the notebook executes the fusion checkpoint and oxygen PINN. No retraining, Earth Engine login, satellite downloads, or GPU is needed.

The first cell verifies/restores the bundle without overwriting modified files. The notebook verifies BOD checkpoint hashes, runs three models on the same 20 familiar and 26 strict unseen cases, compares every row with saved outputs, and recomputes MAE, RMSE and R-squared. It verifies all 21 saved SHAP reconstructions and runs the oxygen checkpoint with analytical references and initial-condition checks.

The setup cell loads `restore_demo_bundle.py` and `three_model_support.py` directly from the repository's `scripts/` directory. It does not depend on importing a package named `scripts`. The cell prints the detected project root and notebook Python executable. If it reports a missing dependency, install `requirements-notebook.txt` in that same Python environment and restart the kernel. After updating this notebook, use **Restart Kernel**, then **Run All**.

## Document structure

1. Real matches: station counts, dates, quality information, units.
2. Split audit: 80 training, 20 familiar test, 26 strict unseen; no training acquisition overlap in the strict set.
3. Architectures: separate diagrams for four-input MLP, satellite-ratio MLP and fusion.
4. Live checkpoints: the same observation predicted with all three models.
5. Evaluation: recomputed metrics, baseline, MAE chart and scatter plots.
6. Feature selection and ablations: training-only selection and mixed component results.
7. SHAP: matching explanations for all three models.
8. Oxygen PINN: live conditional scenarios, analytical references and extrapolation status.
9. Discussion, limitations, and conclusion.
10. Computational reproducibility and environment.
11. References.

Fill the blank school/campus, date, team names, and registration numbers before submission. The title-page table has four blank member rows; adjust it for the actual team size. Sections 4, 7 and 8 use the same `OBSERVATION_ID` for consistent case-study analysis. The default is the Varanasi observation dated 15 September 2021.

Recording directions are kept separately in [video_demo_script.md](../docs/video_demo_script.md); they are not part of the submitted notebook.

The training-mean baseline beats all three neural models on strict unseen stations. PINN/reference agreement is equation-solver evidence, not future river forecasting validation. BOD is not an overall water-safety score.

## Rebuild notebook source

```powershell
.venv\Scripts\python.exe scripts/build_demo_notebook.py
```

This clears embedded outputs. Run All and save again before submission. The generator is the editable source; keep changes there when rebuilding. Notebook execution does not change model checkpoints or data files.

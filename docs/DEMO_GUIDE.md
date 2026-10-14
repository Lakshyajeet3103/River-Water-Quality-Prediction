\# College Demo Guide — River Water Quality Prediction



\## 1. Environment



Open PowerShell in the repository root.



```powershell

$env:PYTHONPATH = (Resolve-Path .\\src).Path

.\\.venv\\Scripts\\python.exe -m pytest -q

```



\## 2. Project Configuration



The Sentinel-2 extraction configuration is stored in `configs/extraction\_pilot.yaml`.



The cloud-threshold pilot configurations are:

\- `configs/extraction\_pilot\_cloud20\_subset.yaml`

\- `configs/extraction\_pilot\_cloud30.yaml`



Both pilot configurations use the Google Cloud project `river-water-quality-fusion`.



\## 3. Run Sentinel-2 Extraction



Run the 20% cloud-threshold pilot:



```powershell

.\\.venv\\Scripts\\python.exe scripts\\extract\_sentinel\_pilot.py --config configs\\extraction\_pilot\_cloud20\_subset.yaml

```



Run the 30% cloud-threshold pilot:



```powershell

.\\.venv\\Scripts\\python.exe scripts\\extract\_sentinel\_pilot.py --config configs\\extraction\_pilot\_cloud30.yaml

```



Check the output paths printed by each run. They may create new run directories.



\## 4. Results and Audit Files



Each extraction run produces files such as:

\- `audit.csv` — records extraction decisions.

\- `features.csv` — extracted feature values.

\- `features.provenance.json` — provenance information.

\- `run.json` — run metadata.



Existing pilot outputs are under `artifacts/restored-demo/data-interim/`.



\## 5. Pilot Comparison



The current pilot attempted three events at each threshold. Both runs selected three events and produced 30 spectral rows. The generated feature files had identical SHA-256 hashes.



This is a small pilot result, not proof that the two thresholds perform equally well in general. More events and independent water-quality validation are needed for broader conclusions.



\## 6. Testing



Run the test suite from the repository root:



```powershell

$env:PYTHONPATH = (Resolve-Path .\\src).Path

.\\.venv\\Scripts\\python.exe -m pytest -q

```



Record the actual test output during the final demo.



\## 7. Important Limitation



Reported carbon emissions and other simulated metrics should be described as simulator outputs, not direct physical measurements. Explain the scope and limitations of each experiment honestly.

## Advanced Feature Demos

### Prerequisites

From the repository root, install the demo dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,ml,advanced-demo]"
```

### Run the advanced demo

```powershell
.\.venv\Scripts\python.exe scripts\demo_advanced_features.py
```

For a quicker smoke run, use:

```powershell
.\.venv\Scripts\python.exe scripts\demo_advanced_features.py --quick
```

### Outputs to show during the presentation

Inspect `artifacts/advanced_demo_001/`:

- `model_comparison.csv` — model metrics for the synthetic comparisons.
- `model_comparison.png` — comparison plot.
- `quantization_benchmark.csv` — model size, latency, and output-difference measurements.
- `report.json` — machine-readable summary.
- `temporal_gate_fp32.pt` and `temporal_gate_int8.pt` — demo checkpoints.

### Suggested explanation

**DANN:** Explain that the encoder learns a representation used by both
the regression task and domain classifier. Gradient reversal makes the
encoder learn against domain discrimination while retaining the
regression objective. The demo compares this setup with a source-only
MLP.

**Temporal gating:** Explain the separate satellite and ground branches,
the observation-age inputs, and the learned sigmoid gate. Compare the
results with ordinary concatenation fusion. Do not claim the gate is
better when the recorded synthetic results show otherwise.

**Quantization:** Compare the recorded FP32 and INT8 model sizes and
CPU timings. In the recorded run, INT8 was larger and slower for this
tiny model. This is a useful demonstration that quantization benefits
depend on model architecture and execution environment.

### Important caveat

All three advanced workflows use synthetic data. Their outputs prove
that the demo pipeline runs, not that it is validated for real river
water-quality prediction. Keep the existing CPCB/Sentinel workflow,
SHAP and Integrated Gradients material distinct from these demos.

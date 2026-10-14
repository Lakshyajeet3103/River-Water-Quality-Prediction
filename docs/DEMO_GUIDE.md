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

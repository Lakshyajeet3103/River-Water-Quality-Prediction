\# River Water Quality Prediction — Project Status



\*\*Last updated:\*\* October 2026  

\*\*Repository:\*\* River Water Quality Prediction  

\*\*Primary goal:\*\* Investigate river water quality estimation by combining satellite-derived observations with measured water-quality data.



\## 1. Current Project Status



The project has progressed beyond initial planning. It contains a data-processing workflow, satellite and water-quality data preparation components, reproducible experiment artifacts, automated tests, and an initial scikit-learn fusion baseline.



The current fusion results are exploratory. The paired dataset is very small, so the reported metrics must not be interpreted as evidence of reliable real-world prediction performance.



\### Phase checklist



\- \[x] Initial project structure and documentation

\- \[x] Data-processing and experiment workflow components

\- \[x] Paired dataset and chronological split artifacts for the current pilot

\- \[x] Initial scikit-learn MLP fusion baseline

\- \[x] Baseline metrics, test predictions, split assignments, and run metadata saved

\- \[x] Automated tests for the fusion baseline

\- \[ ] Expanded paired dataset across more dates, stations, and river systems

\- \[ ] More comprehensive satellite-data validation

\- \[ ] Advanced dual-branch multimodal fusion architecture

\- \[ ] Robust external and temporal validation

\- \[ ] Final research evaluation and deployment assessment



\## 2. Implemented Fusion Baseline



An initial scikit-learn multilayer perceptron (MLP) fusion model has been implemented.



The current baseline uses:

\- Ten Sentinel-2 satellite bands as satellite-derived features.

\- Measured fecal coliform as an additional input feature.

\- Biochemical oxygen demand (BOD) as the prediction target.

\- Training-only feature and target preprocessing to reduce data leakage.

\- Chronological training, validation, and test splits.

\- Saved metrics, test predictions, split assignments, run metadata, and a serialized model artifact.



\*\*Important:\*\* This is a baseline experiment, not the final proposed research architecture. Because measured fecal coliform is an input, this experiment estimates BOD when that measurement is available. It does not establish a satellite-only BOD prediction system.



\## 3. Current Fusion Experiment



The current paired pilot contains 14 events from three Ganga River stations.



| Split | Number of events |

|---|---:|

| Training | 6 |

| Validation | 5 |

| Test | 3 |

| Total | 14 |



\### Scikit-learn MLP fusion results



| Model | Split | MAE (mg/L) | RMSE (mg/L) | R² |

|---|---|---:|---:|---:|

| MLP fusion | Training | 0.186381 | 0.240811 | 0.954784 |

| MLP fusion | Validation | 0.409292 | 0.477922 | 0.816154 |

| MLP fusion | Test | 0.687921 | 0.776960 | 0.315743 |

| Training-mean baseline | Test | 0.850000 | 0.981071 | -0.090995 |



The baseline outperforms the training-mean baseline on this particular three-event test set. This is an encouraging preliminary observation, not sufficient evidence of generalization.



\### Separate exploratory model



An earlier exploratory model reported test MAE of 0.5864 mg/L, RMSE of 0.9329 mg/L, and R² of 0.0135.



These results belong to a separate experiment and must not be combined with or presented as results from the scikit-learn MLP baseline. Model comparisons require consistent datasets, splits, targets, and evaluation procedures.



\## 4. Reproducibility



The scikit-learn fusion baseline can be reproduced from the repository root using PowerShell:



```powershell

$env:PYTHONPATH = (Resolve-Path .\\src).Path



.\\.venv\\Scripts\\python.exe scripts\\train\_fusion.py `

&#x20; --paired artifacts\\restored-demo\\outputs\\hybrid\_demo\_001\\paired.csv `

&#x20; --splits artifacts\\restored-demo\\outputs\\fusion\_demo\_001\\split\_assignments.csv `

&#x20; --output artifacts\\restored-demo\\outputs\\fusion\_sklearn\_001

```



The output directory contains the generated experiment results, including metrics, predictions, split assignments, run metadata, and the serialized model.



The experiment depends on the referenced input artifacts and the project's Python environment.



\## 5. Testing



The latest recorded development test run completed successfully:



```text

176 passed in 12.28s

```



This records the result of that particular test run; it should not be treated as a guarantee that every future code change passes. Run the test suite again after making changes.



```powershell

.\\.venv\\Scripts\\python.exe -m pytest

```



\## 6. Data and Validation Limitations



\- The current paired pilot contains only 14 events.

\- The test split contains only three events, making test metrics highly sensitive to individual observations.

\- The current experiment covers only three Ganga River stations.

\- The baseline is a same-event estimation experiment, not a validated future-time forecasting system.

\- Measured fecal coliform is required as an input to the current fusion baseline.

\- Satellite-to-reference matching, measurement quality, temporal alignment, and geographic coverage need further validation.

\- Performance on one small test split does not establish transferability to other rivers, seasons, regions, or sensors.

\- Reported metrics are experimental software outputs and should not be interpreted as proof of operational water-quality accuracy.



\## 7. Next Priorities



1\. Increase the number of quality-controlled paired satellite and water-quality observations.

2\. Verify station coordinates, sampling dates, satellite acquisition dates, and matching tolerances.

3\. Establish leakage-resistant temporal and geographic evaluation protocols.

4\. Compare against simple baselines and the separate exploratory model using consistent data and splits.

5\. Develop and evaluate the advanced dual-branch fusion architecture.

6\. Report uncertainty, per-station performance, and external-validation results where sufficient data are available.

7\. Update the documentation and experiment audit whenever new runs are completed.



\## 8. Completion Criteria



The project should not be described as research-validated or deployment-ready until it has sufficient paired data, reproducible comparisons, robust validation, and clearly documented limitations.



\*\*Current overall status:\*\* Initial fusion baseline implemented; advanced multimodal fusion and comprehensive validation remain in progress.




## Advanced Feature Demo Update — 14 October 2026

### Implemented demo components

- **DANN:** Feature encoder, regression head, domain classifier, and
  gradient-reversal path, with a source-only comparison.
- **Temporal attention gating:** Satellite and ground branches with
  observation-age inputs and a learned sigmoid gate, compared with
  ordinary concatenation fusion.
- **Dynamic INT8 quantization:** FP32-versus-INT8 checkpoint-size,
  CPU-latency, and output-difference benchmark.
- **Automated tests:** The latest local run completed with 179 passed
  tests in 26.20 seconds.

### Recorded synthetic-demo results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Source-only MLP | 0.342608 | 0.448436 | 0.583656 |
| DANN demo | 0.353243 | 0.454208 | 0.572869 |
| Ordinary concatenation fusion | 0.245843 | 0.299567 | 0.814202 |
| Temporal attention gate | 0.270442 | 0.346150 | 0.751926 |

These are results from the recorded synthetic experiment, not the
real Ganga/CPCB dataset. In this run, DANN and temporal gating did not
outperform their respective baselines.

### Quantization result

The recorded FP32 state dictionary was 8,933 bytes, while the INT8
state dictionary was 12,645 bytes. Recorded CPU latency was about
0.087 ms per batch for FP32 and 0.214 ms for INT8. The maximum output
difference in standardized target units was about 0.0608. This small
model therefore showed no size or latency advantage from quantization.

### Reproducibility and remaining work

Run `scripts/demo_advanced_features.py` from the repository root after
installing the `advanced-demo` extra. Outputs are written to
`artifacts/advanced_demo_001/`.

These components are implementation demonstrations using synthetic
data. Real-data training, independent cross-region validation, and
evidence of operational benefit remain outstanding. The SHAP workflow
requires its corresponding saved model and paired-data artifacts;
pixel-level satellite explainability is not implemented by this update.

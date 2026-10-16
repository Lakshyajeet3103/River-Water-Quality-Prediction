# Video script: notebook and dashboard

**Length:** approximately 8 minutes. Replace bracketed names. Screen directions are recording instructions, not spoken narration.

## Prepare before recording

- Open `notebooks/river_water_quality_demo.ipynb` in VS Code. Select the project `.venv`, Run All, and save. Outputs are embedded, so you can scroll through them while recording.
- Keep the notebook's default example: Varanasi, 15 September 2021. This case has SHAP explanations for all three models.
- Start the API and frontend using `docs/dashboard.md`. Open http://127.0.0.1:5173 and select the matching station/date.
- Keep VS Code, the dashboard, and the GitHub startup guide ready for switching. Collapse code or enlarge figures where useful. You do not need to retrain on camera.

## 0:00-0:30 — Introduction

**Show:** Notebook title and opening summary.

“Hello, I am [name], and our team members are [names]. Our Artificial Intelligence project is River Water Quality Prediction Using Satellite and Ground Observations.

The project estimates biochemical oxygen demand, or BOD, using three neural models. It also provides prediction explanations and a conditional oxygen simulation.

I will first show the notebook, which makes our data and experiments inspectable, and then demonstrate the interactive dashboard.”

## 0:30-1:15 — Real data and split audit

**Show:** Notebook headings **Dataset Description and Observation Matching** and **Experimental Design and Data Partitioning**. Point to station counts, paired dates, and the cohort table.

“BOD describes oxygen consumed as microorganisms break down organic matter. It helps assess organic loading, but it is not a complete water-safety assessment.

Our original dataset contains 100 CPCB ground observations paired with Sentinel-2 reflectance at three Ganga stations. These tables show the matched dates, quality information, and measured chemical values.

The oldest 80 observations train the models, and the newest 20 evaluate familiar-station performance. We also have 26 strict observations from four unseen stations.

The notebook checks that unseen station identities and satellite acquisition groups do not overlap training. Satellite and laboratory dates may differ, so the matching information remains visible.”

## 1:15-2:00 — Three architectures

**Show:** Notebook heading **Model Architectures and Preprocessing**. Scroll through the three dedicated diagrams.

“The four-input model uses satellite bands B8, B8A and B11 together with fecal coliform. It has eight ReLU hidden neurons.

The satellite-ratio model uses ten bands and five calculated ratios, followed by an eight-neuron hidden layer. It estimates BOD without ground chemistry inputs.

The fusion model has separate satellite and chemical branches. A learned gate uses the absolute date gap to combine their representations. Its station-adversarial objective and negative-output penalty operate during training.

The diagrams also show preprocessing because the saved scalers are part of actual inference. The oxygen PINN is a separate downstream model.”

## 2:00-2:40 — Actual checkpoint predictions

**Show:** Notebook heading **Checkpoint Inference and Prediction Verification**, selected observation and output table. Optionally rerun this cell once.

“Here the notebook runs all three saved checkpoints on the same Varanasi observation dated 15 September 2021.

Each model receives only its defined inputs. Measured BOD stays outside the inference data and is shown afterwards for comparison.

The table compares live predictions with frozen evaluation values, and the checks confirm they match. These are actual checkpoint predictions.

The source data and model hashes are also verified to support reproducibility.”

## 2:40-3:35 — Results and generalization

**Show:** Notebook heading **Predictive Performance and Result Analysis**: metric table, MAE chart, and scatter plots. Briefly show the **Feature Selection and Ablation Study** table.

“The notebook reruns all three models on the same 46 evaluation observations and recomputes the metrics.

On familiar stations, the four-input model is strongest, with mean absolute error approximately 0.185 milligrams per litre and R-squared 0.947.

Performance drops at unseen stations. Fusion has the lowest neural-model unseen error at approximately 0.694 milligrams per litre. However, the simple training-mean baseline performs better, at approximately 0.520.

This is our central finding: strong familiar-station performance does not establish reliable transfer to new stations.

The ablations show mixed results too. Gate-only is slightly better than the all-components network in this saved experiment. Additional components did not automatically improve accuracy.”

## 3:35-4:15 — Explain the same prediction

**Show:** Notebook heading **Local Explanation Analysis**, the three SHAP charts and contribution table.

“These are model-specific KernelSHAP explanations for the selected observation.

Positive contributions raise the estimated BOD, and negative contributions lower it. The reference value plus the contributions reconstructs the prediction.

The background examples come from training data only. There are seven saved cases per model, and the notebook verifies all 21 reconstructions.

SHAP explains model behaviour relative to a reference background. It does not prove physical causation or predictive accuracy.”

## 4:15-5:00 — Oxygen PINN in the notebook

**Show:** Notebook heading **Conditional Oxygen-Balance Modelling**, oxygen and demand curves, scenario table, and solver metrics.

“The separate oxygen PINN takes time, initial demand derived from predicted BOD, and measured initial dissolved oxygen.

It approximates oxygen consumption and replenishment over five days. Solid curves show the neural-network output; dashed curves show the analytical reference.

The notebook checks initial conditions and displays equation errors and extrapolation status. The fixed rates and BOD5 conversion are disclosed.

Close agreement shows that the network approximates these assumed equations. It does not establish accurate future river oxygen forecasts. Even the satellite-only BOD route needs measured initial oxygen for this scenario.”

## 5:00-5:45 — Interactive predictions

**Show:** Dashboard Prediction page. Select Varanasi, 15 September 2021. Run a model, change models, and rerun. If showing live inference, explicitly choose the live-checkpoint option.

“The React dashboard presents the same models and evidence through a FastAPI backend.

I select a station, date, and model, then run the prediction. The result includes predicted and measured BOD, input requirements, and the result source.

Saved evaluation results are the default. The explicit live option executes the saved checkpoint with its preprocessing.

The explanation matches the selected observation and model. Changing the selection clears the previous result to prevent a mismatched display.”

## 5:45-6:20 — Accuracy and data matching

**Show:** Accuracy; switch familiar to strict unseen. Then open Data & matching.

“The Accuracy view lets us switch cohorts and compare the selected model with the training-mean baseline. It also includes a scatter plot, individual results, and CSV download.

Data and matching shows the coordinates, sampling and acquisition dates, signed date gap, spectral values, and quality metadata.

The map uses OpenStreetMap. Satellite thumbnails are unavailable in the saved bundle and are clearly labelled.”

## 6:20-6:50 — Oxygen interface

**Show:** Oxygen simulation. Run the scenario and point to assumptions and curves.

“This view exposes the same conditional oxygen model from the notebook.

We can inspect predicted BOD, measured initial oxygen, fixed rates, and the neural and analytical curves.

Extrapolation is identified, and unsupported oxygen-exhaustion scenarios are rejected. The interpretation remains tied to the displayed assumptions.”

## 6:50-7:30 — Architecture dropdown

**Show:** Architecture. Select four-input, satellite-ratio, and fusion in sequence.

“The architecture dropdown displays one selected BOD model at a time.

The four-input option shows the compact network and preprocessing. The satellite-ratio option shows the fifteen-feature route. Fusion shows its two branches, date gate, and training mechanisms.

These correspond to the notebook diagrams. The experiment table reports saved variants and MAE. Selecting a diagram does not retrain or modify the checkpoints.”

## 7:30-8:10 — Reproducibility and conclusion

**Show:** Notebook heading **Computational Reproducibility**, verification tables, then GitHub and `docs/dashboard.md`.

“The notebook's reproducibility checks confirm that predictions and metrics reproduce saved results, SHAP contributions reconstruct predictions, oxygen initial conditions hold, and data and checkpoint hashes remain unchanged.

The repository includes the notebook and startup instructions for restoring the demonstration bundle and running both services.

Our project integrates BOD prediction, explanation, evaluation, and an equation-based oxygen scenario. Its main limitation is transfer to unseen stations.

Future work should expand station coverage, use independent station-held-out validation, and calibrate oxygen dynamics against field measurements.

Thank you.”

## Shorter recording option

For approximately six minutes, omit the ablation discussion, explain one SHAP chart, and briefly combine dashboard Accuracy and Data views. Retain the unseen-station baseline result and the oxygen-validation limitation.

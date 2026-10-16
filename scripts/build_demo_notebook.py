"""Build the academic project notebook; execute after regenerating sources."""
import hashlib
import json
from pathlib import Path

cells = []

def add(kind, source):
    cell = {"cell_type": kind, "id": hashlib.sha256(f"{len(cells)}:{source}".encode()).hexdigest()[:8],
            "metadata": {}, "source": source.strip() + "\n"}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)

add('markdown', '''# River Water Quality Prediction Using Satellite and Ground Observations
### Computational Project Submission

**Vellore Institute of Technology**<br>
**Course:** Artificial Intelligence (BCSE306L)<br>
**Slot:** C1 + TC1<br>
**School / Campus:** ______________________________<br>
**Submission date:** ______________________________

| Team member | Registration number |
| :-- | :-- |
| ______________________________ | ______________________________ |
| ______________________________ | ______________________________ |
| ______________________________ | ______________________________ |
| ______________________________ | ______________________________ |

---

## Abstract

This study evaluates neural estimation of biochemical oxygen demand (BOD) from paired satellite and ground observations at selected River Ganga monitoring stations. A corpus of 100 observations is partitioned into 80 chronological training records and 20 familiar-station test records; 26 additional observations assess transfer to four unseen stations. Three architectures are examined: a four-input multilayer perceptron, a satellite-band-and-ratio perceptron, and a gated satellite-chemical fusion network. Evaluation includes a training-mean baseline, mean absolute error, root mean squared error, and R-squared. KernelSHAP provides local explanations, while a separate conditional physics-informed neural network approximates an assumed oxygen-balance system. The four-input model achieves familiar-station MAE of 0.185 mg/L. Fusion gives the lowest neural-model unseen-station MAE of 0.694 mg/L, but is outperformed by the training-mean baseline at 0.520 mg/L. Numerical oxygen agreement concerns the assumed equations rather than measured future conditions. The findings demonstrate a substantial spatial generalization gap and motivate broader independent validation.

**Keywords:** water quality; biochemical oxygen demand; Sentinel-2; neural regression; spatial generalization; KernelSHAP; physics-informed neural networks.

## Research objectives

1. Compare BOD regression using satellite-only and satellite-chemical inputs.
2. Assess performance separately at familiar and unseen monitoring stations.
3. Explain individual predictions with reference-dependent feature attributions.
4. Evaluate a conditional oxygen PINN against an analytical solution.
5. Verify consistency between executable inference and archived experimental results.

## Contents

1. Dataset description and observation matching
2. Experimental design and data partitioning
3. Model architectures and preprocessing
4. Checkpoint inference and prediction verification
5. Predictive performance and result analysis
6. Feature selection and ablation study
7. Local explanation analysis
8. Conditional oxygen-balance modelling
9. Discussion, limitations, and conclusion
10. Computational reproducibility
11. References

## Computational scope

This notebook reproduces inference and evaluates previously trained checkpoints. It does not retrain the models or recompute KernelSHAP values. Saved explanations are checked for additive reconstruction. Raw acquisition, training, and provenance are documented in the accompanying source repository [6]. BOD is expressed in mg/L and is not a complete drinking-water safety indicator.

---

### Environment and artifact initialization

The following cell resolves repository-local helper files, verifies the archived artifact bundle, and loads the saved models and cohorts. Package versions and data-integrity checks are reported in Section 10.''')

add('code', '''from pathlib import Path
import os
import sys
import json
import importlib.util

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'pyproject.toml').exists() and (p / 'demo_bundle').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Open the notebook from within the project checkout.')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/notebook-matplotlib'))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Resolve helper files explicitly to avoid collisions with generic package names.
def load_project_helper(filename):
    path = ROOT / 'scripts' / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing project helper: {path}. Open the complete project checkout.')
    spec = importlib.util.spec_from_file_location('river_submission_' + path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load project helper: {path}')
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f'Project helper needs dependency {exc.name!r}. Notebook Python: {sys.executable}. '
            'Install requirements-notebook.txt in this environment and restart the kernel.'
        ) from exc
    return module

restore = load_project_helper('restore_demo_bundle.py').restore

artifact_count = restore(ROOT / 'demo_bundle', ROOT)
model_support = load_project_helper('three_model_support.py')
load, sha = model_support.load, model_support.sha
train, familiar, unseen, examples, models, source_hashes = load()
checkpoint_hashes = {name: sha(model['path']) for name, model in models.items()}
LABELS = {'selected_four_inputs': 'Four-input MLP', 'all_components': 'Fusion network',
          'satellite_ratios': 'Satellite ratios', 'training_mean': 'Training mean'}
PREFIX = {'selected_four_inputs': 'selected_four_inputs', 'all_components': 'network_all_components',
          'satellite_ratios': 'satellite_ratios', 'training_mean': 'training_mean'}
COHORTS = {'familiar_20': familiar, 'new_stations_strict_26': unseen}
RESULTS = ROOT / 'docs/saved_models_unseen_results'
saved_predictions = pd.read_csv(RESULTS / 'predictions.csv')
saved_metrics = pd.read_csv(RESULTS / 'metrics.csv')
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.constrained_layout.use': True})
pd.set_option('display.max_columns', 14)
pd.set_option('display.precision', 4)
print(f'Artifact initialization complete: {artifact_count} verified files and three hash-verified BOD checkpoints.')''')

add('markdown', '''## 1. Dataset Description and Observation Matching

The original corpus has **100 matched observations from three Ganga stations**, dated 25 November 2018 to 28 December 2021. CPCB monitoring records provide measured BOD and chemical inputs [1]. Sentinel-2 SR Harmonized provides ten reflectance bands [2].

Extraction uses a 60 m neighbourhood on a common 20 m grid; the temporal search is first +/-3 days and then +/-7 days. Cloud, water, coverage, and valid-pixel thresholds filter the extraction. These checks do not prove the satellite and laboratory sample describe identical conditions.

Measured BOD is the supervised target and is excluded from all BOD-model input frames. Coordinates, dates, identifiers, and extraction-quality fields provide traceability. Table 1 reports station coverage, Table 2 illustrates matched observations, and Table 3 summarizes the measured variables. Optional total coliform is missing in 71 original records and is excluded from these models.''')

add('code', '''paired = pd.read_csv(ROOT / 'outputs/india_100_chemical_dataset_20261014/paired.csv')
assert len(paired) == 100 and paired.observation_id.is_unique
display(Markdown('**Table 1. Original matched-observation counts by monitoring station.**'))
display(paired.groupby('station_name').size().rename('Matched observations').to_frame())
display(Markdown('**Table 2. Sample ground-satellite pairs and extraction metadata.**'))
display(paired[['station_name', 'sample_date', 'satellite_local_date', 'temporal_offset_days',
                'cloud_fraction', 'water_fraction', 'coverage_fraction', 'pixel_count',
                'ground_bod', 'ground_fecal_coliform', 'ground_ph', 'ground_dissolved_oxygen']].head(6))
display(Markdown('**Table 3. Descriptive statistics of the original 100 observations.**'))
display(paired[['ground_bod', 'ground_fecal_coliform', 'ground_ph',
                'ground_dissolved_oxygen', 'temporal_offset_days']].describe().loc[['min', 'max', 'mean']])
print('BOD and DO: mg/L. Fecal coliform: MPN/100 mL. pH and reflectance: unitless.')''')

add('markdown', '''## 2. Experimental Design and Data Partitioning

The oldest **80** original observations train the models; the newest **20** evaluate familiar-station performance. Scaling and station chemistry profiles use training data only. The familiar test cases have been inspected during development, so they are not an untouched final validation set.

Four new stations provide 30 observations. Four with satellite acquisitions overlapping training are excluded, leaving **26 strict unseen observations**. This is a retrospective station-transfer test, not a future-period forecast. The 30-row pre-exclusion cohort is not mixed into the strict comparison.''')

add('code', '''from river_water_quality.training.splits import acquisition_group

assert len(train) == 80 and len(familiar) == 20 and len(unseen) == 26
assert not set(train.observation_id) & set(familiar.observation_id)
assert not set(train.station_id) & set(unseen.station_id)
assert not set(train.scene_id.map(acquisition_group)) & set(unseen.scene_id.map(acquisition_group))
assert pd.to_datetime(train.sample_date).max() < pd.to_datetime(familiar.sample_date).min()
audit = []
for label, frame in [('Train', train), ('Familiar test', familiar), ('Strict unseen test', unseen)]:
    audit.append({'Cohort': label, 'Rows': len(frame), 'Stations': frame.station_id.nunique(),
                  'First date': frame.sample_date.min(), 'Last date': frame.sample_date.max()})
display(Markdown('**Table 4. Chronological training and evaluation cohort definitions.**'))
display(pd.DataFrame(audit).set_index('Cohort'))
display(Markdown('**Table 5. Strict unseen-station observation counts.**'))
display(unseen.groupby('station_name').size().rename('Strict unseen observations').to_frame())
print('Partition checks satisfied: disjoint train/test observations, unseen station identities, and strict acquisition separation.')''')

add('markdown', '''## 3. Model Architectures and Preprocessing

Three input configurations assess the trade-off between measurement availability and predictive performance. The first two models use scikit-learn MLP regression with L-BFGS optimization, alpha 1, maximum 3,000 iterations, and random seed 42 [3, 6]. Input and target scalers are fitted on training data only; fecal coliform undergoes a log1p transformation. Predictions are transformed back to BOD units and clipped to non-negative values.

The fusion model uses independent tanh branches and a date-dependent sigmoid gate. Its non-negativity penalty and gradient-reversal station classifier operate during training. Station identity is not an inference feature. Figures 1-3 document the architectures and preprocessing pathways. The oxygen PINN in Section 8 is a separate downstream model.''')

add('code', '''from matplotlib.patches import FancyBboxPatch

def architecture(title, stages, note):
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    width = 0.205
    for i, text in enumerate(stages):
        x = 0.015 + i * 0.25
        ax.add_patch(FancyBboxPatch((x, 0.30), width, 0.40, boxstyle='round,pad=0.012',
                                   facecolor='#edf4f7', edgecolor='#779aaa'))
        ax.text(x + width/2, 0.50, text, ha='center', va='center', fontsize=9)
        if i < len(stages)-1:
            ax.annotate('', xy=(x+0.245, .50), xytext=(x+width+.013, .50),
                        arrowprops={'arrowstyle': '->', 'color': '#008577', 'lw': 1.8})
    ax.set_title(title, loc='left', fontweight='bold', color='#15384e')
    ax.text(.015, .09, note, fontsize=9, color='#344d5a')
    plt.show()

architecture('Figure 1. Four-input MLP architecture (49 parameters)',
             ['B8, B8A, B11\\n+ fecal coliform', 'log1p fecal\\n+ saved scaling',
              '4 -> 8 ReLU\\n-> 1 linear', 'Inverse target scale\\nclip BOD >= 0'],
             'Requires fecal coliform at prediction time; output is BOD in mg/L.')
architecture('Figure 2. Satellite-ratio MLP architecture (137 parameters)',
             ['Ten spectral bands', 'Five ratios added\\n+ saved scaling',
              '15 -> 8 ReLU\\n-> 1 linear', 'Inverse target scale\\nclip BOD >= 0'],
             'Ratios: B3/B4, B8/B4, B5/B4, (B3-B8)/(B3+B8), (B4-B3)/(B4+B3).')
architecture('Figure 3. Gated satellite-chemical fusion architecture',
             ['10 bands -> 4 tanh\\n3 chemicals -> 4 tanh', 'w = sigmoid(linear\\n(abs(date gap)/7))',
              'Satellite x w; ground\\nx (1-w); concatenate 8', '8 -> 4 tanh -> 1\\nnon-negative BOD'],
             'Chemicals: fecal coliform, pH, DO. Training only: negative-output penalty and station gradient reversal.')''')

add('markdown', '''## 4. Checkpoint Inference and Prediction Verification

One observation from the archived explanation subset is selected for a consistent case study across prediction, attribution, and oxygen analysis. The subset comprises one observation from each of three familiar and four unseen stations (Table 6). The same observation identifier is used in Sections 4, 7, and 8.

Inputs are restricted to the saved model feature definitions. The target is kept outside each inference frame. Live predictions are compared with their frozen counterparts.''')

add('code', '''display(Markdown('**Table 6. Observations available for matched local-explanation analysis.**'))
display(examples[['observation_id', 'station_name', 'sample_date', 'cohort']])
OBSERVATION_ID = str(examples.iloc[0].observation_id)
selected = examples.set_index('observation_id').loc[[OBSERVATION_ID]].reset_index()
selected_row = selected.iloc[0]
saved_case = saved_predictions[saved_predictions.observation_id.eq(OBSERVATION_ID)].iloc[0]
live_values = {}
rows = []
for name, model in models.items():
    inputs = selected[model['features']].copy()
    assert 'ground_bod' not in inputs and 'observation_id' not in inputs
    value = float(model['predict'](inputs)[0])
    live_values[name] = value
    frozen = float(saved_case[PREFIX[name] + '_bod_mg_l'])
    np.testing.assert_allclose(value, frozen, atol=1e-5, rtol=1e-5)
    rows.append({'Model': LABELS[name], 'Measured BOD': selected_row.ground_bod,
                 'Recomputed BOD': value, 'Archived BOD': frozen,
                 'Absolute error': abs(value-selected_row.ground_bod)})
display(Markdown(f'**Case-study observation:** {selected_row.station_name}, {selected_row.sample_date}'))
display(Markdown('**Table 7. Recomputed and archived case-study predictions (mg/L).**'))
display(pd.DataFrame(rows).set_index('Model'))
print('Prediction verification satisfied for all three checkpoints (absolute/relative tolerance 1e-5).')''')

add('markdown', '''## 5. Predictive Performance and Result Analysis

All three models are rerun on the same 46 evaluation observations. MAE, RMSE, and R-squared are recomputed from the resulting rows and checked against the frozen metric table. The baseline predicts the mean of the 80 training BOD targets for every case.

For measured values y and predictions y-hat, MAE = mean(abs(y-hat - y)), RMSE = sqrt(mean((y-hat - y)^2)), and R-squared = 1 - sum((y-hat - y)^2) / sum((y - mean(y))^2).

Lower MAE/RMSE is better. A negative R-squared means worse squared error than predicting that evaluation cohort's own mean; the training-mean baseline uses only training targets. Table 8 reports the recomputed results, Figure 4 compares MAE, and Figure 5 examines prediction distributions.''')

add('code', '''def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    error = predicted-actual
    return {'mae': np.abs(error).mean(), 'rmse': np.sqrt(np.mean(error**2)),
            'r2': 1-np.sum(error**2)/np.sum((actual-actual.mean())**2)}

evaluation = []
live_cohorts = {}
for cohort, frame in COHORTS.items():
    live_cohorts[cohort] = {}
    frozen_rows = saved_predictions[saved_predictions.cohort.eq(cohort)].set_index('observation_id').loc[frame.observation_id]
    for name in [*models, 'training_mean']:
        yhat = (np.full(len(frame), train.ground_bod.mean()) if name=='training_mean'
                else np.asarray(models[name]['predict'](frame[models[name]['features']].copy())))
        live_cohorts[cohort][name] = yhat
        np.testing.assert_allclose(yhat, frozen_rows[PREFIX[name]+'_bod_mg_l'], atol=1e-5, rtol=1e-5)
        result = metrics(frame.ground_bod, yhat)
        frozen_metric = saved_metrics[saved_metrics.cohort.eq(cohort) & saved_metrics.model.eq(PREFIX[name])].iloc[0]
        for metric, value in result.items():
            np.testing.assert_allclose(value, frozen_metric[metric], atol=1e-5, rtol=1e-5)
        evaluation.append({'Cohort': cohort, 'Model': LABELS[name], 'n': len(frame), **result})
evaluation = pd.DataFrame(evaluation)
display(Markdown('**Table 8. Recomputed cohort performance: MAE/RMSE in mg/L; R-squared unitless.**'))
display(evaluation.set_index(['Cohort', 'Model']))
chart = evaluation.pivot(index='Model', columns='Cohort', values='mae').reindex(list(LABELS.values()))
ax = chart.plot.bar(figsize=(10, 4), color=['#008577', '#22639b'], rot=0)
ax.set_ylabel('BOD MAE (mg/L)'); ax.set_xlabel(''); ax.set_title('Figure 4. BOD error across familiar and strict unseen stations')
ax.legend(['Familiar (n=20)', 'Strict unseen (n=26)'], frameon=False)
for bars in ax.containers: ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
ax.set_ylim(0, 1.2); plt.show()
print('Evaluation verification satisfied: 46 observations per model and all reported metrics reproduce archived results.')''')

add('code', '''fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
fig.suptitle('Figure 5. Measured versus predicted BOD; dashed line denotes perfect agreement')
for r, (cohort, frame) in enumerate(COHORTS.items()):
    for col, name in enumerate(models):
        ax = axes[r, col]
        ax.scatter(frame.ground_bod, live_cohorts[cohort][name], s=25, alpha=.8,
                   color=['#008577', '#22639b', '#d88b32'][col])
        ax.plot([0, 5], [0, 5], '--', color='gray', linewidth=1)
        ax.set(xlim=(0,5), ylim=(0,5))
        if r==0: ax.set_title(LABELS[name])
        if r==1: ax.set_xlabel('Measured BOD (mg/L)')
        if col==0: ax.set_ylabel(('Familiar' if r==0 else 'Unseen')+' predicted BOD (mg/L)')
plt.show()''')

add('markdown', '''**Interpretation.** The four-input model achieves familiar MAE **0.185 mg/L** and R-squared **0.947**. Fusion has the lowest neural-model unseen MAE, **0.694 mg/L**, but the training mean is better at **0.520 mg/L**. The satellite-ratio model's lower input burden does not compensate for its larger errors in this experiment. These results do not establish broad river-transfer accuracy.''')

add('markdown', '''## 6. Feature Selection and Ablation Study

The compact input selection uses a split inside the 80 training observations, then refits the chosen model on all 80. The outer test answers do not choose its four features. Feature ranking remains dataset-specific.

The five archived fusion variants isolate the baseline architecture, date gate, non-negativity constraint, station-adversarial objective, and their combination. Tables 9-11 record feature rankings, leading validation candidates, and variant MAE. Figure 6 compares ablations under identical evaluation cohorts.''')

add('code', '''demos = ROOT / 'outputs/straightforward_demos_20261014'
ranking = pd.read_csv(demos / 'feature_ranking.csv')
candidates = pd.read_csv(demos / 'feature_candidates.csv')
display(Markdown('**Table 9. Training-subset feature ranking and selected inputs.**'))
display(ranking[['feature', 'modality', 'score', 'selected']])
display(Markdown('**Table 10. Five leading feature candidates by internal validation MAE.**'))
display(candidates[['candidate', 'input_count', 'validation_mae', 'selected']].sort_values('validation_mae').head(5))
variants = ['network_baseline', 'network_gate_only', 'network_constraint_only',
            'network_station_adversarial_only', 'network_all_components']
ablation = saved_metrics[saved_metrics.cohort.isin(COHORTS) & saved_metrics.model.isin(variants)]
display(Markdown('**Table 11. Fusion ablation MAE by evaluation cohort (mg/L).**'))
display(ablation.pivot(index='model', columns='cohort', values='mae').reindex(variants))
ax = ablation.pivot(index='model', columns='cohort', values='mae').reindex(variants).plot.bar(
    figsize=(11,4), color=['#008577','#22639b'], rot=15)
ax.set_ylabel('BOD MAE (mg/L)'); ax.set_xlabel(''); ax.set_title('Figure 6. Fusion-network ablation performance')
plt.show()''')

add('markdown', '''**Interpretation.** Gate-only performs slightly better than all-components on both cohorts. Constraint-only and adversarial-only do not establish an accuracy gain. No repeated-seed uncertainty estimate is available, so the small gap between gate-only and all-components should not be overstated. The non-negativity penalty is distinct from the separate oxygen PINN.''')

add('markdown', '''## 7. Local Explanation Analysis

Each model has seven saved cases: three familiar and four unseen. KernelSHAP explanations use 20 training-background rows, seed 42, and 512 samples [4, 6]. Satellite ratios are recomputed from masked raw bands rather than treated as independent raw measurements.

Figure 7 shows signed contributions for the case-study observation, and Table 12 groups them by modality. Reference plus contributions reconstructs predicted BOD. Contributions are background-dependent local explanations; they do not establish causality or global feature importance.''')

add('code', '''shap_dir = ROOT / 'docs/three_model_shap_results'
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Figure 7. KernelSHAP contributions for the shared case-study observation')
shap_summary = []
for ax, name in zip(axes, models):
    frame = pd.read_csv(shap_dir / f'{name}_explanations.csv')
    assert len(frame)==7 and frame.observation_id.is_unique
    cols = [c for c in frame if c.startswith('shap_')]
    np.testing.assert_allclose(frame.reference_bod_mg_l + frame[cols].sum(axis=1),
                               frame.predicted_bod_mg_l, atol=1e-5)
    case = frame.set_index('observation_id').loc[OBSERVATION_ID]
    np.testing.assert_allclose(case.predicted_bod_mg_l, live_values[name], atol=1e-5)
    values = case[cols].astype(float).sort_values()
    labels = [c.removeprefix('shap_').replace('satellite_', '').replace('ground_', '') for c in values.index]
    ax.barh(labels, values, color=['#22639b' if v<0 else '#cf6650' for v in values])
    ax.axvline(0, color='gray', lw=.8); ax.set_xlabel('Contribution to BOD (mg/L)')
    ax.set_title(LABELS[name]+f'\\nReference {case.reference_bod_mg_l:.3f} -> {case.predicted_bod_mg_l:.3f}')
    shap_summary.append({'Model': LABELS[name], 'Reference': case.reference_bod_mg_l,
                         'Satellite': case.satellite_contribution_mg_l,
                         'Ground': case.ground_contribution_mg_l,
                         'Date gap': case.date_gap_contribution_mg_l,
                         'Prediction': case.predicted_bod_mg_l})
plt.show()
display(Markdown('**Table 12. Reference values and modality-level BOD contributions (mg/L).**'))
display(pd.DataFrame(shap_summary).set_index('Model'))
print('Attribution verification satisfied: 21 reconstructions and three case-study predictions agree within tolerance.')''')

add('markdown', '''## 8. Conditional Oxygen-Balance Modelling

The conditional network is **3 inputs -> 24 tanh -> 24 tanh -> 2 outputs**, with hard initial conditions. Inputs encode time, ultimate oxygen demand, and measured initial DO. It predicts remaining demand and DO over 0-5 days.

The assumed equations are dL/dt = -kD L and dO/dt = kR(Cs-O)-kD L. Fixed values are kD=0.23/day, kR=0.50/day, and Cs=8.26 mg/L. Predicted BOD is treated as BOD5, with L0=BOD5/(1-exp(-5 kD)). Training follows the physics-informed principle of penalizing equation residuals [5]; this implementation uses Adam for 1,500 epochs, seed 42, and a non-negativity penalty [6]. It does not fit measured future trajectories.

Measured initial DO is required for every oxygen scenario, including those driven by satellite-only BOD estimates. Figure 8 compares PINN and analytical trajectories. Table 13 reports case-study equation errors and extrapolation status; Table 14 summarizes archived cohort-level equation error.''')

add('code', '''import torch
from river_water_quality.training.oxygen_pinn import predict as oxygen_predict, analytical, initial_demand, KD, KR, CS

oxygen_dir = ROOT / 'docs/oxygen_pinn_results'
oxygen_path = oxygen_dir / 'oxygen_pinn.pt'
oxygen_hash = sha(oxygen_path)
oxygen_provenance = json.loads((oxygen_dir / 'provenance.json').read_text())
assert oxygen_hash == oxygen_provenance['model_sha256'], 'Oxygen checkpoint hash mismatch'
checkpoint = torch.load(oxygen_path, map_location='cpu', weights_only=True)
t = np.linspace(0, 5, 21)
o0 = float(selected_row.ground_dissolved_oxygen)
fig, axes = plt.subplots(1,2,figsize=(12,4))
fig.suptitle('Figure 8. Conditional oxygen and demand trajectories: PINN and analytical reference')
oxygen_checks = []
for color, (name, bod) in zip(['#008577','#22639b','#d88b32'], live_values.items()):
    curve = oxygen_predict(checkpoint, t, bod, o0)
    demand_ref, do_ref = analytical(t, initial_demand(bod), o0)
    np.testing.assert_allclose(curve[0], [initial_demand(bod), o0], atol=1e-5)
    for ax, values, ref in [(axes[0],curve[:,1],do_ref),(axes[1],curve[:,0],demand_ref)]:
        ax.plot(t,values,color=color,label=LABELS[name]+' PINN')
        ax.plot(t,ref,'--',color=color,lw=1.2)
    outside = not (checkpoint['training_l0_range'][0] <= initial_demand(bod) <= checkpoint['training_l0_range'][1]
                   and checkpoint['training_do_range'][0] <= o0 <= checkpoint['training_do_range'][1])
    oxygen_checks.append({'Scenario':LABELS[name],'BOD5 mg/L':bod,'Initial DO mg/L':o0,
                          'Equation MAE mg/L':np.abs(curve[:,1]-do_ref).mean(),'Extrapolation':outside})
axes[0].set_ylabel('Dissolved oxygen (mg/L)'); axes[1].set_ylabel('Remaining ultimate demand (mg/L)')
for ax in axes: ax.set_xlabel('Scenario time (days)')
axes[0].legend(fontsize=8,frameon=False); axes[0].set_title('Solid = PINN; dashed = analytical reference')
plt.show()
display(Markdown('**Table 13. Case-study oxygen-solver error and extrapolation status.**'))
display(pd.DataFrame(oxygen_checks).set_index('Scenario'))
display(Markdown('**Table 14. Archived oxygen-solver error by BOD source and cohort (mg/L).**'))
display(pd.read_csv(oxygen_dir / 'solver_metrics.csv'))
print(f'Assumptions: kD={KD}/day, kR={KR}/day, saturation={CS} mg/L. Initial conditions checked.')''')

add('markdown', '''**Interpretation.** Saved cohort-level oxygen equation MAE is approximately 0.007-0.011 mg/L. This measures agreement with the assumed equations. It excludes uncertainty in predicted BOD, initial measurements, physical rates, flow, temperature changes, external loading, and spatial transport. Extrapolation is marked and oxygen-exhaustion scenarios are unsupported.''')

add('markdown', '''## 9. Discussion, Limitations, and Conclusion

### 9.1 Predictive findings

The four-input model provides the strongest familiar-station performance among the evaluated networks. Its use of fecal coliform, however, entails a ground-measurement requirement. The satellite-ratio model reduces that input requirement but has larger errors on both cohorts. Fusion improves unseen-station performance relative to the other neural models without surpassing the training-mean baseline. This finding limits claims of operational transfer and shows why a simple baseline is necessary.

The ablations associate the largest observed fusion improvement with the date gate. Adding the constraint and adversarial objective does not improve the saved all-components score over gate-only. These observations concern one experimental configuration, not a statistically established universal ranking.

### 9.2 Limitations

- **Sample coverage:** training covers only three stations and 80 observations, with limited representation of hydrological and seasonal variation.
- **Evaluation independence:** familiar test cases were inspected during development. A new independent final evaluation is required. The unseen cohort tests retrospective spatial transfer rather than future-period forecasting.
- **Measurement correspondence:** date and footprint differences remain after matching. Complete laboratory quality-control information is not established by the paired dataset.
- **Uncertainty:** no repeated-seed distribution or calibrated prediction interval is reported. Small numerical differences should not be treated as conclusive superiority.
- **Explanation scope:** seven cases per model cannot establish global importance; correlated bands and background choice affect attribution.
- **Physical scope:** oxygen curves assume fixed rates and omit external loading, variable temperature, discharge, photosynthesis, and spatial transport. Equation agreement does not validate future river oxygen.

The associated application's BOD-led category is a separate custom indicator using training-only station chemistry profiles. It is unavailable for unseen stations and is not evaluated here as an independently validated full water quality index.

### 9.3 Conclusion and future work

The study establishes an executable workflow linking paired observations, three BOD regressors, local explanations, and a conditional oxygen model. Its main empirical result is a substantial spatial generalization gap despite strong familiar-station accuracy. Priority extensions are additional station coverage, station-held-out model selection, repeated-seed analysis, an independent final test set, and field-calibrated oxygen parameters.

## 10. Computational Reproducibility

The analysis uses frozen model parameters and training-fitted transformations. No optimizer updates occur during notebook execution. SHA-256 checks verify checkpoint and source-data integrity. Every evaluation prediction is compared with its archived value; MAE, RMSE, and R-squared are independently recomputed. All 21 saved SHAP explanations are checked for additive reconstruction, and oxygen scenarios are checked at their initial conditions.

Table 15 records software versions from the active kernel. The verification output below reports the executed checks; it does not constitute independent external validation.

**Supporting files:** [dataset description](../docs/chemical_dataset.md), [evaluation provenance](../docs/saved_models_unseen.md), [SHAP and oxygen methodology](../docs/oxygen_pinn_and_three_model_shap.md), and [execution instructions](README.md).''')

add('code', '''assert checkpoint_hashes == {name: sha(model['path']) for name, model in models.items()}
assert oxygen_hash == sha(oxygen_path)
for relative, expected in source_hashes.items():
    assert sha(ROOT / relative) == expected
from importlib.metadata import version
display(Markdown('**Table 15. Computational software environment.**'))
environment = [{'Component': 'Python', 'Version': sys.version.split()[0]}]
environment += [{'Component': name, 'Version': version(name)}
                for name in ['numpy', 'pandas', 'scikit-learn', 'torch', 'matplotlib', 'nbformat']]
display(pd.DataFrame(environment).set_index('Component'))
display(Markdown('**Table 16. Executed reproducibility and integrity checks.**'))
verification = [
    ('BOD checkpoint inference', '3 models x 46 evaluation observations', 'Satisfied'),
    ('Regression metrics', 'MAE, RMSE, R-squared for 2 cohorts and 4 estimators', 'Satisfied'),
    ('SHAP reconstruction', '21 archived explanations; 3 matched case-study predictions', 'Satisfied'),
    ('Oxygen initial conditions', 'Demand and DO at time zero for 3 scenarios', 'Satisfied'),
    ('Artifact integrity', 'Source data and BOD/oxygen checkpoint SHA-256 values', 'Satisfied'),
]
display(pd.DataFrame(verification, columns=['Check', 'Scope', 'Outcome']).set_index('Check'))
print('Computational verification completed. Model parameters and source data remain unchanged.')''')

add('markdown', '''## 11. References

[1] National Water Informatics Centre. "Surface Water Quality (Manual - Chemical Parameters), Central Pollution Control Board (CPCB)." *National Water Data Portal*, Ministry of Jal Shakti. [Official dataset portal](https://nwdp.nwic.gov.in/en/dataset/surface-water-quality-manual-chemical-parameters-cpcb). Accessed 15 Oct. 2026.

[2] Google. "Harmonized Sentinel-2 MSI: MultiSpectral Instrument, Level-2A (SR)." *Earth Engine Data Catalog*. [Collection documentation](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED). Accessed 15 Oct. 2026.

[3] scikit-learn Developers. "MLPRegressor." *scikit-learn Documentation*. [Estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html). Accessed 15 Oct. 2026.

[4] SHAP Developers. "shap.KernelExplainer." *SHAP Documentation*. [KernelExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html). Accessed 15 Oct. 2026.

[5] Karniadakis, George Em, et al. "Physics-Informed Machine Learning." *Nature Reviews Physics*, vol. 3, 2021, pp. 422-440. Springer Nature. [doi:10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5).

[6] "River-Water-Quality-Prediction." *GitHub*, Lakshyajeet3103, branch `frontend-professor-demo`. [Project source and archived artifacts](https://github.com/Lakshyajeet3103/River-Water-Quality-Prediction/tree/frontend-professor-demo). Accessed 15 Oct. 2026.

---

**End of computational submission.**''')

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    path = root / 'notebooks/river_water_quality_demo.ipynb'
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3 (project .venv)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13'}}, 'cells': cells}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'Built {len(cells)} cells: {path}')

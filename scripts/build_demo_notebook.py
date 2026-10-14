"""Build the classroom demonstration notebook; execution is verified separately."""
import hashlib
import json
from pathlib import Path

cells = []


def add(kind, source):
    cell = {'cell_type': kind, 'id': hashlib.sha256(f'{len(cells)}:{source}'.encode()).hexdigest()[:8],
            'metadata': {}, 'source': source.strip() + '\n'}
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add('markdown', '''# River Water Quality Prediction — demonstration

**A small proof of concept using 100 real matched observations from three Ganga stations.**

The main model estimates **BOD in mg/L from satellite readings**. BOD describes oxygen used while microorganisms break down organic matter. It is one water-quality measurement, not an overall cleanliness or drinking-water safety score.

This notebook shows: matching → predictions → fair model comparison → feature selection → SHAP explanations → experimental gate, constraint and station-adversarial results.

**Presentation:** saved outputs and charts are included. Scroll through them to present immediately. To rerun in VS Code, install `requirements-notebook.txt`, select Python from `.venv-check` (or your environment), and click **Run All**. It restores missing demo files safely, performs a saved-model prediction, and reads saved experiment results. No training, satellite downloads, Google login or GPU is needed.

The existing predictor stays preserved. Experimental combined models require ground readings; they are separate from satellite-only prediction.''')
add('code', '''from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'pyproject.toml').exists() and (p / 'demo_bundle').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Open this notebook from within the project repository.')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.restore_demo_bundle import restore

count = restore(ROOT / 'demo_bundle', ROOT)
print(f'Verified {count} bundled files; restored only missing identical demo files.')
pd.set_option('display.max_columns', 12)
pd.set_option('display.precision', 3)

DATA = ROOT / 'outputs/india_100_chemical_dataset_20261014'
DEMOS = ROOT / 'outputs/straightforward_demos_20261014'
EXPERIMENT = ROOT / 'outputs/experimental_network_20261014'
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
model_paths = [ROOT / 'outputs/satellite_only_80_20_20261014/model.joblib',
               ROOT / 'outputs/fusion_sklearn_expanded_20261014/fusion_model.joblib']
original_model_hashes = {str(p.relative_to(ROOT)): sha(p) for p in model_paths}''')
add('markdown', '''## 1. Real satellite–ground matches

Each row joins a CPCB sampling event with a nearby Sentinel-2 acquisition. Images must pass the existing cloud, coverage and usable-water-pixel checks. Ground chemical readings are matched by the recorded station name, date and coordinates.

**Inputs available:** ten satellite bands, fecal coliform, pH and dissolved oxygen. **Answer to predict:** measured BOD. Never give the BOD answer to the model as an input.''')
add('code', '''paired = pd.read_csv(DATA / 'paired.csv')
provenance = read_json(DATA / 'provenance.json')
assert sha(DATA / 'paired.csv') == provenance['output_sha256']
assert len(paired) == 100 and paired.observation_id.is_unique
station_labels = {
    'cpcb-0ecd9698ec1a2eb1': 'Anupshahar',
    'cpcb-1c6a3feabae131a4': 'Varanasi',
    'cpcb-baa51b840df3bdc2': 'Bijnor',
}
display(paired.groupby('station_id').size().rename('Matched events').rename(index=station_labels).to_frame())
matching = paired.copy()
matching['Station'] = matching.station_id.map(station_labels)
display(matching[['Station', 'sample_date', 'satellite_local_date', 'temporal_offset_days',
                  'cloud_fraction', 'water_fraction', 'ground_bod', 'ground_fecal_coliform',
                  'ground_ph', 'ground_dissolved_oxygen']].head(9))
print('100 complete pH/DO matches; exact source records are retained in chemical_audit.csv.')''')
add('markdown', '''## 2. Make a satellite-only prediction

The saved working model uses ten satellite band values. Its example input file contains **no fecal-coliform reading or BOD answer**. This cell runs the model rather than merely reading an old prediction.

For a new river section, coordinates and a date first need to be converted into matching satellite features. The model has only been evaluated at these three station areas.''')
add('code', '''import joblib
from scripts.train_satellite_only import predict

satellite_dir = ROOT / 'outputs/satellite_only_80_20_20261014'
satellite_model = joblib.load(satellite_dir / 'model.joblib')
inputs = pd.read_csv(satellite_dir / 'example_inputs.csv')
assert 'ground_bod' not in inputs and 'ground_fecal_coliform' not in inputs
predicted = predict(satellite_model, inputs)
saved = pd.read_csv(satellite_dir / 'test_predictions.csv')
np.testing.assert_allclose(predicted, saved.predicted_bod_mg_l, rtol=1e-10, atol=1e-10)
live_predictions = saved[['station_name', 'sample_date', 'actual_bod_mg_l']].copy()
live_predictions['Predicted BOD (mg/L)'] = predicted
live_predictions['Absolute error (mg/L)'] = np.abs(predicted - saved.actual_bod_mg_l)
display(live_predictions.head(8))
print(f'Predicted all {len(inputs)} satellite-only examples. No training occurred.')''')
add('markdown', '''## 3. Fair satellite-only versus combined comparison

All comparison models use the **same 80 training and 20 testing observations**, the same scikit-learn model settings, and training-only scaling. The combined models additionally use measured ground readings. Lower errors are better; these are not classification-accuracy percentages.''')
add('code', '''metrics = pd.read_csv(DEMOS / 'metrics.csv')
comparison = pd.read_csv(DEMOS / 'test_predictions.csv')
assert len(comparison) == 20 and metrics.n.eq(20).all()
display(metrics[['model', 'input_count', 'n', 'mae', 'rmse', 'r2']])
display(Image(filename=str(DEMOS / 'model_comparison.png')))
display(comparison[['sample_date', 'actual_bod_mg_l', 'satellite_only_bod_mg_l',
                    'combined_all_chemicals_bod_mg_l', 'selected_features_bod_mg_l']].head(6))''')
add('markdown', '''**What happened:** adding ground readings lowered average error on these cases. Satellite-only remains useful when ground readings are unavailable. The combined model's better result cannot be used without supplying its ground inputs.

These test events have been examined in earlier experiments. This is an exploratory demonstration, not an untouched final validation.''')
add('markdown', '''## 4. Feature selection

Inside the 80 training observations, 60 rank/train candidate inputs and 20 validate them. Twelve candidates compare top 3/5/10 bands with top 0/1/2/3 ground readings. The outer 20 test answers never choose the features. The chosen model is refitted using the outer 80 training cases.''')
add('code', '''ranking = pd.read_csv(DEMOS / 'feature_ranking.csv')
candidates = pd.read_csv(DEMOS / 'feature_candidates.csv')
selected = ranking.loc[ranking.selected, 'feature'].tolist()
print('Selected inputs:', ', '.join(selected))
display(ranking[['feature', 'modality', 'score', 'selected']])
display(candidates[['candidate', 'input_count', 'validation_mae', 'selected']].sort_values('validation_mae'))
display(Image(filename=str(DEMOS / 'feature_ranking.png')))''')
add('markdown', '''The selected model kept **B8, B8A, B11 and fecal coliform**. Ranking pH or DO highly does not guarantee that including it improves the smaller model. Feature selection is based on this small dataset, not a universal statement about river chemistry.''')
add('markdown', '''## 5. SHAP: why did the combined model predict that number?

Actual Kernel SHAP explains three fixed test examples, one per station, relative to a background sampled only from training data. Red contributions raise predicted BOD; blue contributions lower it. The full chart explains the all-chemical combined model, not the satellite-only or selected-input model.

Contributions explain model behavior and are reference-dependent. They do not establish physical causes. Correlated bands can make individual contributions harder to interpret.''')
add('code', '''explanations = pd.read_csv(DEMOS / 'shap_explanations.csv')
display(Image(filename=str(DEMOS / 'shap_examples.png')))
display(explanations[['station_name', 'predicted_bod_mg_l', 'reference_bod_mg_l',
                      'satellite_contribution_mg_l', 'ground_contribution_mg_l']])
reconstructed = (explanations.reference_bod_mg_l + explanations.satellite_contribution_mg_l
                 + explanations.ground_contribution_mg_l)
np.testing.assert_allclose(reconstructed, explanations.predicted_bod_mg_l, atol=1e-6)
print('Reference + satellite contributions + ground contributions reconstruct each estimate.')''')
add('markdown', '''## 6. Small experimental network: gate, constraint and station adversary

A separate two-branch PyTorch model has independent switches:

- **Gate:** learn satellite/ground weights from the date difference.
- **Constraint:** penalize negative raw BOD during training.
- **Station adversary:** use a station classifier with reversed gradients to encourage less station-specific representations.

These are modest mechanism demonstrations. The constraint is not a full physics-equation PINN. Station regularization is not proven transfer to a foreign river. The original scikit-learn predictor stays preserved.''')
add('code', '''experimental = read_json(EXPERIMENT / 'run.json')
experimental_metrics = pd.read_csv(EXPERIMENT / 'metrics.csv')
display(experimental_metrics[['model', 'epochs', 'n', 'mae', 'rmse', 'r2']])
display(Image(filename=str(EXPERIMENT / 'comparison.png')))
print('Baseline, each switch alone, and all three together were evaluated on the same 20 cases.')''')
add('markdown', '''### Gate weights

The chart holds one example's readings constant and changes only the satellite–sampling gap. With the gate off, branch weights stay at 0.5. The learned gate reduced satellite weight as the gap increased in this run. These weights are not confidence percentages or SHAP contributions.''')
add('code', '''display(Image(filename=str(EXPERIMENT / 'gate_weights.png')))
sweep = pd.read_csv(EXPERIMENT / 'gate_sweep.csv')
display(sweep[sweep.model.eq('all_components')][['absolute_gap_days', 'satellite_weight', 'ground_weight', 'predicted_bod_mg_l']])
np.testing.assert_allclose(sweep.satellite_weight + sweep.ground_weight, 1.0, atol=1e-6)''')
add('markdown', '''### Training constraint and station classifier

The saved loss histories show that the constraint was actually active and the classifier trained. A mechanism can run without improving accuracy. In particular, the classifier still identifies stations well: station invariance was not achieved.''')
add('code', '''mechanisms = []
for name, result in experimental['variants'].items():
    mechanisms.append({'Variant': name, 'Gate': result['switches']['gate'],
                       'Constraint': result['switches']['constraint'],
                       'Adversarial': result['switches']['station_adversarial'],
                       'Epochs with applied penalty': result['epochs_with_constraint_penalty'],
                       'Station training accuracy': result['station_classifier_training_accuracy']})
display(pd.DataFrame(mechanisms))
display(Image(filename=str(EXPERIMENT / 'training_losses.png')))
display(Image(filename=str(EXPERIMENT / 'station_adversarial_loss.png')))''')
add('markdown', '''## 7. What can we demonstrate?

**Working:** real data matching, saved-model predictions, fair input comparison, training-only spectral/chemical feature selection, actual SHAP explanations, learned date-gap weights, a training penalty, and station-adversarial gradient reversal.

**Results are mixed:** combined inputs and selected features helped this sample; the gate helped the small experimental network; the constraint and adversarial-only variants did not improve their scores. All results, including ineffective variants, are retained.

**Scope:** three Ganga stations, 100 observations, BOD estimation. This does not establish overall water safety, future forecasting, an image CNN, broad unseen-river performance or a full physics-informed model. Source sampling/QC, coordinate/time and water-pixel assumptions remain documented.

The comparison models are separate experiments; no saved working model was automatically replaced.''')
add('code', '''assert {str(p.relative_to(ROOT)): sha(p) for p in model_paths} == original_model_hashes
assert sha(DATA / 'paired.csv') == provenance['output_sha256']
print('Demonstration completed: original models unchanged, dataset verified, predictions reproduced, SHAP sums verified, gate weights normalized.')''')
add('markdown', '''## Reproduction and supporting documentation

- [Dataset sources and units](../docs/chemical_dataset.md)
- [Comparisons, feature selection and SHAP](../docs/straightforward_demonstrations.md)
- [Experimental mechanisms and on/off commands](../docs/experimental_network.md)
- [Saved satellite-only model](../docs/working_model.md)

To rerun experiments, follow those guides and use fresh output directories. This notebook reads their saved outputs so the presentation remains fast and independent of live satellite services.''')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    path = root / 'notebooks/river_water_quality_demo.ipynb'
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13.2'}}, 'cells': cells}
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'Built {len(cells)} cells: {path}')

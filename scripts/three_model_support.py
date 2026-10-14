"""Verified frozen models and train-only reference data for presentation demos."""
import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
import torch

from river_water_quality.training.experimental_network import predict_network
from river_water_quality.training.satellite_refinements import predict as ratio_predict
from river_water_quality.training.sklearn_demonstrations import predict as combined_predict
from river_water_quality.training.sklearn_fusion import SATELLITE_FEATURES
from river_water_quality.training.splits import acquisition_group

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    paired_path = ROOT / 'outputs/india_100_chemical_dataset_20261014/paired.csv'
    split_path = ROOT / 'outputs/satellite_only_80_20_20261014/split_assignments.csv'
    new_path = ROOT / 'outputs/new_station_matches_v2_20261014/new_paired.csv'
    d, splits, new = pd.read_csv(paired_path), pd.read_csv(split_path), pd.read_csv(new_path)
    train = d.set_index('observation_id').loc[splits.loc[splits.split.eq('train'), 'observation_id']].reset_index()
    seen = d.set_index('observation_id').loc[splits.loc[splits.split.eq('test'), 'observation_id']].reset_index()
    if len(train) != 80 or len(seen) != 20 or len(new) != 30:
        raise ValueError('Unexpected data cohorts')
    if set(train.station_id) & set(new.station_id):
        raise ValueError('New station overlap')
    strict = new.loc[~new.scene_id.map(acquisition_group).isin(set(train.scene_id.map(acquisition_group)))].copy()
    examples = pd.concat([seen.groupby('station_id').head(1).assign(cohort='familiar'),
        strict.groupby('station_id').head(1).assign(cohort='unseen')], ignore_index=True)
    spec = json.loads((ROOT / 'configs/working_satellite_model.json').read_text())
    demos = json.loads((ROOT / 'outputs/straightforward_demos_20261014/run.json').read_text())
    networks = json.loads((ROOT / 'outputs/experimental_network_20261014/run.json').read_text())
    if sha(paired_path) != networks['input_sha256'] or sha(split_path) != networks['split_sha256']:
        raise ValueError('Training data hash mismatch')
    definitions = [
        ('selected_four_inputs', ROOT / 'outputs/straightforward_demos_20261014/selected_features.joblib',
         demos['model_hashes']['selected_features'], 'combined'),
        ('all_components', ROOT / 'outputs/experimental_network_20261014/all_components.pt',
         networks['variants']['all_components']['model_sha256'], 'network'),
        ('satellite_ratios', ROOT / spec['model_path'], spec['model_sha256'], 'ratio')]
    models = {}
    torch.set_num_threads(1)
    for name, path, expected, kind in definitions:
        if sha(path) != expected:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
        if kind == 'network':
            model = torch.load(path, map_location='cpu', weights_only=True)
            features = model['features'] + ['temporal_offset_days']
            predictor = lambda f, m=model: predict_network(m, f)['predicted']
        else:
            model = joblib.load(path)
            features = SATELLITE_FEATURES if kind == 'ratio' else model['features']
            fn = ratio_predict if kind == 'ratio' else combined_predict
            predictor = lambda f, m=model, fn=fn: fn(m, f)
        models[name] = {'path': path, 'sha256': expected, 'features': features, 'predict': predictor}
    return train, seen, strict, examples, models, {
        str(p.relative_to(ROOT)): sha(p) for p in [paired_path, split_path, new_path]}

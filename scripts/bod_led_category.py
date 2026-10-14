"""Run the BOD-led category demo or classify raw satellite rows with station_id."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from river_water_quality.bod_category import METHOD, WEIGHTS, assess, classify, training_profiles
from river_water_quality.training.satellite_refinements import predict

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/bod_led_category_results'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='CSV containing station_id and ten satellite bands')
    parser.add_argument('--output', type=Path, help='New output CSV; default prints categories')
    parser.add_argument('--demo', action='store_true', help='Evaluate saved 20 cases and save report')
    args = parser.parse_args()
    if args.demo == bool(args.input) or (args.demo and args.output):
        parser.error('Choose --demo or --input CSV [--output CSV]')
    specification = json.loads((ROOT / 'configs/working_satellite_model.json').read_text())
    model_path = ROOT / specification['model_path']
    if digest(model_path) != specification['model_sha256']:
        raise ValueError('Model hash mismatch; restore demo bundle')
    model = joblib.load(model_path)
    if args.demo:
        paired_path = ROOT / 'outputs/india_100_chemical_dataset_20261014/paired.csv'
        split_path = ROOT / 'outputs/satellite_only_80_20_20261014/split_assignments.csv'
        paired, split = pd.read_csv(paired_path), pd.read_csv(split_path)
        profiles = training_profiles(paired, split)
        test = paired.set_index('observation_id').loc[
            split.loc[split.split.eq('test'), 'observation_id']].reset_index()
        result = classify(test, predict(model, test), profiles)
        result['measured_bod_mg_l'] = test.ground_bod.to_numpy()
        # Reference uses the actual held-out chemistry, never the station averages.
        result['reference_category'] = [assess(r.ground_bod, r.ground_ph,
            r.ground_dissolved_oxygen, r.ground_fecal_coliform)['bod_led_category']
            for r in test.itertuples()]
        train = paired[paired.observation_id.isin(split.loc[split.split.eq('train'), 'observation_id'])]
        baseline = classify(test, [train.ground_bod.mean()] * len(test), profiles)
        training_categories = pd.Series([assess(r.ground_bod, r.ground_ph,
            r.ground_dissolved_oxygen, r.ground_fecal_coliform)['bod_led_category']
            for r in train.itertuples()])
        majority = training_categories.mode().iloc[0]
        result['mean_bod_baseline_category'] = baseline.bod_led_category
        result['category_agrees'] = result.bod_led_category.eq(result.reference_category)
        metrics = {'method': METHOD, 'weights': WEIGHTS,
            'thresholds': {'Best': 'score >= 80', 'Good': '60 <= score < 80', 'Bad': 'score < 60'},
            'assumed_water_conditions': {'temperature_c': 25, 'pressure_atm': 1, 'salinity_ppt': 0},
            'test_count': len(test), 'category_agreement': float(result.category_agrees.mean()),
            'mean_bod_baseline_agreement': float(result.mean_bod_baseline_category.eq(result.reference_category).mean()),
            'training_majority_category': majority,
            'training_majority_agreement': float(result.reference_category.eq(majority).mean()),
            'reference_counts': result.reference_category.value_counts().to_dict(),
            'predicted_counts': result.bod_led_category.value_counts().to_dict(),
            'boundary_sensitive_count': int(result.boundary_sensitive.sum()),
            'model_sha256': digest(model_path), 'paired_sha256': digest(paired_path),
            'split_sha256': digest(split_path),
            'limitations': ['Custom categories; no official WQI or safety certification.',
                'Reference also uses assumed temperature, pressure and salinity.',
                'Repeatedly examined 20 cases: exploratory agreement, not independent accuracy.',
                'Boundary flag varies BOD by historical MAE; not a confidence interval.']}
        REPORT.mkdir(parents=True, exist_ok=True)
        profiles.to_csv(REPORT / 'station_profiles.csv', index=False)
        test[['observation_id', 'station_id'] + [c for c in test if c.startswith('satellite_B')]].to_csv(
            REPORT / 'example_inputs.csv', index=False)
        result.to_csv(REPORT / 'test_predictions.csv', index=False)
        pd.crosstab(result.reference_category, result.bod_led_category).reindex(
            index=['Best', 'Good', 'Bad'], columns=['Best', 'Good', 'Bad'], fill_value=0
        ).fillna(0).astype(int).to_csv(REPORT / 'confusion.csv')
        metrics['station_profiles_sha256'] = digest(REPORT / 'station_profiles.csv')
        (REPORT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        print(json.dumps(metrics, indent=2))
    else:
        metrics = json.loads((REPORT / 'metrics.json').read_text())
        profile_path = REPORT / 'station_profiles.csv'
        if digest(profile_path) != metrics['station_profiles_sha256']:
            raise ValueError('Station profiles changed; rerun --demo')
        if digest(model_path) != metrics['model_sha256']:
            raise ValueError('Profiles/report refer to a different model')
        frame = pd.read_csv(args.input)
        if frame.empty:
            raise ValueError('Empty input')
        result = classify(frame, predict(model, frame), pd.read_csv(profile_path))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8', newline='') as stream:
                result.to_csv(stream, index=False)
        print(result[['station_id', 'bod_led_category', 'boundary_sensitive']].to_string(index=False))


if __name__ == '__main__':
    main()

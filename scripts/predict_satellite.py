"""Predict using the explicitly selected satellite model, verifying its hash."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from river_water_quality.training.satellite_refinements import predict


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    specification = json.loads((root / 'configs/working_satellite_model.json').read_text(encoding='utf-8'))
    model_path = (root / specification['model_path']).resolve()
    if not model_path.is_relative_to(root / 'outputs'):
        raise ValueError('Selected checkpoint must remain within project outputs')
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != specification['model_sha256']:
        raise ValueError('Selected model hash mismatch; restore the preserved bundle')
    model = joblib.load(model_path)
    if model['input_mode'] != specification['input_mode'] or model['target'] != specification['target']:
        raise ValueError('Selected model schema mismatch')
    frame = pd.read_csv(args.input)
    if frame.empty:
        raise ValueError('Input must contain at least one observation')
    values = predict(model, frame)
    result = pd.DataFrame({'predicted_bod_mg_l': values})
    if 'observation_id' in frame:
        result.insert(0, 'observation_id', frame.observation_id.to_numpy())
    if args.output:
        if args.output.exists():
            raise ValueError('Output exists; choose a new file')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8', newline='') as stream:
            result.to_csv(stream, index=False)
        print(f'Saved {len(result)} predictions to {args.output}')
    else:
        print(result.to_string(index=False))


if __name__ == '__main__':
    main()

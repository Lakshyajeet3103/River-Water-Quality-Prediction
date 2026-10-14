"""Add explicitly assumption-based WQI scenarios to existing BOD predictions.

No training, reference-WQI evaluation or notebook changes are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from river_water_quality.training.sklearn_demonstrations import predict
from river_water_quality.wqi import calculate_scenario_wqi, METHOD_SOURCE, OXYGEN_SOURCE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def demo_input():
    """One saved test observation for a calculator smoke demonstration only."""
    paired = Path('outputs/india_100_chemical_dataset_20261014/paired.csv')
    model_path = Path('outputs/satellite_only_80_20_20261014/model.joblib')
    provenance = json.loads(paired.with_name('provenance.json').read_text(encoding='utf-8'))
    if digest(paired) != provenance['output_sha256']:
        raise ValueError('Saved chemical dataset changed')
    if digest(model_path) != provenance['preserved_model_hashes'][str(model_path)]:
        raise ValueError('Saved model changed')
    splits = pd.read_csv('outputs/satellite_only_80_20_20261014/split_assignments.csv')
    identifier = splits.loc[splits.split.eq('test'), 'observation_id'].iloc[0]
    frame = pd.read_csv(paired).query('observation_id == @identifier').copy()
    if len(frame) != 1:
        raise ValueError('Expected one unique demo event')
    model = joblib.load(model_path)
    frame['predicted_bod_mg_l'] = predict(model, frame[model['features']])
    inputs = frame[['observation_id', 'station_name', 'sample_date', 'predicted_bod_mg_l',
                    'ground_ph', 'ground_dissolved_oxygen', 'ground_fecal_coliform']]
    return inputs, {'paired_sha256': digest(paired), 'model_path': str(model_path),
                    'model_sha256': digest(model_path), 'note': 'One saved satellite-only test event; no accuracy evaluation.'}


def calculate_rows(frame, temperatures, pressure, salinity):
    required = ['predicted_bod_mg_l', 'ground_ph', 'ground_dissolved_oxygen', 'ground_fecal_coliform']
    if any(name not in frame for name in required) or frame.empty:
        raise ValueError(f'Nonempty input requires columns: {required}')
    rows = []
    for index, record in enumerate(frame.to_dict('records'), 1):
        for t in temperatures:
            row = {'input_record': index, 'observation_id': record.get('observation_id', ''),
                   'station_name': record.get('station_name', ''), 'sample_date': record.get('sample_date', ''),
                   **{name: record[name] for name in required},
                   'assumed_temperature_c': t, 'assumed_pressure_atm': pressure,
                   'assumed_salinity_ppt': salinity, 'calculation_type': 'assumption-based WQI scenario'}
            try:
                result = calculate_scenario_wqi(bod_mg_l=record['predicted_bod_mg_l'],
                    ph=record['ground_ph'], fecal_per_100ml=record['ground_fecal_coliform'],
                    do_mg_l=record['ground_dissolved_oxygen'], assumed_temperature_c=t,
                    assumed_pressure_atm=pressure, assumed_salinity_ppt=salinity)
                row.update({k: result[k] for k in ['wqi', 'category', 'equilibrium_do_mg_l', 'do_percent_saturation']})
                row.update({f'sub_index_{k}': value for k, value in result['sub_indices'].items()})
                row.update({f'contribution_{k}': value for k, value in result['weighted_contributions'].items()})
                row['status'] = 'formula_warning' if result['formula_warnings'] else 'ok'
                row['message'] = '; '.join(result['formula_warnings'])
            except ValueError as exc:
                # Unsupported/missing data remains visible with no fabricated score/category.
                row.update({'status': 'invalid_input', 'message': str(exc), 'wqi': None, 'category': None})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=Path)
    group.add_argument('--demo', action='store_true')
    parser.add_argument('--assumed-temperatures', type=float, nargs='+', required=True)
    parser.add_argument('--assumed-pressure-atm', type=float, required=True)
    parser.add_argument('--assumed-salinity-ppt', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; choose a new directory')
    if args.demo:
        frame, source = demo_input()
    else:
        frame, source = pd.read_csv(args.input), {'input_path': str(args.input), 'input_sha256': digest(args.input)}
    result = calculate_rows(frame, args.assumed_temperatures,
                            args.assumed_pressure_atm, args.assumed_salinity_ppt)
    args.output.mkdir(parents=True)
    frame.to_csv(args.output / 'inputs.csv', index=False)
    result.to_csv(args.output / 'wqi_scenarios.csv', index=False)
    report = {'source': source, 'method_source': METHOD_SOURCE, 'oxygen_source': OXYGEN_SOURCE,
              'temperatures_c': args.assumed_temperatures, 'pressure_atm': args.assumed_pressure_atm,
              'salinity_ppt': args.assumed_salinity_ppt, 'input_rows': len(frame), 'scenario_rows': len(result),
              'status_counts': result.status.value_counts().to_dict(),
              'limitations': ['Water temperature, pressure and salinity are assumed, not measured.',
                              'Reported fecal MPN/100mL is used as a bacterial concentration estimate.',
                              'Predicted BOD plus measured station inputs; not coordinates-only or safe-to-drink assessment.',
                              'No scores outside the supported oxygen-saturation range; inspect status before using WQI.',
                              'Temperature sweep is illustrative, not a confidence interval or full uncertainty analysis.'],
              'output_sha256': digest(args.output / 'wqi_scenarios.csv')}
    (args.output / 'provenance.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(result[['assumed_temperature_c', 'status', 'wqi', 'category']].to_string(index=False))
    print('Assumption-based scenario only; conditions are not measured.')


if __name__ == '__main__':
    main()

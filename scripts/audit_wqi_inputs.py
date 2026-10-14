"""Audit the existing WQI inputs and exact-match public CPCB physical records.

Downloads only the two bounded UP physical exports when --download is supplied.
Does not calculate WQI, change models, fill missing values or overwrite outputs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = 'https://nwdp.nwic.gov.in'
DATASET = BASE + '/en/dataset/surface-water-quality-manual-physical-parameters-cpcb'
RESOURCES = ['3aac0149-d815-42a6-975a-2e5c757256f6', 'caf05460-e4bc-498d-85f3-2ed21a595e89']
KEYS = ['station_name', 'sample_date', 'latitude', 'longitude']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for resource_id in RESOURCES:
        response = requests.get(BASE + '/api/3/action/resource_show',
                                params={'id': resource_id}, timeout=(20, 45))
        response.raise_for_status()
        payload = response.json()
        if not payload.get('success'):
            raise ValueError('Portal resource lookup failed')
        metadata = payload['result']
        url = metadata['url']
        if not url.startswith(BASE + '/dataset/') or not url.endswith('.csv'):
            raise ValueError('Unexpected download URL')
        filename = url.rsplit('/', 1)[-1]
        path = folder / filename
        if path.exists():
            saved = json.loads(path.with_suffix('.download.json').read_text(encoding='utf-8'))
            if saved['sha256'] != digest(path) or saved['resource_id'] != resource_id:
                raise ValueError('Cached download changed')
            continue
        response = requests.get(url, timeout=(20, 60), stream=True)
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > 5 * 1024 * 1024:
                raise ValueError('Physical export exceeded 5 MiB')
            chunks.append(chunk)
        content = b''.join(chunks)
        # Validate the payload before saving; error pages must not masquerade as CSV.
        if b'Temperature' not in content.splitlines()[0]:
            raise ValueError('Missing temperature CSV header')
        with path.open('xb') as stream:
            stream.write(content)
        sidecar = {'resource_id': resource_id, 'dataset_url': DATASET,
                   'url': response.url, 'sha256': digest(path), 'size_bytes': size,
                   'retrieved_utc': datetime.now(timezone.utc).isoformat()}
        path.with_suffix('.download.json').write_text(json.dumps(sidecar, indent=2), encoding='utf-8')


def join_physical(paired, physical):
    """Exact one-to-one audit; incomplete matches are reported instead of filled."""
    if not paired.observation_id.is_unique or physical.duplicated(KEYS).any():
        raise ValueError('Ambiguous station/date/coordinate join')
    result = paired.merge(physical, on=KEYS, how='left', validate='one_to_one', sort=False,
                          indicator='physical_match')
    if result.observation_id.tolist() != paired.observation_id.tolist():
        raise ValueError('Event order changed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--physical-dir', default='data/raw/cpcb_physical_wqi_20261014')
    parser.add_argument('--paired', default='outputs/india_100_chemical_dataset_20261014/paired.csv')
    parser.add_argument('--output', default='docs/wqi_input_audit')
    args = parser.parse_args()
    folder, out, source = Path(args.physical_dir), Path(args.output), Path(args.paired)
    if out.exists():
        raise ValueError('Output already exists; choose a fresh directory')
    if args.download:
        fetch(folder)
    paired = pd.read_csv(source)
    parent = json.loads(source.with_name('provenance.json').read_text(encoding='utf-8'))
    if digest(source) != parent['output_sha256'] or len(paired) != 100:
        raise ValueError('Expected the preserved 100-event chemical dataset')
    models = parent['preserved_model_hashes']
    for path, expected in models.items():
        if digest(path) != expected:
            raise ValueError('Preserved model hash changed')
    inputs = {}
    for name in ['ground_bod', 'ground_ph', 'ground_dissolved_oxygen', 'ground_fecal_coliform']:
        values = pd.to_numeric(paired[name], errors='coerce')
        finite = np.isfinite(values)
        valid = finite & (values.between(0, 14) if name == 'ground_ph'
                          else values.ge(1) if name == 'ground_fecal_coliform' else values.ge(0))
        inputs[name] = {'finite_count': int(finite.sum()), 'valid_count': int(valid.sum()),
                        'minimum': float(values[finite].min()), 'maximum': float(values[finite].max())}
    tables, sources = [], []
    for path in sorted(folder.glob('*.csv')):
        saved = json.loads(path.with_suffix('.download.json').read_text(encoding='utf-8'))
        if digest(path) != saved['sha256']:
            raise ValueError('Physical source hash mismatch')
        raw = pd.read_csv(path)
        field_names = list(raw.columns)
        temp = [c for c in field_names if c.startswith('Temperature')]
        conductance = [c for c in field_names if c.startswith('Electric Conductivity')]
        if len(temp) != 1 or len(conductance) != 1:
            raise ValueError('Unexpected physical parameter schema')
        table = raw[['Station', 'Latitude', 'Longitude', 'Data Acquisition Time',
                     temp[0], conductance[0]]].copy()
        table = table.rename(columns={'Station': 'station_name', 'Latitude': 'latitude',
                                      'Longitude': 'longitude', temp[0]: 'water_temperature_c',
                                      conductance[0]: 'conductivity_as_reported'})
        table['sample_date'] = pd.to_datetime(table['Data Acquisition Time'],
            format='%d-%m-%Y %H:%M', errors='raise').dt.strftime('%Y-%m-%d')
        table['physical_source_record'] = np.arange(1, len(table) + 1)
        table['physical_source_file'] = path.name
        table['physical_source_sha256'] = saved['sha256']
        table['physical_source_url'] = saved['url']
        table['conductivity_source_column'] = conductance[0]
        table['temperature_source_column'] = temp[0]
        for name in ['water_temperature_c', 'conductivity_as_reported']:
            table[name] = pd.to_numeric(table[name], errors='coerce')
        tables.append(table)
        sources.append({**saved, 'source_rows': len(raw), 'headers': field_names,
                        'temperature_source_column': temp[0], 'conductivity_source_column': conductance[0]})
    if len(tables) != 2:
        raise ValueError('Expected both physical exports')
    matched = join_physical(paired, pd.concat(tables, ignore_index=True))
    temperature_ok = np.isfinite(matched.water_temperature_c) & matched.water_temperature_c.between(0, 40)
    matched['temperature_valid_for_usgs_range'] = temperature_ok
    matched['measured_pressure_available'] = False
    matched['measured_DO_saturation_available'] = False
    matched['wqi_ready_without_conversion_assumptions'] = False
    audit = matched[['observation_id', 'station_id', *KEYS, 'physical_match',
                     'Data Acquisition Time', 'water_temperature_c', 'conductivity_as_reported',
                     'temperature_source_column', 'conductivity_source_column',
                     'temperature_valid_for_usgs_range', 'physical_source_record',
                     'physical_source_file', 'physical_source_sha256', 'physical_source_url',
                     'measured_pressure_available', 'measured_DO_saturation_available',
                     'wqi_ready_without_conversion_assumptions']]
    summary = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'rows': len(paired),
               'paired_sha256': digest(source), 'ground_inputs': inputs,
               'physical_sources': sources, 'exact_physical_matches': int(matched.physical_match.eq('both').sum()),
               'valid_temperature_count': int(temperature_ok.sum()),
               'valid_temperature_range_c': [float(matched.loc[temperature_ok, 'water_temperature_c'].min()),
                                             float(matched.loc[temperature_ok, 'water_temperature_c'].max())]
                   if temperature_ok.any() else None,
               'conductivity_nonnegative_finite_count': int((np.isfinite(matched.conductivity_as_reported)
                                                          & matched.conductivity_as_reported.ge(0)).sum()),
               'measured_pressure_count': 0, 'measured_DO_saturation_count': 0,
               'ready_without_conversion_assumptions_count': 0,
               'model_hashes_preserved': models,
               'join_rule': 'Exact station name, displayed date, latitude, longitude; no filling or rounding.',
               'limitations': ['Temperature and conductance headers/units require source-method confirmation.',
                               'No measured barometric pressure or DO percent saturation in inspected exports.',
                               'Fecal MPN/100mL is used as reported; compatibility with index counts/100mL must be documented.',
                               'DO conversion would require explicit pressure and salinity/conductivity assumptions.',
                               'Date/location matching does not establish same bottle or laboratory QC.']}
    out.mkdir(parents=True)
    audit.to_csv(out / 'physical_matches.csv', index=False)
    summary['audit_sha256'] = digest(out / 'physical_matches.csv')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['rows', 'ground_inputs', 'exact_physical_matches',
        'valid_temperature_count', 'valid_temperature_range_c', 'conductivity_nonnegative_finite_count',
        'ready_without_conversion_assumptions_count']}, indent=2))


if __name__ == '__main__':
    main()

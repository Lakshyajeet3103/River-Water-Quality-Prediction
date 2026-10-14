"""Resumable real Sentinel-2 extraction: 30 matches across at least three new stations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from river_water_quality.data.ingestion import load_dataset
from river_water_quality.data.hybrid import paired_table
from river_water_quality.data.matching import MatchingSettings, match_observations
from river_water_quality.data.schemas import validate_satellite
from river_water_quality.data.sentinel import extract_event, validate_extraction_settings
from river_water_quality.utils.config import load_yaml

STATIONS = ['cpcb-802a0bf7ab0a50d0', 'cpcb-65a5002389d94ee5', 'cpcb-d9f0323fba4e4a7c',
            'cpcb-06c5a8086252cc40', 'cpcb-a78396bed449fa69', 'cpcb-d80ef48e84ee36a6']
KEYS = ['station_name', 'sample_date', 'latitude', 'longitude']
CHEMICAL = {'Potential of Hydrogen (pH)': 'ground_ph', 'Dissolved oxygen (mg/L)': 'ground_dissolved_oxygen'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def chemicals():
    tables, hashes = [], {}
    for path in sorted(Path('data/raw/cpcb_chemical_check_20261014').glob('*.csv')):
        saved = json.loads(path.with_suffix('.download.json').read_text(encoding='utf-8'))
        if sha(path) != saved['sha256']:
            raise ValueError('Chemical source changed')
        hashes[path.name] = sha(path)
        raw = pd.read_csv(path)
        table = raw[['Station', 'Latitude', 'Longitude', 'Data Acquisition Time', *CHEMICAL]].copy()
        table = table.rename(columns={'Station': 'station_name', 'Latitude': 'latitude', 'Longitude': 'longitude', **CHEMICAL})
        table['sample_date'] = pd.to_datetime(table['Data Acquisition Time'], format='%d-%m-%Y %H:%M', errors='raise').dt.strftime('%Y-%m-%d')
        table['chemical_source_record'] = np.arange(1, len(raw) + 1)
        table['chemical_source_file'] = path.name
        table['chemical_source_sha256'] = saved['sha256']
        table['chemical_source_url'] = saved['url']
        tables.append(table)
    if len(tables) != 2:
        raise ValueError('Expected both preserved chemical exports')
    table = pd.concat(tables, ignore_index=True)
    if table.duplicated(KEYS).any():
        raise ValueError('Duplicate chemical join keys')
    for name in CHEMICAL.values():
        table[name] = pd.to_numeric(table[name], errors='coerce')
    table = table[np.isfinite(table[list(CHEMICAL.values())]).all(axis=1)
                  & table.ground_ph.between(0, 14) & table.ground_dissolved_oxygen.ge(0)]
    return table, hashes


def save(out, state, original, ground, chemistry):
    pd.DataFrame(state['attempts']).to_csv(out / 'event_outcomes.csv', index=False)
    pd.DataFrame(state['audits']).to_csv(out / 'extraction_audit.csv', index=False)
    new = pd.DataFrame(state['pairs'])
    if new.empty:
        return
    if not new.observation_id.is_unique or new.observation_id.isin(original.observation_id).any():
        raise ValueError('Duplicate or existing observation in expansion')
    enriched = new.merge(chemistry, on=KEYS, how='left', validate='one_to_one', indicator=True)
    if not enriched._merge.eq('both').all():
        raise ValueError('Missing exact chemical match')
    paired = enriched[[*new.columns, *CHEMICAL.values()]]
    paired.to_csv(out / 'new_paired.csv', index=False)
    pd.concat([original, paired], ignore_index=True).sort_values(['station_id', 'sample_date', 'observation_id']).to_csv(out / 'paired.csv', index=False)
    audit = enriched[['observation_id', *KEYS, *CHEMICAL.values(), 'Data Acquisition Time',
                      'chemical_source_file', 'chemical_source_record', 'chemical_source_sha256', 'chemical_source_url']]
    audit.to_csv(out / 'chemical_audit.csv', index=False)
    pd.DataFrame(state['features']).drop_duplicates(['station_id', 'scene_id', 'band']).to_csv(out / 'satellite_features.csv', index=False)
    ground[ground.observation_id.isin(new.observation_id)].to_csv(out / 'new_ground.csv', index=False)
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'inputs': state['signature'],
              'new_matches': len(new), 'total_matches': len(original) + len(new),
              'station_count': int(pd.concat([original.station_id, new.station_id]).nunique()),
              'new_station_counts': new.groupby(['station_id', 'station_name']).size().reset_index(name='matches').to_dict('records'),
              'attempts': len(state['attempts']), 'status_counts': pd.Series([r['status'] for r in state['attempts']]).value_counts().to_dict(),
              'paired_sha256': sha(out / 'paired.csv'), 'new_paired_sha256': sha(out / 'new_paired.csv'),
              'selection': 'Named new Ganga candidates; complete BOD/fecal/pH/DO, newest-first dates; no target-magnitude ranking; up to ten accepted events per station, stop at 30 total. Skip all-land ROIs after three attempts with zero water and no successes.',
              'limitations': ['Station coordinates, sample timezone and source laboratory QC remain provisional.',
                              'Real ROI spectral means, not raw image patches; same SCL water/cloud thresholds as previous extraction.',
                              'Retrospective matching within seven days, not forecasting.',
                              'Multiple observations may share sensing acquisitions; group-aware splits are required.',
                              'Existing dataset and saved models remain unchanged; WQI assumptions/categories unchanged.']}
    write_json(out / 'run.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/new_station_matches_v2_20261014'))
    args = parser.parse_args()
    out = args.output
    source = load_dataset('data/interim/cpcb_preparation_001/ground.csv',
                          'data/interim/cpcb_preparation_001/ground.provenance.json', kind='cpcb')
    original_path = Path('outputs/india_100_chemical_dataset_20261014/paired.csv')
    original = pd.read_csv(original_path)
    if len(original) != 100 or set(STATIONS) & set(original.station_id):
        raise ValueError('Expected 100 preserved events and genuinely new stations')
    chemistry, chemistry_hashes = chemicals()
    ground = source.data
    values = ground.pivot(index='observation_id', columns='parameter', values='value')
    eligible = values.dropna(subset=['bod', 'fecal_coliform']).index
    events = ground[ground.observation_id.isin(eligible) & ground.station_id.isin(STATIONS)][
        ['observation_id', 'station_id', *KEYS]].drop_duplicates('observation_id')
    events = events.merge(chemistry[KEYS], on=KEYS, how='inner', validate='one_to_one')
    events = events.sort_values(['station_id', 'sample_date', 'observation_id'], ascending=[True, False, True])
    settings = load_yaml('configs/extraction_pilot.yaml')
    settings['max_events'] = 120
    validate_extraction_settings(settings)
    matching = MatchingSettings.from_config(load_yaml('configs/pilot_matching.yaml'))
    signature = {'ground_sha256': source.source_sha256, 'ground_manifest_sha256': source.manifest_sha256,
                 'original_sha256': sha(original_path), 'chemical_sha256': chemistry_hashes,
                 'settings': settings, 'matching_config_sha256': sha('configs/pilot_matching.yaml'),
                 'stations': STATIONS, 'quota_per_station': 10, 'attempt_cap_per_station': 40,
                 'total_attempt_cap': 120, 'target_new_matches': 30}
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / 'checkpoint.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
        'signature': signature, 'attempts': [], 'audits': [], 'features': [], 'pairs': []}
    if state['signature'] != signature:
        raise ValueError('Resume input/settings changed')
    events.to_csv(out / 'planned_events.csv', index=False)
    ee.Initialize(project=settings['project_id'])
    for station in STATIONS:
        if len(state['pairs']) >= 30 or len(state['attempts']) >= 120:
            break
        completed = {a['observation_id'] for a in state['attempts']}
        count = sum(r['station_id'] == station for r in state['pairs'])
        attempts = sum(r['station_id'] == station for r in state['attempts'])
        errors = 0
        for event in events[events.station_id.eq(station)].to_dict('records'):
            if count >= 10 or attempts >= 40 or len(state['pairs']) >= 30 or len(state['attempts']) >= 120:
                break
            if event['observation_id'] in completed:
                continue
            print(f"{event['station_name']} {event['sample_date']}: {count}/10 accepted, attempt {attempts + 1}", flush=True)
            rows, logs = [], []
            try:
                rows, logs = extract_event(ee, event, settings)
                if rows:
                    spectral = pd.DataFrame(rows)
                    validate_satellite(spectral)
                    result = match_observations(ground[ground.observation_id.eq(event['observation_id'])], spectral, matching)
                    paired = paired_table(result)
                    if len(paired) != 1:
                        raise ValueError('Expected exactly one real pair')
                    state['pairs'].extend(paired.to_dict('records'))
                    state['features'].extend(rows)
                    count += 1
                status = logs[-1]['status'] if logs else 'no_audit'
                errors = 0
            except Exception as exc:
                status = 'query_error'
                logs.append({'observation_id': event['observation_id'], 'status': status,
                             'error_type': type(exc).__name__, 'message': str(exc)[:300]})
                errors += 1
            state['attempts'].append({**event, 'status': status})
            state['audits'].extend(logs)
            attempts += 1
            write_json(state_path, state)
            save(out, state, original, ground, chemistry)
            print(f'  {status}; {count}/10 accepted', flush=True)
            if errors >= 3:
                raise RuntimeError('Three consecutive query errors; saved checkpoint, stopping requests')
            station_audits = [a for a in state['audits'] if a.get('station_id') == station and 'roi_water' in a]
            if count == 0 and attempts >= 3 and station_audits and all(a['roi_water'] == 0 for a in station_audits):
                print('Skipping candidate: every inspected ROI has zero classified water.', flush=True)
                break
    save(out, state, original, ground, chemistry)
    if sha(original_path) != signature['original_sha256']:
        raise ValueError('Original dataset changed')
    if len(state['pairs']) != 30 or len({p['station_id'] for p in state['pairs']}) < 3:
        raise RuntimeError(f"Only {len(state['pairs'])}/30 matches within bounded attempts; audit saved")
    print(f"Finished: 30 new real matches, {3 + len({p['station_id'] for p in state['pairs']})} total stations, 130 observations.", flush=True)


if __name__ == '__main__':
    main()

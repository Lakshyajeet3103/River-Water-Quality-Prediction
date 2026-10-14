"""Add exact-location/date CPCB pH and dissolved oxygen to the saved 100 pairs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_COLUMNS = {
    'Potential of Hydrogen (pH)': 'ground_ph',
    'Dissolved oxygen (mg/L)': 'ground_dissolved_oxygen',
}
KEYS = ['station_name', 'sample_date', 'latitude', 'longitude']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def join_chemicals(paired, chemical):
    """Preserve event order and all existing values; refuse ambiguous/missing joins."""
    if paired.empty or not paired.observation_id.is_unique:
        raise ValueError('Paired events must be nonempty and unique')
    if set(SOURCE_COLUMNS.values()) & set(paired.columns):
        raise ValueError('Chemical columns already exist; refusing replacement')
    if chemical.duplicated(KEYS).any():
        raise ValueError('Duplicate chemical station/date/coordinate keys')
    joined = paired.merge(chemical, on=KEYS, how='left', validate='one_to_one', sort=False,
                          indicator='_chemical_match')
    if not joined._chemical_match.eq('both').all():
        raise ValueError('Some paired events have no exact chemical match')
    for original, renamed in SOURCE_COLUMNS.items():
        values = pd.to_numeric(joined[original], errors='raise')
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f'Missing/non-finite chemical values: {original}')
        if renamed == 'ground_ph' and not values.between(0, 14).all():
            raise ValueError('pH outside the configured basic plausibility range 0–14')
        if renamed == 'ground_dissolved_oxygen' and values.lt(0).any():
            raise ValueError('Negative dissolved oxygen')
        joined[renamed] = values
    enriched = joined[[*paired.columns, *SOURCE_COLUMNS.values()]].copy()
    pd.testing.assert_frame_equal(paired.reset_index(drop=True), enriched[paired.columns].reset_index(drop=True))
    return enriched, joined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', default='outputs/matches_expansion_100_20261014/paired.csv')
    parser.add_argument('--chemical-dir', default='data/raw/cpcb_chemical_check_20261014')
    parser.add_argument('--output', default='outputs/india_100_chemical_dataset_20261014')
    args = parser.parse_args()
    source, folder, out = Path(args.paired), Path(args.chemical_dir), Path(args.output)
    if out.exists():
        raise ValueError('Output exists; choose a fresh directory')
    paired = pd.read_csv(source)
    if len(paired) != 100:
        raise ValueError('Expected the saved 100-event cohort')
    raw_files = sorted(folder.glob('*.csv'))
    if len(raw_files) != 2:
        raise ValueError('Expected both original downloaded chemical exports')
    tables, sources = [], []
    for path in raw_files:
        sidecar = path.with_suffix('.download.json')
        download = json.loads(sidecar.read_text(encoding='utf-8'))
        source_hash = digest(path)
        if source_hash != download['sha256']:
            raise ValueError(f'Original chemical file hash changed: {path.name}')
        raw = pd.read_csv(path)
        table = raw[['Station', 'Latitude', 'Longitude', 'Data Acquisition Time', *SOURCE_COLUMNS]].copy()
        table = table.rename(columns={'Station': 'station_name', 'Latitude': 'latitude', 'Longitude': 'longitude'})
        table['sample_date'] = pd.to_datetime(table['Data Acquisition Time'], format='%d-%m-%Y %H:%M', errors='raise').dt.strftime('%Y-%m-%d')
        table['chemical_source_file'] = path.name
        table['chemical_source_sha256'] = source_hash
        table['chemical_source_record'] = np.arange(1, len(table) + 1)
        tables.append(table)
        sources.append({'file': path.name, 'sha256': source_hash, 'url': download['url'],
                        'retrieved_utc': download['retrieved_utc'], 'source_records': len(table)})
    chemical = pd.concat(tables, ignore_index=True)
    enriched, joined = join_chemicals(paired, chemical)
    audits = []
    for row in joined.to_dict('records'):
        for source_column, parameter in SOURCE_COLUMNS.items():
            audits.append({'observation_id': row['observation_id'], 'station_id': row['station_id'],
                           **{key: row[key] for key in KEYS}, 'parameter': parameter,
                           'value': row[parameter], 'unit': 'pH' if parameter == 'ground_ph' else 'mg/L',
                           'source_column': source_column, 'source_value': row[source_column],
                           'source_timestamp_displayed': row['Data Acquisition Time'],
                           'source_file': row['chemical_source_file'], 'source_sha256': row['chemical_source_sha256'],
                           'source_record': int(row['chemical_source_record']), 'quality_flag': 'source_not_reported'})
    model_paths = [Path('outputs/fusion_sklearn_expanded_20261014/fusion_model.joblib'),
                   Path('outputs/satellite_only_80_20_20261014/model.joblib')]
    model_hashes = {str(p): digest(p) for p in model_paths}
    out.mkdir(parents=True)
    enriched.to_csv(out / 'paired.csv', index=False)
    pd.DataFrame(audits).to_csv(out / 'chemical_audit.csv', index=False)
    excerpt = joined[[*KEYS, 'Data Acquisition Time', *SOURCE_COLUMNS,
                      'chemical_source_file', 'chemical_source_sha256', 'chemical_source_record']]
    excerpt.to_csv(out / 'matched_chemical_source_records.csv', index=False)
    units = {'ground_bod': 'mg/L', 'ground_fecal_coliform': 'MPN/100mL',
             'ground_total_coliform': 'MPN/100mL', 'ground_ph': 'pH (dimensionless)',
             'ground_dissolved_oxygen': 'mg/L',
             **{c: 'scaled surface reflectance' for c in enriched if c.startswith('satellite_B')}}
    report = {'rows': len(enriched), 'added_columns': list(SOURCE_COLUMNS.values()), 'units': units,
              'target_column': 'ground_bod',
              'available_ground_predictors': ['ground_fecal_coliform', 'ground_ph', 'ground_dissolved_oxygen'],
              'optional_incomplete_ground_column': 'ground_total_coliform',
              'parent_paired_path': str(source), 'parent_paired_sha256': digest(source),
              'source_dataset_url': 'https://nwdp.nwic.gov.in/en/dataset/surface-water-quality-manual-chemical-parameters-cpcb',
              'chemical_sources': sources, 'output_sha256': digest(out / 'paired.csv'),
              'audit_sha256': digest(out / 'chemical_audit.csv'), 'preserved_model_hashes': model_hashes,
              'join_rule': 'Exact station name, displayed calendar date, latitude and longitude; one-to-one; no rounding/fuzzy matching, averaging, filling or unit conversion.',
              'validation': '100/100 pH and DO values finite; pH 0–14, DO >=0; existing columns/order preserved.',
              'limitations': ['Exact same-station/date coordinates do not establish that chemical and biological readings came from the same physical bottle.',
                              'Displayed 08:30 times and source timezone/CRS/laboratory quality conventions remain unverified.',
                              'Numeric and basic plausibility checks do not establish laboratory QC acceptance.',
                              'Portal source terms are recorded as Other (Open) in earlier inspection; full reuse terms remain to be confirmed.',
                              'BOD is the target and must never be used as an input. Existing model input contracts are unchanged; using pH/DO requires training a new model.']}
    (out / 'provenance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert {str(p): digest(p) for p in model_paths} == model_hashes
    print(f'Prepared {len(enriched)} events with pH and dissolved oxygen: {out}')
    print(enriched[['ground_ph', 'ground_dissolved_oxygen']].describe().to_string())
    print('Existing data columns preserved; both saved models unchanged.')


if __name__ == '__main__':
    main()

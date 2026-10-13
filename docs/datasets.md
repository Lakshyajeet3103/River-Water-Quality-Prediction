# Dataset Documentation

## Current evidence status

Real-data feasibility experiments have been run in a separate local
`restored-demo` artifact bundle. These results are exploratory and are not
automatically part of this Git repository or a reproducible release.

### India: CPCB-labelled events and Sentinel-2

- Ground data: 36 rows across 18 events at three Ganga stations.
- Matched events: 14; unmatched events: 4.
- Satellite inputs: Sentinel-2 surface reflectance bands B2, B3, B4, B5,
  B6, B7, B8, B8A, B11 and B12.
- Satellite extraction: 14 selected events; the extraction audit records
  cloud-related exclusions and events without usable scenes.
- Water-quality target used in the exploratory fusion experiment: BOD (mg/L).
- Ground predictor used in the fusion and feature-selection experiments:
  fecal coliform (MPN/100mL).

The CPCB source URL, license, laboratory quality flags, coordinate reference
system and timezone still require verification. Supplied station coordinates
are not independently verified sampling locations. The selected satellite
pixels may contain bridge or bank mixtures. Matching is retrospective and
must not be described as forecasting.

### United States: USGS/WQP and Sentinel-2

An exploratory cross-region artifact records four matched US events from
three stations. USGS-origin water-quality records were obtained through the
Water Quality Portal, and Sentinel-2 features were extracted for the
matched events.

The US evaluation is too small to establish regional generalization.
Laboratory-method comparability with the India dataset is unresolved, and
one site is classified as a tidal stream. Source query sidecars and manifests
must be retained when reproducing the experiment.

## Satellite processing

The exploratory runs use the Sentinel-2 `COPERNICUS/S2_SR_HARMONIZED`
collection. Recorded pilot settings include a 60 m region of interest,
20 m processing scale, a primary three-day matching window, a maximum
seven-day window, cloud-fraction threshold 0.2 and minimum water-fraction
threshold 0.2.

These are pilot settings, not yet independently validated scientific
defaults. Water classification, pixel contamination, coordinate accuracy
and source time conventions require further review.

## Reproducibility requirements

- Keep real observations separate from synthetic software fixtures.
- Preserve source manifests, download-query sidecars, extraction audits,
  configuration files, split assignments, input hashes and model hashes.
- Record preprocessing, units, exclusions, software versions and random seeds.
- Check spatial and temporal leakage, including reuse of satellite scenes.
- Distinguish retrospective same-event estimation from future forecasting.
- Do not claim research-grade performance from tiny convenience samples.
- Treat the restored experiment outputs as external artifacts until they are
  deliberately copied into a versioned, documented repository location.

## Outstanding work

1. Verify CPCB provenance, quality flags, CRS and timezone.
2. Validate station coordinates and satellite water-pixel selection.
3. Reproduce the extraction and experiments from documented source inputs.
4. Run stronger station/time-aware baselines on a sufficiently large cohort.
5. Evaluate on a genuinely untouched test set.

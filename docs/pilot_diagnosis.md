# Pilot diagnosis and handoff

The failed extractions are location/ROI failures under the current SCL rule,
not authentication failures. Live diagnostics used the existing project, original
coordinates, 60 m sampling circles, and a separate 500 m context for inspection.
No coordinate, measurement, threshold, or original CSV was changed.

## Evidence

| Station | Clear scene date | Water / covered ROI pixels | Nearest SCL-water pixel centre |
|---|---|---:|---:|
| Trighat, Ghazipur | 2021-03-16 | 0 / 27 | 223.1 m |
| Varanasi upstream, Assighat | 2021-03-16 | 2 / 28 | 57.1 m |
| Anoopshahar upstream | 2021-03-07 | 22 / 30 | 12.0 m |

True-colour imagery places the Ghazipur coordinate on land and Assighat on the
bank. All Ghazipur ROI pixels and 26 Assighat pixels are SCL class 5; the remaining
two Assighat pixels are class 6. The latter fails both the 20% water-fraction rule
and the minimum of three usable water pixels. Anoopshahar passes.
These are observations of these dated images, not proof of CPCB's actual sampling
locations or of classification accuracy at every date. A nearby water pixel might
also belong to a different channel or water body; its coordinate is not a correction.

Evidence is saved locally in `data/interim/sentinel_diagnostics_002`:
`diagnostics.json`, input hashes/configuration in `provenance.json`, and RGB/SCL
PNG pairs named by station ID. Yellow outlines show the original 60 m ROI, red
marks the supplied point. Version 001 omitted nearest-pixel properties during
serialization; use version 002 for distances. Images are from Copernicus Sentinel-2
via Earth Engine. The [official catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
defines SCL classes and explains the tile components of asset IDs.

## Defects fixed

1. Quality validation now checks finite, nonnegative integer counts and their
   relationships before division. Missing counts cannot silently mean zero cloud;
   zero coverage cannot divide by zero. Invalid spectral values fail closed.
2. Extraction audits include counts for each SCL class, distinguishing an ROI
   classified as land from one rejected for cloud/shadow.
3. Baseline splits now reject recognized Sentinel-2 tile/processing aliases with
   the same sensing timestamp across folds. Four overlapping tile IDs at Assighat
   exposed this gap: different asset strings can represent the same acquisition.
   The old exact-scene guard remains. Bare and full catalog asset IDs are supported.
   This conservative alias guard is not a universal orbit/spatial-independence
   test; other providers and acquisition naming schemes still need explicit review.

All 173 tests passed, including regression tests for invalid counts, zero coverage,
bad spectra, cross-fold tile/processing aliases, and permissible within-fold aliases.
The initial sandboxed test attempt could not access pytest's Windows temporary
directory; the authorized rerun passed. Raw source hashes remain unchanged.

The live `sentinel_pilot_003` rerun produced the same ten spectral rows as version
002: one accepted event, two rejected events, no query errors. Version 003 adds SCL
class diagnostics. This remains insufficient data for research training.

## Continue from here

The difficult diagnostic and leakage fixes are complete. Remaining immediate
work is source verification, a small batch of verified stations, and configuration:

1. Confirm Ghazipur and Assighat sampling coordinates with the original station
   metadata/provider. Until verified, keep them excluded or use other verified
   stations. Do not snap automatically to the nearest classified water pixel or
   widen the ROI just to obtain a match.
2. Confirm download source, usage terms, coordinate CRS and date/time convention.
   Inspect the existing diagnostics before changing extraction policy.
3. Prepare an explicit next small subset of verified station events and run the
   existing extraction/matching commands into fresh directories. Report paired
   coverage and exclusions before scaling up.
4. Choose a measured target and justified predictors, then configure station or
   buffered temporal folds once enough paired observations exist. WQI remains
   unavailable without a documented definition and required measurements.

Reproduce diagnostics from the repository root (use a new output directory):

```powershell
.\.venv-check\Scripts\python.exe scripts/diagnose_sentinel_pilot.py data/interim/cpcb_preparation_001/pilot_ground.csv data/interim/sentinel_pilot_003/audit.csv --output data/interim/sentinel_diagnostics_003
```

The diagnostic helper inspects only one closest clear audited candidate per event.
It does not alter production sampling geometry or exhaustively validate seasonal
water masks. Future scientific model/evaluation design still needs review; this
handoff concerns the current extraction blocker.

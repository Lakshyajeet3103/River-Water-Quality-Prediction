# Expanded real CPCB–Sentinel-2 matches

Exactly 50 additional matches were collected, then extraction stopped.

- `paired.csv`: 64 total events (14 original + 50 new), ready for the friend's scikit-learn model inputs.
- `new_paired.csv`: only the 50 additional events.
- `new_ground.csv`: original prepared ground measurements for the new events.
- `new_satellite_features.csv`: ten extracted reflectance bands, deduplicated by station/scene/band.
- `event_outcomes.csv`: all 68 attempted events; 50 passed, 18 had no usable scene.
- `extraction_audit.csv`: candidate scene identifiers, date gaps, pixel counts and rejection reasons.
- `run.json`: settings, input hashes, date range and station counts.
- `checkpoint.json`: resumable per-event extraction results.

New observations span 2020-10-19 through 2021-12-28. New counts are Anupshahar 15, Varanasi 18 and Bijnor 17.

The original dataset remains unchanged. Every row has finite BOD, fecal coliform and ten satellite bands. All selected scenes satisfy the existing cloud <= 0.2, water >= 0.2, coverage >= 0.9, valid water pixels >= 3 and absolute date gap <= 7 days checks.

This expansion does not retrain or change either model. Future training must create new chronological or station-aware splits. There are 60 distinct satellite acquisition timestamps among the 64 rows; shared acquisition groups must not cross training/test folds. The older three-event test set has already been examined and is not a fresh independent test.

Data-source, timezone, coordinate and SCL/bridge/bank assumptions remain those documented in the existing pilot. No thresholds were relaxed and no failed values were fabricated. Optional total-coliform values are present for some rows but are not required inputs for the friend's model.

Verification: exact row counts, unique observation IDs and station/date pairs, original-row preservation, complete required inputs, original-file SHA256, and all image thresholds passed. Existing preparation/matching/hybridization software tests: 73 passed.

Resume command (from the repository root): `.\.venv-check\Scripts\python.exe scripts/expand_real_matches.py`. Since the target has been reached, it will not extract additional events.

# Process: Matching Sentinel-2 Satellite Data to CPCB Ground-Station Readings

## Goal

For each real CPCB water-quality measurement (a specific river location, on a
specific date), obtain the corresponding Sentinel-2 satellite reflectance
values for that same location and date — so each row of training data has
both a satellite "view" and a real chemical measurement of the same spot in
time.

## Inputs

- A CPCB dataset containing, at minimum: station name/ID, latitude,
  longitude, date of measurement, and water-quality parameters (e.g. pH, DO,
  BOD, conductivity).
- Access to Google Earth Engine (GEE), which hosts the full Sentinel-2
  archive and lets you query values at specific coordinates/dates without
  downloading full satellite scenes.

## Step-by-Step Process

### 1. Clean and deduplicate CPCB station list
- Extract unique (latitude, longitude) pairs from the CPCB dataset.
- Verify each coordinate actually falls on or near a river (not a landlocked
  error or lake mistakenly labeled as a river station).
- Flag and drop stations with missing or clearly invalid coordinates
  (e.g. 0,0 or out-of-range values).

### 2. Define a spatial query region per station
- Sentinel-2 pixels are ~10-20m resolution depending on band.
- For each station coordinate, define a small buffer/box around the point
  (e.g. 30-60m radius) rather than a single pixel, to reduce GPS/registration
  error and avoid picking a pixel that's actually riverbank/land, not water.
- Optionally apply a water mask (e.g. using the NDWI index or a pre-built
  water-body layer) to confirm the queried pixels are actually water, not
  adjacent land.

### 3. Define a temporal matching window per measurement

The two-satellite Sentinel-2 constellation has a nominal revisit cadence of
~5 days at the equator, but this is **not** the same as guaranteed usable
data on any given date at any given station. Cloud cover, missing scene
coverage, and water-masking failures all reduce the number of actually
usable observations well below the nominal revisit rate. The pipeline must
never assume a CPCB date directly implies an available Sentinel-2 image —
it must search and explicitly fall back or exclude:

```
CPCB date
   ↓
search ±3 days
   ↓
usable image?
   ├── yes → select closest usable image
   └── no → optionally widen search to ±7 days
                 ↓
              usable image found?
                 ├── yes → select closest usable image
                 └── no → EXCLUDE this CPCB measurement
```

- "Usable" means: passes the cloud-cover filter (Step 4) and passes the
  water-mask check (Step 2) — not merely "an image exists for that date."
- If a match is found at ±7 days rather than ±3, record the actual temporal
  offset used as metadata (see Step 6/7) rather than treating all matches as
  equally reliable.
- Never substitute a stale or future image outside the maximum allowed
  window — water conditions can change significantly within days after
  rainfall or upstream discharge events.

### 4. Filter for cloud cover
- Sentinel-2 imagery includes a cloud probability/QA band.
- Discard any candidate image where cloud cover over the query region
  exceeds a defined threshold (e.g. 20%).
- If no cloud-free image exists within the temporal window, that CPCB
  measurement is excluded rather than matched to a degraded image.

### 5. Extract band values
- For each valid station-date match, extract the relevant Sentinel-2 bands
  (e.g. B02, B03, B04, B08 at 10m; B05, B06, B07, B8A, B11, B12 at 20m).
- Aggregate pixel values within the query region (e.g. mean or median across
  the buffer) rather than a single pixel, to reduce noise.

### 6. Join with CPCB measurement
- Combine the extracted band values with the corresponding CPCB row
  (same station, same date) into a single record: latitude, longitude,
  date, satellite bands, water-quality parameters.
- Record which temporal offset was used (0 days, 2 days, etc.) **and**
  which search tier produced the match (±3 day pass vs. ±7 day fallback)
  as metadata, so downstream analysis can check whether prediction accuracy
  degrades with larger offsets or fallback-tier matches.

### 7. Log exclusions
- Track and report how many CPCB measurements were excluded and why
  (no cloud-free image, no coordinate match, outside temporal window, etc.).
- This exclusion rate itself is a data-quality signal worth reporting later.

## Known Limitations to Flag

- Only surface reflectance is measured; it is a proxy for water quality
  (e.g. turbidity, chlorophyll, some organic matter), not a direct
  measurement of chemical parameters like pH or BOD — the AI model has to
  learn a statistical/physical relationship, not a direct readout.
- Small rivers/streams narrower than the Sentinel-2 pixel size may not
  produce reliable pure-water pixels, especially with adjacent land/bank
  contamination in the pixel.
- Temporal offset (even a few days) between satellite pass and ground
  measurement introduces some inherent noise, especially after rain events.
- Cloud-cover filtering can create seasonal bias (e.g. monsoon season readings
  underrepresented in India due to persistent cloud cover).

## What This Process Does NOT Cover

- It does not cover model training, feature selection, or fusion
  architecture — this is strictly the data acquisition/matching step
  (Phase 1-2 of the broader project).
- It does not cover the USGS dataset, which already provides pre-matched
  Sentinel-2 + water-quality data and does not need this pipeline.

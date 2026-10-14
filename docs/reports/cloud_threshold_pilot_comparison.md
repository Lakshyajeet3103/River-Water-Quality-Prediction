# River Water Quality Prediction
## Sentinel-2 Cloud-Threshold Pilot Comparison

### 1. Objective
Compare the extraction pipeline at 20% and 30% cloud-fraction thresholds
using the same three ground-observation events.

### 2. Experimental Configuration

| Setting | Value |
|---|---|
| Earth Engine project | river-water-quality-fusion |
| Ground events | 3 |
| Spatial radius | 60 m |
| Extraction scale | 20 m |
| Primary time window | 3 days |
| Maximum time window | 7 days |
| Minimum usable pixels | 3 |
| Threshold A | 20% |
| Threshold B | 30% |
| Spectral bands | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 |

### 3. Results

| Metric | 20% threshold | 30% threshold |
|---|---:|---:|
| Events attempted | 3 | 3 |
| Events selected | 3 | 3 |
| Spectral rows | 30 | 30 |
| Audit status: selected | 3 | 3 |
| Audit status: cloud_fraction | 4 | 4 |

Both runs processed the same three ground-observation events.

### 4. Feature File Integrity

Both exported features.csv files produced the same SHA-256 digest:

`8254C6C72F2ED05748996F726D4E5157CE860DB361C094293A5E5F8578C3DFA2`

This confirms that the two exported feature files are byte-for-byte identical.

### 5. Interpretation

For this three-event pilot, changing the configured cloud threshold from
20% to 30% did not change the exported feature file. Both runs selected
three events and generated 30 spectral rows.

The pilot does not demonstrate that either threshold is generally better.
The result applies only to the events and configurations tested.

### 6. Software Testing

The repository test suite completed successfully:

- Tests passed: 176
- Tests failed: 0
- Reported duration: 12.64 seconds

These tests provide evidence about software behavior, not independent
validation of river-water-quality prediction accuracy.

### 7. Generated Deliverables

Each extraction run produced:

- `audit.csv` — candidate selection and filtering statuses
- `features.csv` — extracted spectral features
- `features.provenance.json` — feature provenance metadata
- `run.json` — run settings and execution metadata

### 8. Reproducibility

The two output directories are:

- `artifacts/restored-demo/data-interim/cloud20_extraction_001`
- `artifacts/restored-demo/data-interim/cloud30_extraction_001`

The experimental configuration files are:

- `configs/extraction_pilot_cloud20_subset.yaml`
- `configs/extraction_pilot_cloud30.yaml`

The runs used the same three-event ground-data subset. Earth Engine access
and the Sentinel-2 collection query were verified before extraction.

### 9. Limitations

- Only three ground-observation events were used in the paired comparison.
- The ground-data timezone remains provisional.
- The source provenance has unresolved URL and licensing details.
- Identical feature files do not establish general equivalence between thresholds.
- Prediction accuracy requires separate evaluation using suitable ground-truth data.

### 10. Conclusion

The extraction pipeline successfully generated outputs at both configured
cloud thresholds, and the exported feature files were identical for the
three tested events. The implementation and outputs are suitable for a
college-level workflow demonstration, with the pilot scope and limitations
clearly documented.

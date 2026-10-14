# WQI input compatibility audit (step 2)

Audited the saved 100-event Indian cohort without changing data or model artifacts. The selected method is specified in [wqi_method.md](../wqi_method.md).

## Findings

| Input | Complete, numerically valid observations | Recorded range |
|---|---:|---|
| BOD, mg/L | 100/100 | 0.9-4.1 |
| pH | 100/100 | 6.8-8.7 |
| DO, mg/L | 100/100 | 6.2-10.8 |
| Fecal coliform, MPN/100 mL | 100/100 | 110-23,000 |
| Water temperature | 0/100 | Missing |
| Measured DO percent saturation | 0/100 | Not provided |
| Measured barometric pressure | 0/100 | Not provided |

The two chemical exports and biological export have no temperature or saturation columns. Downloaded the two relevant [NWIC/CPCB physical exports for Uttar Pradesh](https://nwdp.nwic.gov.in/en/dataset/surface-water-quality-manual-physical-parameters-cpcb), using portal resource metadata to discover download URLs.

Found 63 exact station/date/coordinate physical matches. The other 37 remain unmatched. Their temperature column exists but is empty; both entire physical exports also have no reported temperatures. Valid conductance is present for 28 cohort observations; the downloaded header specifies microSiemens/cm, not milliSiemens/cm. Values are retained as reported with the exact source column name; none are used for a conversion.

| Station | Matched physical records | Unmatched |
|---|---:|---:|
| Anupshahar | 27 | 11 |
| Varanasi, Malviya Bridge | 24 | 13 |
| Bijnor barrage | 12 | 13 |

Fecal readings are MPN estimates per 100 mL, whereas the index table says counts per 100 mL. Their use must be disclosed as the reported bacterial concentration estimate; MPN is not an exact colony count. No laboratory-method equivalence or QC certification has been established by this audit.

## Decision

**Zero observations are ready for the selected WQI without oxygen-conversion assumptions.** Matching more physical rows would not solve the empty temperature column. No WQI scores or categories have been generated.

Preferred route: obtain measured water temperature and pressure (or directly reported DO saturation) for the corresponding events. Pressure/salinity approximations also need explicit provenance. [USGS DOTABLES](https://www.usgs.gov/tools/dotables) describes the required oxygen-solubility inputs.

For a classroom demonstration, an alternative is an explicitly labelled assumption-based calculation with a stated temperature, pressure and freshwater assumption, accompanied by a sensitivity comparison across plausible temperatures. This would be an estimated scenario, not an observed WQI. That choice is not implemented or silently assumed in this step. A universal saturation denominator, an air-temperature substitution or a missing oxygen term must not be presented as the selected index applied to measured conditions.

## Evidence and reproduction

- `physical_matches.csv`: 100-row audit, exact-match status, missing temperature, source conductance, source filenames, URLs, record numbers and hashes.
- `summary.json`: input ranges, original paired-data hash, physical-file hashes/headers/retrieval times and unchanged saved-model hashes.
- Original exports and download sidecars: ignored `data/raw/cpcb_physical_wqi_20261014/`.

From the repository root, using a new output path:

```powershell
.\.venv-check\Scripts\python.exe scripts/audit_wqi_inputs.py --download --output outputs/wqi_audit_reproduction
```

The download option reuses only hash-verified cached exports. Omitting it performs the audit offline when both raw exports and sidecars exist. The script refuses to overwrite an output directory. Original saved inputs/models are hash-verified. Tests cover duplicate-key rejection, nearby-coordinate non-matches and preservation of missing temperature; all three passed.

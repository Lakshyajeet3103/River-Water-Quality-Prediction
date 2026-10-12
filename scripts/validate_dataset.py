"""Validate local CSV input without generating observations or output artifacts."""

import argparse

from river_water_quality.data.ingestion import load_dataset


def main() -> None:
    """Validate an explicitly supplied local dataset and provenance manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["cpcb", "usgs", "sentinel2"])
    parser.add_argument("csv_path")
    parser.add_argument("manifest_path")
    args = parser.parse_args()
    try:
        result = load_dataset(args.csv_path, args.manifest_path, kind=args.kind)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")
    print(f"Validated {len(result.data)} rows for {result.provenance['dataset_id']}")
    print(f"Source SHA256: {result.source_sha256}")
    print(f"Manifest SHA256: {result.manifest_sha256}")
    print("Schema validation only; scientific suitability and quality flags require review.")


if __name__ == "__main__":
    main()

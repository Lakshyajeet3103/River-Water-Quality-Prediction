"""Run the scikit-learn fusion baseline on a paired dataset."""
import argparse
from pathlib import Path

from river_water_quality.training.fusion import train_fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", required=True, help="Path to paired.csv")
    parser.add_argument("--splits", required=True, help="Path to split_assignments.csv")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train_fusion(args.paired, args.splits, args.output, args.seed)
    print(metrics.to_string(index=False))
    print(f"\nSaved model and reports to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

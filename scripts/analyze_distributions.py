#!/usr/bin/env python3
"""
Analyze distribution statistics for text features dataset.

Reads the features Parquet file and prints distribution statistics
for flesch_score, avg_sentence_length, and sentence_count to help
define human writing ranges.
"""

import sys
from pathlib import Path

import polars as pl


def print_statistics(df: pl.DataFrame, field: str) -> None:
    """
    Print distribution statistics for a given field.

    Args:
        df: Polars DataFrame containing the data
        field: Name of the field to analyze
    """
    print(f"\n{'=' * 60}")
    print(f"Statistics for: {field}")
    print(f"{'=' * 60}")

    # Get the series for the field
    series = df[field]

    # Compute statistics
    min_val = series.min()
    max_val = series.max()
    mean_val = series.mean()
    median_val = series.median()

    # Compute percentiles
    p10 = series.quantile(0.10)
    p25 = series.quantile(0.25)
    p75 = series.quantile(0.75)
    p90 = series.quantile(0.90)

    # Print statistics with clear labels
    print(f"  Minimum:           {min_val:>12.2f}")
    print(f"  10th Percentile:   {p10:>12.2f}")
    print(f"  25th Percentile:   {p25:>12.2f}")
    print(f"  Median (50th):     {median_val:>12.2f}")
    print(f"  Mean:              {mean_val:>12.2f}")
    print(f"  75th Percentile:   {p75:>12.2f}")
    print(f"  90th Percentile:   {p90:>12.2f}")
    print(f"  Maximum:           {max_val:>12.2f}")


def main():
    """Main execution function."""
    # Path to features dataset
    features_path = Path(__file__).parent.parent / "datasets" / "features.parquet"

    if not features_path.exists():
        print(f"Error: Features file not found at {features_path}")
        sys.exit(1)

    # Read the features Parquet file
    print(f"Reading features dataset: {features_path}")
    df = pl.read_parquet(features_path)

    total_rows = len(df)
    print(f"Total rows: {total_rows}")

    # Analyze the three specified fields
    fields_to_analyze = [
        "flesch_score",
        "avg_sentence_length",
        "sentence_count",
    ]

    # Verify all fields exist
    missing_fields = [f for f in fields_to_analyze if f not in df.columns]
    if missing_fields:
        print(f"Error: Missing fields in dataset: {missing_fields}")
        sys.exit(1)

    # Print statistics for each field
    for field in fields_to_analyze:
        print_statistics(df, field)

    print(f"\n{'=' * 60}")
    print("Analysis complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

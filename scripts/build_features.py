#!/usr/bin/env python3
"""
Build structural text features dataset from writing_events.

Reads the canonical writing_events Parquet file, computes text features using
TextAnalyzer, and stores results in a derived features dataset.

Features computed:
- sentence_count: Number of sentences
- word_count: Number of words
- avg_sentence_length: Average words per sentence (word_count / sentence_count)
- syllable_count: Total syllables across all words
- flesch_score: Flesch Reading Ease score (0-100, or null)

The derived dataset is fully regenerable from the canonical writing_events dataset.
"""

import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
from data.arrow import DatasetManager
from text import TextAnalyzer


def compute_features_for_row(row: dict) -> dict:
    """
    Compute text features for a single writing event row.

    Args:
        row: Dictionary with at least 'id' and 'text' keys

    Returns:
        Dictionary with computed features:
        - event_id: Original event ID
        - sentence_count: Number of sentences (int)
        - word_count: Number of words (int)
        - avg_sentence_length: Average words per sentence (float, or null if no sentences)
        - syllable_count: Total syllables (int)
        - flesch_score: Flesch score (float, or null)
    """
    text = row.get("text", "")
    event_id = row.get("id", "")

    # Initialize TextAnalyzer with the text
    analyzer = TextAnalyzer(text)

    # Compute features
    sentence_count = analyzer.sentence_count
    word_count = analyzer.word_count
    syllable_count = analyzer.syllable_count
    flesch_score = analyzer.flesch_score

    # Calculate average sentence length
    # Handle division by zero: if no sentences, return null
    if sentence_count > 0:
        avg_sentence_length = word_count / sentence_count
    else:
        avg_sentence_length = None

    return {
        "event_id": event_id,
        "sentence_count": sentence_count,
        "word_count": word_count,
        "avg_sentence_length": avg_sentence_length,
        "syllable_count": syllable_count,
        "flesch_score": flesch_score,
    }


def build_features_dataset(
    input_path: str = "datasets/data.parquet",
    output_path: str = "datasets/features.parquet",
) -> None:
    """
    Build the features dataset from writing_events.

    Args:
        input_path: Path to the canonical writing_events Parquet file
        output_path: Path where the derived features dataset will be written
    """
    # Step 1: Read the canonical writing_events dataset
    print(f"Reading canonical dataset: {input_path}")
    dataset_manager = DatasetManager(input_path)
    events_df = dataset_manager.read()

    total_events = len(events_df)
    print(f"Found {total_events} writing events")

    if total_events == 0:
        print("Warning: No events found in the dataset")
        return

    # Step 2: Compute features for each row
    print("Computing text features for each event...")
    features_records = []

    for idx, row in enumerate(events_df.iter_rows(named=True), start=1):
        features = compute_features_for_row(row)
        features_records.append(features)

        # Progress indicator for large datasets
        if idx % 100 == 0 or idx == total_events:
            print(f"  Processed {idx}/{total_events} events", end="\r")

    print(f"\nComputed features for {len(features_records)} events")

    # Step 3: Create Polars DataFrame with correct schema
    # Define schema to ensure correct types
    features_df = pl.DataFrame(
        features_records,
        schema={
            "event_id": pl.String,
            "sentence_count": pl.Int64,
            "word_count": pl.Int64,
            "avg_sentence_length": pl.Float64,
            "syllable_count": pl.Int64,
            "flesch_score": pl.Float64,
        },
    )

    # Step 4: Ensure output directory exists
    output_filepath = Path(output_path)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Step 5: Write derived features dataset to Parquet
    print(f"Writing features dataset: {output_path}")
    features_df.write_parquet(output_path)

    print(f"Successfully created features dataset with {len(features_df)} rows")

    # Display summary statistics
    print("\nSummary statistics:")
    print(f"  Events processed: {len(features_df)}")
    print(f"  Average sentences per event: {features_df['sentence_count'].mean():.2f}")
    print(f"  Average words per event: {features_df['word_count'].mean():.2f}")
    print(f"  Average syllables per event: {features_df['syllable_count'].mean():.2f}")
    flesch_mean = features_df["flesch_score"].mean()
    if flesch_mean is not None:
        print(f"  Average Flesch score: {flesch_mean:.2f}")


def main():
    """Main execution function."""
    # Default paths
    input_path = "datasets/data.parquet"
    output_path = "datasets/features.parquet"

    # Build the features dataset
    build_features_dataset(input_path, output_path)


if __name__ == "__main__":
    main()

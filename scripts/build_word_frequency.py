#!/usr/bin/env python3
"""
Build word frequency table from writing_events dataset.

Reads the canonical writing_events Parquet file, tokenizes text, normalizes words,
filters out numbers and acronyms, computes word frequencies and syllable counts,
and stores results in a derived word frequency dataset.

The word frequency table is used by LexicalSimplifier to identify rare, complex
words that may be candidates for simplification.

Schema:
- word: lowercase normalized word (str)
- frequency: count of occurrences across all writing events (int64)
- syllable_count: number of syllables in the word (int64)

The derived dataset is fully regenerable from the canonical writing_events dataset.
"""

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import polars as pl

# Add parent directory to path for imports
# Scripts are in scripts/, package is at root level
sys.path.insert(0, str(Path(__file__).parent.parent))

from humanize.data.arrow import DatasetManager
from humanize.text.analyzer import TextAnalyzer


def count_syllables_in_word(word: str) -> int:
    """Count syllables in a single word using TextAnalyzer's heuristic rules.

    This duplicates the logic from TextAnalyzer._count_syllables_in_word()
    to avoid requiring a TextAnalyzer instance for each word.

    Args:
        word: The word to count syllables for

    Returns:
        Number of syllables (minimum 1)
    """
    # Lowercase and remove non-alphabetic characters
    cleaned = re.sub(r"[^a-zA-Z]", "", word.lower())

    if not cleaned:
        return 1

    # Count vowel groups (a, e, i, o, u, y)
    # Pattern matches consecutive vowels as a single group
    vowel_groups = re.findall(r"[aeiouy]+", cleaned)
    syllable_count = len(vowel_groups)

    # Subtract one syllable if word ends with silent 'e'
    if cleaned.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    # Minimum 1 syllable per word
    return max(1, syllable_count)


def split_words(text: str) -> List[str]:
    """Split text into words, handling punctuation.

    This duplicates the logic from TextAnalyzer._split_words()
    to avoid requiring a TextAnalyzer instance for tokenization.

    Args:
        text: The text to tokenize

    Returns:
        List of word strings with punctuation stripped
    """
    # Split on whitespace
    words = re.split(r"\s+", text)

    result = []
    for word in words:
        # Strip leading/trailing punctuation
        cleaned = word.strip(".,!?;:()[]{}'\"-—_")
        # Include if not empty
        if cleaned:
            result.append(cleaned)

    return result


def is_acronym(word: str) -> bool:
    """Check if a word is an acronym (ALL CAPS, 2+ characters).

    Args:
        word: The word to check

    Returns:
        True if the word is an acronym, False otherwise
    """
    # Check if word is ALL CAPS and has 2+ characters
    return bool(re.match(r"^[A-Z]{2,}$", word))


def is_number(word: str) -> bool:
    """Check if a word consists entirely of digits.

    Args:
        word: The word to check

    Returns:
        True if the word is a number, False otherwise
    """
    return bool(re.match(r"^\d+$", word))


def tokenize_and_normalize(text: str) -> List[str]:
    """Tokenize text into normalized words, filtering out numbers and acronyms.

    Args:
        text: The text to tokenize

    Returns:
        List of normalized lowercase words (numbers and acronyms filtered out)
    """
    words = split_words(text)
    normalized = []

    for word in words:
        # Filter out numbers
        if is_number(word):
            continue

        # Filter out acronyms (ALL CAPS words)
        if is_acronym(word):
            continue

        # Normalize to lowercase
        normalized_word = word.lower()

        # Filter out empty strings
        if not normalized_word.strip():
            continue

        normalized.append(normalized_word)

    return normalized


def build_word_frequency_table(
    input_path: str = "datasets/data.parquet",
    output_path: str = "datasets/word_frequency.parquet",
) -> None:
    """
    Build the word frequency table from writing_events dataset.

    Args:
        input_path: Path to the canonical writing_events Parquet file
        output_path: Path where the word frequency dataset will be written
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

    # Step 2: Tokenize and collect word frequencies
    print("Tokenizing text and computing word frequencies...")
    word_counter: Counter[str] = Counter()
    word_syllable_counts: Dict[str, int] = {}

    for idx, row in enumerate(events_df.iter_rows(named=True), start=1):
        text = row.get("text", "")

        # Tokenize and normalize
        words = tokenize_and_normalize(text)

        # Count word frequencies
        word_counter.update(words)

        # Compute syllable counts for unique words (cache results)
        for word in words:
            if word not in word_syllable_counts:
                word_syllable_counts[word] = count_syllables_in_word(word)

        # Progress indicator
        if idx % 100 == 0 or idx == total_events:
            print(f"  Processed {idx}/{total_events} events", end="\r")

    print(f"\nFound {len(word_counter)} unique words")

    # Step 3: Build frequency records
    frequency_records = []
    for word, frequency in word_counter.items():
        syllable_count = word_syllable_counts.get(word, 1)
        frequency_records.append(
            {
                "word": word,
                "frequency": frequency,
                "syllable_count": syllable_count,
            }
        )

    # Step 4: Create Polars DataFrame with correct schema
    frequency_df = pl.DataFrame(
        frequency_records,
        schema={
            "word": pl.String,
            "frequency": pl.Int64,
            "syllable_count": pl.Int64,
        },
    )

    # Sort by frequency descending for easier inspection
    frequency_df = frequency_df.sort("frequency", descending=True)

    # Step 5: Ensure output directory exists
    output_filepath = Path(output_path)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Step 6: Write word frequency dataset to Parquet
    print(f"Writing word frequency dataset: {output_path}")
    frequency_df.write_parquet(output_path)

    print(f"Successfully created word frequency dataset with {len(frequency_df)} words")

    # Display summary statistics
    print("\nSummary statistics:")
    print(f"  Unique words: {len(frequency_df)}")
    print(f"  Total word occurrences: {frequency_df['frequency'].sum()}")
    print(
        f"  Most common word: {frequency_df['word'][0]} ({frequency_df['frequency'][0]} occurrences)"
    )
    print(f"  Average frequency: {frequency_df['frequency'].mean():.2f}")
    print(f"  Average syllable count: {frequency_df['syllable_count'].mean():.2f}")


def main():
    """Main execution function."""
    # Default paths
    input_path = "datasets/data.parquet"
    output_path = "datasets/word_frequency.parquet"

    # Build the word frequency table
    build_word_frequency_table(input_path, output_path)


if __name__ == "__main__":
    main()

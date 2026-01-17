#!/usr/bin/env python3
"""
Build curated replacements table for lexical simplification.

Populates the replacements parquet file from various sources:
- WordNet/NLTK synonyms (if available)
- JSON/TSV import files
- Interactive manual entry

The replacements table schema:
- original: word to be replaced (str, lowercase)
- replacement: simpler replacement word (str, lowercase)
- confidence: quality/verification level (str: "manual", "wordnet", "frequency")
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional
import ssl

import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def create_empty_replacements_table(output_path: str) -> pl.DataFrame:
    """Create an empty replacements table with correct schema.

    Args:
        output_path: Path where the table will be written

    Returns:
        Empty DataFrame with correct schema
    """
    return pl.DataFrame(
        schema={
            "original": pl.String,
            "replacement": pl.String,
            "confidence": pl.String,
        }
    )


def load_existing_replacements(output_path: str) -> Dict[str, str]:
    """Load existing replacements from parquet file.

    Args:
        output_path: Path to replacements parquet file

    Returns:
        Dictionary mapping original -> replacement
    """
    if not Path(output_path).exists():
        return {}

    try:
        df = pl.read_parquet(output_path)
        return {
            row["original"].lower(): row["replacement"].lower() for row in df.iter_rows(named=True)
        }
    except Exception:
        return {}


def generate_from_wordnet(
    output_path: str, existing_replacements: Optional[Dict[str, str]] = None
) -> None:
    """Generate replacement suggestions from WordNet synonyms (if available).

    Args:
        output_path: Path to write replacements parquet file
        existing_replacements: Existing replacements to merge with
    """
    try:
        import nltk
        from nltk.corpus import wordnet
    except ImportError:
        print(
            "NLTK/WordNet not available. Install with: pip install nltk\n"
            "Then download WordNet data: python -c 'import nltk; nltk.download(\"wordnet\")'"
        )
        return

    # Try to ensure WordNet is loaded
    try:
        wordnet.ensure_loaded()
    except LookupError:
        print("WordNet data not found. Attempting to download...")
        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception as e:
            print(
                f"Failed to download WordNet data: {e}\n"
                "This may be due to network/SSL issues. You can:\n"
                "1. Download manually: python -c 'import nltk; nltk.download(\"wordnet\")'\n"
                "2. Use import-json or import-tsv to add replacements manually\n"
                "3. Use the interactive mode to enter replacements"
            )
            return

    # Verify WordNet is actually available
    try:
        # Test access to WordNet
        _ = wordnet.synsets("test")
    except (LookupError, Exception) as e:
        print(
            f"WordNet data is not available: {e}\n"
            "Please download WordNet data manually:\n"
            "  python -c 'import nltk; nltk.download(\"wordnet\")'\n"
            "Or use alternative methods (import-json, import-tsv, interactive)"
        )
        return

    print("Generating replacements from WordNet...")
    new_replacements = existing_replacements.copy() if existing_replacements else {}
    suggestions = []

    # Get all WordNet synsets
    try:
        all_words = set(wordnet.all_lemma_names())
    except Exception as e:
        print(f"Error accessing WordNet data: {e}")
        return

    # For each word, find simpler synonyms
    # Strategy: find words with fewer syllables or shorter length
    for word in sorted(all_words):
        word_lower = word.lower()

        # Skip if already has replacement
        if word_lower in new_replacements:
            continue

        # Get synonyms
        synsets = wordnet.synsets(word_lower)
        if not synsets:
            continue

        # Collect all synonyms
        synonyms = set()
        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().lower().replace("_", " ")
                if synonym != word_lower and len(synonym.split()) == 1:
                    synonyms.add(synonym)

        # Find simpler synonym (fewer syllables or shorter)
        # This is a heuristic - manual review recommended
        word_syllables = _count_syllables_heuristic(word_lower)
        word_length = len(word_lower)

        best_replacement = None
        for synonym in synonyms:
            syn_syllables = _count_syllables_heuristic(synonym)
            syn_length = len(synonym)

            # Prefer shorter or fewer syllables
            if (syn_syllables < word_syllables) or (
                syn_syllables == word_syllables and syn_length < word_length
            ):
                if best_replacement is None:
                    best_replacement = synonym
                elif _count_syllables_heuristic(best_replacement) > syn_syllables:
                    best_replacement = synonym
                elif (
                    _count_syllables_heuristic(best_replacement) == syn_syllables
                    and len(best_replacement) > syn_length
                ):
                    best_replacement = synonym

        if best_replacement:
            suggestions.append((word_lower, best_replacement))

    print(f"Found {len(suggestions)} WordNet replacement suggestions")
    print("Note: These are suggestions and should be manually reviewed.")

    # Merge with existing
    for original, replacement in suggestions:
        if original not in new_replacements:
            new_replacements[original] = replacement

    # Write to parquet
    _write_replacements(new_replacements, output_path, "wordnet")


def _count_syllables_heuristic(word: str) -> int:
    """Simple syllable counting heuristic for WordNet suggestions.

    Args:
        word: Word to count syllables for

    Returns:
        Estimated syllable count
    """
    import re

    word = re.sub(r"[^a-zA-Z]", "", word.lower())
    if not word:
        return 1

    vowel_groups = re.findall(r"[aeiouy]+", word)
    syllable_count = len(vowel_groups)

    if word.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    return max(1, syllable_count)


def import_from_json(json_path: str, output_path: str) -> None:
    """Import replacements from JSON file.

    Expected JSON format:
    {
        "original": "replacement",
        ...
    }

    Args:
        json_path: Path to JSON file
        output_path: Path to write replacements parquet file
    """
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"Error: JSON file not found: {json_path}")
        return

    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print("Error: JSON file must contain a dictionary/object")
            return

        # Load existing replacements
        existing = load_existing_replacements(output_path)

        # Merge new replacements
        for original, replacement in data.items():
            existing[original.lower()] = replacement.lower()

        _write_replacements(existing, output_path, "manual")
        print(f"Imported {len(data)} replacements from {json_path}")

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
    except Exception as e:
        print(f"Error importing from JSON: {e}")


def import_from_tsv(tsv_path: str, output_path: str) -> None:
    """Import replacements from TSV file.

    Expected TSV format (tab-separated):
    original\treplacement
    ...

    Args:
        tsv_path: Path to TSV file
        output_path: Path to write replacements parquet file
    """
    tsv_file = Path(tsv_path)
    if not tsv_file.exists():
        print(f"Error: TSV file not found: {tsv_path}")
        return

    try:
        existing = load_existing_replacements(output_path)

        with open(tsv_file, "r") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) != 2:
                    print(f"Warning: Skipping line {line_num} (expected 2 columns): {line}")
                    continue

                original, replacement = parts
                existing[original.lower()] = replacement.lower()

        _write_replacements(existing, output_path, "manual")
        print(f"Imported replacements from {tsv_path}")

    except Exception as e:
        print(f"Error importing from TSV: {e}")


def interactive_entry(output_path: str) -> None:
    """Interactive mode for manually entering replacements.

    Args:
        output_path: Path to write replacements parquet file
    """
    existing = load_existing_replacements(output_path)
    print("Interactive replacement entry mode")
    print("Enter replacements (original -> replacement). Type 'done' to finish.")
    print()

    while True:
        original = input("Original word (or 'done'): ").strip().lower()
        if original == "done":
            break

        if not original:
            continue

        replacement = input("Replacement word: ").strip().lower()
        if not replacement:
            print("Replacement cannot be empty, skipping...")
            continue

        existing[original] = replacement
        print(f"Added: {original} -> {replacement}\n")

    _write_replacements(existing, output_path, "manual")
    print(f"\nSaved {len(existing)} replacements to {output_path}")


def _write_replacements(replacements: Dict[str, str], output_path: str, confidence: str) -> None:
    """Write replacements dictionary to parquet file.

    Args:
        replacements: Dictionary mapping original -> replacement
        output_path: Path to write parquet file
        confidence: Confidence level for all replacements
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for orig, repl in sorted(replacements.items()):
        # Auto-detect phrases (multi-word entries)
        is_phrase = " " in orig
        records.append(
            {
                "original": orig,
                "replacement": repl,
                "confidence": confidence,
                "is_phrase": is_phrase,
            }
        )

    df = pl.DataFrame(
        records,
        schema={
            "original": pl.String,
            "replacement": pl.String,
            "confidence": pl.String,
            "is_phrase": pl.Boolean,
        },
    )

    df.write_parquet(output_path)
    print(f"Wrote {len(replacements)} replacements to {output_path}")


def seed_initial_replacements(output_path: str) -> None:
    """Seed initial replacements with a small set of safe, verified replacements.

    Includes both word-level and phrase-level replacements.

    Args:
        output_path: Path to write replacements parquet file
    """
    seed_replacements = {
        # Word-level replacements
        "delineate": "describe",
        "negate": "remove",
        "inherent": "built-in",
        # Phrase-level replacements (to avoid awkward grammar)
        "inherent in": "built into",
    }

    _write_replacements(seed_replacements, output_path, "manual")
    print(f"Seeded {len(seed_replacements)} initial replacements")


def main():
    """Main execution function."""
    parser = ArgumentParser(
        description="Build curated replacements table for lexical simplification"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="datasets/replacements.parquet",
        help="Output parquet file path (default: datasets/replacements.parquet)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # WordNet command
    subparsers.add_parser("wordnet", help="Generate replacements from WordNet synonyms")

    # Import JSON command
    json_parser = subparsers.add_parser("import-json", help="Import from JSON file")
    json_parser.add_argument("file", type=str, help="Path to JSON file")

    # Import TSV command
    tsv_parser = subparsers.add_parser("import-tsv", help="Import from TSV file")
    tsv_parser.add_argument("file", type=str, help="Path to TSV file")

    # Interactive command
    subparsers.add_parser("interactive", help="Interactive manual entry mode")

    # Seed command
    subparsers.add_parser("seed", help="Seed initial replacements")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "wordnet":
        existing = load_existing_replacements(args.output)
        generate_from_wordnet(args.output, existing)
    elif args.command == "import-json":
        import_from_json(args.file, args.output)
    elif args.command == "import-tsv":
        import_from_tsv(args.file, args.output)
    elif args.command == "interactive":
        interactive_entry(args.output)
    elif args.command == "seed":
        seed_initial_replacements(args.output)


if __name__ == "__main__":
    main()

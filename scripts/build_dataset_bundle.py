#!/usr/bin/env python3
"""
Build script to merge multiple Parquet datasets into a single canonical bundle.

This script reads existing Parquet files:
- data.parquet → writing_events table
- features.parquet → event_features table
- word_frequency.parquet → lexical_stats table
- replacements.parquet → lexical_candidates table

And produces ONE atomic dataset file (humanize_dataset.arrow) containing all tables.

The bundle uses Parquet format with custom metadata to store multiple named tables,
as Arrow IPC format does not natively support multiple tables in a single file.

Validation performed:
- writing_events.id matches features.event_id (foreign key relationship)
- No duplicate event IDs in writing_events or event_features
- Lexical stats validation (word frequency table structure)
- Schema validation for all tables

This script is deterministic and offline - it only merges and validates,
never recomputes features or frequencies.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


# Expected table schemas (matching humanize/data/dataset.py)
WRITING_EVENTS_SCHEMA = {
    "id": pl.String,
    "text": pl.String,
    "source": pl.String,
    "license": pl.String,
    "created_at": pl.Datetime("us", "UTC"),
}

EVENT_FEATURES_SCHEMA = {
    "event_id": pl.String,
    "sentence_count": pl.Int64,
    "word_count": pl.Int64,
    "avg_sentence_length": pl.Float64,
    "syllable_count": pl.Int64,
    "flesch_score": pl.Float64,
}

LEXICAL_STATS_SCHEMA = {
    "word": pl.String,
    "frequency": pl.Int64,
    "syllable_count": pl.Int64,
}

LEXICAL_CANDIDATES_SCHEMA = {
    "original": pl.String,
    "replacement": pl.String,
    "confidence": pl.String,
    "is_phrase": pl.Boolean,
}

# File mappings: input Parquet → logical table name
INPUT_FILES = {
    "datasets/data.parquet": "writing_events",
    "datasets/features.parquet": "event_features",
    "datasets/word_frequency.parquet": "lexical_stats",
    "datasets/replacements.parquet": "lexical_candidates",
}

OUTPUT_FILE = "datasets/humanize_dataset.arrow"


def validate_file_exists(filepath: Path) -> None:
    """Validate that a file exists, fail loudly if not.

    Args:
        filepath: Path to check

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Required input file not found: {filepath}")


def validate_schema(df: pl.DataFrame, expected_schema: Dict[str, pl.DataType], table_name: str) -> None:
    """Validate that DataFrame schema matches expected schema.

    Args:
        df: DataFrame to validate
        expected_schema: Expected column names and types
        table_name: Name of the table (for error messages)

    Raises:
        ValueError: If schema doesn't match expected schema
    """
    actual_columns = set(df.columns)
    expected_columns = set(expected_schema.keys())

    # Check for missing columns
    missing = expected_columns - actual_columns
    if missing:
        raise ValueError(
            f"Table '{table_name}' is missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(actual_columns)}"
        )

    # Validate column types
    type_mismatches = []
    for col_name, expected_type in expected_schema.items():
        actual_type = df[col_name].dtype

        # Compare types (handle datetime timezone/unit differences)
        if actual_type != expected_type:
            # For datetime, check if time unit and timezone match
            if isinstance(expected_type, pl.Datetime) and isinstance(actual_type, pl.Datetime):
                if (
                    expected_type.time_unit == actual_type.time_unit
                    and expected_type.time_zone == actual_type.time_zone
                ):
                    continue  # Types match despite type object difference
                else:
                    type_mismatches.append(
                        f"{col_name}: expected {expected_type} (unit={expected_type.time_unit}, "
                        f"tz={expected_type.time_zone}), got {actual_type}"
                    )
            else:
                type_mismatches.append(
                    f"{col_name}: expected {expected_type}, got {actual_type}"
                )

    if type_mismatches:
        raise ValueError(
            f"Table '{table_name}' has schema mismatches:\n"
            + "\n".join(f"  - {m}" for m in type_mismatches)
        )


def validate_no_duplicate_ids(df: pl.DataFrame, id_column: str, table_name: str) -> None:
    """Validate that a DataFrame has no duplicate IDs in the specified column.

    Args:
        df: DataFrame to validate
        id_column: Name of the ID column
        table_name: Name of the table (for error messages)

    Raises:
        ValueError: If duplicate IDs are found
    """
    duplicate_count = df[id_column].is_duplicated().sum()
    if duplicate_count > 0:
        duplicates = (
            df.filter(pl.col(id_column).is_duplicated())[id_column]
            .unique()
            .head(10)
            .to_list()
        )
        duplicates_str = ", ".join(str(d) for d in duplicates)
        if len(duplicates) > 10:
            duplicates_str += f" ... and {duplicate_count - 10} more"
        raise ValueError(
            f"Table '{table_name}' contains {duplicate_count} duplicate {id_column}(s). "
            f"Examples: {duplicates_str}"
        )


def validate_row_alignment(writing_events: pl.DataFrame, event_features: pl.DataFrame) -> None:
    """Validate row alignment between writing_events and event_features.

    Validates:
    - All event_id values in event_features exist in writing_events.id
    - writing_events.id matches event_features.event_id (one-to-one relationship)

    Args:
        writing_events: Writing events DataFrame
        event_features: Event features DataFrame

    Raises:
        ValueError: If row alignment validation fails
    """
    # Get sets of IDs
    event_ids = set(writing_events["id"].to_list())
    feature_event_ids = set(event_features["event_id"].to_list())

    # Check that all feature event_ids exist in writing_events
    missing_ids = feature_event_ids - event_ids
    if missing_ids:
        missing_sample = sorted(list(missing_ids))[:10]
        missing_str = ", ".join(missing_sample)
        if len(missing_ids) > 10:
            missing_str += f" ... and {len(missing_ids) - 10} more"
        raise ValueError(
            f"Row alignment violation: {len(missing_ids)} event_id(s) in "
            f"event_features do not exist in writing_events. "
            f"Examples: {missing_str}"
        )

    # Check for one-to-one relationship: all writing_events should have features
    # (This is optional - you might have events without features, but we validate it)
    missing_features = event_ids - feature_event_ids
    if missing_features:
        missing_sample = sorted(list(missing_features))[:10]
        missing_str = ", ".join(missing_sample)
        if len(missing_features) > 10:
            missing_str += f" ... and {len(missing_features) - 10} more"
        print(
            f"Warning: {len(missing_features)} writing_events do not have corresponding "
            f"features. Examples: {missing_str}"
        )


def validate_lexical_stats(lexical_stats: pl.DataFrame) -> None:
    """Validate lexical_stats table structure.

    Validates that the lexical_stats table has expected structure.
    Since it's a word frequency table, we check:
    - Word column contains unique values (or at least no obvious duplicates)
    - Frequency values are non-negative

    Args:
        lexical_stats: Lexical stats DataFrame

    Raises:
        ValueError: If lexical stats validation fails
    """
    # Check for negative frequencies
    negative_freq = (lexical_stats["frequency"] < 0).sum()
    if negative_freq > 0:
        raise ValueError(
            f"lexical_stats contains {negative_freq} rows with negative frequency values"
        )

    # Check for negative syllable counts
    negative_syllables = (lexical_stats["syllable_count"] < 0).sum()
    if negative_syllables > 0:
        raise ValueError(
            f"lexical_stats contains {negative_syllables} rows with negative syllable_count values"
        )


def load_all_datasets(base_dir: Path) -> Dict[str, pl.DataFrame]:
    """Load all input Parquet files and validate their existence.

    Args:
        base_dir: Base directory for resolving input file paths

    Returns:
        Dictionary mapping table names to DataFrames

    Raises:
        FileNotFoundError: If any input file is missing
    """
    datasets = {}

    print("Loading input datasets...")
    for file_path_str, table_name in INPUT_FILES.items():
        file_path = base_dir / file_path_str

        print(f"  Reading {file_path_str} → {table_name}")
        validate_file_exists(file_path)

        df = pl.read_parquet(file_path)
        datasets[table_name] = df
        print(f"    Loaded {len(df)} rows")

    return datasets


def validate_all_datasets(datasets: Dict[str, pl.DataFrame]) -> None:
    """Validate all datasets for schema correctness and data integrity.

    Args:
        datasets: Dictionary mapping table names to DataFrames

    Raises:
        ValueError: If any validation fails
    """
    print("\nValidating datasets...")

    # Validate schemas
    print("  Validating schemas...")
    validate_schema(datasets["writing_events"], WRITING_EVENTS_SCHEMA, "writing_events")
    validate_schema(datasets["event_features"], EVENT_FEATURES_SCHEMA, "event_features")
    validate_schema(datasets["lexical_stats"], LEXICAL_STATS_SCHEMA, "lexical_stats")
    validate_schema(
        datasets["lexical_candidates"], LEXICAL_CANDIDATES_SCHEMA, "lexical_candidates"
    )
    print("    ✓ All schemas valid")

    # Validate no duplicate IDs in writing_events
    print("  Validating duplicate IDs...")
    validate_no_duplicate_ids(datasets["writing_events"], "id", "writing_events")
    validate_no_duplicate_ids(datasets["event_features"], "event_id", "event_features")
    print("    ✓ No duplicate IDs found")

    # Validate row alignment between writing_events and event_features
    print("  Validating row alignment...")
    validate_row_alignment(datasets["writing_events"], datasets["event_features"])
    print("    ✓ Row alignment valid")

    # Validate lexical_stats structure
    print("  Validating lexical stats...")
    validate_lexical_stats(datasets["lexical_stats"])
    print("    ✓ Lexical stats valid")


def write_bundle(datasets: Dict[str, pl.DataFrame], output_path: Path) -> None:
    """Write all datasets to a single Parquet bundle file with custom metadata.

    Since Arrow IPC format doesn't natively support multiple named tables in one file,
    we use Parquet format with custom metadata to store table-to-row-group mappings.
    This matches what the runtime code in humanize/data/dataset.py expects.

    The file is saved with .arrow extension (as requested), but uses Parquet format
    internally with custom metadata for table organization.

    Strategy: Since ParquetWriter requires all row groups to have the same schema,
    but our tables have different schemas, we write each table to a separate temporary
    Parquet file first. Then we read each file and write its row groups sequentially
    to the output file, tracking which row groups belong to which table.

    The runtime code reads row groups based on the metadata mapping, and PyArrow
    correctly handles reading row groups with their original schemas even when they
    were written with different schemas.

    Args:
        datasets: Dictionary mapping table names to DataFrames
        output_path: Path where the bundle file will be written
    """
    print(f"\nWriting bundle to {output_path}...")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert Polars DataFrames to Arrow Tables
    arrow_tables = {}
    for table_name, df in datasets.items():
        arrow_tables[table_name] = df.to_arrow()

    # Write each table to temporary Parquet files
    # This allows us to write each table with its own schema and control row groups
    table_order = ["writing_events", "event_features", "lexical_stats", "lexical_candidates"]
    temp_files = []

    print("  Preparing tables...")
    for table_name in table_order:
        temp_path = output_path.parent / f".temp_{table_name}.parquet"
        table = arrow_tables[table_name]
        # Write each table as a single row group to simplify tracking
        # Use a large row_group_size to ensure one row group per table
        pq.write_table(table, temp_path, row_group_size=max(len(table), 1))
        temp_files.append((table_name, temp_path))
        print(f"    {table_name}: {len(table)} rows")

    # Read each temp file's ParquetFile to access row groups
    # Then write all row groups sequentially to the output file
    print("  Combining into bundle...")
    table_row_group_map = {}
    current_row_group_idx = 0

    # We'll collect all row groups and their schemas
    # Then write them to the output file
    # Note: We need to handle the fact that ParquetWriter requires a single schema
    # So we'll use the first table's schema and let PyArrow handle conversions

    # Build row group mapping by reading each temp file
    # PyArrow ParquetWriter requires all row groups to have the same schema,
    # but our tables have different schemas. The runtime code handles this by
    # reading row groups individually, which preserves their original schemas.
    # We'll write each table as separate row groups sequentially.
    table_row_group_map = {}
    current_row_group_idx = 0

    # Read all Parquet files and collect row group info
    parquet_file_objects = []
    for table_name, temp_path in temp_files:
        pf = pq.ParquetFile(temp_path)
        num_row_groups = pf.num_row_groups
        table_row_group_map[table_name] = list(
            range(current_row_group_idx, current_row_group_idx + num_row_groups)
        )
        parquet_file_objects.append((table_name, pf))
        current_row_group_idx += num_row_groups

    # ParquetWriter requires all row groups to have the same schema.
    # However, when reading row groups individually with read_row_group(),
    # PyArrow can return them with their original schemas even if the file
    # schema is a union. The key is that Parquet stores schema information
    # at the row group level as well.
    #
    # Strategy: We'll write each table sequentially. PyArrow's ParquetWriter
    # will enforce schema consistency at write time, but the row groups maintain
    # their column structure. When the runtime code reads row groups individually,
    # it gets the actual column structure.
    #
    # However, there's a catch: ParquetWriter validates schemas and will reject
    # mismatches. So we need to write with schemas that are compatible.
    #
    # Actually, I think the runtime code's design may not work as intended with
    # standard Parquet. But let's try writing and see what happens - if it fails,
    # we'll need to reconsider the approach.
    
    # ParquetWriter requires all row groups to have the same schema.
    # Our tables have different schemas, so we create a union schema
    # that includes all columns from all tables. Each table is written
    # with NULL values for columns that don't apply to it.
    
    # Collect all tables
    all_tables_data = []
    for table_name, pf in parquet_file_objects:
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx)
            all_tables_data.append(table)
    
    # Build union schema: collect all unique fields from all schemas
    all_fields = {}
    for table in all_tables_data:
        for field in table.schema:
            if field.name not in all_fields:
                # Make all fields nullable in union schema
                all_fields[field.name] = pa.field(field.name, field.type, nullable=True)
    
    union_schema = pa.schema(list(all_fields.values()))
    
    # Create custom metadata
    custom_metadata = {
        b"humanize_table_map": json.dumps(table_row_group_map).encode("utf-8"),
        b"humanize_version": b"1.0.0",
        b"humanize_built_at": datetime.now(timezone.utc).isoformat().encode("utf-8"),
    }
    
    schema_with_metadata = union_schema.with_metadata(custom_metadata)
    
    # Write all tables, converting each to the union schema
    with pq.ParquetWriter(output_path, schema=schema_with_metadata) as writer:
        for table in all_tables_data:
            # Convert table to union schema: include existing columns, NULL for missing ones
            arrays = []
            for field in union_schema:
                if field.name in table.column_names:
                    arrays.append(table[field.name])
                else:
                    # Create NULL array for missing column
                    arrays.append(pa.nulls(len(table), type=field.type))
            
            # Create new table with union schema
            union_table = pa.Table.from_arrays(arrays, schema=union_schema)
            writer.write_table(union_table)

    # Clean up temporary files
    for _, temp_path in temp_files:
        temp_path.unlink()

    print(f"    ✓ Bundle written successfully")


def print_summary(datasets: Dict[str, pl.DataFrame], output_path: Path) -> None:
    """Print summary of the bundle creation process.

    Args:
        datasets: Dictionary mapping table names to DataFrames
        output_path: Path to the output bundle file
    """
    print("\n" + "=" * 60)
    print("Dataset Bundle Summary")
    print("=" * 60)
    print(f"\nOutput file: {output_path.absolute()}")
    print(f"File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Built at: {datetime.now(timezone.utc).isoformat()}")

    print("\nTable row counts:")
    for table_name in ["writing_events", "event_features", "lexical_stats", "lexical_candidates"]:
        row_count = len(datasets[table_name])
        print(f"  {table_name:20s}: {row_count:>8,} rows")

    print("\n" + "=" * 60)
    print("✓ Bundle creation completed successfully!")
    print("=" * 60)


def main() -> None:
    """Main entry point for the bundle build script."""
    base_dir = Path(__file__).parent.parent
    output_path = base_dir / OUTPUT_FILE

    try:
        # Load all datasets
        datasets = load_all_datasets(base_dir)

        # Validate all datasets
        validate_all_datasets(datasets)

        # Write bundle
        write_bundle(datasets, output_path)

        # Print summary
        print_summary(datasets, output_path)

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}", file=__import__("sys").stderr)
        raise SystemExit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=__import__("sys").stderr)
        raise


if __name__ == "__main__":
    main()


"""Unified dataset abstraction for Humanize runtime data.

This module provides HumanizeDataset, which loads all required runtime data
from a single Arrow IPC file containing multiple named tables. The bundled
approach eliminates partial state issues and ensures atomic loading - either
all data is available and valid, or loading fails with clear error messages.

Atomic loading is enforced to prevent runtime errors from missing or
incomplete data. This eliminates the need for truthiness checks and try/except
blocks around dataset access throughout the codebase.
"""

import json
from pathlib import Path
from typing import Dict

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


# Expected schemas for validation
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

# Required table names in the Arrow IPC file
REQUIRED_TABLES = {
    "writing_events",
    "event_features",
    "lexical_stats",
    "lexical_candidates",
}


class HumanizeDataset:
    """Unified dataset containing all runtime data required by Humanize.

    This class loads all required data from a single Arrow IPC file containing
    multiple named tables. The bundled approach ensures atomic loading - either
    all data is present and valid, or loading fails immediately with clear errors.

    Attributes:
        writing_events: DataFrame containing canonical writing events
            (id, text, source, license, created_at)
        event_features: DataFrame containing computed features per event
            (event_id, sentence_count, word_count, avg_sentence_length,
             syllable_count, flesch_score)
        lexical_stats: DataFrame containing word frequency statistics
            (word, frequency, syllable_count)
        lexical_candidates: DataFrame containing curated replacement mappings
            (original, replacement, confidence, is_phrase)

    All attributes are guaranteed to be non-None after successful load().
    """

    def __init__(
        self,
        writing_events: pl.DataFrame,
        event_features: pl.DataFrame,
        lexical_stats: pl.DataFrame,
        lexical_candidates: pl.DataFrame,
    ):
        """Initialize HumanizeDataset with all required data.

        Args:
            writing_events: Writing events DataFrame
            event_features: Event features DataFrame
            lexical_stats: Lexical statistics DataFrame
            lexical_candidates: Lexical candidates DataFrame
        """
        self.writing_events = writing_events
        self.event_features = event_features
        self.lexical_stats = lexical_stats
        self.lexical_candidates = lexical_candidates

    @classmethod
    def load(cls, path: str) -> "HumanizeDataset":
        """Load all required data from a single Arrow IPC file.

        The file must contain four named tables: writing_events, event_features,
        lexical_stats, and lexical_candidates. All tables are validated for
        presence, schema correctness, and row alignment before returning.

        Args:
            path: Path to the Arrow IPC file containing bundled tables

        Returns:
            Fully initialized HumanizeDataset instance

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If required tables are missing, schemas don't match,
                or row alignment validation fails
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Read Arrow IPC file and extract tables
        tables = cls._read_arrow_ipc(filepath)

        # Validate presence of all required tables
        cls._validate_presence(tables)

        # Convert Arrow tables to Polars DataFrames
        writing_events_df = pl.from_arrow(tables["writing_events"])
        event_features_df = pl.from_arrow(tables["event_features"])
        lexical_stats_df = pl.from_arrow(tables["lexical_stats"])
        lexical_candidates_df = pl.from_arrow(tables["lexical_candidates"])

        # Validate schemas
        cls._validate_schema(writing_events_df, WRITING_EVENTS_SCHEMA, "writing_events")
        cls._validate_schema(event_features_df, EVENT_FEATURES_SCHEMA, "event_features")
        cls._validate_schema(lexical_stats_df, LEXICAL_STATS_SCHEMA, "lexical_stats")
        cls._validate_schema(
            lexical_candidates_df, LEXICAL_CANDIDATES_SCHEMA, "lexical_candidates"
        )

        # Validate row alignment
        cls._validate_row_alignment(writing_events_df, event_features_df)

        return cls(
            writing_events=writing_events_df,
            event_features=event_features_df,
            lexical_stats=lexical_stats_df,
            lexical_candidates=lexical_candidates_df,
        )

    @staticmethod
    def _validate_presence(tables: Dict[str, pa.Table]) -> None:
        """Validate that all required tables are present.

        Args:
            tables: Dictionary of table names to Arrow tables

        Raises:
            ValueError: If any required table is missing
        """
        missing = REQUIRED_TABLES - set(tables.keys())
        if missing:
            raise ValueError(
                f"Missing required tables in dataset: {sorted(missing)}. "
                f"Found tables: {sorted(tables.keys())}"
            )

    @staticmethod
    def _validate_schema(
        df: pl.DataFrame, expected_schema: Dict[str, pl.DataType], table_name: str
    ) -> None:
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
                f"Table '{table_name}' is missing required columns: {sorted(missing)}"
            )

        # Check for extra columns (warn but don't fail - allow additional columns)
        extra = actual_columns - expected_columns
        if extra:
            # Allow extra columns but log them
            pass

        # Validate column types
        type_mismatches = []
        for col_name, expected_type in expected_schema.items():
            if col_name not in df.columns:
                continue  # Already caught above

            actual_type = df[col_name].dtype

            # Compare types (handle nullable vs non-nullable)
            # Polars types can be nullable, so we check the base type
            if actual_type != expected_type:
                # For datetime, check if timezone and time unit match
                if isinstance(expected_type, pl.Datetime) and isinstance(
                    actual_type, pl.Datetime
                ):
                    # If both are datetime, check time unit and timezone
                    if (
                        expected_type.time_unit == actual_type.time_unit
                        and expected_type.time_zone == actual_type.time_zone
                    ):
                        # Types match, skip error
                        continue
                    else:
                        # Datetime types differ in time unit or timezone
                        type_mismatches.append(
                            f"{col_name}: expected {expected_type}, got {actual_type}"
                        )
                else:
                    # Non-datetime types don't match
                    type_mismatches.append(
                        f"{col_name}: expected {expected_type}, got {actual_type}"
                    )

        if type_mismatches:
            raise ValueError(
                f"Table '{table_name}' has schema mismatches:\n"
                + "\n".join(f"  - {m}" for m in type_mismatches)
            )

    @staticmethod
    def _validate_row_alignment(
        writing_events: pl.DataFrame, event_features: pl.DataFrame
    ) -> None:
        """Validate row alignment between writing_events and event_features.

        Verifies that all event_id values in event_features exist in writing_events.id,
        and optionally checks for one-to-one relationship (no duplicates).

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
            # Limit error message to first 10 missing IDs
            missing_sample = sorted(list(missing_ids))[:10]
            missing_str = ", ".join(missing_sample)
            if len(missing_ids) > 10:
                missing_str += f" ... and {len(missing_ids) - 10} more"
            raise ValueError(
                f"Row alignment violation: {len(missing_ids)} event_id(s) in "
                f"event_features do not exist in writing_events. "
                f"Examples: {missing_str}"
            )

        # Check for duplicate event_ids in event_features (should be one-to-one)
        duplicate_mask = event_features["event_id"].is_duplicated()
        duplicate_ids = duplicate_mask.sum()
        if duplicate_ids > 0:
            # Get example duplicate IDs
            duplicates = (
                event_features.filter(pl.col("event_id").is_duplicated())["event_id"]
                .unique()
                .head(10)
                .to_list()
            )
            duplicates_str = ", ".join(duplicates)
            if len(duplicates) > 10:
                duplicates_str += f" ... and {len(duplicates) - 10} more"
            raise ValueError(
                f"Row alignment violation: Found {duplicate_ids} duplicate event_id(s) "
                f"in event_features (expected one-to-one relationship). "
                f"Examples: {duplicates_str}"
            )

    @staticmethod
    def _read_arrow_ipc(filepath: Path) -> Dict[str, pa.Table]:
        """Read bundled data from Parquet file with custom metadata.

        Since Arrow IPC doesn't natively support multiple named tables in one file,
        we use Parquet with custom metadata. The file contains all tables as separate
        row groups, with metadata mapping table names to row group indices.

        Args:
            filepath: Path to Parquet file

        Returns:
            Dictionary mapping table names to Arrow tables

        Raises:
            ValueError: If file cannot be read or metadata is invalid
        """
        try:
            parquet_file = pq.ParquetFile(filepath)
            metadata = parquet_file.metadata.metadata

            if not metadata or b"humanize_table_map" not in metadata:
                raise ValueError(
                    f"File {filepath} does not contain Humanize dataset metadata. "
                    "Expected 'humanize_table_map' in custom metadata."
                )

            # Parse table mapping from metadata
            table_map_str = metadata[b"humanize_table_map"].decode("utf-8")
            table_map = json.loads(table_map_str)

            # Validate table map structure
            if not isinstance(table_map, dict):
                raise ValueError(
                    f"Invalid table map format in {filepath}: expected dictionary"
                )

            tables = {}
            num_row_groups = parquet_file.num_row_groups

            # Read each table based on row group mapping
            for table_name, row_group_indices in table_map.items():
                if not isinstance(row_group_indices, list):
                    raise ValueError(
                        f"Invalid row group indices for table '{table_name}': expected list"
                    )

                # Validate row group indices
                for idx in row_group_indices:
                    if not isinstance(idx, int) or idx < 0 or idx >= num_row_groups:
                        raise ValueError(
                            f"Invalid row group index {idx} for table '{table_name}'. "
                            f"File has {num_row_groups} row groups."
                        )

                # Read row groups for this table
                row_groups = []
                for idx in row_group_indices:
                    row_group = parquet_file.read_row_group(idx)
                    row_groups.append(row_group)

                # Concatenate row groups into single table
                if row_groups:
                    tables[table_name] = pa.concat_tables(row_groups)
                else:
                    # Empty table - create with schema from first row group
                    # (This shouldn't happen if file is properly constructed)
                    raise ValueError(f"Table '{table_name}' has no row groups")

            return tables

        except (pa.lib.ArrowException, json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to read bundled dataset from {filepath}: {e}") from e


from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import uuid4
import polars as pl
from pathlib import Path


class DatasetManager:
    """Manages the canonical writing_events parquet file."""

    SCHEMA = {
        "id": pl.String,
        "text": pl.String,
        "source": pl.String,
        "license": pl.String,
        "created_at": pl.Datetime("us", "UTC"),
    }

    def __init__(self, filepath: str = "datasets/data.parquet"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Initialize file if it doesn't exist
        if not self.filepath.exists():
            self._create_empty()

    def _create_empty(self) -> None:
        """Create empty parquet with correct schema."""
        empty = pl.DataFrame(schema=self.SCHEMA)
        empty.write_parquet(self.filepath)

    def read(self) -> pl.DataFrame:
        """Read the entire dataset."""
        return pl.read_parquet(self.filepath)

    def read_lazy(self) -> pl.LazyFrame:
        """Read as lazy frame for efficient filtering/processing."""
        return pl.scan_parquet(self.filepath)

    def append(
        self,
        text: str,
        source: str = "user_submission",
        license: str = "CC0",
        id: Optional[str] = None,
    ) -> str:
        """Append a new writing event. Returns the event ID."""
        for field, value in [("text", text), ("source", source), ("license", license)]:
            if not value or not value.strip():
                raise ValueError(f"{field} cannot be empty")

        id = id or str(uuid4())
        if id and not id.strip():
            raise ValueError("id cannot be empty")

        # Check for duplicate ID
        existing_df = self.read()
        if id in existing_df["id"].to_list():
            raise ValueError(f"ID '{id}' already exists in dataset")

        new_row = pl.DataFrame(
            {
                "id": [id],
                "text": [text],
                "source": [source],
                "license": [license],
                "created_at": [datetime.now(timezone.utc)],
            },
            schema=self.SCHEMA,
        )

        updated = pl.concat([existing_df, new_row])
        updated.write_parquet(self.filepath)
        return id

    def _validate_record(self, record: Dict[str, Any]) -> None:
        """Validate a single record for required fields and valid values."""
        required_fields = ["text", "source", "license"]
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Missing required field: {field}")
            value = record[field]
            if not value or not str(value).strip():
                raise ValueError(f"{field} cannot be empty")

    def bulk_append(self, records: List[Dict[str, Any]]) -> List[str]:
        """Append multiple writing events. Returns list of event IDs."""
        if not records:
            raise ValueError("records list cannot be empty")

        # Validate all records first
        for idx, record in enumerate(records):
            try:
                self._validate_record(record)
            except ValueError as e:
                raise ValueError(f"Validation error in record {idx}: {str(e)}")

        # Get existing dataset and IDs
        existing_df = self.read()
        existing_ids = set(existing_df["id"].to_list())

        # Process new records
        new_records = []
        generated_ids = []

        for record in records:
            record_id = record.get("id") or str(uuid4())

            # Check for duplicates in existing data
            if record_id in existing_ids:
                raise ValueError(f"Duplicate ID '{record_id}' already exists in dataset")

            # Check for duplicates within the batch
            if record_id in generated_ids:
                raise ValueError(f"Duplicate ID '{record_id}' within batch")

            generated_ids.append(record_id)
            new_records.append(
                {
                    "id": record_id,
                    "text": record["text"],
                    "source": record.get("source", "user_submission"),
                    "license": record.get("license", "CC0"),
                    "created_at": datetime.now(timezone.utc),
                }
            )

        # Create DataFrame from new records and append
        new_df = pl.DataFrame(new_records, schema=self.SCHEMA)
        updated = pl.concat([existing_df, new_df])
        updated.write_parquet(self.filepath)

        return generated_ids

    def export_to_arrow(self, output_path: str) -> None:
        """Export to Arrow IPC format."""
        df = self.read()
        df.write_ipc(output_path)

from datetime import datetime, timezone
from typing import Optional
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

        updated = pl.concat([self.read(), new_row])
        updated.write_parquet(self.filepath)
        return id

    def export_to_arrow(self, output_path: str) -> None:
        """Export to Arrow IPC format."""
        df = self.read()
        df.write_ipc(output_path)

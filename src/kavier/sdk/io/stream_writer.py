"""Incremental writer that appends DataFrames to a single parquet file."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class StreamingParquetWriter:
    """Append DataFrames to one snappy parquet ``path``; schema is fixed from the first non-empty ``write``."""

    def __init__(self, path: str):
        self._path = path
        self._writer = None

    def write(self, df: pd.DataFrame):
        """Append ``df`` to the parquet file; skips empty frames and infers schema from the first write."""
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._path, table.schema, compression="snappy")
        self._writer.write_table(table)

    def close(self):
        """Finalise and flush the parquet file; no-op if nothing was written."""
        if self._writer is not None:
            self._writer.close()

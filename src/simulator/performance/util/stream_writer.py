import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class StreamingParquetWriter:
    def __init__(self, path: str):
        self._path = path
        self._writer = None

    def write(self, df: pd.DataFrame):
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._path, table.schema, compression="snappy")
        self._writer.write_table(table)

    def close(self):
        if self._writer is not None:
            self._writer.close()

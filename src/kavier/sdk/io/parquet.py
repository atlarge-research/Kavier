"""Parquet reading."""

from __future__ import annotations

import os
from collections.abc import Sequence

import pandas as pd
import pyarrow.parquet as pq


def read_parquet(path: str | os.PathLike[str], columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a parquet file into a DataFrame.

    Arrow opens the path itself and reads on one thread. ``pd.read_parquet`` hands Arrow a Python
    file object instead, and a threaded read of one can abort the process at exit
    (apache/arrow#34314).
    """
    table = pq.read_table(os.fspath(path), columns=None if columns is None else list(columns), use_threads=False)
    return table.to_pandas(use_threads=False)

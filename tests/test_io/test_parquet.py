"""Tests for kavier.sdk.io.parquet.read_parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from kavier.sdk.io.parquet import read_parquet

SRC = Path(__file__).resolve().parents[2] / "src" / "kavier"


def _sample(tmp_path: Path) -> Path:
    path = tmp_path / "sample.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="15min"),
            "power_draw": [1.5, 2.0, 2.5],
            "host_id": ["a", "b", "c"],
            "tokens": [1, 2, 3],
        }
    ).to_parquet(path)
    return path


def test_matches_pandas(tmp_path):
    path = _sample(tmp_path)
    pd.testing.assert_frame_equal(read_parquet(path), pd.read_parquet(path))


def test_reads_selected_columns(tmp_path):
    path = _sample(tmp_path)
    assert list(read_parquet(path, columns=["tokens"]).columns) == ["tokens"]


def test_arrow_opens_the_path_and_reads_single_threaded(tmp_path, monkeypatch):
    # A Python file object passed to Arrow, read with threads, can abort the process at exit
    # (apache/arrow#34314); pd.read_parquet passes one for local paths.
    calls = []
    real = pq.read_table

    def spy(source, *args, **kwargs):
        calls.append((source, kwargs))
        return real(source, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", spy)
    read_parquet(_sample(tmp_path))
    ((source, kwargs),) = calls
    assert isinstance(source, str)
    assert kwargs["use_threads"] is False


def test_src_does_not_call_pandas_read_parquet():
    offenders = [str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if "pd.read_parquet(" in p.read_text()]
    assert offenders == []

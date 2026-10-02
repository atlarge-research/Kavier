"""Tests for InputSpec trace loading from CSV and parquet.

input_tokens, output_tokens and session_id are optional in both formats. Regression, issue #6:
the parquet path requested all five columns and raised ArrowInvalid on minimal traces.
A full five-column trace loads the same from CSV and parquet.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from kavier.sdk.io.input_spec import InputSpec

_MINIMAL = pd.DataFrame(
    {
        "num_input_tokens": [10, 20, 30],
        "num_output_tokens": [1, 2, 3],
    }
)

_IN_TOKENS = [[1, 2, 3], [4, 5], [6]]
_OUT_TOKENS = [[9], [8, 7], [6, 5, 4]]

_FULL = pd.DataFrame(
    {
        "num_input_tokens": [3, 2, 1],
        "num_output_tokens": [1, 2, 3],
        "input_tokens": [json.dumps(t) for t in _IN_TOKENS],
        "output_tokens": [json.dumps(t) for t in _OUT_TOKENS],
        "session_id": ["a", "b", "a"],
    }
)


def _write(df: pd.DataFrame, tmp_path, fmt: str) -> str:
    path = tmp_path / f"trace.{fmt}"
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return str(path)


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_minimal_two_column_trace_loads(tmp_path, fmt) -> None:
    spec = InputSpec(_write(_MINIMAL, tmp_path, fmt))
    assert spec.num_in_t.tolist() == [10, 20, 30]
    assert spec.num_out_t.tolist() == [1, 2, 3]
    # num_tot_t = num_in_t + num_out_t
    assert spec.num_tot_t.tolist() == [11, 22, 33]
    # No extra columns: token lists empty, no sessions.
    assert spec.in_t == [] and spec.out_t == []
    assert spec.sessions is None


def test_minimal_csv_and_parquet_load_equivalently(tmp_path) -> None:
    # pandas.read_csv and the pyarrow dataset reader agree.
    csv_spec = InputSpec(_write(_MINIMAL, tmp_path, "csv"))
    pq_spec = InputSpec(_write(_MINIMAL, tmp_path, "parquet"))
    assert csv_spec.num_in_t.tolist() == pq_spec.num_in_t.tolist()
    assert csv_spec.num_out_t.tolist() == pq_spec.num_out_t.tolist()
    assert csv_spec.num_tot_t.tolist() == pq_spec.num_tot_t.tolist()
    assert csv_spec.in_t == pq_spec.in_t == []
    assert csv_spec.out_t == pq_spec.out_t == []
    assert csv_spec.sessions is None and pq_spec.sessions is None


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_full_five_column_roundtrip(tmp_path, fmt) -> None:
    spec = InputSpec(_write(_FULL, tmp_path, fmt))
    assert spec.num_in_t.tolist() == [3, 2, 1]
    assert spec.num_out_t.tolist() == [1, 2, 3]
    # JSON-encoded lists of ints decode back to lists of ints.
    assert spec.in_t == _IN_TOKENS
    assert spec.out_t == _OUT_TOKENS
    assert spec.sessions == ["a", "b", "a"]


def test_full_csv_and_parquet_load_equivalently(tmp_path) -> None:
    # Token parsing and session extraction agree across both readers.
    csv_spec = InputSpec(_write(_FULL, tmp_path, "csv"))
    pq_spec = InputSpec(_write(_FULL, tmp_path, "parquet"))
    assert csv_spec.in_t == pq_spec.in_t
    assert csv_spec.out_t == pq_spec.out_t
    assert csv_spec.sessions == pq_spec.sessions
    assert csv_spec.num_tot_t.tolist() == pq_spec.num_tot_t.tolist()


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_token_lists_load_without_session_id(tmp_path, fmt) -> None:
    df = pd.DataFrame(
        {
            "num_input_tokens": [3],
            "num_output_tokens": [1],
            "input_tokens": [json.dumps([1, 2, 3])],
            "output_tokens": [json.dumps([9])],
        }
    )
    spec = InputSpec(_write(df, tmp_path, fmt))
    assert spec.in_t == [[1, 2, 3]]
    assert spec.out_t == [[9]]
    assert spec.sessions is None


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_input_tokens_load_without_output_tokens(tmp_path, fmt, capsys) -> None:
    # The prefix cache reads only the input tokens, so output_tokens is optional on its own.
    df = pd.DataFrame(
        {
            "num_input_tokens": [3],
            "num_output_tokens": [1],
            "input_tokens": [json.dumps([1, 2, 3])],
            "session_id": ["a"],
        }
    )
    spec = InputSpec(_write(df, tmp_path, fmt))
    assert spec.in_t == [[1, 2, 3]]
    assert spec.out_t == []
    assert spec.sessions == ["a"]
    # The log names the column that is missing.
    out = capsys.readouterr().out
    assert "'output_tokens'" in out
    assert "'input_tokens'" not in out


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_extra_columns_are_optional_one_by_one(tmp_path, fmt) -> None:
    # Each optional column loads when present, whatever the other two are.
    df = pd.DataFrame(
        {
            "num_input_tokens": [3],
            "num_output_tokens": [1],
            "output_tokens": [json.dumps([9])],
        }
    )
    spec = InputSpec(_write(df, tmp_path, fmt))
    assert spec.in_t == []
    assert spec.out_t == [[9]]
    assert spec.sessions is None


def test_parquet_list_token_columns_load_like_json_strings(tmp_path) -> None:
    # A parquet file can store token lists as list<int64> columns instead of JSON strings.
    native = _FULL.assign(input_tokens=_IN_TOKENS, output_tokens=_OUT_TOKENS)
    path = tmp_path / "native.parquet"
    native.to_parquet(path, index=False)
    spec = InputSpec(str(path))
    assert spec.in_t == _IN_TOKENS
    assert spec.out_t == _OUT_TOKENS
    assert all(type(t) is int for row in spec.in_t for t in row)


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_scalar_token_value_raises_value_error(tmp_path, fmt) -> None:
    # "5" is valid JSON but not a list of tokens; read_csv turns it into the int 5.
    df = _FULL.iloc[:1].assign(input_tokens="5")
    with pytest.raises(ValueError, match="Bad token string"):
        InputSpec(_write(df, tmp_path, fmt))


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_empty_token_cell_parses_to_empty_list(tmp_path, fmt) -> None:
    # An empty or NaN token cell yields [] instead of failing in json.loads("").
    df = pd.DataFrame(
        {
            "num_input_tokens": [3, 2],
            "num_output_tokens": [1, 1],
            "input_tokens": ["", json.dumps([4, 5])],
            "output_tokens": [json.dumps([9]), ""],
            "session_id": ["a", "b"],
        }
    )
    spec = InputSpec(_write(df, tmp_path, fmt))
    assert spec.in_t == [[], [4, 5]]
    assert spec.out_t == [[9], []]


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_malformed_token_string_raises(tmp_path, fmt) -> None:
    # A non-JSON token string raises ValueError instead of JSONDecodeError.
    df = pd.DataFrame(
        {
            "num_input_tokens": [3],
            "num_output_tokens": [1],
            "input_tokens": ["not-json"],
            "output_tokens": [json.dumps([9])],
            "session_id": ["a"],
        }
    )
    with pytest.raises(ValueError, match="Bad token string"):
        InputSpec(_write(df, tmp_path, fmt))


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_missing_required_column_raises(tmp_path, fmt) -> None:
    df = pd.DataFrame({"num_input_tokens": [1, 2]})
    with pytest.raises(ValueError, match="requires"):
        InputSpec(_write(df, tmp_path, fmt))


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_empty_trace_raises(tmp_path, fmt) -> None:
    empty = _MINIMAL.iloc[0:0]
    with pytest.raises(ValueError, match="empty"):
        InputSpec(_write(empty, tmp_path, fmt))


def test_unsupported_extension_raises(tmp_path) -> None:
    path = tmp_path / "trace.txt"
    path.write_text("num_input_tokens,num_output_tokens\n1,2\n")
    with pytest.raises(ValueError, match="csv or .parquet"):
        InputSpec(str(path))

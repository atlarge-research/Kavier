"""InputSpec trace loading: the optional columns (input_tokens/output_tokens/session_id) are
really optional for BOTH formats (regression, issue #6: the parquet path requested all five
columns and crashed with ArrowInvalid on minimal traces the CSV path accepted), and a full
five-column trace round-trips identically through CSV and parquet."""

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
    # Data preserved verbatim by the loader.
    assert spec.num_in_t.tolist() == [10, 20, 30]
    assert spec.num_out_t.tolist() == [1, 2, 3]
    # num_tot_t = num_in_t + num_out_t, elementwise: 10+1, 20+2, 30+3.
    assert spec.num_tot_t.tolist() == [11, 22, 33]
    # No extra columns -> token lists stay empty, sessions untracked.
    assert spec.in_t == [] and spec.out_t == []
    assert spec.sessions is None


def test_minimal_csv_and_parquet_load_equivalently(tmp_path) -> None:
    # Cross-check: two independent readers (pandas.read_csv vs pyarrow dataset) must agree.
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
    # We JSON-encoded lists of ints; the loader must decode them back to lists of ints.
    assert spec.in_t == _IN_TOKENS
    assert spec.out_t == _OUT_TOKENS
    assert spec.sessions == ["a", "b", "a"]


def test_full_csv_and_parquet_load_equivalently(tmp_path) -> None:
    # Cross-check: JSON token parsing and session extraction agree across both readers.
    csv_spec = InputSpec(_write(_FULL, tmp_path, "csv"))
    pq_spec = InputSpec(_write(_FULL, tmp_path, "parquet"))
    assert csv_spec.in_t == pq_spec.in_t
    assert csv_spec.out_t == pq_spec.out_t
    assert csv_spec.sessions == pq_spec.sessions
    assert csv_spec.num_tot_t.tolist() == pq_spec.num_tot_t.tolist()


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_token_parsing_gated_on_all_three_extra_columns(tmp_path, fmt) -> None:
    # The loader only parses token lists when input_tokens AND output_tokens AND session_id
    # are all present (the gate is set(extra_cols).issubset). A trace with token columns but
    # NO session_id therefore leaves in_t/out_t empty despite the data being present.
    df = pd.DataFrame(
        {
            "num_input_tokens": [3],
            "num_output_tokens": [1],
            "input_tokens": [json.dumps([1, 2, 3])],
            "output_tokens": [json.dumps([9])],
        }
    )
    spec = InputSpec(_write(df, tmp_path, fmt))
    assert spec.in_t == [] and spec.out_t == []
    assert spec.sessions is None


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_empty_token_cell_parses_to_empty_list(tmp_path, fmt) -> None:
    # An empty/NaN token cell must yield [] rather than crashing json.loads("").
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
    # First input cell empty -> []; second output cell empty -> [].
    assert spec.in_t == [[], [4, 5]]
    assert spec.out_t == [[9], []]


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_malformed_token_string_raises(tmp_path, fmt) -> None:
    # A non-JSON token string must surface as a ValueError, not a raw JSONDecodeError.
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
    df = pd.DataFrame({"num_input_tokens": [1, 2]})  # no num_output_tokens
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

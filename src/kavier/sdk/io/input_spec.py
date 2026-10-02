import json
import os
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
from pandas.api.types import is_list_like
from tqdm.auto import tqdm

from kavier.sdk.io.log import log


def _to_token_list(s: Any) -> list[int]:
    """Return one cell as a list of int tokens; a cell is a JSON list string, a list-like, or missing."""
    if isinstance(s, str):
        return [int(t) for t in json.loads(s)] if s else []
    # Parquet list columns arrive as numpy arrays.
    if is_list_like(s):
        return [int(t) for t in s]
    if pd.isna(s):
        return []
    raise TypeError(f"expected a list of tokens, got {type(s).__name__}")


def _string_array_to_tokens(strings, tqdm_message=""):
    tokens = []
    for s in tqdm(strings, desc=tqdm_message, unit="row"):
        try:
            tokens.append(_to_token_list(s))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Bad token string: {s!r}: {e}") from e
    return tokens


class InputSpec:
    """Inference trace from .csv or .parquet; needs num_input_tokens and num_output_tokens, rest optional."""

    def __init__(self, path: str):
        self.path = path
        self.num_in_t = self.num_out_t = self.num_tot_t = None
        self.in_t, self.out_t = [], []
        self.sessions = None
        self.df = None
        self._load(path)

    def _load(self, path):
        filetype = os.path.splitext(path)[-1].lstrip(".").lower()

        base_cols = ["num_input_tokens", "num_output_tokens"]
        extra_cols = ["input_tokens", "output_tokens", "session_id"]
        cols_needed = base_cols + extra_cols

        if filetype == "parquet":
            dataset = ds.dataset(path, format="parquet")
            # Extra columns are optional; read only those present, as the CSV path does.
            cols = [c for c in cols_needed if c in dataset.schema.names]
            tbl = dataset.to_table(columns=cols)
            self.df = tbl.to_pandas(self_destruct=True)
        elif filetype == "csv":
            self.df = pd.read_csv(
                path,
                usecols=lambda c: c in cols_needed,
            )
        else:
            raise ValueError("Trace must be .csv or .parquet")

        if self.df.empty:
            raise ValueError("Trace file is empty.")

        if not set(base_cols).issubset(self.df.columns):
            raise ValueError("Trace requires 'num_input_tokens' & 'num_output_tokens'.")

        self.num_in_t = self.df["num_input_tokens"]
        self.num_out_t = self.df["num_output_tokens"]
        self.num_tot_t = self.num_in_t + self.num_out_t

        # Each optional column loads on its own; the prefix cache needs only input_tokens.
        if "input_tokens" in self.df.columns:
            self.in_t = _string_array_to_tokens(self.df["input_tokens"].tolist(), tqdm_message="Loading input tokens")
        else:
            log("[yellow]No 'input_tokens' column; prefix cache not used.")
        if "output_tokens" in self.df.columns:
            self.out_t = _string_array_to_tokens(
                self.df["output_tokens"].tolist(), tqdm_message="Loading output tokens"
            )
        else:
            log("[yellow]No 'output_tokens' column; skipping output token lists.")
        if self.in_t or self.out_t:
            log("Token lists loaded.")

        if "session_id" in self.df.columns:
            self.sessions = self.df["session_id"].tolist()
            log(f"Found {len(set(self.sessions))} unique sessions in the trace.")
        else:
            self.sessions = None
            log("[yellow]No 'session_id' column; sessions not tracked.")

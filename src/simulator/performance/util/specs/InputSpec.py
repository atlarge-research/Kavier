import json
import os

import pandas as pd
import pyarrow.dataset as ds
from tqdm.auto import tqdm

from simulator.log import log


def _string_array_to_tokens(strings, tqdm_message=""):
    tokens = []
    for s in tqdm(strings, desc=tqdm_message, unit="row"):
        if s in (None, "") or (isinstance(s, float) and pd.isna(s)):
            tokens.append([])
            continue
        try:
            tokens.append([int(t) for t in json.loads(s)])
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Bad token string: {s!r} → {e}")
    return tokens


class InputSpec:
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
            tbl = ds.dataset(path, format="parquet").to_table(columns=cols_needed)
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

        if set(extra_cols).issubset(self.df.columns):
            self.in_t = _string_array_to_tokens(
                self.df["input_tokens"].tolist(), tqdm_message="[1/2] Loading input tokens"
            )
            self.out_t = _string_array_to_tokens(
                self.df["output_tokens"].tolist(), tqdm_message="[2/2] Loading output tokens"
            )
            log(":white_check_mark:  Token lists loaded successfully.")
        else:
            log("[yellow]⚠️  'input_tokens' / 'output_tokens' missing → skipping token lists.")
            self.in_t, self.out_t = [], []

        if "session_id" in self.df.columns:
            self.sessions = self.df["session_id"].tolist()
            log(f"Found {len(set(self.sessions))} unique sessions in the trace.")
        else:
            self.sessions = None
            log("[yellow]No 'session_id' column found → sessions not tracked.")

import pandas as pd

from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY
from kavier.sdk.library.llm import LLM_SPEC_LIBRARY

MAX_TOTAL_GPUS = 32


def throughput_column(df: pd.DataFrame) -> str:
    if "measured_throughput" in df.columns:
        return "measured_throughput"
    if "actual_throughput" in df.columns:
        return "actual_throughput"
    raise KeyError("CSV must have measured_throughput or actual_throughput")


def simulatable_mask(df: pd.DataFrame) -> pd.Series:
    total_gpus = df["number_gpus"].astype(int) * df["number_nodes"].astype(int)
    tcol = throughput_column(df)
    return (
        df["model_name"].isin(LLM_SPEC_LIBRARY)
        & df["gpu_model"].isin(GPU_SPEC_LIBRARY)
        & (total_gpus <= MAX_TOTAL_GPUS)
        & (df[tcol] > 0)
    )

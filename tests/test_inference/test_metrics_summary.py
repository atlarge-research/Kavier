"""Tests for the run summary text returned by ``Metrics.summary``."""

from __future__ import annotations

import pytest

from kavier.sdk.inference.core.cache import PrefixCache
from kavier.sdk.inference.core.config import SimConfig
from kavier.sdk.inference.core.metrics import Metrics
from kavier.sdk.library.gpu import GPU_SPEC_LIBRARY
from kavier.sdk.library.llm import LLM_SPEC_LIBRARY


def _summary(export_rate: float) -> str:
    metrics = Metrics()
    metrics.add(0.1, 0.2, 300.0)
    cfg = SimConfig(export_rate=export_rate)
    return metrics.summary(PrefixCache(cfg.cache), 1, GPU_SPEC_LIBRARY["A10"], LLM_SPEC_LIBRARY["Llama-3-8B"], cfg)


@pytest.mark.parametrize(("rate", "shown"), [(0.01, "0.01"), (0.001, "0.001"), (0.1, "0.1"), (2.5, "2.5")])
def test_summary_shows_small_export_rates(rate, shown) -> None:
    (line,) = [ln for ln in _summary(rate).splitlines() if ln.startswith("Export rate (s)")]
    assert line.split()[-1] == shown


def test_summary_is_plain_ascii() -> None:
    # The summary is printed and written to _sim_results.txt, so it must encode in any locale.
    assert _summary(0.1).isascii()

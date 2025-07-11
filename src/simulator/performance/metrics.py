from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from simulator.config import SimConfig
from simulator.performance.cache import PrefixCache
from simulator.performance.util.specs.GPUSpec import GPUSpec
from simulator.performance.util.specs.LLMSpec import LLMSpec


@dataclass
class Metrics:
    sum_prefill: float = 0.0
    sum_decode: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def add(self, prefill_s: float, decode_s: float, latency_ms: float) -> None:
        self.sum_prefill += prefill_s
        self.sum_decode += decode_s
        self.latencies.append(latency_ms)

    def summary(
            self,
            cache: PrefixCache,
            n_req: int,
            fragments: pd.DataFrame,
            gpu: GPUSpec,
            llm: LLMSpec,
            cfg: SimConfig,
    ) -> str:
        p95_lat = np.percentile(self.latencies, 95)
        total_s = self.sum_prefill + self.sum_decode

        hdr = "─" * 46
        return (
            f"\n{hdr}\n"
            f"{'SIMULATION SUMMARY':^46}\n"
            f"{hdr}\n"
            f"{'GPU':28}│ {gpu.name}\n"
            f"{'Model':28}│ {llm.name}\n"
            f"{'Export rate (s)':28}│ {cfg.export_rate:,.1f}\n"
            f"{'KV cache enabled':28}│ {cfg.kv_cache}\n"
            f"{'Prefix cache':28}│ {cfg.cache.action} | ≥{cfg.cache.min_len}t | {cfg.cache.scope}\n"
            f"{'Prefill time':28}│ {self.sum_prefill:>9,.1f}s ({self.sum_prefill / 3600:>6.2f} h)\n"
            f"{'Decode time':28}│ {self.sum_decode:>9,.1f}s ({self.sum_decode / 3600:>6.2f} h)\n"
            f"{'Total time':28}│ {total_s:>9,.1f} s  ({total_s / 3600:>6.2f} h)\n"
            f"{'p95 latency':28}│ {p95_lat:>9,.0f} ms\n"
            f"{'Cache hit ratio':28}│ {cache.hits / n_req:>9.2%}\n"
            f"{'LRU evictions':28}│ {cache.evictions:>9}\n"
            f"{hdr}\n"
        )

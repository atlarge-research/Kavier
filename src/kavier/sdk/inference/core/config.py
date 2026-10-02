"""Immutable simulation configuration dataclasses (KV cache, prefix cache, export rate)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CacheAction(StrEnum):
    """What a prefix-cache hit skips. Members compare equal to their string values."""

    NONE = "none"
    PREFILL = "prefill"
    FULL = "full"


class CacheScope(StrEnum):
    """Whether the prefix-cache key includes the session id."""

    SESSION = "session"
    GLOBAL = "global"


@dataclass(frozen=True)
class CacheCfg:
    """Prefix-cache settings: minimum token length, hit action, key scope, and LRU cap."""

    min_len: int = 1024
    action: CacheAction = CacheAction.PREFILL
    scope: CacheScope = CacheScope.SESSION
    max_entries: int = 10_000


@dataclass(frozen=True)
class SimConfig:
    """Simulation config; export_rate in seconds."""

    export_rate: float = 0.1
    kv_cache: bool = True
    cache: CacheCfg = CacheCfg()

    @staticmethod
    def from_cli(args) -> "SimConfig":
        return SimConfig(
            export_rate=args.export_rate,
            kv_cache=(args.kv_cache == "on"),
            cache=CacheCfg(
                min_len=args.prefix_cache_min_tokens,
                # argparse choices restrict these to valid members.
                action=CacheAction(args.prefix_cache_policy),
                scope=CacheScope(args.cache_scope),
                max_entries=args.max_cached_prompts,
            ),
        )

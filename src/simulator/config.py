from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CacheCfg:
    min_len: int = 1024
    action: Literal["none", "prefill", "full"] = "prefill"
    scope: Literal["session", "global"] = "session"
    max_entries: int = 10_000


@dataclass(frozen=True)
class SimConfig:
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
                action=args.prefix_cache_policy,
                scope=args.cache_scope,
                max_entries=args.max_cached_prompts,
            ),
        )

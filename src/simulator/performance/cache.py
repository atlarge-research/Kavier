from typing import Any, Tuple

from cachetools import LRUCache

from simulator.config import CacheCfg


class PrefixCache:
    def __init__(self, cfg: CacheCfg):
        self.cfg = cfg
        self._store: LRUCache[Tuple[Any, Tuple[int, ...]], None] = LRUCache(maxsize=cfg.max_entries)

        self.hits: int = 0
        self.evictions: int = 0

    def _evict(self) -> None:
        self.evictions += 1

    def _key(self, sid, tokens):
        core = tuple(tokens[: self.cfg.min_len])
        return (sid, core) if self.cfg.scope == "session" else core

    def lookup(self, sid, tokens) -> bool:
        k = self._key(sid, tokens)
        if k in self._store:  # hit
            _ = self._store[k]
            self.hits += 1
            return True

        if len(self._store) >= self.cfg.max_entries:
            self._store.popitem()
            self._evict()

        self._store[k] = None

        return False

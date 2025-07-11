from simulator.config import CacheCfg
from simulator.performance.cache import PrefixCache


def make_cache(max_entries=3, scope="session"):
    return PrefixCache(CacheCfg(max_entries=max_entries, scope=scope, min_len=2))


def test_hit_and_miss():
    c = make_cache()
    assert not c.lookup("s1", [1, 2, 3])
    assert c.lookup("s1", [1, 2, 3])
    assert c.hits == 1 and c.evictions == 0


def test_evictions_lru_order():
    c = make_cache(max_entries=2)
    c.lookup("s", [1, 2])
    c.lookup("s", [2, 3])
    assert c.evictions == 0
    c.lookup("s", [3, 4])
    assert c.evictions == 1
    assert not c.lookup("s", [1, 2])

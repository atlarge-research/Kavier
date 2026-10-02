"""Tests for the LRU prefix cache (kavier.sdk.inference.core.cache.PrefixCache).

The cache holds M entries keyed on the first ``min_len`` prompt tokens, optionally per session id.
"""

from hypothesis import given
from hypothesis import strategies as st

from kavier.sdk.inference.core.cache import PrefixCache
from kavier.sdk.inference.core.config import CacheCfg


def make_cache(max_entries=3, scope="session", min_len=2):
    return PrefixCache(CacheCfg(max_entries=max_entries, scope=scope, min_len=min_len))


def test_first_lookup_misses_second_lookup_hits():
    # A new key misses and is inserted; the same lookup then hits.
    c = make_cache()
    assert c.lookup("s1", [1, 2, 3]) is False  # cold key -> miss
    assert c.lookup("s1", [1, 2, 3]) is True  # same key -> hit
    assert c.hits == 1  # the miss is not counted
    assert c.evictions == 0  # capacity 3, only one distinct key -> nothing evicted


def test_key_uses_only_first_min_len_tokens():
    # Key is tuple(tokens[:min_len]). With min_len=2, [1,2,3] and [1,2,99] share prefix (1,2).
    c = make_cache(min_len=2)
    assert c.lookup("s", [1, 2, 3]) is False  # inserts prefix (1,2)
    assert c.lookup("s", [1, 2, 99]) is True  # same first-2 tokens -> prefix hit
    assert c.lookup("s", [1, 7, 3]) is False  # prefix (1,7) differs -> miss


def test_session_scope_namespaces_by_session_id():
    # scope="session" adds the sid to the key; equal tokens in another session miss.
    c = make_cache(scope="session")
    assert c.lookup("sessionA", [1, 2]) is False
    assert c.lookup("sessionB", [1, 2]) is False  # different session -> not a hit
    assert c.hits == 0


def test_global_scope_ignores_session_id():
    # scope="global" leaves the sid out of the key; equal tokens in another session hit.
    c = make_cache(scope="global")
    assert c.lookup("sessionA", [1, 2]) is False  # cold -> miss, inserts prefix (1,2)
    assert c.lookup("sessionB", [1, 2]) is True  # same tokens, sid ignored -> hit
    assert c.hits == 1


def test_eviction_removes_least_recently_used_entry():
    # Capacity 2. Insert (1,2) then (2,3); a third key evicts the oldest, (1,2).
    c = make_cache(max_entries=2)
    c.lookup("s", [1, 2])  # store LRU->MRU: (1,2)
    c.lookup("s", [2, 3])  # store LRU->MRU: (1,2),(2,3)
    assert c.evictions == 0  # still within capacity
    c.lookup("s", [3, 4])  # full -> evict LRU (1,2), insert (3,4)
    assert c.evictions == 1
    # Check the survivor first: a miss re-inserts and evicts.
    assert c.lookup("s", [2, 3]) is True  # (2,3) survived -> still a hit
    assert c.lookup("s", [1, 2]) is False  # (1,2) was the evicted one -> miss


def test_hit_refreshes_recency_and_protects_from_eviction():
    # A hit moves its key to most-recently-used.
    # Capacity 2: insert A,B -> LRU order A,B. Hit A -> order B,A. Insert C evicts B.
    c = make_cache(max_entries=2)
    c.lookup("s", [1, 1])  # A
    c.lookup("s", [2, 2])  # B ; order A,B
    assert c.lookup("s", [1, 1]) is True  # hit A -> A becomes MRU ; order B,A
    c.lookup("s", [3, 3])  # C evicts LRU == B
    assert c.evictions == 1
    # Check the survivor first: a miss re-inserts and evicts.
    assert c.lookup("s", [1, 1]) is True  # A survived because of the hit
    assert c.lookup("s", [2, 2]) is False  # B was evicted despite being inserted after A


@given(
    n_keys=st.integers(min_value=1, max_value=40),
    capacity=st.integers(min_value=1, max_value=20),
)
def test_distinct_inserts_evict_count_equals_overflow(n_keys, capacity):
    # n_keys distinct keys into a capacity-M cache: max(0, n_keys - M) evictions, zero hits.
    c = make_cache(max_entries=capacity, min_len=2)
    for i in range(n_keys):
        assert c.lookup("s", [i, i]) is False  # each prefix (i,i) is unique -> always a miss
    assert c.hits == 0
    assert c.evictions == max(0, n_keys - capacity)
    assert len(c._store) == min(n_keys, capacity)

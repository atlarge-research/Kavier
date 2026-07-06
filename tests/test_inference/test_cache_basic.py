"""Behavioral tests for the LRU prefix cache (kavier.sdk.inference.core.cache.PrefixCache).

Oracles are derived independently from the documented contract: an LRU cache of capacity
M keyed on the first ``min_len`` prompt tokens (optionally namespaced by session id). None of
the expected values are copied from PrefixCache's own output.
"""

from hypothesis import given
from hypothesis import strategies as st

from kavier.sdk.inference.core.cache import PrefixCache
from kavier.sdk.inference.core.config import CacheCfg


def make_cache(max_entries=3, scope="session", min_len=2):
    return PrefixCache(CacheCfg(max_entries=max_entries, scope=scope, min_len=min_len))


def test_first_lookup_misses_second_lookup_hits():
    # Contract: a never-seen key is a miss (returns False, gets inserted); the identical
    # follow-up lookup is a hit (True). Oracle: exactly one hit, zero evictions occurred.
    c = make_cache()
    assert c.lookup("s1", [1, 2, 3]) is False  # cold key -> miss
    assert c.lookup("s1", [1, 2, 3]) is True  # same key -> hit
    assert c.hits == 1  # exactly one hit was recorded (the miss must not count)
    assert c.evictions == 0  # capacity 3, only one distinct key -> nothing evicted


def test_key_uses_only_first_min_len_tokens():
    # Contract: the key is tuple(tokens[:min_len]). With min_len=2, [1,2,3] and [1,2,99]
    # collapse to the same prefix (1,2); [1,7,...] is a different prefix.
    c = make_cache(min_len=2)
    assert c.lookup("s", [1, 2, 3]) is False  # inserts prefix (1,2)
    assert c.lookup("s", [1, 2, 99]) is True  # same first-2 tokens -> prefix hit
    assert c.lookup("s", [1, 7, 3]) is False  # prefix (1,7) differs -> miss


def test_session_scope_namespaces_by_session_id():
    # Contract: scope="session" folds the sid into the key, so identical tokens under a
    # different session are a distinct key -> miss. Oracle: no hit despite equal tokens.
    c = make_cache(scope="session")
    assert c.lookup("sessionA", [1, 2]) is False
    assert c.lookup("sessionB", [1, 2]) is False  # different session -> not a hit
    assert c.hits == 0


def test_global_scope_ignores_session_id():
    # Contract: scope="global" drops the sid from the key, so identical tokens under a
    # different session collide -> hit. This is the direct complement of session scope.
    c = make_cache(scope="global")
    assert c.lookup("sessionA", [1, 2]) is False  # cold -> miss, inserts prefix (1,2)
    assert c.lookup("sessionB", [1, 2]) is True  # same tokens, sid ignored -> hit
    assert c.hits == 1


def test_eviction_removes_least_recently_used_entry():
    # Capacity 2. Insert (1,2) then (2,3); the 3rd distinct key must evict the LRU entry.
    # Oracle (pure LRU accounting): (1,2) is oldest, so it is the one evicted; (2,3) survives.
    c = make_cache(max_entries=2)
    c.lookup("s", [1, 2])  # store LRU->MRU: (1,2)
    c.lookup("s", [2, 3])  # store LRU->MRU: (1,2),(2,3)
    assert c.evictions == 0  # still within capacity
    c.lookup("s", [3, 4])  # full -> evict LRU (1,2), insert (3,4)
    assert c.evictions == 1
    # Assert survivor before evicted: a miss re-inserts and would evict, perturbing state.
    assert c.lookup("s", [2, 3]) is True  # (2,3) survived -> still a hit
    assert c.lookup("s", [1, 2]) is False  # (1,2) was the evicted one -> miss


def test_hit_refreshes_recency_and_protects_from_eviction():
    # A hit must move its key to most-recently-used, changing which key dies next.
    # Capacity 2: insert A,B -> LRU order A,B. Hit A -> order becomes B,A. Insert C evicts B.
    # Oracle: without recency-refresh A (oldest) would die; the refresh makes B die instead.
    c = make_cache(max_entries=2)
    c.lookup("s", [1, 1])  # A
    c.lookup("s", [2, 2])  # B ; order A,B
    assert c.lookup("s", [1, 1]) is True  # hit A -> A becomes MRU ; order B,A
    c.lookup("s", [3, 3])  # C evicts LRU == B
    assert c.evictions == 1
    # Assert survivor before evicted (a miss would re-insert and evict, perturbing state).
    assert c.lookup("s", [1, 1]) is True  # A survived thanks to the recency refresh
    assert c.lookup("s", [2, 2]) is False  # B was evicted despite being inserted after A


@given(
    n_keys=st.integers(min_value=1, max_value=40),
    capacity=st.integers(min_value=1, max_value=20),
)
def test_distinct_inserts_evict_count_equals_overflow(n_keys, capacity):
    # Property: inserting n_keys *distinct*, never-before-seen keys into a capacity-M cache
    # produces max(0, n_keys - M) evictions and zero hits (every lookup is a cold miss).
    # Oracle is pure cache arithmetic, independent of the implementation's branch condition.
    c = make_cache(max_entries=capacity, min_len=2)
    for i in range(n_keys):
        assert c.lookup("s", [i, i]) is False  # each prefix (i,i) is unique -> always a miss
    assert c.hits == 0
    assert c.evictions == max(0, n_keys - capacity)
    assert len(c._store) == min(n_keys, capacity)  # cache never exceeds capacity

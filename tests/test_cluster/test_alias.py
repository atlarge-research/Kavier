"""``kavier.cluster`` is a convenience alias for the ``kavier.sdk.cluster`` package (not a copy).

Mirrors the inference/training alias contract: a typo in ``_LAZY_ALIASES`` (pointing elsewhere, or a
missing entry) breaks the ``is`` identity below.
"""

from __future__ import annotations


def test_cluster_alias_is_the_sdk_package() -> None:
    import kavier
    import kavier.sdk.cluster

    assert kavier.cluster is kavier.sdk.cluster
    assert callable(kavier.cluster.schedule)

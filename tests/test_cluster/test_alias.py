"""``kavier.cluster`` is an alias of the ``kavier.sdk.cluster`` package and the same module object.

A wrong or missing entry in ``_LAZY_ALIASES`` breaks the ``is`` identity below.
"""

from __future__ import annotations


def test_cluster_alias_is_the_sdk_package() -> None:
    import kavier
    import kavier.sdk.cluster

    assert kavier.cluster is kavier.sdk.cluster
    assert callable(kavier.cluster.schedule)

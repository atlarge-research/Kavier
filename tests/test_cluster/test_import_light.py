"""A bare ``import kavier.sdk.cluster`` does not import heavy dependencies.

As in the training package, ``__init__`` re-exports the verb lazily through PEP 562 ``__getattr__``.
Each check runs in a fresh interpreter so pandas imported by other tests cannot hide a failure.
"""

from __future__ import annotations

import subprocess
import sys

_HEAVY = "{'pandas', 'numpy', 'scipy', 'sklearn', 'matplotlib'}"


def test_bare_import_does_not_import_heavy_deps() -> None:
    code = (
        "import sys\n"
        "import kavier.sdk.cluster\n"
        f"heavy = sorted(m for m in sys.modules if m.split('.')[0] in {_HEAVY})\n"
        "assert not heavy, heavy\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_importing_plot_timeline_does_not_import_matplotlib() -> None:
    # plot_timeline imports matplotlib inside the function body, so importing the name loads neither
    # matplotlib nor numpy.
    code = (
        "import sys\n"
        "from kavier.sdk.cluster import plot_timeline\n"
        f"heavy = sorted(m for m in sys.modules if m.split('.')[0] in {_HEAVY})\n"
        "assert not heavy, heavy\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

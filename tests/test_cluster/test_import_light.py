"""Import-light contract: a bare ``import kavier.sdk.cluster`` must not pull heavy deps.

Mirrors the training package's contract — the lazy ``__init__`` re-exports the verb via PEP-562
``__getattr__`` so importing the package (or a sibling that transitively imports it) stays cheap and
stdlib-light. Run in a fresh interpreter so it is not masked by pandas already imported elsewhere.
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
    # plot_timeline imports matplotlib lazily INSIDE the function, so referencing the symbol must not
    # pull matplotlib (nor its numpy) until the function is actually called.
    code = (
        "import sys\n"
        "from kavier.sdk.cluster import plot_timeline\n"
        f"heavy = sorted(m for m in sys.modules if m.split('.')[0] in {_HEAVY})\n"
        "assert not heavy, heavy\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

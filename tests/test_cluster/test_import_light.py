"""Import-light contract: a bare ``import kavier.sdk.cluster`` must not pull heavy deps.

Mirrors the training package's contract — the lazy ``__init__`` re-exports the verb via PEP-562
``__getattr__`` so importing the package (or a sibling that transitively imports it) stays cheap and
stdlib-light. Run in a fresh interpreter so it is not masked by pandas already imported elsewhere.
"""

from __future__ import annotations

import subprocess
import sys


def test_bare_import_does_not_import_pandas() -> None:
    code = (
        "import sys\n"
        "import kavier.sdk.cluster\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in {'pandas', 'numpy', 'scipy', 'sklearn'})\n"
        "assert not heavy, heavy\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

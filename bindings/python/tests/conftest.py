"""pytest bootstrap: make the ``kernel_set`` package importable without an
editable install, so the suite runs from the repo root via

    python3 -m pytest bindings/python/tests/... -q

The package lives in the parent directory of this ``tests/`` folder.
"""

import os
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

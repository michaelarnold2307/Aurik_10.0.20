"""Legacy launcher compatibility wrapper.

Keeps existing user/developer docs and scripts working while routing all GUI starts
through the canonical Aurik10.main entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Aurik10.main import main

if __name__ == "__main__":
    raise SystemExit(main())

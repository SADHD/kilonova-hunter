"""Put this folder and the repository root on sys.path.

Run the checks from the repository root:

    uv run python checks/mosfit/band_sidebyside.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def install() -> None:
    """Make ``sidebyside_lib`` and ``kilonova_hunter`` importable."""
    for path in (HERE, ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

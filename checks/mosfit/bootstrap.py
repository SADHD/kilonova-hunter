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


def load_waxman_l_int(path: Path | None = None) -> tuple[list[float], list[float]]:
    """Table B2 L_int from the committed CSV.

    Comment lines and the ``time_days`` header are skipped. ``numpy.genfromtxt``
    with ``names=True`` treats the first comment as the header, so this
    reader does not use it.
    """
    source = (
        ROOT / "docs" / "at2017gfo_bolometric_waxman2018.csv" if path is None else path
    )
    days: list[float] = []
    luminosities: list[float] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("time_days"):
            continue
        day_text, lum_text = stripped.split(",")
        days.append(float(day_text))
        luminosities.append(float(lum_text))
    return days, luminosities

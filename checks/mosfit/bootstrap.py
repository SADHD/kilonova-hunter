"""Put this folder and the repository root on sys.path.

Run the checks from the repository root:

    uv run python checks/mosfit/band_sidebyside.py
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# sncosmo reads this before it fetches a bandpass. The files under it are
# the CSP, SDSS, DES, PS1, and LSST curves these checks evaluate.
SNCOSMO_DATA = HERE / "sncosmo_bandpasses"


def install() -> None:
    """Make ``sidebyside_lib`` and ``kilonova_hunter`` importable.

    Also point sncosmo at the bandpasses committed in this folder, and
    send library logs and warnings to stderr. The result table stays on
    stdout.
    """
    for path in (HERE, ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    os.environ["SNCOSMO_DATA_DIR"] = str(SNCOSMO_DATA)
    _library_output_to_stderr()


def _library_output_to_stderr() -> None:
    """redback, bilby, and Python warnings write to stderr, not the table."""
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s : %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    def showwarning(message, category, filename, lineno, file=None, line=None):
        sys.stderr.write(
            warnings.formatwarning(message, category, filename, lineno, line)
        )

    warnings.showwarning = showwarning


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

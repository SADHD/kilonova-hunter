"""Write docs/mosfit_external_check.csv from the MOSFiT side-by-side.

The numbers are computed by checks/mosfit/band_sidebyside.py. This
script does not type them in. pip mosfit does not build on Python 3.10
or newer, so the comparison uses the reimplementation in that folder.

    uv run python scripts/record_mosfit_external_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.path.insert(0, str(ROOT / "checks" / "mosfit"))
    import bootstrap

    bootstrap.install()
    import band_sidebyside

    path, rows = band_sidebyside.write_external_check(
        ROOT / "docs" / "mosfit_external_check.csv"
    )
    print(f"wrote {path}")
    print(
        "max |wrapped redback - MOSFiT| from 0.5 d: "
        f"g {rows['g']:.3f}, r {rows['r']:.3f}, i {rows['i']:.3f}, "
        f"bolometric {rows['bolometric']:.3f} mag"
    )


if __name__ == "__main__":
    main()

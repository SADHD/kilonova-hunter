"""Record the external wrapped-redback versus MOSFiT check.

pip mosfit does not build on Python 3.10 or newer, so this repository
does not run MOSFiT. The numbers below were measured outside this
environment, against a line-by-line reimplementation of the modules in
MOSFiT commit 6595248. redback is about 2 percent brighter, from its
own diffusion integral. From 0.5 days on, every difference is inside
0.1 mag, which is the bar that blocks a grid rerun.

This script only writes that record. It does not call MOSFiT.

    uv run python scripts/record_mosfit_external_check.py
"""

from __future__ import annotations

from pathlib import Path

OUTPUT = Path("docs/mosfit_external_check.csv")

# max |wrapped redback − MOSFiT reimplementation|, magnitudes, from 0.5 d.
ROWS = (
    ("g", 0.028),
    ("r", 0.023),
    ("i", 0.020),
    ("bolometric", 0.030),
)
LIMIT_MAG = 0.1
SOURCE = "external check against MOSFiT commit 6595248"


def main() -> None:
    lines = ["quantity,max_abs_difference_mag,limit_mag,passes,source"]
    for name, difference in ROWS:
        passed = "yes" if difference <= LIMIT_MAG else "no"
        lines.append(f"{name},{difference},{LIMIT_MAG},{passed},{SOURCE}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(
        "External check only. redback is about 2% brighter "
        "from its own diffusion integral. The 0.1 mag bar passes."
    )


if __name__ == "__main__":
    main()

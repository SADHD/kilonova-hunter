"""Rerun every checks/mosfit script and compare its stdout with the committed .out.

Library logs and warnings go to stderr and are not part of the comparison.
The bandpasses are in the repository, so sncosmo does not download them.
A number may differ by one unit in the last printed decimal of the
reference value. Numbers with no printed fraction use an absolute
tolerance of 1e-3. Any other stdout difference fails.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "checks" / "mosfit"
# Used when the reference token has no digits after the decimal point.
_ABSOLUTE_FALLBACK = 1e-3
_NUMBER = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_EXPECTED_SCRIPTS = (
    "band_sidebyside.py",
    "dense_test.py",
    "gr_gap.py",
    "harness_sparse_check.py",
    "late_blue.py",
    "mosfit_grid.py",
    "reproduce_pr_table.py",
    "sidebyside.py",
    "temperature.py",
)


def _printed_unit(reference: str) -> float:
    """One unit in the last printed decimal, or 1e-3 when there is none."""
    token = reference.strip()
    if token[:1] in "+-":
        token = token[1:]
    mantissa, separator, exponent = token.lower().partition("e")
    power = int(exponent) if separator else 0
    if "." in mantissa:
        fraction = mantissa.split(".", 1)[1]
        if fraction:
            return 10 ** (-len(fraction) + power)
    return _ABSOLUTE_FALLBACK


def _same_number(found: str, expected: str) -> bool:
    if _integer(found) and _integer(expected):
        return found.lstrip("+") == expected.lstrip("+")
    left = float(found)
    right = float(expected)
    return abs(left - right) <= _printed_unit(expected)


def _integer(token: str) -> bool:
    body = token[1:] if token[:1] in "+-" else token
    return body.isdigit()


def _line_differs(found: str, expected: str) -> str | None:
    found_numbers = list(_NUMBER.finditer(found))
    expected_numbers = list(_NUMBER.finditer(expected))
    if [item.group() for item in found_numbers] == [
        item.group() for item in expected_numbers
    ] and _NUMBER.sub("#", found) == _NUMBER.sub("#", expected):
        return None
    if len(found_numbers) != len(expected_numbers):
        return "different number of values"
    if _NUMBER.sub("#", found) != _NUMBER.sub("#", expected):
        return "text differs"
    for new, old in zip(found_numbers, expected_numbers, strict=True):
        if not _same_number(new.group(), old.group()):
            return f"{new.group()} vs {old.group()}"
    return None


def compare(found: str, expected: str) -> list[str]:
    """Human-readable mismatches. Empty when the texts agree."""
    got = found.splitlines()
    want = expected.splitlines()
    problems = []
    if len(got) != len(want):
        problems.append(f"line count {len(got)} vs {len(want)}")
    for index, (new, old) in enumerate(zip(got, want, strict=False), start=1):
        reason = _line_differs(new, old)
        if reason is None:
            continue
        problems.append(f"line {index}: {reason}\n  new: {new}\n  old: {old}")
        if len(problems) >= 8:
            problems.append("further differences omitted")
            break
    if len(got) != len(want) and len(problems) < 8:
        extra = got[len(want) :] or want[len(got) :]
        for line in extra[:3]:
            problems.append(f"unpaired: {line}")
    return problems


def _scripts() -> list[Path]:
    """The nine committed checks. Each one has to have its .out file."""
    missing = [
        name
        for name in _EXPECTED_SCRIPTS
        if not (CHECKS / name).is_file()
        or not (CHECKS / name).with_suffix(".out").is_file()
    ]
    assert not missing, "checks/mosfit is missing a script or its .out: " + ", ".join(
        missing
    )
    return [CHECKS / name for name in _EXPECTED_SCRIPTS]


def main() -> int:
    failed = False
    for script in _scripts():
        expected_path = script.with_suffix(".out")
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            print(f"{script.name} exited {completed.returncode}", file=sys.stderr)
            print(completed.stdout[-2000:], file=sys.stderr)
            print(completed.stderr[-2000:], file=sys.stderr)
            failed = True
            continue
        problems = compare(completed.stdout, expected_path.read_text(encoding="utf-8"))
        if problems:
            print(
                f"{script.name} does not match {expected_path.name}:",
                file=sys.stderr,
            )
            print("\n".join(problems), file=sys.stderr)
            failed = True
        else:
            print(f"{script.name} matches {expected_path.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

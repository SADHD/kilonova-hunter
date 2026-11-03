"""Rerun every checks/mosfit script and compare it with the committed .out.

Log lines from redback and bilby carry a timestamp, so those lines are
dropped. A fresh machine also prints one line per sncosmo bandpass it
downloads. Numbers may differ by float noise. Anything else has to match.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "checks" / "mosfit"
# A last-digit change in a value printed to about three significant
# figures. Integers, which are counts, are compared exactly.
RELATIVE_TOLERANCE = 1e-3
ABSOLUTE_TOLERANCE = 1e-4
_NUMBER = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?")
_LOG_MARKERS = ("redback WARNING", "redback INFO", "bilby INFO")


def _scientific_lines(text: str) -> list[str]:
    kept = []
    for line in text.splitlines():
        if any(marker in line for marker in _LOG_MARKERS):
            continue
        # sncosmo prints this only when the bandpass is not cached yet.
        if line.strip().startswith("Downloading http"):
            continue
        kept.append(line)
    return kept


def _same_number(found: str, expected: str) -> bool:
    if _integer(found) and _integer(expected):
        return found.lstrip("+") == expected.lstrip("+")
    left = float(found)
    right = float(expected)
    return abs(left - right) <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(right)


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
    got = _scientific_lines(found)
    want = _scientific_lines(expected)
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
    return sorted(
        path for path in CHECKS.glob("*.py") if path.with_suffix(".out").is_file()
    )


def main() -> int:
    failed = False
    for script in _scripts():
        expected_path = script.with_suffix(".out")
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            print(f"{script.name} exited {completed.returncode}", file=sys.stderr)
            print(completed.stdout[-2000:], file=sys.stderr)
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

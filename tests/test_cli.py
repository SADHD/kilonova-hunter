"""Tests for the kilohunt command."""

import shutil
import subprocess
from pathlib import Path


def _kilohunt() -> str:
    """Return the kilohunt executable installed in this environment."""
    found = shutil.which("kilohunt")
    assert found is not None, (
        "kilohunt is not on PATH. Install the project with `uv sync`."
    )
    return found


def test_help_runs() -> None:
    """`kilohunt --help` exits successfully and mentions the demo command."""
    result = subprocess.run(
        [_kilohunt(), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "demo" in result.stdout
    assert "kilonova" in result.stdout.lower()


def test_demo_writes_a_plot(tmp_path: Path) -> None:
    """`kilohunt demo` saves the made-up light curve where --output points."""
    destination = tmp_path / "demo.png"
    result = subprocess.run(
        [_kilohunt(), "demo", "--output", str(destination), "--seed", "42"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert destination.is_file()
    assert destination.stat().st_size > 0

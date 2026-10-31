"""Tests for the kilohunt command.

These call ``main`` directly, which is the function the ``kilohunt`` command
runs. The test then does not need the shell to find a ``kilohunt`` program
on ``PATH``.
"""

from pathlib import Path

import pytest

from kilonova_hunter.cli import main


def test_help_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` exits successfully and mentions the demo command.

    argparse prints the help and then exits the program. From inside a test
    that exit arrives as ``SystemExit``. A code of 0 means "this was help,
    not a failure." ``capsys`` is pytest's copy of what was printed.
    """
    with pytest.raises(SystemExit) as caught:
        main(["--help"])

    assert caught.value.code == 0
    printed = capsys.readouterr().out
    assert "demo" in printed
    assert "kilonova" in printed.lower()


def test_demo_writes_a_plot(tmp_path: Path) -> None:
    """``demo`` saves the made-up light curve where ``--output`` points."""
    destination = tmp_path / "demo.png"
    main(["demo", "--output", str(destination), "--seed", "42"])
    assert destination.is_file()
    assert destination.read_bytes().startswith(b"\x89PNG")

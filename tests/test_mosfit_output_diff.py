"""The MOSFiT output check ignores log timestamps and float noise."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _compare():
    path = Path(__file__).resolve().parents[1] / "scripts" / "diff_mosfit_outputs.py"
    spec = importlib.util.spec_from_file_location("diff_mosfit_outputs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.compare


def test_float_noise_and_log_lines_are_not_failures() -> None:
    compare = _compare()
    found = (
        "11:01 redback WARNING : missing\n"
        "Downloading https://sncosmo.github.io/data/bandpasses/csp/g.dat [Done]\n"
        "value +0.423\n"
    )
    expected = "09:00 redback INFO    : ready\nvalue +0.4231\n"
    assert compare(found, expected) == []


def test_a_real_number_change_or_a_count_change_fails() -> None:
    compare = _compare()
    assert compare("gap +0.41\n", "gap +0.42\n")
    assert compare("n_g 25\n", "n_g 26\n")
    assert compare("label changed\n", "label original\n")

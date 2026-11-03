"""The MOSFiT output check allows float noise and nothing else on stdout."""

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


def test_float_noise_is_allowed_and_extra_stdout_is_not() -> None:
    compare = _compare()
    assert compare("value +0.423\n", "value +0.4231\n") == []
    download = (
        "Downloading https://sncosmo.github.io/data/bandpasses/csp/"
        "i_tel_ccd_atm_ext_1.2.dat [Done]\n"
        "value +0.423\n"
    )
    assert compare(download, "value +0.423\n")
    assert compare("redback WARNING : missing\nvalue +0.423\n", "value +0.423\n")


def test_a_real_number_change_or_a_count_change_fails() -> None:
    compare = _compare()
    assert compare("gap +0.41\n", "gap +0.42\n")
    assert compare("n_g 25\n", "n_g 26\n")
    assert compare("label changed\n", "label original\n")

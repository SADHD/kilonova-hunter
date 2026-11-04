"""The wrapped model stays within 0.1 mag of the MOSFiT reimplementation."""

import pytest

pytestmark = pytest.mark.slow
pytest.importorskip("redback")
pytest.importorskip("sncosmo")


def test_side_by_side_bands_stay_within_a_tenth_of_a_magnitude() -> None:
    """From 0.5 days, g, r, and i each stay within 0.1 mag.

    The measured maxima are 0.028, 0.023, and 0.020, and the bolometric
    epoch gap is 0.030, each within 0.005 mag.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "checks" / "mosfit"))
    import bootstrap

    bootstrap.install()
    import band_sidebyside

    rows = band_sidebyside.max_abs_differences()
    assert rows["g"] <= 0.1
    assert rows["r"] <= 0.1
    assert rows["i"] <= 0.1
    assert rows["g"] == pytest.approx(0.028, abs=0.005)
    assert rows["r"] == pytest.approx(0.023, abs=0.005)
    assert rows["i"] == pytest.approx(0.020, abs=0.005)
    assert rows["bolometric"] == pytest.approx(0.030, abs=0.005)

"""Sky cuts, the 0.86 dust scale, Planck18, and the ZTF night split."""

import numpy as np
import pandas as pd
import pytest

from kilonova_hunter.alerts import (
    average_flags_per_night,
    split_nights,
    ztf_night,
)
from kilonova_hunter.sky import (
    SFD_RECALIBRATION,
    draw_footprint_positions,
    ebv_from_sfd,
    redshift_at_luminosity_distance,
)


def test_footprint_positions_stay_in_the_strip_and_off_the_plane() -> None:
    """Declination is between -70 and +12, and |b| is above 15 degrees."""
    ra, dec, galactic_b = draw_footprint_positions(np.random.default_rng(7), 80)
    assert len(ra) == 80
    assert np.all(dec > -70.0)
    assert np.all(dec < 12.0)
    assert np.all(np.abs(galactic_b) > 15.0)


def test_sfd_ebv_is_scaled_by_0_86() -> None:
    """The raw map value is multiplied by the Schlafly and Finkbeiner factor."""

    def query(_coordinates):
        return np.array([1.0, 0.5])

    ebv = ebv_from_sfd(np.array([10.0, 20.0]), np.array([0.0, 1.0]), query=query)
    assert ebv[0] == pytest.approx(SFD_RECALIBRATION)
    assert ebv[1] == pytest.approx(0.5 * SFD_RECALIBRATION)


def test_planck18_redshift_at_40_mpc_is_near_0_009() -> None:
    """40 Mpc is the AT2017gfo distance. The redshift is about 0.009.

    ``astropy.cosmology.Planck18`` needs scipy to turn a distance into
    a redshift. The default ``uv sync`` used by CI does not install the
    sim extra, so this test skips there. ``uv sync --extra sim`` runs it.
    """
    pytest.importorskip("scipy")
    redshift = redshift_at_luminosity_distance(40.0)
    assert 0.008 < redshift < 0.010


def test_fitzpatrick_dims_g_more_than_r(monkeypatch: pytest.MonkeyPatch) -> None:
    """R_V = 3.1, and g is extinguished more than r at the same E(B-V)."""
    pytest.importorskip("extinction")
    from kilonova_hunter.sky import fitzpatrick_extinction_mag

    assert fitzpatrick_extinction_mag(0.0, "g") == 0.0
    g = fitzpatrick_extinction_mag(0.1, "g")
    r = fitzpatrick_extinction_mag(0.1, "r")
    i = fitzpatrick_extinction_mag(0.1, "i")
    assert g > r > i > 0.0


def test_ztf_night_changes_at_20_utc_and_slices_do_not_share_nights() -> None:
    """Palomar noon in standard time is 20:00 UTC, which is MJD fraction 20/24."""
    noon = 60000.0 + 20.0 / 24.0
    assert int(ztf_night(noon)) == 60000
    assert int(ztf_night(noon - 1e-6)) == 59999
    alerts = pd.DataFrame({"night": [5, 5, 6, 7, 8]})
    tuning, test = split_nights(alerts)
    assert set(tuning).isdisjoint(test)
    assert tuning == [5, 7]
    assert test == [6, 8]


def test_flags_per_night_counts_objects_not_rows() -> None:
    """Two alerts of one candidate on two nights are one flag, averaged over nights."""
    # A curve the placeholder cuts accept: the noise-free demo, placed
    # on two nights as if it were two alerts of one object. Easier to
    # build a tiny table the filter rejects, and one it accepts.
    from kilonova_hunter.inject import simulate_kilonova_light_curve

    demo = simulate_kilonova_light_curve(noise_mag=0.0)
    demo = demo.loc[demo["time_days"] <= 3.0]
    rows = []
    for point in demo.itertuples(index=False):
        band = {"g": 1, "r": 2}[point.filter]
        rows.append(
            {
                "i:objectId": "good",
                "i:jd": 2450000.0 + point.time_days,
                "i:fid": band,
                "i:magpsf": point.magnitude,
                "i:sigmapsf": 0.02,
                "mjd": 50000.0 + point.time_days,
                "night": 1,
            }
        )
    rows.append(
        {
            "i:objectId": "quiet",
            "i:jd": 2450001.0,
            "i:fid": 1,
            "i:magpsf": 20.0,
            "i:sigmapsf": 0.02,
            "mjd": 50001.0,
            "night": 2,
        }
    )
    summary = average_flags_per_night(pd.DataFrame(rows))
    assert summary["n_nights"] == 2
    assert summary["n_flagged"] == 1
    assert summary["flags_per_night"] == pytest.approx(0.5)

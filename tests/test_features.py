"""Tests for fade-rate and color measurements on one light curve.

A larger magnitude is fainter. A positive fade rate means the object is
getting fainter. g minus r below zero means g is the brighter band,
which is a blue color. A positive color rise means the object got redder.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from kilonova_hunter.config import THRESHOLDS
from kilonova_hunter.features import (
    fade_rates,
    g_minus_r_at_time,
    g_minus_r_over_time,
    g_minus_r_rise,
    peak_time_days,
)
from kilonova_hunter.inject import (
    G_FADE_PER_DAY,
    R_FADE_PER_DAY,
    simulate_kilonova_light_curve,
)


def _curve(
    rows: list[tuple[float, str, float]],
) -> pd.DataFrame:
    """Build a light curve from (time, filter, magnitude) triples."""
    return pd.DataFrame(
        {
            "time_days": [row[0] for row in rows],
            "filter": [row[1] for row in rows],
            "magnitude": [row[2] for row in rows],
        }
    )


def test_fade_fit_uses_only_the_points_after_the_peak() -> None:
    """A steep rise before the peak must not flatten the fade line.

    The magnitudes climb by 1.0 per day after day 1. The point at day 0
    is much fainter, which is the rise seen from before the peak. Fitting
    through that point as well would not return 1.0.
    """
    curve = _curve(
        [
            (0.0, "g", 25.0),
            (1.0, "g", 20.0),
            (2.0, "g", 21.0),
            (3.0, "g", 22.0),
        ]
    )
    g_fade = fade_rates(curve)["g"]
    assert g_fade.mag_per_day == pytest.approx(1.0)
    assert "not enough data" not in g_fade.explanation.lower()


def test_noise_free_demo_fade_matches_the_built_in_rates() -> None:
    """The straight-line fit recovers the slopes the demo was built with.

    Noise is left off, so this checks the fit rather than one random seed.
    """
    curve = simulate_kilonova_light_curve(noise_mag=0.0)
    rates = fade_rates(curve)
    assert rates["g"].mag_per_day == pytest.approx(G_FADE_PER_DAY)
    assert rates["r"].mag_per_day == pytest.approx(R_FADE_PER_DAY)


def test_duplicate_index_labels_still_find_one_brightest_row() -> None:
    """Two tables stuck together can reuse the same row labels.

    The brightest g point is day 1 (magnitude 20). Day 3 is fainter, but
    it carries the same label as day 1, because the two tables were
    concatenated without building a fresh index. A lookup by label would
    return both rows. The fade after day 1 is 1 magnitude per day.
    """
    early = pd.DataFrame(
        {
            "time_days": [0.0, 1.0],
            "filter": ["g", "g"],
            "magnitude": [21.0, 20.0],
        }
    )
    later = pd.DataFrame(
        {
            "time_days": [2.0, 3.0],
            "filter": ["g", "g"],
            "magnitude": [21.0, 22.0],
        }
    )
    curve = pd.concat([early, later])
    assert curve.index.duplicated().any()
    assert peak_time_days(curve, "g") == pytest.approx(1.0)
    assert fade_rates(curve)["g"].mag_per_day == pytest.approx(1.0)


def test_one_point_has_no_fade_rate() -> None:
    """A single detection cannot tell a fade from a flat line."""
    curve = _curve([(1.0, "r", 19.6)])
    r_fade = fade_rates(curve)["r"]
    assert r_fade.mag_per_day is None
    assert "not enough data" in r_fade.explanation.lower()


def test_points_at_one_time_have_no_slope() -> None:
    """Two visits at the same time are still one moment, so no slope."""
    curve = _curve(
        [
            (1.0, "r", 19.6),
            (1.0, "r", 19.7),
        ]
    )
    r_fade = fade_rates(curve)["r"]
    assert r_fade.mag_per_day is None
    assert "not enough data" in r_fade.explanation.lower()


def test_color_at_a_shared_time_is_g_minus_r() -> None:
    """When g and r land on the same time, the color is their difference."""
    curve = _curve(
        [
            (1.0, "g", 19.0),
            (1.0, "r", 20.0),
        ]
    )
    color = g_minus_r_at_time(curve, 1.0)
    assert color.g_minus_r == pytest.approx(-1.0)


def test_color_between_points_is_read_off_the_joining_line() -> None:
    """Halfway between two samples, each band is halfway in magnitude.

    g goes from 20 to 22, so at the midpoint it is 21. r goes from 19 to
    21, so at the midpoint it is 20. g minus r is 1.
    """
    curve = _curve(
        [
            (0.0, "g", 20.0),
            (2.0, "g", 22.0),
            (0.0, "r", 19.0),
            (2.0, "r", 21.0),
        ]
    )
    color = g_minus_r_at_time(curve, 1.0)
    assert color.g_minus_r == pytest.approx(1.0)


def test_color_outside_the_data_is_not_enough_data() -> None:
    """We do not guess a magnitude past the last point we actually have."""
    curve = _curve(
        [
            (1.0, "g", 19.5),
            (1.0, "r", 19.6),
        ]
    )
    color = g_minus_r_at_time(curve, 2.0)
    assert color.g_minus_r is None
    assert "not enough data" in color.explanation.lower()


def test_color_without_r_is_not_enough_data() -> None:
    """g minus r cannot be computed when r was never observed."""
    curve = _curve([(1.0, "g", 19.5), (2.0, "g", 20.5)])
    color = g_minus_r_at_time(curve, 1.5)
    assert color.g_minus_r is None
    assert "not enough data" in color.explanation.lower()


def test_missing_column_is_a_clear_error() -> None:
    """A table that is not a light curve says which column is missing."""
    not_a_curve = pd.DataFrame({"time_days": [1.0], "filter": ["g"]})
    with pytest.raises(ValueError, match="magnitude"):
        fade_rates(not_a_curve)


def test_color_over_time_stays_inside_the_data() -> None:
    """A color is read only where both bands have coverage.

    g is measured at day 0 and day 2. r is measured at day 1 and day 3.
    The overlap is day 1 to day 2. Day 0 would require guessing r before
    its first point, and day 3 would require guessing g after its last
    point. Neither is included. The magnitudes in between are read off
    the straight line that joins the neighbors.
    """
    curve = _curve(
        [
            (0.0, "g", 20.0),
            (2.0, "g", 22.0),
            (1.0, "r", 19.0),
            (3.0, "r", 21.0),
        ]
    )
    samples = g_minus_r_over_time(curve)
    assert [sample.time_days for sample in samples] == pytest.approx([1.0, 2.0])
    # At day 1, g is halfway from 20 to 22, so 21, and r is 19.
    assert samples[0].g_minus_r == pytest.approx(2.0)
    # At day 2, g is 22, and r is halfway from 19 to 21, so 20.
    assert samples[1].g_minus_r == pytest.approx(2.0)


def test_color_rise_stops_at_the_end_of_the_window() -> None:
    """The fit uses colors inside the window, not a later one.

    g minus r is 0.0 at day 0, 0.5 at day 2, and 1.0 at day 4. The
    default window is 2 days, so day 4 is left out. With two colors the
    straight line goes through both, and the rise is still 0.5. A
    shorter window, read from the config, leaves only the first color
    and the rise cannot be measured.
    """
    curve = _curve(
        [
            (0.0, "g", 20.0),
            (0.0, "r", 20.0),
            (2.0, "g", 20.5),
            (2.0, "r", 20.0),
            (4.0, "g", 21.0),
            (4.0, "r", 20.0),
        ]
    )
    change = g_minus_r_rise(curve)
    assert change.rise == pytest.approx(0.5)
    assert change.earlier_time_days == pytest.approx(0.0)
    assert change.later_time_days == pytest.approx(2.0)
    assert change.color_count == 3

    short_window = replace(THRESHOLDS, reddening_window_days=1.0)
    too_soon = g_minus_r_rise(curve, short_window)
    assert too_soon.rise is None
    assert too_soon.color_count == 3


def test_one_color_measurement_has_no_rise() -> None:
    """A single g point against a longer r curve is one color, not a rise.

    r continues after that point, but g does not, and we do not guess
    g past its only measurement.
    """
    curve = _curve(
        [
            (1.0, "g", 20.0),
            (1.0, "r", 19.5),
            (2.0, "r", 20.0),
            (3.0, "r", 20.5),
        ]
    )
    samples = g_minus_r_over_time(curve)
    assert len(samples) == 1
    assert samples[0].g_minus_r == pytest.approx(0.5)
    change = g_minus_r_rise(curve)
    assert change.rise is None
    assert change.color_count == 1
    assert "only once" in change.explanation.lower()


def test_one_noisy_first_color_does_not_flip_a_reddening_curve() -> None:
    """One bad first point no longer decides whether the object got redder.

    The true color climbs by 0.25 mag every half day, from 0.00 at day 0
    to 1.00 at day 2. The day-0 measurement is wrong: it says 1.20, which
    is redder than the end of the curve, and its g-band uncertainty is
    0.50 mag. Every later point has an uncertainty of 0.02 mag in each
    band.

    The old check subtracted the raw first color from the raw last color.
    That difference is 1.00 - 1.20 = -0.20, so a clearly reddening curve
    looked like it got bluer. The weighted line trusts the precise points
    (the weight is one over the uncertainty) and the rise stays near the
    true 1.00 mag, which is more than the placeholder cut of 0.10.
    """
    times = (0.0, 0.5, 1.0, 1.5, 2.0)
    # The first color is the bad one. The rest sit on a line of slope 0.5.
    colors = (1.2, 0.25, 0.50, 0.75, 1.00)
    g_errors = (0.50, 0.02, 0.02, 0.02, 0.02)
    rows: list[dict[str, float | str]] = []
    for time_days, color, g_error in zip(times, colors, g_errors, strict=True):
        rows.append(
            {
                "time_days": time_days,
                "filter": "r",
                "magnitude": 20.0,
                "magnitude_error": 0.02,
            }
        )
        rows.append(
            {
                "time_days": time_days,
                "filter": "g",
                "magnitude": 20.0 + color,
                "magnitude_error": g_error,
            }
        )
    curve = pd.DataFrame(rows)

    samples = g_minus_r_over_time(curve)
    assert samples[0].g_minus_r == pytest.approx(1.2)
    assert samples[-1].g_minus_r == pytest.approx(1.0)
    # The raw two-point rise the old code reported. Negative means bluer.
    raw_endpoint_rise = samples[-1].g_minus_r - samples[0].g_minus_r
    assert raw_endpoint_rise == pytest.approx(-0.2)
    # g and r are separate measurements, so their errors combine in
    # quadrature: the square root of the sum of the squares.
    assert samples[0].uncertainty == pytest.approx(float(np.hypot(0.50, 0.02)))

    change = g_minus_r_rise(curve)
    assert change.rise is not None
    assert change.rise == pytest.approx(0.99, abs=0.02)
    assert change.rise > THRESHOLDS.min_g_minus_r_rise
    assert "one over its uncertainty" in change.explanation


def test_color_uncertainty_between_points_mixes_the_neighbors() -> None:
    """A color read between visits inherits the neighbors' uncertainties.

    g is measured at day 0 and day 2, each with an error of 0.10. Halfway
    between them the magnitude is an equal mix of the two, so each error
    is halved before they are combined. r is measured at that halfway
    time with an error of 0.20, and the color error combines g and r.
    """
    curve = pd.DataFrame(
        {
            "time_days": [0.0, 2.0, 1.0],
            "filter": ["g", "g", "r"],
            "magnitude": [20.0, 22.0, 19.0],
            "magnitude_error": [0.10, 0.10, 0.20],
        }
    )
    samples = g_minus_r_over_time(curve)
    assert [sample.time_days for sample in samples] == pytest.approx([1.0])
    g_error = float(np.hypot(0.5 * 0.10, 0.5 * 0.10))
    assert samples[0].uncertainty == pytest.approx(float(np.hypot(g_error, 0.20)))


def test_nan_magnitude_is_skipped_instead_of_breaking_the_fit() -> None:
    """A blank magnitude is dropped, and the remaining line is still fit."""
    curve = _curve(
        [
            (1.0, "g", 20.0),
            (2.0, "g", np.nan),
            (3.0, "g", 22.0),
        ]
    )
    g_fade = fade_rates(curve)["g"]
    assert g_fade.mag_per_day == pytest.approx(1.0)

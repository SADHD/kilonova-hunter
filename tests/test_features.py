"""Tests for fade-rate and color measurements on one light curve.

A larger magnitude is fainter. A positive fade rate means the object is
getting fainter. g minus r below zero means g is the brighter band,
which is a blue color.
"""

import numpy as np
import pandas as pd
import pytest

from kilonova_hunter.features import fade_rates, g_minus_r_at_time, peak_time_days
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

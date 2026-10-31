"""Tests for the made-up kilonova light curve.

Magnitudes are backwards from everyday charts: a bigger number is fainter.
"Peaks then fades" means the magnitude drops to a minimum near day 1
(that minimum is the brightest moment) and climbs afterward.
"""

import numpy as np
import pandas as pd

from kilonova_hunter.filter import FADE_FASTER_THAN_MAG_PER_DAY
from kilonova_hunter.inject import (
    _FILTERS,
    PEAK_DAY,
    _brightness_without_noise,
    simulate_kilonova_light_curve,
)


def _brightest_time(group: pd.DataFrame) -> float:
    """Return the time of the smallest magnitude in one filter."""
    row = group.loc[group["magnitude"].idxmin()]
    return float(row["time_days"])


def _smooth_magnitudes(filter_name: str, times: np.ndarray) -> np.ndarray:
    """Return one filter's magnitudes with the random scatter left off."""
    params = _FILTERS[filter_name]
    return _brightness_without_noise(
        times,
        peak_mag=params["peak_mag"],
        rise_per_day=params["rise_per_day"],
        fade_per_day=params["fade_per_day"],
    )


def test_light_curve_has_both_filters() -> None:
    """The demo follows a blue filter (g) and a redder filter (r)."""
    curve = simulate_kilonova_light_curve()
    assert set(curve["filter"]) == {"g", "r"}
    assert len(curve) > 4


def test_noise_free_curve_peaks_at_one_day() -> None:
    """Without noise, both filters are brightest exactly one day after the explosion.

    The public light curve adds random scatter on top of this shape. Checking
    the smooth curve keeps this test about the shape we designed, not about
    one particular random seed.
    """
    # Same half-day spacing the demo uses, and long enough to see the week-long fade.
    times = np.arange(0.0, 10.0 + 1e-9, 0.5)
    for filter_name in _FILTERS:
        magnitudes = _smooth_magnitudes(filter_name, times)
        # A smaller magnitude is brighter, so the peak is the minimum.
        peak_time = float(times[int(np.argmin(magnitudes))])
        assert peak_time == PEAK_DAY, filter_name


def test_light_curve_peaks_then_fades() -> None:
    """Each noisy filter is fainter at the start and fainter a week later.

    The exact day of the brightest noisy point is allowed to slip by a sample.
    ``test_noise_free_curve_peaks_at_one_day`` is the one that pins day 1.
    """
    curve = simulate_kilonova_light_curve()
    for filter_name, group in curve.groupby("filter"):
        group = group.sort_values("time_days")
        peak_time = _brightest_time(group)
        brightest = float(group["magnitude"].min())
        first = float(group.iloc[0]["magnitude"])
        last = float(group.iloc[-1]["magnitude"])
        # The rise and the fade are both large compared with the 0.05 mag noise.
        assert first > brightest + 0.3, filter_name
        assert last > brightest + 0.5, filter_name

        # Still fading after the peak, not bouncing back up in brightness.
        later = group.loc[group["time_days"] >= peak_time, "magnitude"]
        assert float(later.iloc[-1]) > float(later.iloc[0])


def test_color_turns_redder_after_peak() -> None:
    """After the peak, g minus r grows on the smooth curve.

    A larger g minus r means the bluer filter has become faint compared
    with the redder one. The check walks every step after the peak with
    the noise left off, so it follows the fade rates themselves. It does
    not pin the day the color crosses zero: a faster r fade moves that
    crossing later, and one noisy seed could nudge a close pair of points
    the wrong way.
    """
    # From the peak through the rest of the demo, on the demo's half-day grid.
    times = np.arange(PEAK_DAY, 10.0 + 1e-9, 0.5)
    color = _smooth_magnitudes("g", times) - _smooth_magnitudes("r", times)
    # Each later sample is redder than the one just before it.
    assert np.all(np.diff(color) > 0.0)


def test_noise_free_r_fade_beats_the_filter_cut() -> None:
    """The smooth r curve fades faster than the v1 filter's cut.

    The rate is how far the magnitude climbs over the first few days after
    the peak, with the noise left off. A larger magnitude is fainter, so a
    bigger rate is a faster fade. The cut is the named constant the filter
    will use, so this test and that rule cannot drift apart.
    """
    # Three days is long enough to be "the first few days" and short enough
    # to stay in the stretch AT2017gfo was fast in r.
    times = np.arange(PEAK_DAY, PEAK_DAY + 3.0 + 1e-9, 0.5)
    magnitudes = _smooth_magnitudes("r", times)
    fade_per_day = float(magnitudes[-1] - magnitudes[0]) / float(times[-1] - times[0])
    assert fade_per_day > FADE_FASTER_THAN_MAG_PER_DAY, (
        f"r faded {fade_per_day:.3f} mag/day, "
        f"which is not faster than {FADE_FASTER_THAN_MAG_PER_DAY}"
    )


def test_same_seed_redraws_the_same_curve() -> None:
    """The noise is random, and a fixed seed makes it the same random every time."""
    first = simulate_kilonova_light_curve(seed=7)
    second = simulate_kilonova_light_curve(seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_changes_the_noise() -> None:
    """Two seeds are two different noisy realizations of the same shape."""
    first = simulate_kilonova_light_curve(seed=1)
    second = simulate_kilonova_light_curve(seed=2)
    assert not first["magnitude"].equals(second["magnitude"])

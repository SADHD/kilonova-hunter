"""Tests for the made-up kilonova light curve.

Magnitudes are backwards from everyday charts: a bigger number is fainter.
"Peaks then fades" means the magnitude drops to a minimum near day 1
(that minimum is the brightest moment) and climbs afterward.
"""

import numpy as np
import pandas as pd

from kilonova_hunter.inject import (
    _FILTERS,
    _brightness_without_noise,
    simulate_kilonova_light_curve,
)


def _brightest_time(group: pd.DataFrame) -> float:
    """Return the time of the smallest magnitude in one filter."""
    row = group.loc[group["magnitude"].idxmin()]
    return float(row["time_days"])


def test_light_curve_has_both_filters() -> None:
    """The demo follows a blue filter (g) and a redder filter (r)."""
    curve = simulate_kilonova_light_curve()
    assert set(curve["filter"]) == {"g", "r"}
    assert len(curve) > 4


def test_noise_free_curve_peaks_at_one_day() -> None:
    """Without noise, both filters are brightest exactly one day after the explosion.

    The public light curve adds random scatter on top of this shape. On the
    r filter the fade just after day 1 is small, only a little larger than
    that scatter, so the brightest noisy point sometimes lands at day 1.5
    and, less often, at day 2. Checking the smooth curve keeps this test
    about the shape we designed, not about one lucky random seed.
    """
    # Same half-day spacing the demo uses, and long enough to see the week-long fade.
    times = np.arange(0.0, 10.0 + 1e-9, 0.5)
    for filter_name, params in _FILTERS.items():
        magnitudes = _brightness_without_noise(
            times,
            peak_mag=params["peak_mag"],
            rise_per_day=params["rise_per_day"],
            fade_per_day=params["fade_per_day"],
        )
        # A smaller magnitude is brighter, so the peak is the minimum.
        peak_time = float(times[int(np.argmin(magnitudes))])
        assert peak_time == 1.0, filter_name


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


def test_color_turns_redder() -> None:
    """g minus r is larger at the end of the week than at the start.

    A growing g - r means the bluer filter has become faint compared with
    the redder one, which is the cooling the filter will eventually look for.
    """
    curve = simulate_kilonova_light_curve()

    def color_at(day: float) -> float:
        """Return g magnitude minus r magnitude at one timestamp."""
        magnitudes = {}
        for filter_name in ("g", "r"):
            group = curve[curve["filter"] == filter_name]
            row = group.loc[group["time_days"] == day].iloc[0]
            magnitudes[filter_name] = float(row["magnitude"])
        return magnitudes["g"] - magnitudes["r"]

    assert color_at(8.0) > color_at(1.0) > color_at(0.0)


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

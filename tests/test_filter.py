"""Tests for the two v1 kilonova rules.

The fade cut is "faster than 0.3 mag/day in any one of g, r, or i".
The color cut is a placeholder: g minus r greater than 0.0 after the
peak. Both numbers are read from ``config.py``.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from kilonova_hunter.config import THRESHOLDS
from kilonova_hunter.filter import (
    FADE_FASTER_THAN_MAG_PER_DAY,
    evaluate_color_rule,
    evaluate_fade_rule,
    is_kilonova_candidate,
)
from kilonova_hunter.inject import simulate_kilonova_light_curve


def _noise_free_demo() -> pd.DataFrame:
    """The teaching kilonova with the random scatter left off."""
    return simulate_kilonova_light_curve(noise_mag=0.0)


def _fading_curve(
    *,
    fade_per_day: float,
    g_minus_r: float,
    bands: tuple[str, ...] = ("g", "r"),
    peak_day: float = 1.0,
    duration_days: float = 6.0,
    step_days: float = 0.5,
    rise_per_day: float = 1.0,
    peak_r_mag: float = 19.6,
) -> pd.DataFrame:
    """A rise, then a straight fade, with a color that does not change.

    ``g_minus_r`` is added to the r magnitudes to make g. A negative
    value keeps the object blue. Every requested band shares the same
    fade rate, so a test can blame the fade or the color on purpose.
    """
    times = np.arange(0.0, duration_days + step_days * 0.5, step_days)
    times = times[times <= duration_days + 1e-9]
    days_from_peak = times - peak_day
    rate = np.where(days_from_peak < 0.0, rise_per_day, fade_per_day)
    r_magnitude = peak_r_mag + rate * np.abs(days_from_peak)

    frames: list[pd.DataFrame] = []
    for band in bands:
        if band == "g":
            magnitude = r_magnitude + g_minus_r
        else:
            magnitude = r_magnitude
        frames.append(
            pd.DataFrame(
                {
                    "time_days": times,
                    "filter": band,
                    "magnitude": magnitude,
                    "magnitude_error": 0.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _one_point_each() -> pd.DataFrame:
    """One g point and one r point, which is too little to fit or to fade."""
    return pd.DataFrame(
        {
            "time_days": [1.0, 1.0],
            "filter": ["g", "r"],
            "magnitude": [19.5, 19.6],
            "magnitude_error": [0.05, 0.05],
        }
    )


def test_noise_free_demo_passes_both_rules() -> None:
    """The made-up kilonova, with the noise left off, is a candidate.

    g fades at 0.90 mag/day and r at 0.45, both faster than 0.3. One day
    after the peak, g minus r is about 0.35, which is redder than the
    placeholder cut of 0.0.
    """
    verdict = is_kilonova_candidate(_noise_free_demo())
    assert verdict.fade.passed
    assert verdict.color.passed
    assert verdict.is_candidate
    assert verdict.fade.explanation == (
        "It faded 0.90 mag/day in g, which is faster than 0.30 mag/day, "
        "so the fade rule passes."
    )
    assert verdict.color.explanation == (
        "1 day after the r-band peak, g minus r is 0.35, which is redder "
        "than 0.00, so the color rule passes."
    )


def test_slow_fade_fails_the_fade_rule() -> None:
    """A supernova-like 0.1 mag/day fade is not fast enough.

    The color is held red, so this failure is the fade rule and not the
    color rule.
    """
    curve = _fading_curve(fade_per_day=0.1, g_minus_r=0.4)
    verdict = is_kilonova_candidate(curve)
    assert not verdict.fade.passed
    assert verdict.color.passed
    assert not verdict.is_candidate
    # g and r were given the same slope. The fitted numbers differ only
    # by floating-point noise, so either band may be named as the fastest.
    assert verdict.fade.explanation in (
        (
            "The fastest fade was 0.10 mag/day in g, which is not faster than "
            "0.30 mag/day, so the fade rule fails."
        ),
        (
            "The fastest fade was 0.10 mag/day in r, which is not faster than "
            "0.30 mag/day, so the fade rule fails."
        ),
    )


def test_blue_source_fails_the_color_rule() -> None:
    """A source that fades quickly but stays blue fails only the color rule.

    g minus r is kept at -0.40, so g stays the brighter band. The fade
    is 0.80 mag/day, which would pass on its own.
    """
    curve = _fading_curve(fade_per_day=0.8, g_minus_r=-0.4)
    verdict = is_kilonova_candidate(curve)
    assert verdict.fade.passed
    assert not verdict.color.passed
    assert not verdict.is_candidate
    assert verdict.color.explanation == (
        "1 day after the r-band peak, g minus r is -0.40, which is not "
        "redder than 0.00, so the color rule fails."
    )


def test_empty_light_curve_says_not_enough_data() -> None:
    """Zero rows is an empty night, not a crash.

    Both rules fail, and each sentence says there was not enough data.
    """
    empty = pd.DataFrame(
        columns=["time_days", "filter", "magnitude", "magnitude_error"]
    )
    assert len(empty) == 0
    verdict = is_kilonova_candidate(empty)
    assert not verdict.is_candidate
    assert not verdict.fade.passed
    assert not verdict.color.passed
    assert "not enough data" in verdict.fade.explanation.lower()
    assert "not enough data" in verdict.color.explanation.lower()


def test_too_few_points_says_not_enough_data() -> None:
    """One point per band cannot support either rule, and it must not crash."""
    verdict = is_kilonova_candidate(_one_point_each())
    assert not verdict.fade.passed
    assert not verdict.color.passed
    assert not verdict.is_candidate
    assert "not enough data" in verdict.fade.explanation.lower()
    assert "not enough data" in verdict.color.explanation.lower()


def test_fade_rule_reads_the_cut_from_the_config() -> None:
    """The same curve passes a loose cut and fails a strict one.

    Nothing inside the rule types the number 0.3. The only change
    between these two calls is the value on the config object.
    """
    curve = _noise_free_demo()
    loose = replace(THRESHOLDS, fade_faster_than_mag_per_day=0.3)
    strict = replace(THRESHOLDS, fade_faster_than_mag_per_day=2.0)
    assert evaluate_fade_rule(curve, loose).passed
    assert not evaluate_fade_rule(curve, strict).passed


def test_color_rule_reads_the_cut_from_the_config() -> None:
    """The placeholder 0.0 passes the demo. A cut of 1.0 does not.

    One day after the peak the demo's g minus r is about 0.35, so the
    verdict flips only because the threshold changed.
    """
    curve = _noise_free_demo()
    placeholder = replace(THRESHOLDS, redder_than_g_minus_r=0.0)
    stricter = replace(THRESHOLDS, redder_than_g_minus_r=1.0)
    assert evaluate_color_rule(curve, placeholder).passed
    assert not evaluate_color_rule(curve, stricter).passed


def test_color_delay_is_read_from_the_config() -> None:
    """At the peak the demo is still blue. One day later it is red.

    Both calls use the placeholder color cut. Only the helper setting
    that says when to measure has changed.
    """
    curve = _noise_free_demo()
    at_the_peak = replace(THRESHOLDS, color_days_after_peak=0.0)
    one_day_later = replace(THRESHOLDS, color_days_after_peak=1.0)
    assert not evaluate_color_rule(curve, at_the_peak).passed
    assert evaluate_color_rule(curve, one_day_later).passed


def test_fade_window_is_read_from_the_config() -> None:
    """A window shorter than the cadence cannot see a fade.

    The demo is sampled every half day. A window of 0.1 days keeps only
    the peak, which is not enough points. The default window of 3 days
    is enough, and the fade rule passes.
    """
    curve = _noise_free_demo()
    too_short = replace(THRESHOLDS, fade_window_days=0.1)
    short_result = evaluate_fade_rule(curve, too_short)
    assert not short_result.passed
    assert "not enough data" in short_result.explanation.lower()
    assert evaluate_fade_rule(curve).passed


def test_minimum_point_count_is_read_from_the_config() -> None:
    """Requiring more points than the curve has turns a pass into "not enough data"."""
    curve = _fading_curve(fade_per_day=0.5, g_minus_r=0.2)
    assert evaluate_fade_rule(curve).passed
    needs_many = replace(THRESHOLDS, min_points_per_band=50)
    result = evaluate_fade_rule(curve, needs_many)
    assert not result.passed
    assert "not enough data" in result.explanation.lower()


def test_fade_constant_matches_the_config() -> None:
    """``FADE_FASTER_THAN_MAG_PER_DAY`` is the config value, not a second copy.

    ``test_inject`` compares the demo against this name. Keeping the two
    equal means that test and the rule cannot drift apart.
    """
    assert FADE_FASTER_THAN_MAG_PER_DAY == THRESHOLDS.fade_faster_than_mag_per_day
    assert FADE_FASTER_THAN_MAG_PER_DAY == 0.3


def test_placeholder_color_cut_is_zero() -> None:
    """The color cut stays 0.0 until the science lead replaces it."""
    assert THRESHOLDS.redder_than_g_minus_r == 0.0


def test_i_band_alone_can_pass_the_fade_rule() -> None:
    """ZTFReST accepts a fast fade in any one of g, r, or i.

    This curve has only i, fading at 0.50 mag/day. There is no g or r,
    so the color rule cannot pass, but the fade rule can.
    """
    curve = _fading_curve(fade_per_day=0.5, g_minus_r=0.0, bands=("i",))
    fade = evaluate_fade_rule(curve)
    assert fade.passed
    assert fade.explanation == (
        "It faded 0.50 mag/day in i, which is faster than 0.30 mag/day, "
        "so the fade rule passes."
    )
    color = evaluate_color_rule(curve)
    assert not color.passed
    assert "not enough data" in color.explanation.lower()

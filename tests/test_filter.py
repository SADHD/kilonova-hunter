"""Tests for the three v1 kilonova rules.

The fade cut is "faster than 0.3 mag/day in any one of g, r, or i".
The color cut is a placeholder: g minus r greater than 0.0 after the
peak. The reddening cut is also a placeholder: g minus r must climb by
more than 0.1 mag inside a 2-day window. Every number is read from
``config.py``.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from kilonova_hunter.config import THRESHOLDS
from kilonova_hunter.filter import (
    FADE_FASTER_THAN_MAG_PER_DAY,
    evaluate_color_rule,
    evaluate_fade_rule,
    evaluate_reddening_rule,
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


def test_noise_free_demo_passes_every_rule() -> None:
    """The made-up kilonova, with the noise left off, is a candidate.

    g fades at 0.90 mag/day and r at 0.45, both faster than 0.3. One day
    after the peak, g minus r is about 0.35, which is redder than the
    placeholder cut of 0.0. g minus r also climbs over the first two
    days, so the reddening rule passes.
    """
    verdict = is_kilonova_candidate(_noise_free_demo())
    assert verdict.fade.passed
    assert verdict.color.passed
    assert verdict.reddening.passed is True
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

    The color is held red, so the single-time color rule passes. The
    failure this test checks is the fade rule.
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
    """A source that fades quickly but stays blue fails the color rule.

    g minus r is kept at -0.40, so g stays the brighter band. The fade
    is 0.80 mag/day, which would pass on its own. The color does not
    change, so the reddening rule fails as well. This test checks the
    single-time color sentence, which is unchanged.
    """
    curve = _fading_curve(fade_per_day=0.8, g_minus_r=-0.4)
    verdict = is_kilonova_candidate(curve)
    assert verdict.fade.passed
    assert not verdict.color.passed
    # The color is held still, so the rise is 0.00 and reddening fails.
    assert verdict.reddening.passed is False
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


def test_reddening_cuts_are_placeholders() -> None:
    """The reddening numbers stay put until the science lead replaces them.

    Tuesday Nov 3, 2026 is when the real AT2017gfo numbers are expected.
    Until then these are the stand-ins the tests read.
    """
    assert THRESHOLDS.min_g_minus_r_rise == 0.1
    assert THRESHOLDS.reddening_window_days == 2.0
    assert THRESHOLDS.min_color_separation_days == 0.5


def test_one_color_skips_reddening_and_still_passes() -> None:
    """One color measurement does not reject an otherwise good candidate.

    r fades quickly, so the fade rule passes. The only g point sits one
    day after the r peak, and it is red, so the single-time color rule
    passes. There is no second color, so the reddening rule is skipped.
    A skip is not a failure: the object is still a candidate.
    """
    # r peaks at day 1 (magnitude 19.6) and fades by 0.5 mag/day.
    curve = pd.DataFrame(
        {
            "time_days": [0.0, 1.0, 2.0, 3.0, 2.0],
            "filter": ["r", "r", "r", "r", "g"],
            "magnitude": [20.6, 19.6, 20.1, 20.6, 20.5],
            "magnitude_error": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    verdict = is_kilonova_candidate(curve)
    assert verdict.fade.passed is True
    assert verdict.color.passed is True
    assert verdict.color.explanation == (
        "1 day after the r-band peak, g minus r is 0.40, which is redder "
        "than 0.00, so the color rule passes."
    )
    assert verdict.reddening.passed is None
    assert verdict.reddening.explanation == (
        "Skipped: g minus r was measured only once, so we cannot tell "
        "whether it keeps getting redder. This rule does not reject the object."
    )
    assert verdict.is_candidate is True


def test_stays_equally_red_fails_reddening() -> None:
    """Red at the start and still the same red later is not "getting redder".

    Two or more color measurements exist, the fade is fast, and the
    single-time color cut passes. The reddening rule is the one that
    fails, and that failure rejects the object.
    """
    curve = _fading_curve(fade_per_day=0.8, g_minus_r=0.4)
    verdict = is_kilonova_candidate(curve)
    assert verdict.fade.passed is True
    assert verdict.color.passed is True
    assert verdict.reddening.passed is False
    assert verdict.reddening.explanation == (
        "From day 0.00 to day 2.00, g minus r went from 0.40 to 0.40, "
        "a rise of 0.00 mag, which is not more than 0.10 mag, "
        "so the reddening rule fails."
    )
    assert verdict.is_candidate is False


def test_noise_free_demo_passes_reddening() -> None:
    """The made-up kilonova gets redder across the placeholder window.

    g fades faster than r, so g minus r climbs. From the first visit to
    the visit two days later, the rise is larger than the placeholder
    cut of 0.1 mag.
    """
    result = evaluate_reddening_rule(_noise_free_demo())
    assert result.passed is True
    assert result.explanation == (
        "From day 0.00 to day 2.00, g minus r went from -0.40 to 0.35, "
        "a rise of 0.75 mag, which is more than 0.10 mag, "
        "so the reddening rule passes."
    )


def test_reddening_rule_reads_the_cuts_from_the_config() -> None:
    """The same curve passes a loose rise cut and fails a strict one.

    Nothing inside the rule types the number 0.1. The only change
    between the first two calls is the value on the config object.
    A window too short to hold a second visit skips the rule instead
    of failing it, and that window is also read from the config.
    """
    curve = _noise_free_demo()
    loose = replace(THRESHOLDS, min_g_minus_r_rise=0.1)
    strict = replace(THRESHOLDS, min_g_minus_r_rise=5.0)
    assert evaluate_reddening_rule(curve, loose).passed is True
    failed = evaluate_reddening_rule(curve, strict)
    assert failed.passed is False
    assert "not more than 5.00 mag" in failed.explanation

    # The demo is sampled every half day. 0.1 days keeps only the first
    # color, so there is nothing later to compare.
    too_short = replace(
        THRESHOLDS,
        reddening_window_days=0.1,
        min_color_separation_days=0.0,
    )
    skipped = evaluate_reddening_rule(curve, too_short)
    assert skipped.passed is None
    assert skipped.explanation == (
        "Skipped: no later g minus r measurement falls inside the "
        "0.10-day window, at least 0.00 days after the first color, "
        "so we cannot tell whether it keeps getting redder. "
        "This rule does not reject the object."
    )
    # The same curve, asked to compare colors at least a day apart,
    # still finds a later visit inside the default 2-day window.
    wider_gap = replace(THRESHOLDS, min_color_separation_days=1.0)
    assert evaluate_reddening_rule(curve, wider_gap).passed is True


def test_color_separation_cannot_be_longer_than_the_window() -> None:
    """A later color cannot be required to sit outside the window."""
    with pytest.raises(ValueError, match="min_color_separation_days"):
        replace(
            THRESHOLDS,
            reddening_window_days=1.0,
            min_color_separation_days=2.0,
        )


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

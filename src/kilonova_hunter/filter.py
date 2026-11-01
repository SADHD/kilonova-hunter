"""Rules that decide whether one light curve looks like a kilonova.

The module is named ``filter`` because it narrows a night of alerts down
to a short candidate list. It has nothing to do with Python's built-in
``filter`` function.

Version 1 has two rules, and each one comes with a sentence a newcomer
can read:

- Fade: the object fades faster than the cut in ``config.py`` in any
  one of g, r, or i. The cut is 0.3 magnitudes per day, signed off from
  ZTFReST. See ``FADE_FASTER_THAN_MAG_PER_DAY``.
- Color: after the peak, g minus r is redder than the cut in
  ``config.py``. That cut is still a placeholder.

Both rules have to pass. A rule that cannot be measured (too few
points) fails, with a sentence that says there was not enough data.
We do not call something a kilonova when we cannot tell.

``is_kilonova_candidate`` checks one light curve. Scoring a whole
downloaded night, and printing the list from the command line, is not
wired up yet.
"""

from dataclasses import dataclass

import pandas as pd

from kilonova_hunter.config import THRESHOLDS, Thresholds, resolve_thresholds
from kilonova_hunter.features import fade_rates, g_minus_r_at_time, peak_time_days

# The fade cut lives in config.py so there is one place to change it.
# This name stays because tests and comments already use it. Do not
# type a new number here; edit THRESHOLDS instead.
FADE_FASTER_THAN_MAG_PER_DAY = THRESHOLDS.fade_faster_than_mag_per_day


@dataclass(frozen=True)
class RuleResult:
    """One yes-or-no rule and the sentence that explains it.

    Attributes:
        passed: True when this rule is satisfied.
        explanation: One plain-English sentence. It says why the rule
            passed, why it failed, or that there was not enough data.
    """

    passed: bool
    explanation: str


@dataclass(frozen=True)
class CandidateVerdict:
    """Whether one light curve looks like a kilonova, and why.

    Attributes:
        is_candidate: True only when every rule passed.
        fade: The fast-fade rule.
        color: The red-color rule (g minus r after the peak).
    """

    is_candidate: bool
    fade: RuleResult
    color: RuleResult


def is_kilonova_candidate(
    light_curve: pd.DataFrame,
    thresholds: Thresholds | None = None,
) -> CandidateVerdict:
    """Decide whether one light curve looks like a kilonova.

    Args:
        light_curve: One object, with columns ``time_days``, ``filter``,
            and ``magnitude``. This is the same table
            ``simulate_kilonova_light_curve`` returns.
        thresholds: Cuts and helper settings. Omit this to use
            ``THRESHOLDS`` from ``config.py``.

    Returns:
        Pass or fail for each rule, each with one sentence, plus
        ``is_candidate`` for the combination.

    Raises:
        ValueError: If the table is missing a required column.
    """
    settings = resolve_thresholds(thresholds)
    fade = evaluate_fade_rule(light_curve, settings)
    color = evaluate_color_rule(light_curve, settings)
    return CandidateVerdict(
        is_candidate=fade.passed and color.passed,
        fade=fade,
        color=color,
    )


def evaluate_fade_rule(
    light_curve: pd.DataFrame,
    thresholds: Thresholds | None = None,
) -> RuleResult:
    """Apply the fast-fade rule to one light curve.

    The rule passes when any one configured band fades strictly faster
    than the cut. Bands that were not observed, or that do not have
    enough points, are skipped. If no band can be measured, the rule
    fails with a "not enough data" sentence.

    Args:
        light_curve: One object, with the usual light-curve columns.
        thresholds: Cuts and helper settings. Omit this to use the
            project defaults.

    Returns:
        Whether the fade rule passed, and one sentence explaining it.
    """
    settings = resolve_thresholds(thresholds)
    rates = fade_rates(light_curve, settings)

    fastest_band = ""
    fastest_rate: float | None = None
    for rate in rates.values():
        if rate.mag_per_day is None:
            continue
        if fastest_rate is None or rate.mag_per_day > fastest_rate:
            fastest_rate = rate.mag_per_day
            fastest_band = rate.band

    if fastest_rate is None:
        bands = _or_list(settings.fade_bands)
        return RuleResult(
            passed=False,
            explanation=(
                "Not enough data to measure a fade: a fit needs at least "
                f"{settings.min_points_per_band} points after the peak in "
                f"{bands}, and none of those bands has that many."
            ),
        )

    cut = settings.fade_faster_than_mag_per_day
    rate_text = f"{fastest_rate:.2f}"
    cut_text = f"{cut:.2f}"
    if fastest_rate > cut:
        return RuleResult(
            passed=True,
            explanation=(
                f"It faded {rate_text} mag/day in {fastest_band}, which is "
                f"faster than {cut_text} mag/day, so the fade rule passes."
            ),
        )
    return RuleResult(
        passed=False,
        explanation=(
            f"The fastest fade was {rate_text} mag/day in {fastest_band}, "
            f"which is not faster than {cut_text} mag/day, so the fade rule fails."
        ),
    )


def evaluate_color_rule(
    light_curve: pd.DataFrame,
    thresholds: Thresholds | None = None,
) -> RuleResult:
    """Apply the red-color rule to one light curve.

    The color is g minus r, read ``color_days_after_peak`` after the
    brightest r-band point. r is the clock because g minus r is a
    comparison against r, and g often peaks earlier. The rule passes
    when that color is strictly redder than the cut.

    Args:
        light_curve: One object, with the usual light-curve columns.
        thresholds: Cuts and helper settings. Omit this to use the
            project defaults.

    Returns:
        Whether the color rule passed, and one sentence explaining it.
    """
    settings = resolve_thresholds(thresholds)
    peak_time = peak_time_days(light_curve, "r")
    if peak_time is None:
        return RuleResult(
            passed=False,
            explanation=(
                "Not enough data: g minus r needs r-band points to mark "
                "the peak, and this curve has none."
            ),
        )

    when = peak_time + settings.color_days_after_peak
    color = g_minus_r_at_time(light_curve, when)
    if color.g_minus_r is None:
        return RuleResult(passed=False, explanation=color.explanation)

    cut = settings.redder_than_g_minus_r
    color_text = f"{color.g_minus_r:.2f}"
    cut_text = f"{cut:.2f}"
    when_text = _after_peak_phrase(settings.color_days_after_peak)
    if color.g_minus_r > cut:
        return RuleResult(
            passed=True,
            explanation=(
                f"{when_text}, g minus r is {color_text}, which is redder "
                f"than {cut_text}, so the color rule passes."
            ),
        )
    return RuleResult(
        passed=False,
        explanation=(
            f"{when_text}, g minus r is {color_text}, which is not redder "
            f"than {cut_text}, so the color rule fails."
        ),
    )


def score_candidates(features: pd.DataFrame) -> pd.DataFrame:
    """Score a whole night of transients.

    The check for one light curve is ``is_kilonova_candidate``. Applying
    it to a table of many objects, and printing that list, waits until
    the download and the command line are wired up.

    Args:
        features: One row per transient, as ``measure_features`` returns.

    Returns:
        The same rows, plus a score and a plain-language reason for each
        rule the candidate passed or failed.

    Raises:
        NotImplementedError: Until a night of alerts is wired through.
    """
    raise NotImplementedError(
        "Scoring a whole night is not wired up yet. For one light curve, "
        "call is_kilonova_candidate "
        f"(this table has {len(features)} rows)."
    )


def _or_list(names: tuple[str, ...]) -> str:
    """Join names as "g, r, or i"."""
    items = list(names)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def _after_peak_phrase(days_after_peak: float) -> str:
    """Describe when the color was read, as the start of a sentence."""
    if abs(days_after_peak) <= 1e-9:
        return "At the r-band peak"
    if abs(days_after_peak - 1.0) <= 1e-9:
        return "1 day after the r-band peak"
    text = f"{days_after_peak:.2f}".rstrip("0").rstrip(".")
    return f"{text} days after the r-band peak"

"""Rules that decide whether one light curve looks like a kilonova.

The module is named ``filter`` because it narrows a night of alerts down
to a short candidate list. It has nothing to do with Python's built-in
``filter`` function.

Version 1 has three rules, and each one comes with a sentence a newcomer
can read:

- Fade: the object fades faster than the cut in ``config.py`` in any
  one of g, r, or i. The cut is 0.3 magnitudes per day, signed off from
  ZTFReST. See ``FADE_FASTER_THAN_MAG_PER_DAY``.
- Color: the latest same-night g+r detection pair is redder than the cut in
  ``config.py``. That cut is still a placeholder. This is one moment
  in time, not a change.
- Reddening: a straight line fitted to g minus r inside a time window
  climbs by more than the cut in ``config.py``. Those cuts are still
  placeholders. This is the "keeps getting redder" check. If there are
  not two colors spread across the window, this rule is skipped.

The color rule must pass. Reddening skips when there are too few colors.
Fade normally fails when it cannot be measured. Decision 12(22) adds
an optional, default-off fade variant: a band needs two detections
strictly after its brightest point within three days. If no band is
measurable, this variant skips fade and counts the skip.

``is_kilonova_candidate`` checks one light curve. ``kilohunt check``
prints that verdict for a CSV, and ``kilohunt demo`` prints it for the
noise-free toy curve. Scoring a whole downloaded night is not wired
up yet.
"""

from dataclasses import dataclass, fields

import pandas as pd

from kilonova_hunter.config import THRESHOLDS, Thresholds, resolve_thresholds
from kilonova_hunter.features import (
    ColorRise,
    _format_mag,
    fade_rates,
    g_minus_r_rise,
    latest_same_night_color,
)

# The fade cut lives in config.py so there is one place to change it.
# This name stays because tests and comments already use it. Do not
# type a new number here; edit THRESHOLDS instead.
FADE_FASTER_THAN_MAG_PER_DAY = THRESHOLDS.fade_faster_than_mag_per_day


@dataclass(frozen=True)
class RuleResult:
    """One rule and the sentence that explains it.

    Attributes:
        passed: True when this rule is satisfied, False when it is not,
            and None when it was skipped. Reddening skips when there
            are not two colors to compare.
            A skip is not a failure, and it does not reject the object.
            The command line prints that as SKIPPED. The fade rule and
            the single-time color rule still use False when they do not
            have enough data.
        explanation: One plain-English sentence. It says why the rule
            passed, why it failed, or why it was skipped.
    """

    passed: bool | None
    explanation: str


@dataclass(frozen=True)
class CandidateVerdict:
    """Whether one light curve looks like a kilonova, and why.

    Each rule is its own field, holding a ``RuleResult``. The command
    line prints one line per rule by walking those fields. It does not
    have the names fade and color written out by hand, so a new rule
    field (for example reddening) is printed too. A rule that says it
    was skipped is printed as SKIPPED.

    Attributes:
        is_candidate: True when every rule that could be measured
            passed. A skipped rule (``passed`` is ``None``) does not
            count as a failure and does not block the candidate.
        fade: The fast-fade rule.
        color: The latest same-night g minus r detection pair.
        reddening: The "keeps getting redder" rule. A straight line
            through g minus r has to climb.
    """

    is_candidate: bool
    fade: RuleResult
    color: RuleResult
    reddening: RuleResult


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
        Pass, fail, or skip for each rule, each with one sentence, plus
        ``is_candidate`` for the combination. A skipped rule does not
        turn a candidate into a reject.

    Raises:
        ValueError: If the table is missing a required column.
    """
    settings = resolve_thresholds(thresholds)
    fade = evaluate_fade_rule(light_curve, settings)
    color = evaluate_color_rule(light_curve, settings)
    reddening = evaluate_reddening_rule(light_curve, settings)
    # passed is None on a skip. None is not False, so a skip does not
    # block. The fade and color rules still have to pass.
    return CandidateVerdict(
        is_candidate=_measured_rules_passed(fade, color, reddening),
        fade=fade,
        color=color,
        reddening=reddening,
    )


def evaluate_fade_rule(
    light_curve: pd.DataFrame,
    thresholds: Thresholds | None = None,
) -> RuleResult:
    """Apply the fast-fade rule to one light curve.

    The rule passes when any one configured band fades strictly faster
    than the cut. Bands that were not observed, or that do not have
    enough points, are skipped. If no band can be measured, the rule
    fails with a "not enough data" sentence. The default-off candidate
    variant requires two detections strictly after the brightest point
    within the window, preserves the fit including that point, and
    skips if no band is measurable.

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
    """Apply the red-color rule to the latest same-night detection pair.

    Decision: option (ii), Nov 4 2026; reverses 12(3).
    Callers truncate to D first. Absolute Rubin merger time is supplied
    in attrs; ZTF curves carry their own night column. Reddening retains
    its separate interpolated sampling. The cut is unchanged.

    Args:
        light_curve: One object, with the usual light-curve columns.
        thresholds: Cuts and helper settings. Omit this to use the
            project defaults.

    Returns:
        Whether the color rule passed, and one sentence explaining it.
    """
    settings = resolve_thresholds(thresholds)
    color = latest_same_night_color(light_curve)
    if color.g_minus_r is None:
        return RuleResult(passed=False, explanation=color.explanation)

    cut = settings.redder_than_g_minus_r
    color_text = f"{color.g_minus_r:.2f}"
    cut_text = f"{cut:.2f}"
    when_text = "At the latest same-night g+r pair"
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


def evaluate_reddening_rule(
    light_curve: pd.DataFrame,
    thresholds: Thresholds | None = None,
) -> RuleResult:
    """Apply the "keeps getting redder" rule to one light curve.

    g minus r should climb along a straight line fitted to every color
    inside the time window. The climb has to be strictly larger than
    ``min_g_minus_r_rise``. A color that stays the same, or gets bluer,
    fails. The number is the fitted climb, so one noisy endpoint does
    not decide it by itself.

    If there are not two colors spread across the window, the rule is
    skipped. ``passed`` is ``None``, and the sentence says why. One
    snapshot can still be red. We only reject when we can see that the
    color is not climbing. The fade rule and the single-time color rule
    do not skip: too little data for those still fails them.

    Args:
        light_curve: One object, with the usual light-curve columns.
        thresholds: Cuts and helper settings. Omit this to use the
            project defaults.

    Returns:
        Pass, fail, or skip, and one sentence explaining it.
    """
    settings = resolve_thresholds(thresholds)
    change = g_minus_r_rise(light_curve, settings)
    # A rise only exists when both colors were read. If any piece is
    # missing, we skip instead of rejecting the object.
    if (
        change.rise is None
        or change.earlier_time_days is None
        or change.later_time_days is None
        or change.earlier_g_minus_r is None
        or change.later_g_minus_r is None
    ):
        return RuleResult(
            passed=None,
            explanation=_skipped_reddening_sentence(change, settings),
        )

    rise_text = _format_mag(change.rise)
    cut_text = f"{settings.min_g_minus_r_rise:.2f}"
    span = (
        f"From day {change.earlier_time_days:.2f} to day "
        f"{change.later_time_days:.2f}, a straight-line fit of g minus r "
        f"goes from {_format_mag(change.earlier_g_minus_r)} to "
        f"{_format_mag(change.later_g_minus_r)}, a rise of {rise_text} mag, "
    )
    if change.rise > settings.min_g_minus_r_rise:
        ending = f"which is more than {cut_text} mag, so the reddening rule passes."
    else:
        ending = f"which is not more than {cut_text} mag, so the reddening rule fails."
    # The note says whether every color had a real uncertainty, whether
    # an unknown one was given a stand-in, or whether the fit gave up
    # on weights because no uncertainty was known.
    explanation = f"{span}{ending}"
    if change.weighting_note:
        explanation = f"{explanation} {change.weighting_note}"
    return RuleResult(
        passed=change.rise > settings.min_g_minus_r_rise, explanation=explanation
    )


def score_candidates(features: pd.DataFrame) -> pd.DataFrame:
    """Score a whole night of transients.

    The check for one light curve is ``is_kilonova_candidate``.
    ``kilohunt check`` prints that verdict for a CSV. Applying the
    rules to a table of many objects waits until a night of alerts
    can be downloaded.

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


def _measured_rules_passed(*rules: RuleResult) -> bool:
    """True when every rule that was actually measured passed.

    ``passed is None`` means the rule was skipped. A skipped rule is
    left out of this check, so it does not keep the object off the
    candidate list. ``passed is False`` means we measured the rule and
    it failed, and the object is not a candidate.
    """
    return all(rule.passed is not False for rule in rules)


def _skipped_reddening_sentence(change: ColorRise, settings: Thresholds) -> str:
    """Explain why the reddening rule was not applied.

    The sentence starts with "Skipped" so it cannot be read as a
    failure, and it says the object is not rejected.
    """
    if change.color_count <= 0:
        reason = "g minus r could not be measured at any time"
    elif change.color_count == 1:
        reason = "g minus r was measured only once"
    else:
        window = f"{settings.reddening_window_days:.2f}"
        gap = f"{settings.min_color_separation_days:.2f}"
        reason = (
            "no later g minus r measurement falls inside the "
            f"{window}-day window, at least {gap} days after the first color"
        )
    return (
        f"Skipped: {reason}, so we cannot tell whether it keeps getting "
        "redder. This rule does not reject the object."
    )


def _or_list(names: tuple[str, ...]) -> str:
    """Join names as "g, r, or i"."""
    items = list(names)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


def format_verdict(verdict: CandidateVerdict) -> str:
    """Turn a filter verdict into the lines ``kilohunt check`` prints.

    The first line is ``CANDIDATE`` or ``NOT A CANDIDATE``. Each line
    after that is one rule: PASS, FAIL, or SKIPPED, then the sentence.
    ``recall`` uses this same text for the AT2017gfo-like sanity check,
    so the two commands cannot drift apart.
    """
    overall = "CANDIDATE" if verdict.is_candidate else "NOT A CANDIDATE"
    lines = [overall]
    for field in fields(verdict):
        value = getattr(verdict, field.name)
        if not isinstance(value, RuleResult):
            continue
        label = field.name.replace("_", " ").capitalize()
        lines.append(f"{label}: {_rule_word(value)}. {value.explanation}")
    return "\n".join(lines)


def _rule_word(result: RuleResult) -> str:
    """PASS, FAIL, or SKIPPED.

    SKIPPED means ``passed`` is ``None``. Reddening does that when there
    are not two colors to compare. A skip is not a failure. Fade and
    color always pass or fail. Too little data for those two is FAIL.
    """
    if result.passed is None:
        return "SKIPPED"
    if result.passed:
        return "PASS"
    return "FAIL"


def _after_peak_phrase(days_after_peak: float) -> str:
    """Describe when the color was read, as the start of a sentence."""
    if abs(days_after_peak) <= 1e-9:
        return "At the r-band peak"
    if abs(days_after_peak - 1.0) <= 1e-9:
        return "1 day after the r-band peak"
    text = f"{days_after_peak:.2f}".rstrip("0").rstrip(".")
    return f"{text} days after the r-band peak"

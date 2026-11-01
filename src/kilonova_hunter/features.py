"""Measure how one light curve fades, and what color it is.

Kilonova debris brightens within about a day, fades within days, and
turns red as it cools. This module measures those facts for a single
light curve:

- the fade rate in each band, as a straight line fitted after the
  brightest point
- g minus r at one chosen time
- how much g minus r rises between the earliest color and a later one

A negative fade rate means the object is still getting brighter,
because a smaller magnitude is brighter. A positive color rise means
the object got redder. Too few points returns ``None`` for the number
and a sentence that says why, instead of raising. Summarizing a whole
night of objects into one table is still ``measure_features``, which
is not wired up yet.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kilonova_hunter.config import Thresholds, resolve_thresholds

# Columns every measurement reads. magnitude_error is not required:
# the line fit uses the magnitudes themselves.
_REQUIRED_COLUMNS = ("time_days", "filter", "magnitude")

# Two times closer than this count as the same visit. A survey cadence
# is hours to days, so this only absorbs floating-point wobble.
_SAME_VISIT_DAYS = 1e-6


@dataclass(frozen=True)
class BandFade:
    """The fade rate in one filter, or why it could not be measured.

    Attributes:
        band: The filter name, such as ``"g"`` or ``"r"``.
        mag_per_day: Magnitudes per day after the peak. ``None`` when
            there are not enough points. A positive number means the
            object is fading, because magnitude climbs as it gets fainter.
        explanation: One sentence. On success it states the rate. On
            failure it says there was not enough data.
    """

    band: str
    mag_per_day: float | None
    explanation: str


@dataclass(frozen=True)
class ColorMeasurement:
    """g minus r at one time, or why it could not be measured.

    Attributes:
        g_minus_r: g magnitude minus r magnitude. ``None`` when either
            band cannot be read at the requested time. A larger number
            is redder.
        time_days: The time that was asked for, even when the number
            could not be measured.
        explanation: One sentence. On success it states the color. On
            failure it says there was not enough data.
    """

    g_minus_r: float | None
    time_days: float
    explanation: str


@dataclass(frozen=True)
class ColorSample:
    """g minus r at one time where both bands can be read.

    Attributes:
        time_days: When this color was read.
        g_minus_r: g magnitude minus r magnitude. A larger number is
            redder.
    """

    time_days: float
    g_minus_r: float


@dataclass(frozen=True)
class ColorRise:
    """How much g minus r changed between two times.

    Attributes:
        rise: The later color minus the earlier color, in magnitudes.
            ``None`` when there are not two colors inside the window.
            A positive number means the object got redder. A negative
            number means it got bluer.
        earlier_time_days: When the first color was read, or ``None``
            if no color could be read at all.
        later_time_days: When the comparison color was read, or
            ``None`` if there was no later color inside the window.
        earlier_g_minus_r: g minus r at the earlier time, or ``None``.
        later_g_minus_r: g minus r at the later time, or ``None``.
        color_count: How many times g minus r could be read, before
            the window is applied. Two are required before a rise exists.
        explanation: One sentence. On success it states the rise. On
            failure it says there was not enough data.
    """

    rise: float | None
    earlier_time_days: float | None
    later_time_days: float | None
    earlier_g_minus_r: float | None
    later_g_minus_r: float | None
    color_count: int
    explanation: str


def fade_rates(
    light_curve: pd.DataFrame,
    thresholds: Thresholds | None = None,
) -> dict[str, BandFade]:
    """Fit a fade rate for each configured band that appears in the curve.

    Bands in the config that were never observed are left out. A band
    that was observed, but does not have enough points after its peak,
    is included with ``mag_per_day`` set to ``None``.

    Args:
        light_curve: One object. Columns ``time_days``, ``filter``, and
            ``magnitude``. A smaller magnitude is brighter.
        thresholds: Cuts and helper settings. Omit this to use
            ``THRESHOLDS`` from ``config.py``.

    Returns:
        A mapping from band name to its fade. The order follows
        ``thresholds.fade_bands``.

    Raises:
        ValueError: If the table is missing a required column.
    """
    settings = resolve_thresholds(thresholds)
    _require_light_curve(light_curve)
    rates: dict[str, BandFade] = {}
    for band in settings.fade_bands:
        points = light_curve.loc[light_curve["filter"] == band]
        if points.empty:
            continue
        rates[band] = _fade_rate_for_one_band(
            points,
            band=band,
            min_points=settings.min_points_per_band,
            window_days=settings.fade_window_days,
        )
    return rates


def g_minus_r_at_time(
    light_curve: pd.DataFrame,
    time_days: float,
) -> ColorMeasurement:
    """Return g minus r at one time.

    If a band has a point at that time, that magnitude is used. If the
    time sits between two points, the magnitude is read off the straight
    line that joins them. That is linear interpolation: we do not invent
    a curve, we only connect the neighbors. If the time is earlier than
    the first point or later than the last, the result is ``None``,
    because guessing past the data would invent a detection.

    Args:
        light_curve: One object, with the usual light-curve columns.
        time_days: When to read the color, in the same time unit as
            ``time_days`` on the table.

    Returns:
        The color, or ``None`` plus a "not enough data" sentence.

    Raises:
        ValueError: If the table is missing a required column.
    """
    _require_light_curve(light_curve)
    g_points = _usable_points(light_curve, "g")
    r_points = _usable_points(light_curve, "r")
    if g_points.empty or r_points.empty:
        return ColorMeasurement(
            g_minus_r=None,
            time_days=time_days,
            explanation=_missing_band_sentence(g_points.empty, r_points.empty),
        )

    g_magnitude = _magnitude_at_time(g_points, time_days)
    r_magnitude = _magnitude_at_time(r_points, time_days)
    if g_magnitude is None or r_magnitude is None:
        return ColorMeasurement(
            g_minus_r=None,
            time_days=time_days,
            explanation=_outside_coverage_sentence(
                time_days,
                g_points if g_magnitude is None else None,
                r_points if r_magnitude is None else None,
            ),
        )

    color = g_magnitude - r_magnitude
    return ColorMeasurement(
        g_minus_r=color,
        time_days=time_days,
        explanation=(
            f"At day {time_days:.2f}, g minus r is {color:.2f} "
            "(a larger number is redder)."
        ),
    )


def g_minus_r_over_time(light_curve: pd.DataFrame) -> list[ColorSample]:
    """Read g minus r at every time the data can support.

    The times we try are the times when g or r was observed. If the
    other band was not observed at that exact time, its magnitude is
    read off the straight line between its neighbors. That is the same
    reading ``g_minus_r_at_time`` already does. We do not invent extra
    times between visits.

    A time before the first point of either band, or after the last
    point, is left out. Extending the line past the data would invent
    a detection. Two observations closer than a tiny rounding error
    are one visit, so they produce one color, not two.

    Args:
        light_curve: One object, with the usual light-curve columns.

    Returns:
        The colors, earliest first. Empty when g or r is missing, or
        when their times do not overlap.

    Raises:
        ValueError: If the table is missing a required column.
    """
    _require_light_curve(light_curve)
    g_points = _usable_points(light_curve, "g")
    r_points = _usable_points(light_curve, "r")
    if g_points.empty or r_points.empty:
        return []

    samples: list[ColorSample] = []
    for time_days in _visit_times(g_points, r_points):
        # Ask the existing color helper. It returns None when this time
        # would need a guess past the first or last point of either band.
        color = g_minus_r_at_time(light_curve, time_days)
        if color.g_minus_r is None:
            continue
        samples.append(ColorSample(time_days=time_days, g_minus_r=color.g_minus_r))
    return samples


def g_minus_r_rise(
    light_curve: pd.DataFrame,
    thresholds: Thresholds | None = None,
) -> ColorRise:
    """Measure how much g minus r climbed inside the time window.

    The earlier color is the earliest one ``g_minus_r_over_time`` can
    read. The later color is the last one that is still inside
    ``reddening_window_days`` of that first color, and at least
    ``min_color_separation_days`` after it. The rise is the later color
    minus the earlier color. Using the last color in the window asks
    "how much redder did it get over that whole stretch?", rather than
    "did the very next visit move?".

    If there are not two colors that fit that window, ``rise`` is
    ``None``. This function only measures. The filter decides whether
    the rise is large enough, and it skips the rule when ``rise`` is
    ``None``.

    Args:
        light_curve: One object, with the usual light-curve columns.
        thresholds: The window and the minimum separation. Omit this to
            use ``THRESHOLDS`` from ``config.py``.

    Returns:
        The rise, the two times and colors that produced it, how many
        colors could be read, and one sentence.

    Raises:
        ValueError: If the table is missing a required column.
    """
    settings = resolve_thresholds(thresholds)
    samples = g_minus_r_over_time(light_curve)
    if len(samples) == 0:
        return ColorRise(
            rise=None,
            earlier_time_days=None,
            later_time_days=None,
            earlier_g_minus_r=None,
            later_g_minus_r=None,
            color_count=0,
            explanation=(
                "Not enough data: g minus r could not be measured at any time."
            ),
        )
    if len(samples) == 1:
        only = samples[0]
        return ColorRise(
            rise=None,
            earlier_time_days=only.time_days,
            later_time_days=None,
            earlier_g_minus_r=only.g_minus_r,
            later_g_minus_r=None,
            color_count=1,
            explanation=(
                "Not enough data: g minus r was measured only once, "
                f"at day {only.time_days:.2f}."
            ),
        )

    # TODO(before redback recall run): the rise only compares the earliest
    # color with one later color, and it ignores magnitude_error (how
    # uncertain each brightness is). On noisy data one bad first point
    # can swing the result. Options for later, not for now: a fit through
    # all the colors in the window, or an error-weighted rise that trusts
    # a precise point more than a noisy one.
    earliest = samples[0]
    # Walk forward from the first color. A later color that is too close
    # is still the same moment, so it does not show a change. A color
    # past the window is too late for this comparison. The last color we
    # keep is the far end of the window.
    window_end = earliest.time_days + settings.reddening_window_days
    earliest_later = earliest.time_days + settings.min_color_separation_days
    later: ColorSample | None = None
    for sample in samples[1:]:
        if sample.time_days < earliest_later - _SAME_VISIT_DAYS:
            continue
        if sample.time_days > window_end + _SAME_VISIT_DAYS:
            break
        later = sample

    if later is None:
        return ColorRise(
            rise=None,
            earlier_time_days=earliest.time_days,
            later_time_days=None,
            earlier_g_minus_r=earliest.g_minus_r,
            later_g_minus_r=None,
            color_count=len(samples),
            explanation=(
                "Not enough data: no later g minus r measurement falls "
                f"inside the {settings.reddening_window_days:.2f}-day window "
                "after the first color, at least "
                f"{settings.min_color_separation_days:.2f} days after it."
            ),
        )

    rise = later.g_minus_r - earliest.g_minus_r
    return ColorRise(
        rise=rise,
        earlier_time_days=earliest.time_days,
        later_time_days=later.time_days,
        earlier_g_minus_r=earliest.g_minus_r,
        later_g_minus_r=later.g_minus_r,
        color_count=len(samples),
        explanation=(
            f"From day {earliest.time_days:.2f} to day {later.time_days:.2f}, "
            f"g minus r went from {earliest.g_minus_r:.2f} to "
            f"{later.g_minus_r:.2f}, a rise of {rise:.2f} mag."
        ),
    )


def peak_time_days(light_curve: pd.DataFrame, band: str) -> float | None:
    """Return the time of the brightest point in one band.

    Brightest means the smallest magnitude. Returns ``None`` when that
    band has no magnitude to read. If two points tie, the earlier time
    wins, so the fade that follows has one definite start.

    Args:
        light_curve: One object, with the usual light-curve columns.
        band: Which filter to look at, such as ``"r"``.

    Returns:
        The peak time, or ``None`` if the band is missing.

    Raises:
        ValueError: If the table is missing a required column.
    """
    _require_light_curve(light_curve)
    points = _usable_points(light_curve, band)
    if points.empty:
        return None
    # Sort by time so that, if two points are equally bright, the earlier
    # one comes first. argmin keeps the first minimum it sees, so that
    # earlier time is the one we return.
    #
    # Use the row's position, not its label. A label is only a name.
    # Sticking two tables together without resetting the index can give
    # two different rows the same name, and looking that name up would
    # hand back both rows.
    ordered = points.sort_values("time_days")
    magnitudes = ordered["magnitude"].to_numpy(dtype=float)
    brightest_position = int(np.argmin(magnitudes))
    return float(ordered["time_days"].iloc[brightest_position])


def measure_features(light_curves: pd.DataFrame) -> pd.DataFrame:
    """Summarize every transient in a night with a rise, a fade, and a color.

    For one light curve, call ``fade_rates``, ``g_minus_r_at_time``,
    and ``g_minus_r_rise``. Folding a whole downloaded night into one
    row per object is not wired up yet.

    Args:
        light_curves: Cleaned light curves, one row per observation.

    Returns:
        One row per transient, with its rates and color.

    Raises:
        NotImplementedError: Until a night of alerts is wired through.
    """
    raise NotImplementedError(
        "A table of features for many transients is not wired up yet. "
        "For one light curve, call fade_rates, g_minus_r_at_time, "
        "and g_minus_r_rise "
        f"(this table has {len(light_curves)} rows)."
    )


def _fade_rate_for_one_band(
    points: pd.DataFrame,
    *,
    band: str,
    min_points: int,
    window_days: float,
) -> BandFade:
    """Fit one band's fade, using only the points after its own peak.

    The points before the peak are the rise. Mixing them into the line
    would make a fast fade look slower, so they are left out. The peak
    itself is kept: it is the start of the fade.
    """
    peak_time = peak_time_days(points, band)
    if peak_time is None:
        return BandFade(
            band=band,
            mag_per_day=None,
            explanation=(
                f"Not enough data: {band} has no magnitudes to fit a fade to."
            ),
        )

    # Every point in this band that has a time and a magnitude, including
    # the rise before the peak. The next lines keep the peak and the
    # points after it that fall inside the fade window.
    usable_points = _usable_points(points, band)
    window_end = peak_time + window_days
    # A tiny margin keeps the peak, and the point exactly at the end of
    # the window, when floating-point rounding nudges a time.
    in_window = usable_points.loc[
        (usable_points["time_days"] >= peak_time - _SAME_VISIT_DAYS)
        & (usable_points["time_days"] <= window_end + _SAME_VISIT_DAYS)
    ]
    times = in_window["time_days"].to_numpy(dtype=float)
    magnitudes = in_window["magnitude"].to_numpy(dtype=float)

    if len(times) < min_points:
        return BandFade(
            band=band,
            mag_per_day=None,
            explanation=(
                f"Not enough data: {band} has {len(times)} point(s) in the "
                f"{window_days:.1f} days after its peak, and a fade fit needs "
                f"at least {min_points}."
            ),
        )
    if float(times.max() - times.min()) <= _SAME_VISIT_DAYS:
        return BandFade(
            band=band,
            mag_per_day=None,
            explanation=(
                f"Not enough data: the {band} points after the peak are all "
                "at the same time, so there is no slope to measure."
            ),
        )

    # np.polyfit(times, magnitudes, 1) fits a straight line:
    #     magnitude = slope * time + intercept
    # The 1 means "a polynomial of degree 1", which is a straight line.
    # The result lists the slope first and the intercept second. We keep
    # the slope only. It is magnitudes per day.
    #
    # With two points the line goes through both. With more points it is
    # the straight line that passes as close as it can to all of them
    # (a least-squares fit). A positive slope means the magnitude is
    # climbing, and a climbing magnitude means the object is fading.
    slope = float(np.polyfit(times, magnitudes, 1)[0])
    if np.isnan(slope):
        return BandFade(
            band=band,
            mag_per_day=None,
            explanation=(
                f"Not enough data: the {band} magnitudes could not be fit "
                "with a straight line."
            ),
        )
    return BandFade(
        band=band,
        mag_per_day=slope,
        explanation=(
            f"The {band} band changed by {slope:.2f} magnitudes per day after its peak."
        ),
    )


def _magnitude_at_time(points: pd.DataFrame, when_days: float) -> float | None:
    """Read one band's magnitude at one time, or None if it is out of range.

    ``points`` must already be the usable rows for a single band.
    """
    ordered = points.sort_values("time_days")
    times = [float(value) for value in ordered["time_days"]]
    magnitudes = [float(value) for value in ordered["magnitude"]]

    for time_days, magnitude in zip(times, magnitudes, strict=True):
        if abs(time_days - when_days) <= _SAME_VISIT_DAYS:
            return magnitude

    if when_days < times[0] or when_days > times[-1]:
        return None

    earlier_time = times[0]
    earlier_magnitude = magnitudes[0]
    later_time = times[-1]
    later_magnitude = magnitudes[-1]
    for time_days, magnitude in zip(times, magnitudes, strict=True):
        if time_days > when_days:
            later_time = time_days
            later_magnitude = magnitude
            break
        earlier_time = time_days
        earlier_magnitude = magnitude

    gap = later_time - earlier_time
    if gap <= _SAME_VISIT_DAYS:
        return earlier_magnitude
    # 0 means "at the earlier point", 1 means "at the later point".
    fraction = (when_days - earlier_time) / gap
    return earlier_magnitude + fraction * (later_magnitude - earlier_magnitude)


def _visit_times(*bands: pd.DataFrame) -> list[float]:
    """Observation times from one or more bands, one entry per visit.

    Times closer than ``_SAME_VISIT_DAYS`` are the same visit. Keeping
    both would look like two colors when the telescope only looked once.
    """
    raw: list[float] = []
    for points in bands:
        raw.extend(float(value) for value in points["time_days"])
    raw.sort()
    visits: list[float] = []
    for time_days in raw:
        if visits and abs(time_days - visits[-1]) <= _SAME_VISIT_DAYS:
            continue
        visits.append(time_days)
    return visits


def _usable_points(light_curve: pd.DataFrame, band: str) -> pd.DataFrame:
    """Rows for one band that have both a time and a magnitude."""
    points = light_curve.loc[light_curve["filter"] == band]
    return points.dropna(subset=["time_days", "magnitude"])


def _require_light_curve(light_curve: pd.DataFrame) -> None:
    """Raise if the table is not shaped like a light curve."""
    missing = [name for name in _REQUIRED_COLUMNS if name not in light_curve.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "A light curve needs the columns time_days, filter, and magnitude. "
            f"Missing: {joined}."
        )


def _missing_band_sentence(g_missing: bool, r_missing: bool) -> str:
    """Explain which of g and r is absent."""
    if g_missing and r_missing:
        which = "g or r"
    elif g_missing:
        which = "g"
    else:
        which = "r"
    return (
        "Not enough data: g minus r needs both a g point and an r point, "
        f"and this curve has no {which} points."
    )


def _outside_coverage_sentence(
    time_days: float,
    g_points: pd.DataFrame | None,
    r_points: pd.DataFrame | None,
) -> str:
    """Explain that the requested time falls outside a band we do have."""
    parts: list[str] = []
    if g_points is not None:
        parts.append(f"g ({_coverage_phrase(g_points)})")
    if r_points is not None:
        parts.append(f"r ({_coverage_phrase(r_points)})")
    covered = " and ".join(parts)
    return (
        f"Not enough data: g minus r was needed at day {time_days:.2f}, "
        f"which is outside the measurements we have for {covered}."
    )


def _coverage_phrase(points: pd.DataFrame) -> str:
    """Describe the time range of one band in a few words."""
    first = float(points["time_days"].min())
    last = float(points["time_days"].max())
    if abs(first - last) <= _SAME_VISIT_DAYS:
        return f"only at day {first:.2f}"
    return f"from day {first:.2f} to day {last:.2f}"

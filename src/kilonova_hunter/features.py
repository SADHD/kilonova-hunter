"""Measure how one light curve fades, and what color it is.

Kilonova debris brightens within about a day, fades within days, and
turns red as it cools. This module measures two of those facts for a
single light curve:

- the fade rate in each band, as a straight line fitted after the
  brightest point
- g minus r at one chosen time

A negative fade rate means the object is still getting brighter,
because a smaller magnitude is brighter. Too few points returns
``None`` for the number and a sentence that says why, instead of
raising. Summarizing a whole night of objects into one table is still
``measure_features``, which is not wired up yet.
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
    # sort puts the earliest row first, so a tie keeps the earlier time.
    ordered = points.sort_values("time_days")
    brightest = ordered["magnitude"].idxmin()
    return float(ordered.loc[brightest, "time_days"])


def measure_features(light_curves: pd.DataFrame) -> pd.DataFrame:
    """Summarize every transient in a night with a rise, a fade, and a color.

    For one light curve, call ``fade_rates`` and ``g_minus_r_at_time``.
    Folding a whole downloaded night into one row per object is not
    wired up yet.

    Args:
        light_curves: Cleaned light curves, one row per observation.

    Returns:
        One row per transient, with its rates and color.

    Raises:
        NotImplementedError: Until a night of alerts is wired through.
    """
    raise NotImplementedError(
        "A table of features for many transients is not wired up yet. "
        "For one light curve, call fade_rates and g_minus_r_at_time "
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

    after_peak = _usable_points(points, band)
    window_end = peak_time + window_days
    # A tiny margin keeps the peak, and the point exactly at the end of
    # the window, when floating-point rounding nudges a time.
    in_window = after_peak.loc[
        (after_peak["time_days"] >= peak_time - _SAME_VISIT_DAYS)
        & (after_peak["time_days"] <= window_end + _SAME_VISIT_DAYS)
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

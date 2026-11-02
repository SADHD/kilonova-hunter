"""Measure how one light curve fades, and what color it is.

Kilonova debris brightens within about a day, fades within days, and
turns red as it cools. This module measures those facts for a single
light curve:

- the fade rate in each band, as a straight line fitted after the
  brightest point
- g minus r at one chosen time
- how much g minus r rises, from a straight line fitted to every
  color in the time window, trusting precise points more than noisy ones

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
        uncertainty: How unsure this color is, in magnitudes. It is
            built from the ``magnitude_error`` of the g and r points
            that produced it. ``None`` when those errors are missing,
            so a fit cannot give this color a weight. A larger number
            means a noisier measurement.
    """

    time_days: float
    g_minus_r: float
    uncertainty: float | None = None


@dataclass(frozen=True)
class ColorRise:
    """How much g minus r changed between two times.

    Attributes:
        rise: How many magnitudes the fitted line climbs from the first
            color time in the window to the last. This is the slope
            times that time span, which is the same as the fitted color
            at the last time minus the fitted color at the first time.
            ``None`` when there are not two colors spread across the
            window. A positive number means the object got redder. A
            negative number means it got bluer.
        earlier_time_days: When the first color in the fit was read, or
            ``None`` if no color could be read at all.
        later_time_days: When the last color in the fit was read, or
            ``None`` if the window did not hold a second color far
            enough after the first.
        earlier_g_minus_r: The fitted line's color at
            ``earlier_time_days``, or ``None``. This is the line, not
            necessarily the raw measurement at that time. A noisy point
            can sit off the line.
        later_g_minus_r: The fitted line's color at ``later_time_days``,
            or ``None``.
        color_count: How many times g minus r could be read, before
            the window is applied. Two inside the window, far enough
            apart, are required before a rise exists.
        explanation: One or two sentences. On success it states the
            rise. On failure it says there was not enough data.
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
        The colors, earliest first. Each one carries an uncertainty when
        the light curve has ``magnitude_error`` on the points that color
        used. Empty when g or r is missing, or when their times do not
        overlap.

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
        samples.append(
            ColorSample(
                time_days=time_days,
                g_minus_r=color.g_minus_r,
                uncertainty=_color_uncertainty(g_points, r_points, time_days),
            )
        )
    return samples


def g_minus_r_rise(
    light_curve: pd.DataFrame,
    thresholds: Thresholds | None = None,
) -> ColorRise:
    """Measure how much g minus r climbed inside the time window.

    Every color ``g_minus_r_over_time`` can read inside
    ``reddening_window_days`` of the earliest color is kept. A straight
    line is fitted to those colors: g minus r against time. The rise is
    the slope of that line times the time from the first color in the
    window to the last. That is the change the line itself predicts, not
    the raw first point minus the raw last point.

    A raw endpoint comparison is easy for one bad measurement to flip.
    The line has to agree with the colors in between as well. When every
    color has an uncertainty, the fit is weighted: ``numpy.polyfit`` is
    called with ``w = 1/sigma``, so a precise color pulls the line more
    than a noisy one. If any color is missing an uncertainty, or every
    uncertainty is zero (the noise-free demo), the points are weighted
    equally. Dividing by zero is not a weight.

    The line still has to cover at least ``min_color_separation_days``.
    Two snapshots of the same visit are not a change. If there are not
    two colors spread across that window, ``rise`` is ``None``. This
    function only measures. The filter decides whether the rise is large
    enough, and it skips the rule when ``rise`` is ``None``.

    Args:
        light_curve: One object, with the usual light-curve columns.
            ``magnitude_error`` is used when it is present.
        thresholds: The window and the minimum separation. Omit this to
            use ``THRESHOLDS`` from ``config.py``.

    Returns:
        The rise, the two times, the fitted colors at those times, how
        many colors could be read, and one or two sentences.

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

    earliest = samples[0]
    # Colors past the window are too late for this comparison. The last
    # color we keep is the far end of the window.
    window_end = earliest.time_days + settings.reddening_window_days
    in_window = [
        sample
        for sample in samples
        if sample.time_days <= window_end + _SAME_VISIT_DAYS
    ]
    span = in_window[-1].time_days - in_window[0].time_days
    # A span shorter than the minimum separation is still one moment,
    # even if several noisy points landed on it. A span of exactly the
    # minimum is kept: the color has had the time we asked for.
    too_short = span + _SAME_VISIT_DAYS < settings.min_color_separation_days
    if len(in_window) < 2 or span <= _SAME_VISIT_DAYS or too_short:
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

    times = np.array([sample.time_days for sample in in_window], dtype=float)
    colors = np.array([sample.g_minus_r for sample in in_window], dtype=float)
    uncertainties = [sample.uncertainty for sample in in_window]
    fitted = _fit_color_line(times, colors, uncertainties)
    if fitted is None:
        return ColorRise(
            rise=None,
            earlier_time_days=float(times[0]),
            later_time_days=None,
            earlier_g_minus_r=float(colors[0]),
            later_g_minus_r=None,
            color_count=len(samples),
            explanation=(
                "Not enough data: the g minus r measurements could not be "
                "fit with a straight line."
            ),
        )

    slope, intercept, weighted = fitted
    earlier_time = float(times[0])
    later_time = float(times[-1])
    # slope * span is the climb of the line. Writing it as the fitted
    # color at the last time minus the fitted color at the first time
    # is the same number, and it is what the sentence quotes.
    fitted_early = intercept + slope * earlier_time
    fitted_late = intercept + slope * later_time
    rise = fitted_late - fitted_early
    if weighted:
        weight_sentence = "Each color was weighted by one over its uncertainty."
    else:
        weight_sentence = (
            "The colors were weighted equally, because a usable "
            "uncertainty was not available on every color."
        )
    early_text = _format_mag(fitted_early)
    late_text = _format_mag(fitted_late)
    rise_text = _format_mag(rise)
    return ColorRise(
        rise=float(rise),
        earlier_time_days=earlier_time,
        later_time_days=later_time,
        earlier_g_minus_r=float(fitted_early),
        later_g_minus_r=float(fitted_late),
        color_count=len(samples),
        explanation=(
            f"From day {earlier_time:.2f} to day {later_time:.2f}, a "
            f"straight-line fit of g minus r goes from {early_text} "
            f"to {late_text}, a rise of {rise_text} mag. "
            f"The fit uses {len(in_window)} colors inside the "
            f"{settings.reddening_window_days:.2f}-day window. "
            f"{weight_sentence}"
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


def _color_uncertainty(
    g_points: pd.DataFrame,
    r_points: pd.DataFrame,
    when_days: float,
) -> float | None:
    """Uncertainty on g minus r at one time, or None if it cannot be built.

    g minus r is a difference of two magnitudes. When the two measurements
    are independent, the uncertainties combine in quadrature: the square
    root of the sum of the squares. The color is then at least as
    uncertain as either band, and a noisy band makes the color noisy too.
    """
    g_error = _magnitude_uncertainty_at_time(g_points, when_days)
    r_error = _magnitude_uncertainty_at_time(r_points, when_days)
    if g_error is None or r_error is None:
        return None
    return float(np.hypot(g_error, r_error))


def _magnitude_uncertainty_at_time(
    points: pd.DataFrame,
    when_days: float,
) -> float | None:
    """Uncertainty on one band at one time, following ``_magnitude_at_time``.

    ``points`` must already be the usable rows for a single band. Returns
    ``None`` when the table has no ``magnitude_error`` column, or when the
    points this reading depends on do not have a usable error. A returned
    ``0.0`` means the table said that magnitude is exact.

    An exact visit uses that row's error. A time between two visits uses
    the same mix as the interpolated magnitude: if the magnitude is
    ``(1 - f) * earlier + f * later``, the uncertainty is the matching
    combination of the two errors, added in quadrature.
    """
    if "magnitude_error" not in points.columns:
        return None

    ordered = points.sort_values("time_days")
    times = [float(value) for value in ordered["time_days"]]
    # errors="coerce" turns a word in the error column into "missing"
    # instead of crashing the fit. A missing error means this color
    # cannot be weighted, and the line falls back to equal weights.
    numeric_errors = pd.to_numeric(ordered["magnitude_error"], errors="coerce")
    errors = [float(value) for value in numeric_errors]

    for time_days, error in zip(times, errors, strict=True):
        if abs(time_days - when_days) <= _SAME_VISIT_DAYS:
            if not np.isfinite(error) or error < 0:
                return None
            return error

    if when_days < times[0] or when_days > times[-1]:
        return None

    earlier_time = times[0]
    earlier_error = errors[0]
    later_time = times[-1]
    later_error = errors[-1]
    for time_days, error in zip(times, errors, strict=True):
        if time_days > when_days:
            later_time = time_days
            later_error = error
            break
        earlier_time = time_days
        earlier_error = error

    if not _usable_error(earlier_error) or not _usable_error(later_error):
        return None
    gap = later_time - earlier_time
    if gap <= _SAME_VISIT_DAYS:
        return earlier_error
    # 0 means "at the earlier point", 1 means "at the later point".
    fraction = (when_days - earlier_time) / gap
    # (1 - f) of the earlier error and f of the later error. hypot is
    # the square root of the sum of squares, which is how independent
    # uncertainties combine.
    return float(np.hypot((1.0 - fraction) * earlier_error, fraction * later_error))


def _usable_error(error: float) -> bool:
    """True when an error can be used as a weight: finite and not negative."""
    return bool(np.isfinite(error) and error >= 0)


def _fit_color_line(
    times: np.ndarray,
    colors: np.ndarray,
    uncertainties: list[float | None],
) -> tuple[float, float, bool] | None:
    """Fit ``color = slope * time + intercept``.

    Returns ``(slope, intercept, weighted)``. ``weighted`` is True when
    the fit used ``w = 1/sigma``. It is False when the points were
    weighted equally. Returns ``None`` when the fit does not produce a
    finite line.

    ``numpy.polyfit`` treats ``w`` as a multiplier on each point, and the
    statistical choice is ``w = 1/sigma``: a small uncertainty becomes a
    large weight, so that point pulls the line harder. A missing
    uncertainty, a negative one, or a set of uncertainties that are all
    zero cannot be turned into weights. Zero is the noise-free demo,
    which is exact on purpose. Those curves get an ordinary unweighted
    line. A zero mixed in with real uncertainties is replaced by a tiny
    floor, so the exact point is kept and trusted much more than a noisy
    neighbor, without dividing by zero.
    """
    weights = _polyfit_weights(uncertainties)
    try:
        if weights is None:
            slope, intercept = np.polyfit(times, colors, 1)
        else:
            slope, intercept = np.polyfit(times, colors, 1, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None
    slope = float(slope)
    intercept = float(intercept)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return slope, intercept, weights is not None


def _format_mag(value: float) -> str:
    """Format a magnitude to two decimal places, without a negative zero.

    A perfectly flat line can come back from the fit as a tiny negative
    number, around -0.0000000000000002. Rounded to two decimals that is
    still zero, and the sentence should say 0.00 rather than -0.00.
    """
    text = f"{value:.2f}"
    if text == "-0.00":
        return "0.00"
    return text


def _polyfit_weights(uncertainties: list[float | None]) -> np.ndarray | None:
    """Return ``1/sigma`` weights, or None to ask for an unweighted fit."""
    if any(uncertainty is None for uncertainty in uncertainties):
        return None
    sigmas = np.array(uncertainties, dtype=float)
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas < 0):
        return None
    if np.all(sigmas == 0.0):
        return None
    # 1e-6 mag is far smaller than any real survey error, so a point the
    # table called exact still dominates, and 1/sigma stays finite.
    safe = np.maximum(sigmas, 1e-6)
    return 1.0 / safe


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

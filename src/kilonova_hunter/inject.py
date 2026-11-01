"""Fake kilonovae, used to measure what the filter would catch.

Sprint 4 will build these from published redback models and sample them at
the times Rubin actually observed, so the recall number means something.
redback is not installed yet. Until then, ``simulate_kilonova_light_curve``
draws a simple curve with the same rough shape: a one-day rise, a week-long
fade, and a color that turns red. Tests and the demo use that curve.
"""

import numpy as np
import pandas as pd

# Brightest about one day after the explosion. A real kilonova rises
# on that kind of timescale, which is fast for a supernova.
PEAK_DAY = 1.0

# Scatter added to every point, in magnitudes. Small on purpose: the rise
# and the fade should still be obvious, the way a real detection with a
# modest error bar is.
NOISE_MAG = 0.05

# How fast the demo fades in the r band after the peak, in magnitudes
# per day. A larger number means the object gets fainter faster.
#
# 0.45 clears the v1 rule of faster than 0.3 mag/day, and it also sits
# above the stricter r-band cut of about 0.39 mag/day from the ZTFReST
# model grid (Andreoni et al. 2020, quoted in arXiv:2104.06352). The gap
# above 0.39 leaves room for noise, so a noisy slope fitted to this
# curve can still come out faster than that cut. How fast AT2017gfo
# itself faded is approximate, and still to be confirmed by the science
# lead against the 2017 photometry.
R_FADE_PER_DAY = 0.45

# How fast the demo fades in the g band after the peak, in magnitudes
# per day. This is faster than the r band, so g minus r grows and the
# object turns red.
#
# Those same ZTFReST model cuts expect g to fade faster than r: about
# 0.57 mag/day in g against about 0.39 in r. A GW170817-like kilonova
# goes from blue to red within about 1 to 2 days, so this toy uses 0.9,
# inside the 0.8 to 1.0 range, and starts only slightly blue. The day
# the color crosses to red is approximate, pending the science lead's
# check against AT2017gfo photometry.
G_FADE_PER_DAY = 0.90

# peak_mag is the brightness at PEAK_DAY, before noise. g peaks 0.1 mag
# brighter than r, so g minus r is slightly negative there (slightly blue).
# rise_per_day and fade_per_day are positive magnitudes of fading per day
# on either side of the peak (the object is fainter as you move away).
# g fades faster than r, so g minus r grows and the object turns red.
# That reddening is the cooling debris.
_FILTERS: dict[str, dict[str, float]] = {
    "g": {"peak_mag": 19.5, "rise_per_day": 1.0, "fade_per_day": G_FADE_PER_DAY},
    "r": {"peak_mag": 19.6, "rise_per_day": 1.3, "fade_per_day": R_FADE_PER_DAY},
}


def _brightness_without_noise(
    time_days: np.ndarray,
    peak_mag: float,
    rise_per_day: float,
    fade_per_day: float,
) -> np.ndarray:
    """Return the smooth light curve, before noise is added.

    On both sides of the peak the magnitude moves toward fainter values.
    Before the peak that fading-with-distance is the rise seen backwards:
    walking toward the peak, the object gets brighter.

    Args:
        time_days: Days since the explosion.
        peak_mag: Magnitude at ``PEAK_DAY``. A smaller number is brighter.
        rise_per_day: How many magnitudes fainter it was per day before the peak.
        fade_per_day: How many magnitudes fainter it gets per day after the peak.

    Returns:
        One magnitude for each entry in ``time_days``.
    """
    days_from_peak = time_days - PEAK_DAY
    # The rise is steeper than the fade, so the two sides use different rates.
    rate = np.where(days_from_peak < 0.0, rise_per_day, fade_per_day)
    return peak_mag + rate * np.abs(days_from_peak)


def simulate_kilonova_light_curve(
    *,
    duration_days: float = 10.0,
    step_days: float = 0.5,
    seed: int = 42,
    noise_mag: float | None = None,
) -> pd.DataFrame:
    """Build a made-up kilonova light curve in the g and r filters.

    The curve brightens for about a day and fades for about a week. At the
    peak, g minus r is slightly negative (a bit blue). g then fades faster
    than r, so the color turns red. On this toy curve that crossing falls
    about one to two days after the explosion. The crossing day is
    approximate, pending the science lead's check against AT2017gfo
    photometry. The same ``seed`` always produces the same noise, so a plot
    you make today matches a plot you make tomorrow.

    Args:
        duration_days: How long to follow the object, counting from the explosion.
        step_days: Time between observations. The default is a smooth teaching
            curve, not Rubin's real cadence.
        seed: Starting point for the random noise.
        noise_mag: How wide the random scatter is, in magnitudes. Omit
            this to use ``NOISE_MAG``. Pass ``0`` for the smooth curve
            with the noise left off, which is what the filter tests check.

    Returns:
        One row per observation, with columns ``time_days``, ``filter``,
        ``magnitude``, and ``magnitude_error``. A smaller magnitude is brighter.

    Raises:
        ValueError: If the duration is too short to show the fade, the
            step is not a positive number of days, or ``noise_mag`` is
            negative.
    """
    # The fade is the part a kilonova filter cares about, so a curve that
    # ends at the peak is not a useful example.
    if duration_days <= PEAK_DAY:
        raise ValueError(
            "duration_days must be longer than the 1-day rise so the fade is visible, "
            f"got {duration_days}."
        )
    if step_days <= 0:
        raise ValueError(
            f"step_days must be a positive number of days, got {step_days}."
        )
    scatter = NOISE_MAG if noise_mag is None else noise_mag
    if scatter < 0:
        raise ValueError(
            f"noise_mag must be zero or a positive number of magnitudes, got {scatter}."
        )

    # Half a step past the end makes the last day included when it lands on
    # the grid. The comparison then drops anything floating-point rounding
    # pushed past duration_days.
    times = np.arange(0.0, duration_days + step_days * 0.5, step_days)
    times = times[times <= duration_days + 1e-9]

    # default_rng is the modern NumPy generator. A fixed seed makes the
    # "random" noise repeatable, which is what the tests and the docs plot rely on.
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for filter_name, params in _FILTERS.items():
        smooth = _brightness_without_noise(
            times,
            peak_mag=params["peak_mag"],
            rise_per_day=params["rise_per_day"],
            fade_per_day=params["fade_per_day"],
        )
        # Scale zero would also draw zeros, but writing the smooth case
        # out makes it obvious the noise was left off on purpose.
        if scatter == 0:
            noise = np.zeros(times.size)
        else:
            noise = rng.normal(0.0, scatter, size=times.size)
        frames.append(
            pd.DataFrame(
                {
                    "time_days": times,
                    "filter": filter_name,
                    "magnitude": smooth + noise,
                    "magnitude_error": scatter,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def inject_fake_kilonovae(
    observation_times: pd.DataFrame,
    *,
    count: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Build fake kilonovae on the times and filters of real Rubin observations.

    Sprint 4 uses redback for the physical model and adds realistic noise.
    ``simulate_kilonova_light_curve`` is the stand-in you can run today.

    Args:
        observation_times: Real visit times and filters to sample the fakes at.
        count: How many fake events to build. The recall test wants 200 or more.
        seed: Starting point for the random draws.

    Returns:
        Light curves for the fake events, in the same columns as a real fetch.

    Raises:
        NotImplementedError: Until Sprint 4 installs redback.
    """
    raise NotImplementedError(
        "Physical fake kilonovae arrive in Sprint 4 with redback "
        f"({count} events, seed {seed}, {len(observation_times)} real timestamps). "
        "Use simulate_kilonova_light_curve for a made-up curve until then."
    )

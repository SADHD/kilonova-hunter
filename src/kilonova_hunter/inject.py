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
# 0.4 sits a comfortable step above the v1 rule of "faster than 0.3
# magnitudes per day". AT2017gfo, the kilonova that came with the
# neutron-star merger GW170817, faded faster than that in r over its
# first week (about a magnitude per day near the start). Published
# searches such as ZTFReST use the same idea: keep a candidate that
# fades at least 0.3 magnitudes per day, and set aside slower
# supernovae. The g band below is a little faster than this r value,
# so g minus r still grows and the source turns red.
R_FADE_PER_DAY = 0.40

# peak_mag is the brightness at PEAK_DAY, before noise.
# rise_per_day and fade_per_day are positive magnitudes of fading per day
# on either side of the peak (the object is fainter as you move away).
# g is a bluer filter and fades faster than r, so g minus r grows with
# time and the object looks redder. That reddening is the cooling debris.
_FILTERS: dict[str, dict[str, float]] = {
    "g": {"peak_mag": 19.0, "rise_per_day": 1.0, "fade_per_day": 0.50},
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
) -> pd.DataFrame:
    """Build a made-up kilonova light curve in the g and r filters.

    The curve brightens for about a day, fades for about a week, and turns
    redder as the g filter fades faster than r. The same ``seed`` always
    produces the same noise, so a plot you make today matches a plot you
    make tomorrow.

    Args:
        duration_days: How long to follow the object, counting from the explosion.
        step_days: Time between observations. The default is a smooth teaching
            curve, not Rubin's real cadence.
        seed: Starting point for the random noise.

    Returns:
        One row per observation, with columns ``time_days``, ``filter``,
        ``magnitude``, and ``magnitude_error``. A smaller magnitude is brighter.

    Raises:
        ValueError: If the duration is too short to show the fade, or the
            step is not a positive number of days.
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
        noise = rng.normal(0.0, NOISE_MAG, size=times.size)
        frames.append(
            pd.DataFrame(
                {
                    "time_days": times,
                    "filter": filter_name,
                    "magnitude": smooth + noise,
                    "magnitude_error": NOISE_MAG,
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

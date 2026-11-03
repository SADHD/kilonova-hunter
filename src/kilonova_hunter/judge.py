"""Whether a fake is judgeable, using only the survey and the signal-to-noise.

This definition is frozen. It does not read the filter cuts in
``config.py``. A later change to a color cut or a fade cut must not
change who counts as judgeable.

A fake is judgeable by day D, counted from the merger, when both of
these are true on the points with time at most D:

- at least two detections in the same band, each at signal-to-noise
  of 5 or more, and at least 0.5 days apart
- at least one night that has both a g detection and an r detection,
  each at signal-to-noise of 5 or more

For the simulated Rubin fakes, a night is ``floor(MJD - 0.5)``. That
integer changes at noon UTC. It is not the integer MJD, which would
change at midnight, and it is not a local-noon clock at the telescope.
ZTF alert nights use a different rollover, documented in ``alerts.py``.

Signal-to-noise comes from the magnitude error stored on the point.
The error is about ``1.0857 / signal-to-noise``, so a point whose error
is exactly the 5-sigma value has signal-to-noise 5 and counts. A point
just noisier than that does not.

Every row with time after day D is ignored, including a non-detection
or a forced-photometry point if one is present. The simulated tables
usually contain only detections, but the cut does not look at the kind
of row. It drops everything later than D.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 2.5 / ln(10). Divide a magnitude error by this and invert it to get
# the signal-to-noise ratio. At the 5-sigma limit the error is about 0.217 mag.
SNR_PER_INVERSE_ERROR = 2.5 / np.log(10.0)

# A detection at exactly this signal-to-noise counts. One just below does not.
MIN_SIGNAL_TO_NOISE = 5.0

# Two detections in one band count as a pair when they are at least this
# far apart. Exactly 0.5 days counts.
MIN_SAME_BAND_SEPARATION_DAYS = 0.5

# Days the harness asks about. Each one is recomputed from scratch,
# using only the rows at or before that day.
JUDGE_DAYS = (3.0, 5.0, 7.0)

# A time that lands exactly on day D is kept. This absorbs the tiny
# rounding error from adding floats, and nothing more.
_DAY_EDGE = 1e-9

# Same idea for the 0.5 day gap and for signal-to-noise of exactly 5.
_COMPARE_EDGE = 1e-9


def signal_to_noise(magnitude_error: np.ndarray | float) -> np.ndarray:
    """Signal-to-noise ratio implied by a magnitude error.

    An error of zero would be a perfect measurement, so the ratio is
    infinite and the point counts. A missing or negative error is not
    a detection: the ratio is zero and the point does not count.
    """
    error = np.asarray(magnitude_error, dtype=float)
    ratio = np.zeros(error.shape, dtype=float)
    perfect = np.isfinite(error) & (error == 0.0)
    usable = np.isfinite(error) & (error > 0.0)
    ratio[perfect] = np.inf
    ratio[usable] = SNR_PER_INVERSE_ERROR / error[usable]
    return ratio


def rubin_night(mjd: np.ndarray | float) -> np.ndarray:
    """Night index for a simulated Rubin visit.

    ``floor(MJD - 0.5)`` changes when MJD is n + 0.5, which is noon UTC.
    Two visits share a night when this integer is the same. G and r
    taken 30 minutes apart share a night unless that pair sits on top
    of noon.
    """
    values = np.asarray(mjd, dtype=float)
    return np.floor(values - 0.5)


def is_judgeable(
    light_curve: pd.DataFrame,
    day: float,
    *,
    merger_mjd: float,
) -> bool:
    """True when the fake can be judged by ``day`` days after the merger.

    ``merger_mjd`` is the merger time as a Modified Julian Date. Visit
    times on the light curve are days since that merger, so the night
    of a visit is ``rubin_night(merger_mjd + time_days)``.

    The filter cuts are not an input. Passing a different color cut
    cannot change this answer.
    """
    if light_curve is None or light_curve.empty:
        return False
    times = light_curve["time_days"].to_numpy(dtype=float)
    # Drop everything after day D before looking at signal-to-noise.
    early = times <= float(day) + _DAY_EDGE
    if not np.any(early):
        return False
    kept = light_curve.loc[early]
    ratio = signal_to_noise(kept["magnitude_error"].to_numpy(dtype=float))
    loud = ratio + _COMPARE_EDGE >= MIN_SIGNAL_TO_NOISE
    if not np.any(loud):
        return False
    loud_rows = kept.loc[loud]
    visit_times = loud_rows["time_days"].to_numpy(dtype=float)
    bands = loud_rows["filter"].astype(str).to_numpy()
    if not _same_band_pair(visit_times, bands):
        return False
    return _night_has_g_and_r(visit_times, bands, merger_mjd)


def _same_band_pair(times: np.ndarray, bands: np.ndarray) -> bool:
    """True when one band has two points at least 0.5 days apart."""
    for band in set(bands.tolist()):
        band_times = np.sort(times[bands == band])
        if len(band_times) < 2:
            continue
        span = float(band_times[-1] - band_times[0])
        if span + _COMPARE_EDGE >= MIN_SAME_BAND_SEPARATION_DAYS:
            return True
    return False


def _night_has_g_and_r(
    times: np.ndarray,
    bands: np.ndarray,
    merger_mjd: float,
) -> bool:
    """True when some night has both a g point and an r point."""
    nights = rubin_night(float(merger_mjd) + times)
    g_nights = set(nights[bands == "g"].tolist())
    r_nights = set(nights[bands == "r"].tolist())
    return bool(g_nights & r_nights)


# Candidate definition (b). Not official. It does not replace
# ``is_judgeable``. One day after the r-band peak, and
# the color may be read across a gap of at most one day.
CANDIDATE_COLOR_DAYS_AFTER_R_PEAK = 1.0
CANDIDATE_COLOR_MAX_GAP_DAYS = 1.0


def is_candidate_judgeable(
    light_curve: pd.DataFrame,
    day: float,
    *,
    merger_mjd: float,
) -> bool:
    """Stricter candidate for who counts as judgeable. Not the official rule.

    Official definition (a) is ``is_judgeable`` and is unchanged. This
    candidate also requires a usable g minus r at the r-band peak plus
    1 day, using only points at or before day D. The two points the
    color is read between must be at most 1.0 day apart. A wider gap is
    no data, and then this candidate says the fake does not count.

    This is not the official definition.
    """
    if not is_judgeable(light_curve, day, merger_mjd=merger_mjd):
        return False
    from kilonova_hunter.features import g_minus_r_at_time, peak_time_days

    times = light_curve["time_days"].to_numpy(dtype=float)
    visible = light_curve.loc[times <= float(day) + _DAY_EDGE]
    peak = peak_time_days(visible, "r")
    if peak is None:
        return False
    when = float(peak) + CANDIDATE_COLOR_DAYS_AFTER_R_PEAK
    if when > float(day) + _DAY_EDGE:
        return False
    color = g_minus_r_at_time(
        visible,
        when,
        max_gap_days=CANDIDATE_COLOR_MAX_GAP_DAYS,
    )
    return color.g_minus_r is not None

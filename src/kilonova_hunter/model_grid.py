"""The two-component 6-day cap.

redback 1.20.0 ``two_component_kilonova_model`` builds its internal
time grid only out to 6 days in the source frame (``86400 * 6``
seconds). A later requested time is filled with the magnitude at that
last grid point, so the light curve freezes. ``realistic_magnitudes``
and the legacy ``redback_magnitudes`` both call that model, through
``two_component_bns_ejecta_relation`` or directly. This module turns
those later times into NaN. A NaN is not a detection.

The three-component model used for the 2017-like set and the exact
copies is built out to ``7e6`` seconds, about 81 days. Day 7 and day
10 on that path are real outputs. That path is not capped.

Public model functions already evaluate the diffusion integral on
their own dense grid, starting at 0.01 seconds. The harness calls
those public functions. It does not call ``_one_component_kilonova_model``
on the visit times.
"""

from __future__ import annotations

import numpy as np

# redback two_component_kilonova_model, source-frame seconds 86400 * 6.
TWO_COMPONENT_SOURCE_FRAME_DAYS = 6.0

# Kinds whose light curves come from the two-component model.
TWO_COMPONENT_KINDS = frozenset({"realistic", "population", "at2017gfo"})

# Printed next to realistic day-7 numbers. The day-7 window still asks
# "by day 7", but the model has nothing after source-frame day 6.
REALISTIC_DAY7_LABEL = (
    "Realistic day 7 uses only data through 6 days in the source frame. "
    "redback's two-component model has no output after that, so a later "
    "time is no detection, not a frozen magnitude."
)


def observer_cap_days(redshift: float) -> float:
    """Observer-frame days that match 6 days in the source frame."""
    return TWO_COMPONENT_SOURCE_FRAME_DAYS * (1.0 + float(redshift))


def uses_two_component_model(kind: str) -> bool:
    """True for the realistic set and the legacy two-component fake."""
    return str(kind) in TWO_COMPONENT_KINDS


def nan_after_two_component_cap(
    times: np.ndarray,
    magnitudes: np.ndarray,
    redshift: float,
) -> np.ndarray:
    """NaN every magnitude later than source-frame day 6.

    Times at the cap stay. ``observe_visits`` drops a NaN, so the
    survey records no detection there.
    """
    times = np.asarray(times, dtype=float)
    out = np.array(magnitudes, dtype=float, copy=True)
    cap = observer_cap_days(redshift)
    out[times > cap + 1e-9] = np.nan
    return out


def drop_after_two_component_cap(curve, redshift: float):
    """Drop light-curve rows later than source-frame day 6.

    ``curve`` is a table with ``time_days``. Rows at or before the cap
    stay. Day-7 realistic scoring uses this cut, so a cached point past
    the cap cannot count.
    """
    if curve is None or len(curve) == 0 or "time_days" not in curve.columns:
        return curve
    cap = observer_cap_days(redshift)
    times = curve["time_days"].to_numpy(dtype=float)
    return curve.loc[times <= cap + 1e-9].reset_index(drop=True)

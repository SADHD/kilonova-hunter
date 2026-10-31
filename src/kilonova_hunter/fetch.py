"""Pull Rubin Observatory alerts and their light curves.

Rubin sends up to millions of public alerts a night. A broker (Fink, with
ALeRCE as the backup) is the service that hands us a slice of that stream
we can actually download. Sprint 1 fills this module in. The Fink and
ALeRCE clients are not installed yet.
"""

import datetime as dt

import pandas as pd


def fetch_light_curves(
    start: dt.date,
    end: dt.date,
    *,
    broker: str = "fink",
) -> pd.DataFrame:
    """Download light curves for every new transient in a date range.

    A second call for the same range will read the saved download in
    ``data/`` instead of hitting the network again. An empty night or a
    broker outage will return a clear message rather than a crash.

    The star in the signature means ``broker`` has to be written by name
    (``broker="alerce"``). That keeps a date from landing in the broker
    slot by accident.

    Args:
        start: First night to include.
        end: Last night to include.
        broker: ``"fink"`` (the default) or ``"alerce"``.

    Returns:
        One row per observation, including the light-curve columns
        described in the README (``time_days``, ``filter``, ``magnitude``,
        ``magnitude_error``) plus an ``object_id`` for each transient.

    Raises:
        NotImplementedError: Until the Sprint 1 broker client lands.
    """
    # Mention the arguments in the message so a caller can see which
    # request is still waiting on Sprint 1.
    raise NotImplementedError(
        f"Fetching {broker} alerts from {start.isoformat()} to {end.isoformat()} "
        "arrives in Sprint 1. Until then, run `kilohunt demo` for a made-up "
        "light curve."
    )

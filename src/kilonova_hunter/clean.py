"""Drop bad measurements and put brightness into magnitudes.

A broker light curve mixes real detections with upper limits, flagged
pixels, and the occasional missing value. Fitting a fade rate through
those points invents explosions that are not there. Sprint 2 does the
cleaning, and saves a before-and-after plot so you can see what was removed.
"""

import pandas as pd


def clean_light_curves(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a light-curve table that is safe to measure.

    Drops flagged and missing points, converts whatever units the broker
    used into magnitudes, and keeps only real detections.

    Args:
        raw: Light curves as ``fetch_light_curves`` returned them.

    Returns:
        A table with the same light-curve columns, minus the bad rows.

    Raises:
        NotImplementedError: Until Sprint 2.
    """
    raise NotImplementedError(
        "Cleaning light curves arrives in Sprint 2 "
        f"(this table has columns {list(raw.columns)})."
    )

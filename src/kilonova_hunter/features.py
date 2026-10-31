"""Measure how each transient changes: rise, fade, and color.

Kilonova debris brightens within about a day, fades within days, and turns
red as it cools. These three numbers are the evidence the filter reads.
Sprint 2 fits them. Rates are in magnitudes per day, and a negative rate
means the object is getting brighter, because a smaller magnitude is brighter.
"""

import pandas as pd


def measure_features(light_curves: pd.DataFrame) -> pd.DataFrame:
    """Summarize every transient with a rise rate, a fade rate, and a color.

    Rise and fade are fitted per filter. Color is the magnitude difference
    between two filters taken close together in time (``g - r`` growing
    means the object is turning redder). A transient that was never seen
    in the second filter still gets a row; its color is left blank instead
    of raising an error.

    Args:
        light_curves: Cleaned light curves, one row per observation.

    Returns:
        One row per transient, with its rates and color.

    Raises:
        NotImplementedError: Until Sprint 2.
    """
    raise NotImplementedError(
        "Rise, fade, and color arrive in Sprint 2 "
        f"(this table has {len(light_curves)} rows)."
    )

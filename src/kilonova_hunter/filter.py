"""Rules that decide whether a transient looks like a kilonova.

The module is named ``filter`` because it narrows a night of alerts down
to a short candidate list. It has nothing to do with Python's built-in
``filter`` function.

Each rule is a yes-or-no check plus one sentence a newcomer can read, for
example "faded faster than 0.3 magnitudes per day, which is what the debris
of a neutron-star merger does." That 0.3 is
``FADE_FASTER_THAN_MAG_PER_DAY``. Sprint 3 chooses the thresholds from that
physics and records, for every candidate, which rules passed and which failed.
"""

import pandas as pd

# A candidate has to fade faster than this, in magnitudes per day.
# 0.3 is the v1 cut. ZTFReST applies it in any one of the g, r, or i
# filters, not in r alone: fading at least this fast in one of those
# bands keeps the candidate. How fast AT2017gfo itself faded is
# approximate, including near the start, and still to be confirmed by
# the science lead against the 2017 photometry. Sprint 3 will apply
# this number as a yes-or-no check.
FADE_FASTER_THAN_MAG_PER_DAY = 0.3


def score_candidates(features: pd.DataFrame) -> pd.DataFrame:
    """Score each transient and keep the reason for every rule.

    Args:
        features: One row per transient, as ``measure_features`` returns.

    Returns:
        The same rows, plus a score and a plain-language reason for each
        rule the candidate passed or failed.

    Raises:
        NotImplementedError: Until Sprint 3.
    """
    raise NotImplementedError(
        f"The kilonova rules arrive in Sprint 3 (this table has {len(features)} rows)."
    )

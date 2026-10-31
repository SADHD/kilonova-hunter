"""Rules that decide whether a transient looks like a kilonova.

The module is named ``filter`` because it narrows a night of alerts down
to a short candidate list. It has nothing to do with Python's built-in
``filter`` function.

Each rule is a yes-or-no check plus one sentence a newcomer can read, for
example "faded faster than 0.3 magnitudes per day, which is what the debris
of a neutron-star merger does." Sprint 3 chooses the thresholds from that
physics and records, for every candidate, which rules passed and which failed.
"""

import pandas as pd


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

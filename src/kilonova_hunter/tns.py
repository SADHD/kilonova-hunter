"""Confirmed types from the Transient Name Server.

The Transient Name Server (TNS) is the official registry of what each
transient turned out to be. Sprint 5 downloads those labels and writes
them onto the scorecard, so an object our rules liked that astronomers
classified as something else is counted as an impostor. Search results
come down as a CSV. This module only reads; it never submits a report.
"""

import pandas as pd


def lookup_classifications(object_names: list[str]) -> pd.DataFrame:
    """Look up the confirmed type of each named transient.

    An object the TNS has never heard of comes back unlabeled, which the
    scorecard reports as still unidentified. That is a normal result, not
    an error: most Rubin transients are not classified by people.

    Args:
        object_names: Names to look up, such as TNS internal names or
            broker object ids once we know how the two catalogs join.

    Returns:
        One row per name, with the reported type when TNS has one.

    Raises:
        NotImplementedError: Until Sprint 5.
    """
    raise NotImplementedError(
        "Transient Name Server lookups arrive in Sprint 5 "
        f"(requested {len(object_names)} object names)."
    )

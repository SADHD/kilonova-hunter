"""The frozen judgeable rule, including the edge cases from the decision log."""

import pandas as pd

from kilonova_hunter.judge import (
    MIN_SIGNAL_TO_NOISE,
    SNR_PER_INVERSE_ERROR,
    is_candidate_judgeable,
    is_judgeable,
    rubin_night,
)


def _curve(rows: list[tuple[float, str, float]]) -> pd.DataFrame:
    """Rows are (time_days, band, magnitude_error). Magnitude is a bright 18."""
    return pd.DataFrame(
        {
            "time_days": [row[0] for row in rows],
            "filter": [row[1] for row in rows],
            "magnitude": [18.0] * len(rows),
            "magnitude_error": [row[2] for row in rows],
        }
    )


def _error_for_snr(snr: float) -> float:
    return SNR_PER_INVERSE_ERROR / snr


def test_exactly_half_a_day_apart_and_snr_of_exactly_five_is_judgeable() -> None:
    """0.5 days and signal-to-noise 5 are enough. Both are inclusive."""
    error = _error_for_snr(MIN_SIGNAL_TO_NOISE)
    # Merger at MJD 60000. g and r at day 0.10 share a night. The second
    # g is exactly 0.5 days later.
    curve = _curve(
        [
            (0.10, "g", error),
            (0.10, "r", error),
            (0.60, "g", error),
        ]
    )
    assert is_judgeable(curve, 3.0, merger_mjd=60000.0) is True


def test_just_under_half_a_day_or_just_under_snr_five_is_not_judgeable() -> None:
    """0.49 days is too close. Signal-to-noise 4.9 is too faint."""
    loud = _error_for_snr(10.0)
    close = _curve(
        [
            (0.10, "g", loud),
            (0.10, "r", loud),
            (0.59, "g", loud),
        ]
    )
    assert is_judgeable(close, 3.0, merger_mjd=60000.0) is False
    quiet = _error_for_snr(4.9)
    faint = _curve(
        [
            (0.10, "g", quiet),
            (0.10, "r", loud),
            (1.00, "g", loud),
        ]
    )
    # The second g is loud, but the first g is not, so the pair is gone.
    assert is_judgeable(faint, 3.0, merger_mjd=60000.0) is False


def test_g_and_r_on_different_nights_is_not_judgeable() -> None:
    """A same-band pair is not enough when g and r never share a night.

    Noon UTC is the boundary: floor(MJD - 0.5). With the merger at
    MJD 60000, day 0.10 and day 0.60 are different nights.
    """
    loud = _error_for_snr(10.0)
    curve = _curve(
        [
            (0.10, "g", loud),
            (1.60, "g", loud),
            (0.60, "r", loud),
        ]
    )
    assert is_judgeable(curve, 3.0, merger_mjd=60000.0) is False
    # Put r on the first g's night and it becomes judgeable.
    together = _curve(
        [
            (0.10, "g", loud),
            (0.10, "r", loud),
            (1.20, "g", loud),
        ]
    )
    assert is_judgeable(together, 3.0, merger_mjd=60000.0) is True


def test_day_3_and_day_7_are_recomputed_from_the_rows_up_to_that_day() -> None:
    """A pair that arrives on day 6 does not make the fake judgeable at day 3."""
    loud = _error_for_snr(10.0)
    curve = _curve(
        [
            (0.20, "g", loud),
            (0.20, "r", loud),
            (6.00, "g", loud),
        ]
    )
    assert is_judgeable(curve, 3.0, merger_mjd=60000.0) is False
    assert is_judgeable(curve, 7.0, merger_mjd=60000.0) is True


def test_rubin_night_changes_at_noon_utc() -> None:
    """floor(MJD - 0.5) steps at half-days, which is noon UTC."""
    assert int(rubin_night(60000.5)) == 60000
    assert int(rubin_night(60000.5 - 1e-6)) == 59999
    assert int(rubin_night(60000.0)) == 59999


def test_candidate_judgeable_needs_a_color_one_day_after_the_r_peak() -> None:
    """(b) is stricter than (a), and (a) itself does not change.

    Both curves are officially judgeable. Only the one with g and r at
    the r peak plus 1 day, inside the 1.0-day gap, counts under (b).
    """
    loud = _error_for_snr(10.0)
    official_only = _curve(
        [
            (0.2, "g", loud),
            (0.2, "r", loud),
            (1.7, "g", loud),
        ]
    )
    with_color = _curve(
        [
            (0.2, "g", loud),
            (0.2, "r", loud),
            (1.2, "g", loud),
            (1.2, "r", loud),
        ]
    )
    assert is_judgeable(official_only, 3.0, merger_mjd=60000.0) is True
    assert is_candidate_judgeable(official_only, 3.0, merger_mjd=60000.0) is False
    assert is_judgeable(with_color, 3.0, merger_mjd=60000.0) is True
    assert is_candidate_judgeable(with_color, 3.0, merger_mjd=60000.0) is True

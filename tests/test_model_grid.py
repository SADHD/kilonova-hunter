"""The two-component model ends at source-frame day 6. Later times are NaN."""

import numpy as np
import pandas as pd
import pytest

from kilonova_hunter.model_grid import (
    drop_after_two_component_cap,
    nan_after_two_component_cap,
    observer_cap_days,
    uses_two_component_model,
)
from kilonova_hunter.recall import SurveySettings, observe_visits


def test_cap_is_six_source_frame_days_and_grows_with_redshift() -> None:
    """Observer-frame day 6 at z = 0, and 6*(1+z) otherwise."""
    assert observer_cap_days(0.0) == pytest.approx(6.0)
    assert observer_cap_days(0.1) == pytest.approx(6.6)
    assert uses_two_component_model("realistic") is True
    assert uses_two_component_model("2017-like") is False
    assert uses_two_component_model("at2017gfo-exact") is False


def test_times_after_the_cap_are_nan_and_not_detections() -> None:
    """A frozen magnitude past the cap becomes no detection."""
    redshift = 0.1
    cap = observer_cap_days(redshift)
    times = np.array([1.0, cap, cap + 0.25])
    magnitudes = nan_after_two_component_cap(times, np.full(3, 18.0), redshift)
    assert np.isfinite(magnitudes[0])
    assert np.isfinite(magnitudes[1])
    assert np.isnan(magnitudes[2])
    observed = observe_visits(
        times,
        np.array(["g", "g", "g"]),
        magnitudes,
        survey=SurveySettings(),
        rng=np.random.default_rng(0),
    )
    assert observed["time_days"].tolist() == pytest.approx([1.0, cap])


def test_drop_after_the_cap_keeps_only_rows_the_model_computed() -> None:
    """Day-7 scoring of a cached realistic curve ignores the frozen tail."""
    curve = pd.DataFrame(
        {
            "time_days": [1.0, 6.0, 6.6, 7.0, 10.0],
            "filter": ["g"] * 5,
            "magnitude": [18.0] * 5,
            "magnitude_error": [0.05] * 5,
        }
    )
    kept = drop_after_two_component_cap(curve, 0.0)
    assert kept["time_days"].tolist() == [1.0, 6.0]
    # z = 0.1 moves the cap to 6.6 days, so that row stays.
    kept = drop_after_two_component_cap(curve, 0.1)
    assert kept["time_days"].tolist() == [1.0, 6.0, 6.6]


def test_three_component_call_is_not_capped_and_two_component_is(monkeypatch) -> None:
    """The wrapper NaNs only the two-component path. Day 10 stays finite for three."""
    pytest.importorskip("redback")
    from kilonova_hunter.recall import at2017gfo_like_event
    from kilonova_hunter.villar import (
        VillarEvent,
        exact_components,
        realistic_magnitudes,
        villar_magnitudes,
    )

    def constant(times, *args, **kwargs):
        return np.full(len(np.asarray(times)), 18.0)

    monkeypatch.setattr(
        "redback.transient_models.kilonova_models.two_component_bns_ejecta_relation",
        constant,
    )
    monkeypatch.setattr(
        "redback.transient_models.kilonova_models.three_component_kilonova_model",
        constant,
    )
    event = at2017gfo_like_event()
    late = np.array([1.0, 10.0])
    realistic = realistic_magnitudes(late, np.array(["g", "g"]), event)
    assert np.isfinite(realistic[0])
    assert np.isnan(realistic[1])

    blue, purple, red = exact_components()
    copy = VillarEvent(
        object_id="cap-check",
        kind="at2017gfo-exact",
        distance_mpc=40.0,
        redshift=event.redshift,
        ebv=0.0,
        ra_deg=0.0,
        dec_deg=0.0,
        galactic_b_deg=30.0,
        merger_mjd=60000.0,
        viewing_angle_deg=0.0,
        blue=blue,
        purple=purple,
        red=red,
    )
    copied = villar_magnitudes(late, np.array(["g", "g"]), copy)
    assert np.all(np.isfinite(copied))


def test_real_three_component_day_7_is_not_frozen() -> None:
    """Day 7 and day 10 from the three-component model are finite and differ.

    The two-component model repeats its day-6 magnitude after the cap.
    The three-component grid goes to 7e6 seconds, so those later times
    move. This calls the installed model once.
    """
    pytest.importorskip("redback")
    from kilonova_hunter.sky import redshift_at_luminosity_distance
    from kilonova_hunter.villar import (
        EXACT_DISTANCE_MPC,
        VillarEvent,
        exact_components,
        villar_magnitudes,
    )

    blue, purple, red = exact_components()
    event = VillarEvent(
        object_id="day7-real",
        kind="at2017gfo-exact",
        distance_mpc=EXACT_DISTANCE_MPC,
        redshift=redshift_at_luminosity_distance(EXACT_DISTANCE_MPC),
        ebv=0.0,
        ra_deg=0.0,
        dec_deg=0.0,
        galactic_b_deg=30.0,
        merger_mjd=60000.0,
        viewing_angle_deg=0.0,
        blue=blue,
        purple=purple,
        red=red,
    )
    times = np.array([7.0, 10.0])
    magnitudes = villar_magnitudes(times, np.array(["g", "g"]), event)
    assert np.all(np.isfinite(magnitudes))
    assert magnitudes[0] != pytest.approx(magnitudes[1], abs=0.01)

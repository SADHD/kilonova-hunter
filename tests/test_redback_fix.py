"""The diffusion wrapper: factor 2 in the integral, t_d left alone."""

import numpy as np
import pytest

pytest.importorskip("redback")

import matplotlib

# redback turns LaTeX on at import. This machine has no LaTeX install.
matplotlib.rcParams["text.usetex"] = False

from kilonova_hunter.redback_fix import (  # noqa: E402
    REQUIRED_REDBACK_VERSION,
    corrected_diffusion,
    installed_redback_version,
)


def _ratio_at(times: np.ndarray, luminosity: np.ndarray, when: float) -> float:
    return float(np.interp(when, times, luminosity))


def test_installed_redback_is_exactly_1_20_0() -> None:
    """A newer redback must not be silently doubled on top of an upstream fix."""
    assert installed_redback_version() == REQUIRED_REDBACK_VERSION


def test_constant_heating_reaches_the_arnett_curve() -> None:
    """Constant Q gives L/Q = 1 - exp(-(t/t_d)^2), not half of that.

    The points are t/t_d = 1, 2, 5, and 10. The first is about 0.632,
    the second about 0.982, and the last two sit on 1.
    """
    diffusion_time = 1.0
    times = np.linspace(0.0, 12.0, 12001)
    heating = np.ones_like(times)
    with corrected_diffusion():
        from redback.transient_models import kilonova_models

        # Look the name up inside the block. A name imported earlier is
        # still the original function, and would miss the patch.
        luminosity = kilonova_models._kilonova_diffusion_luminosity(
            times, heating, diffusion_time
        )
    ratio = luminosity / heating
    expected = {
        1.0: 1.0 - np.exp(-1.0),
        2.0: 1.0 - np.exp(-4.0),
        5.0: 1.0,
        10.0: 1.0,
    }
    # The integral grid is the array we pass. It starts at 0, and 1000
    # of the 12001 samples sit before t/t_d = 1.
    assert float(times[0]) == pytest.approx(0.0)
    assert int(np.sum(times < 1.0)) == 1000
    early = {
        0.1: 1.0 - np.exp(-(0.1**2)),
        0.25: 1.0 - np.exp(-(0.25**2)),
        0.5: 1.0 - np.exp(-(0.5**2)),
    }
    expected.update(early)
    for time_over_td, target in expected.items():
        found = _ratio_at(times, ratio, time_over_td)
        # Relative, so t/t_d = 0.1 (about 0.01) is not compared with an
        # absolute 0.005 that would also accept half the right answer.
        assert found == pytest.approx(target, rel=0.02)
        half = target / 2.0
        assert half != pytest.approx(target, rel=0.02)


def test_diffusion_time_is_not_doubled() -> None:
    """The 0.632 point stays at t = t_d. Passing 2 t_d moves it.

    Doubling the diffusion time would be the wrong fix. This test fails
    if the wrapper changes t_d instead of the integral.
    """
    diffusion_time = 1.0
    times = np.linspace(0.0, 8.0, 8001)
    heating = np.ones_like(times)
    target = 1.0 - np.exp(-1.0)
    with corrected_diffusion():
        from redback.transient_models import kilonova_models

        diffuse = kilonova_models._kilonova_diffusion_luminosity
        correct = diffuse(times, heating, diffusion_time)
        doubled_time = diffuse(times, heating, 2.0 * diffusion_time)
    at_td = _ratio_at(times, correct, diffusion_time)
    at_td_if_time_were_doubled = _ratio_at(times, doubled_time, diffusion_time)
    assert at_td == pytest.approx(target, abs=0.005)
    assert at_td_if_time_were_doubled < 0.40


def test_decaying_heating_matches_q_at_the_peak() -> None:
    """Where the light curve peaks, dL/dt is 0, so L equals Q.

    The tolerance is 2 percent. The unfixed integral peaks near Q/2,
    which this test rejects.
    """
    diffusion_time = 1.0
    times = np.linspace(0.0, 8.0, 8001)
    heating = np.exp(-times / diffusion_time)
    with corrected_diffusion():
        from redback.transient_models import kilonova_models

        luminosity = kilonova_models._kilonova_diffusion_luminosity(
            times, heating, diffusion_time
        )
    # Skip the first samples. redback copies the second point onto the
    # first, and the peak we want is later than that.
    start = 20
    peak_index = start + int(np.argmax(luminosity[start:]))
    peak_luminosity = float(luminosity[peak_index])
    peak_heating = float(heating[peak_index])
    assert peak_luminosity == pytest.approx(peak_heating, rel=0.02)


def test_wrapper_doubles_bolometric_luminosity_and_restores_usetex() -> None:
    """A real one-component call is twice as bright, and usetex is restored."""
    import matplotlib
    from redback.transient_models.kilonova_models import _one_component_kilonova_model

    from kilonova_hunter.redback_fix import corrected_diffusion

    # Start at 0.01 day. A grid that starts at 0.2 day drops the early
    # part of the diffusion integral and moves the ratio off 2.
    time = np.geomspace(1e-2, 3.0, 80) * 86400.0
    plain, _temperature, _radius = _one_component_kilonova_model(
        time, 0.02, 0.266, 0.5, temperature_floor=674.0
    )
    matplotlib.rcParams["text.usetex"] = True
    with corrected_diffusion():
        fixed, _temperature, _radius = _one_component_kilonova_model(
            time, 0.02, 0.266, 0.5, temperature_floor=674.0
        )
    assert matplotlib.rcParams["text.usetex"] is True
    matplotlib.rcParams["text.usetex"] = False
    ratio = fixed[10:] / plain[10:]
    assert np.allclose(ratio, 2.0)


def test_the_patch_is_removed_after_the_model_call() -> None:
    """redback's own function is back in place once the block ends."""
    from redback.transient_models import kilonova_models

    original = kilonova_models._kilonova_diffusion_luminosity
    with corrected_diffusion():
        assert kilonova_models._kilonova_diffusion_luminosity is not original
    assert kilonova_models._kilonova_diffusion_luminosity is original


def test_villar_call_passes_the_table_temperature_floors(monkeypatch) -> None:
    """674, 1308, and 3745 K are positional arguments, not the 4000 K default."""
    from kilonova_hunter.villar import (
        VillarEvent,
        exact_components,
        villar_magnitudes,
    )

    captured: dict = {}

    def fake_model(*args, **kwargs):
        captured["args"] = args
        from redback.transient_models import kilonova_models

        captured["fixed"] = getattr(
            kilonova_models._kilonova_diffusion_luminosity,
            "_hunter_diffusion_fix",
            False,
        )
        return np.zeros(len(args[0]))

    monkeypatch.setattr(
        "redback.transient_models.kilonova_models.three_component_kilonova_model",
        fake_model,
    )
    blue, purple, red = exact_components()
    event = VillarEvent(
        object_id="floor-check",
        kind="at2017gfo-exact",
        distance_mpc=40.0,
        redshift=0.009,
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
    villar_magnitudes(np.array([1.0]), np.array(["g"]), event)
    assert captured["args"][4] == 674.0
    assert captured["args"][8] == 1308.0
    assert captured["args"][12] == 3745.0
    assert captured["fixed"] is True


def test_nested_patch_stays_at_factor_two() -> None:
    """A second entry does not wrap the wrapper. The factor stays 2, not 4."""
    diffusion_time = 1.0
    times = np.linspace(0.0, 4.0, 4001)
    heating = np.ones_like(times)
    target = 1.0 - np.exp(-1.0)
    with corrected_diffusion():
        with corrected_diffusion():
            from redback.transient_models import kilonova_models

            luminosity = kilonova_models._kilonova_diffusion_luminosity(
                times, heating, diffusion_time
            )
    found = _ratio_at(times, luminosity, diffusion_time)
    assert found == pytest.approx(target, abs=0.005)


def test_an_already_patched_function_is_refused() -> None:
    """A leaked patch is not wrapped again."""
    from redback.transient_models import kilonova_models

    original = kilonova_models._kilonova_diffusion_luminosity

    def already(time, thermalised_luminosity, diffusion_timescale):
        return original(time, thermalised_luminosity, diffusion_timescale)

    already._hunter_diffusion_fix = True  # type: ignore[attr-defined]
    kilonova_models._kilonova_diffusion_luminosity = already
    try:
        with pytest.raises(RuntimeError, match="already applied"):
            with corrected_diffusion():
                pass
    finally:
        kilonova_models._kilonova_diffusion_luminosity = original


def test_realistic_and_legacy_paths_run_with_the_diffusion_fix(monkeypatch) -> None:
    """Both public two-component callers are inside the wrapper."""
    from kilonova_hunter.recall import at2017gfo_like_event, redback_magnitudes
    from kilonova_hunter.villar import realistic_magnitudes

    seen: list[bool] = []

    def fake(*args, **kwargs):
        from redback.transient_models import kilonova_models

        seen.append(
            bool(
                getattr(
                    kilonova_models._kilonova_diffusion_luminosity,
                    "_hunter_diffusion_fix",
                    False,
                )
            )
        )
        return np.zeros(len(args[0]))

    monkeypatch.setattr(
        "redback.transient_models.kilonova_models.two_component_bns_ejecta_relation",
        fake,
    )
    monkeypatch.setattr(
        "redback.transient_models.kilonova_models.two_component_kilonova_model",
        fake,
    )
    event = at2017gfo_like_event()
    realistic_magnitudes(np.array([1.0]), np.array(["g"]), event)
    redback_magnitudes(np.array([1.0]), np.array(["g"]), event)
    assert seen == [True, True]

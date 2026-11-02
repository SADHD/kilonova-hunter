"""Tests for the fake-kilonova recall count.

These use a tiny stand-in light curve. They do not import redback.
One test at the bottom runs redback when that optional library is
installed, and skips when it is not.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kilonova_hunter.cli import main
from kilonova_hunter.recall import (
    CATCH_WITHIN_DAYS,
    PopulationSettings,
    SurveySettings,
    at2017gfo_like_event,
    build_recall_set,
    comoving_volume_mpc3,
    diagnose_catch_window,
    dust_dimming_mag,
    light_curve_for_the_filter,
    magnitude_uncertainty,
    observe_visits,
    redshift_at_comoving_distance,
    run_recall,
    sample_ebv,
    sample_masses,
    sample_redshifts_and_distances,
    sample_viewing_angle_deg,
    score_observed_curve,
    simulate_event,
    stub_magnitudes,
    summarize_recall,
    survey_visits,
    visit_times_for_one_band,
    wilson_interval,
)


def simulate_event_with(event, survey: SurveySettings, seed: int):
    """Run one stub event on a fixed seed. Local to these tests."""
    observed, error_reason, _peaks = simulate_event(
        event,
        survey,
        np.random.default_rng(seed),
        stub_magnitudes,
    )
    return observed, error_reason


def test_masses_stay_inside_the_range_and_the_heavier_star_is_first() -> None:
    """Both stars are between 1.1 and 1.6 solar masses, heavier one first."""
    settings = PopulationSettings()
    rng = np.random.default_rng(1)
    for _ in range(50):
        heavier, lighter = sample_masses(rng, settings)
        assert settings.mass_min_solar <= lighter <= heavier <= settings.mass_max_solar


def test_distances_are_uniform_in_comoving_volume() -> None:
    """More fakes live far away, because there is more volume out there.

    Uniform in distance would put half the fakes closer than the
    midpoint. Uniform in volume puts about half of them below the
    midpoint of the comoving volume, which is much farther out.
    """
    settings = PopulationSettings()
    redshifts, distances = sample_redshifts_and_distances(
        np.random.default_rng(2),
        40,
        settings,
    )
    assert np.all(distances > 0)
    z_near = redshift_at_comoving_distance(settings.distance_min_mpc)
    z_far = redshift_at_comoving_distance(settings.distance_max_mpc)
    volume_near = float(comoving_volume_mpc3(np.array([z_near]))[0])
    volume_far = float(comoving_volume_mpc3(np.array([z_far]))[0])
    midpoint = 0.5 * (volume_near + volume_far)
    volumes = comoving_volume_mpc3(redshifts)
    fraction_below = float(np.mean(volumes < midpoint))
    assert 0.25 < fraction_below < 0.75
    # The stored distance is the luminosity distance, so it can sit a
    # little outside the comoving-distance edges. It should still be
    # in the same ballpark as the range we asked for.
    assert float(distances.min()) > settings.distance_min_mpc * 0.9
    assert float(distances.max()) < settings.distance_max_mpc * 1.15


def test_viewing_angles_are_isotropic() -> None:
    """Every direction is equally likely: the average cosine is near zero."""
    rng = np.random.default_rng(3)
    angles = np.array([sample_viewing_angle_deg(rng) for _ in range(4000)])
    assert angles.min() >= 0.0
    assert angles.max() <= 180.0
    assert float(np.mean(np.cos(np.radians(angles)))) == pytest.approx(0.0, abs=0.05)


def test_dust_draw_is_a_capped_exponential_and_g_is_dimmed_most() -> None:
    """E(B-V) is non-negative, capped, and dims g more than r."""
    settings = PopulationSettings()
    rng = np.random.default_rng(4)
    draws = np.array([sample_ebv(rng, settings) for _ in range(2000)])
    assert draws.min() >= 0.0
    assert draws.max() <= settings.ebv_cap + 1e-12
    # Clipping trims the tail, so the mean sits a little under 0.05.
    assert float(draws.mean()) == pytest.approx(settings.ebv_mean, abs=0.02)
    assert dust_dimming_mag("g", 0.1) == pytest.approx(0.3303)
    assert dust_dimming_mag("g", 0.1) > dust_dimming_mag("r", 0.1)
    assert dust_dimming_mag("r", 0.1) > dust_dimming_mag("i", 0.1)


def test_cadence_gaps_are_not_a_smooth_grid() -> None:
    """Visits wait at least the revisit gap, plus a random extra night."""
    survey = SurveySettings(
        cadence="simple",
        revisit_gap_days=1.0,
        nightly_gap_days=1.0,
        survey_length_days=8.0,
    )
    times = visit_times_for_one_band(np.random.default_rng(1), survey=survey)
    gaps = np.diff(times)
    assert len(gaps) >= 2
    assert float(times[0]) >= 0.0
    assert float(times[0]) <= survey.revisit_gap_days
    assert float(gaps.min()) >= survey.revisit_gap_days - 1e-9
    assert float(gaps.max()) <= survey.revisit_gap_days + survey.nightly_gap_days + 1e-9
    # A smooth grid would have every gap equal. The nightly jitter does not.
    assert float(gaps.max() - gaps.min()) > 0.05
    again = visit_times_for_one_band(np.random.default_rng(1), survey=survey)
    np.testing.assert_allclose(times, again)


def test_points_fainter_than_the_limit_are_dropped() -> None:
    """A point fainter than the g limit is dropped. The bright point is kept.

    The point at magnitude 22 is kept. Its stored error is the error
    the depth formula predicts for magnitude 22, and the magnitude
    itself has been moved by Gaussian noise. The same seed repeats it.
    """
    survey = SurveySettings()
    times = np.array([1.0, 1.0])
    bands = np.array(["g", "g"])
    # 25.5 is fainter than the g limit from Ivezić et al. 2019 Table 2
    # (24.81). 22.0 is brighter than that limit, so it is kept.
    true = np.array([25.5, 22.0])
    observed = observe_visits(
        times,
        bands,
        true,
        survey=survey,
        rng=np.random.default_rng(0),
    )
    assert len(observed) == 1
    assert observed.iloc[0]["magnitude"] != pytest.approx(22.0)
    expected = magnitude_uncertainty(22.0, survey.limit_for("g"))
    assert observed.iloc[0]["magnitude_error"] == pytest.approx(expected)
    again = observe_visits(
        times,
        bands,
        true,
        survey=survey,
        rng=np.random.default_rng(0),
    )
    assert again.iloc[0]["magnitude"] == pytest.approx(observed.iloc[0]["magnitude"])


def test_magnitude_error_is_about_0_22_at_the_five_sigma_limit() -> None:
    """At the limiting magnitude the signal-to-noise ratio is 5.

    The magnitude error is then 2.5 / ln(10) / 5, about 0.22 mag.
    A source two magnitudes brighter has a smaller error. The floor
    is turned off here so the formula itself is what we check.
    """
    at_the_limit = magnitude_uncertainty(24.5, 24.5, floor=0.0)
    assert at_the_limit == pytest.approx((2.5 / np.log(10.0)) / 5.0)
    brighter = magnitude_uncertainty(22.5, 24.5, floor=0.0)
    assert brighter < at_the_limit


def test_the_filter_only_sees_the_first_three_days() -> None:
    """A point at exactly 3 days stays. A point just after 3 days does not.

    A detection that exists only at day 4 was detected, so it is not
    "never detected", but the filter is not shown it, so it is not caught.
    """
    curve = pd.DataFrame(
        {
            "time_days": [1.0, 3.0, 3.01, 5.0],
            "filter": ["r", "r", "r", "r"],
            "magnitude": [18.0, 19.0, 20.0, 21.0],
            "magnitude_error": [0.01, 0.01, 0.01, 0.01],
        }
    )
    seen = light_curve_for_the_filter(curve, CATCH_WITHIN_DAYS)
    assert list(seen["time_days"]) == pytest.approx([1.0, 3.0])

    late_only = curve.loc[curve["time_days"] > CATCH_WITHIN_DAYS]
    never, caught, n_visible = score_observed_curve(late_only)
    assert never is False
    assert caught is False
    assert n_visible == 0


def test_never_detected_fakes_stay_in_the_denominator() -> None:
    """One catch, one rule-failure, one never-detected: recall is 1/3.

    The never-detected fake is also reported on its own, and it is in
    the distance bin it belongs to.
    """
    scored = pd.DataFrame(
        {
            "distance_mpc": [20.0, 80.0, 250.0],
            "is_candidate": [True, False, False],
            "never_detected": [False, False, True],
            "model_failed": [False, False, False],
        }
    )
    numbers = summarize_recall(scored, set_name="test")
    assert numbers.n_total == 3
    assert numbers.n_caught == 1
    assert numbers.n_never_detected == 1
    assert numbers.recall == pytest.approx(1.0 / 3.0)
    low, high = wilson_interval(1, 3)
    assert numbers.ci_low == pytest.approx(low)
    assert numbers.ci_high == pytest.approx(high)
    assert 0.0 <= numbers.ci_low <= numbers.recall <= numbers.ci_high <= 1.0
    far = [item for item in numbers.by_distance if item.low_mpc == 200.0]
    assert far[0].n_total == 1
    assert far[0].n_caught == 0
    assert far[0].n_never_detected == 1


def test_a_nearby_stub_is_caught_and_a_distant_one_is_not() -> None:
    """The stand-in model is bright and reddening at 40 Mpc, and invisible far away.

    The visits are a short regular cadence here so this test is about
    the depth cut and the filter, not about one random gap.
    """
    survey = SurveySettings(
        cadence="simple",
        revisit_gap_days=0.5,
        nightly_gap_days=0.0,
        survey_length_days=3.0,
    )
    nearby = at2017gfo_like_event()
    observed, error_reason = simulate_event_with(nearby, survey, seed=4)
    assert error_reason is None
    never, caught, n_visible = score_observed_curve(observed)
    assert never is False
    assert n_visible >= 4
    assert caught is True

    far = replace(nearby, distance_mpc=8000.0)
    faint, error_reason = simulate_event_with(far, survey, seed=4)
    assert error_reason is None
    never, caught, _n_visible = score_observed_curve(faint)
    assert never is True
    assert caught is False


def test_recall_writes_two_sets_from_different_seeds(tmp_path: Path) -> None:
    """Tuning and test are different draws, and the headline is the test set.

    The sanity fake is written beside them and is not one of the rows
    in either set.
    """
    survey = SurveySettings(
        cadence="simple",
        revisit_gap_days=1.0,
        nightly_gap_days=0.0,
        survey_length_days=3.0,
    )
    text = run_recall(
        which="both",
        set_size=3,
        seed=5,
        output_dir=tmp_path,
        model=stub_magnitudes,
        survey=survey,
    )
    tuning = pd.read_csv(tmp_path / "tuning" / "events.csv")
    test = pd.read_csv(tmp_path / "test" / "events.csv")
    assert len(tuning) == 3
    assert len(test) == 3
    assert not np.allclose(tuning["distance_mpc"], test["distance_mpc"])
    assert "at2017gfo-like" not in set(tuning["object_id"])
    assert "at2017gfo-like" not in set(test["object_id"])
    assert (tmp_path / "at2017gfo" / "events.csv").is_file()
    assert (tmp_path / "report.txt").is_file()
    assert "Test set (this is the recall number)" in text
    assert "not the official number" in text
    assert "Never detected at all" in text
    assert "have not been tuned" in text
    assert "AT2017gfo-like sanity check" in text
    sanity = pd.read_csv(tmp_path / "at2017gfo" / "events.csv")
    assert sanity.iloc[0]["distance_mpc"] == pytest.approx(40.0)
    assert sanity.iloc[0]["ebv"] == pytest.approx(0.105)


def test_recall_help_lists_the_options(capsys: pytest.CaptureFixture[str]) -> None:
    """``kilohunt recall --help`` names the size, the seed, and which set."""
    with pytest.raises(SystemExit) as caught:
        main(["recall", "--help"])
    assert caught.value.code == 0
    printed = capsys.readouterr().out
    assert "--set-size" in printed
    assert "--seed" in printed
    assert "--which" in printed
    assert "--cadence" in printed


def test_recall_without_redback_explains_the_extra(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A normal install does not have redback, and the command says how to add it.

    The check is patched, so this test does the same thing whether or
    not redback happens to be installed in this environment.
    """
    monkeypatch.setattr("kilonova_hunter.cli.redback_available", lambda: False)
    with pytest.raises(SystemExit) as caught:
        main(["recall", "--set-size", "1", "--which", "test"])
    assert caught.value.code == 1
    error = capsys.readouterr().err
    assert "uv sync --extra sim" in error


def test_three_days_are_counted_from_the_merger() -> None:
    """The catch window starts at the merger, not at the first detection.

    The first point is on day 1, so a window that started then would
    still include day 4. The filter is not shown day 4. Day 1 is inside
    the 3 days after the merger, and day 3.0 exactly is kept.
    """
    curve = pd.DataFrame(
        {
            "time_days": [1.0, 3.0, 4.0],
            "filter": ["r", "g", "r"],
            "magnitude": [18.0, 18.5, 19.0],
            "magnitude_error": [0.01, 0.01, 0.01],
        }
    )
    seen = light_curve_for_the_filter(curve, CATCH_WITHIN_DAYS)
    assert list(seen["time_days"]) == pytest.approx([1.0, 3.0])
    late_only = curve.loc[curve["time_days"] >= 4.0]
    never, caught, n_visible = score_observed_curve(late_only)
    assert never is False
    assert caught is False
    assert n_visible == 0


def test_wide_fast_deep_visits_come_in_pairs() -> None:
    """The default cadence takes two different filters about 30 minutes apart.

    The gap from one pair to the next is between 2 and 4 days, and the
    first visit is somewhere in that first gap.
    """
    survey = SurveySettings(cadence="wfd", survey_length_days=12.0)
    times, bands = survey_visits(np.random.default_rng(1), survey)
    assert len(times) >= 4
    assert float(times[0]) >= 0.0
    assert float(times[0]) <= survey.wfd_gap_max_days + 1e-9
    pair_starts = times[::2]
    gaps = np.diff(pair_starts)
    thirty_minutes = 30.0 / (24.0 * 60.0)
    assert np.allclose(times[1::2] - times[::2], thirty_minutes, atol=1e-6)
    assert float(gaps.min()) >= survey.wfd_gap_min_days - 1e-9
    assert float(gaps.max()) <= survey.wfd_gap_max_days + 1e-9
    allowed = {frozenset(pair) for pair in (("g", "r"), ("r", "i"), ("g", "i"))}
    for first, second in zip(bands[::2], bands[1::2], strict=True):
        assert first != second
        assert frozenset((first, second)) in allowed


def test_too_many_model_errors_abort_the_run() -> None:
    """A model that fails for most fakes must not be reported as a low recall."""

    def broken(times, bands, event):
        raise ValueError("bad ejecta mass")

    survey = SurveySettings(cadence="simple", survey_length_days=2.0)
    with pytest.raises(RuntimeError, match="broken model"):
        build_recall_set(
            set_name="test",
            population=PopulationSettings(count=4, seed=1),
            survey=survey,
            model=broken,
        )


def test_one_model_error_is_not_counted_as_never_detected() -> None:
    """A single bad draw is logged and kept apart from fakes the survey missed."""
    calls = {"n": 0}

    def sometimes(times, bands, event):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("one bad draw")
        return stub_magnitudes(times, bands, event)

    survey = SurveySettings(cadence="simple", survey_length_days=2.0)
    events, _curves, numbers = build_recall_set(
        set_name="test",
        population=PopulationSettings(count=5, seed=2),
        survey=survey,
        model=sometimes,
    )
    assert numbers.n_model_failed == 1
    bad = events.loc[events["model_failed"]]
    assert len(bad) == 1
    assert bool(bad.iloc[0]["never_detected"]) is False
    assert "bad draw" in str(bad.iloc[0]["model_error_reason"])
    assert "judgeable" in events.columns
    assert "n_detections_in_window" in events.columns
    assert "window_days" in events.columns
    assert "fade_rate_mag_per_day" in events.columns
    assert "g_minus_r" in events.columns
    assert "reddening_rise_mag" in events.columns


def test_judgeable_needs_two_epochs_and_one_color() -> None:
    """One detection is not enough to judge. Two bands at two times are."""
    one = pd.DataFrame(
        {
            "time_days": [1.0],
            "filter": ["r"],
            "magnitude": [18.0],
            "magnitude_error": [0.05],
        }
    )
    assert diagnose_catch_window(one, 3.0)["judgeable"] is False
    two = pd.DataFrame(
        {
            "time_days": [0.5, 0.5, 2.0, 2.0],
            "filter": ["g", "r", "g", "r"],
            "magnitude": [19.0, 18.5, 20.0, 19.0],
            "magnitude_error": [0.05, 0.05, 0.05, 0.05],
        }
    )
    diagnosis = diagnose_catch_window(two, 3.0)
    assert diagnosis["judgeable"] is True
    assert diagnosis["n_detections_in_window"] == 4
    # A point after day 3 does not count, even though it is the next visit.
    late = pd.concat(
        [
            two,
            pd.DataFrame(
                {
                    "time_days": [4.5, 4.5],
                    "filter": ["g", "r"],
                    "magnitude": [21.0, 20.0],
                    "magnitude_error": [0.05, 0.05],
                }
            ),
        ],
        ignore_index=True,
    )
    still = diagnose_catch_window(late, 3.0)
    assert still["n_detections_in_window"] == 4


def test_redback_sanity_fake_returns_finite_magnitudes() -> None:
    """One real redback call, skipped when redback is not installed.

    Three bands at one time for the AT2017gfo-like fake. This is the
    slow optional check that the model name and the band names still
    work. The counting tests above do not run this.
    """
    import matplotlib

    # importorskip imports redback, and redback turns on LaTeX in
    # matplotlib. Save the settings first and put them back, so the plot
    # tests later in the same run do not go looking for a LaTeX install.
    saved = matplotlib.rcParams.copy()
    try:
        pytest.importorskip("redback")
        from kilonova_hunter.recall import redback_magnitudes

        event = at2017gfo_like_event()
        times = np.array([1.0, 1.0, 1.0])
        bands = np.array(["g", "r", "i"])
        magnitudes = redback_magnitudes(times, bands, event)
        assert magnitudes.shape == (3,)
        assert np.isfinite(magnitudes).all()
    finally:
        matplotlib.rcParams.update(saved)


def test_build_recall_set_columns_match_what_the_report_reads() -> None:
    """The saved event table has the score columns summarize_recall sums."""
    survey = SurveySettings(
        cadence="simple",
        revisit_gap_days=1.0,
        nightly_gap_days=0.0,
        survey_length_days=2.0,
    )
    events, curves, numbers = build_recall_set(
        set_name="tuning",
        population=PopulationSettings(count=2, seed=9),
        survey=survey,
        model=stub_magnitudes,
    )
    for column in (
        "is_candidate",
        "never_detected",
        "distance_mpc",
        "viewing_angle_deg",
    ):
        assert column in events.columns
    assert numbers.n_total == 2
    assert set(curves["object_id"]) <= set(events["object_id"])

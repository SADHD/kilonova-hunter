"""Wilson intervals, the pick order, the stopping rule, and exclusion of copies."""

import numpy as np
import pandas as pd
import pytest

from kilonova_hunter.config import THRESHOLDS
from kilonova_hunter.grid import (
    PreparedFake,
    _combo_sentence,
    _worst_not_judgeable,
    chance_wilson_lower_at_least,
    fast_verdict_matches_filter,
    grid_combinations,
    load_settings,
    pick_by_recall,
    pick_with_flag_limit,
    score_exact_copies,
    score_prepared,
)
from kilonova_hunter.inject import simulate_kilonova_light_curve
from kilonova_hunter.recall import wilson_interval
from kilonova_hunter.sample import (
    SampleSizeSettings,
    next_batch_size,
    sample_status,
)
from kilonova_hunter.villar import (
    EXACT_KIND,
    VillarSpread,
    clipped_lognormal_mass,
    truncated_normal_speed,
)


def test_reports_say_tns_is_not_yet_implemented(tmp_path) -> None:
    """Every report says TNS is not implemented, and the lookup still raises."""
    from kilonova_hunter.grid import format_test_score, summarize_grid
    from kilonova_hunter.recall import format_recall_report
    from kilonova_hunter.tns import TNS_STATUS_LINE, lookup_classifications

    summary = summarize_grid(
        [],
        samples=[],
        dust_mode="sfd",
        dust_reason="",
        copies=None,
        contamination_applied=False,
    )
    score = format_test_score({}, dry_run=False, provisional=True)
    report = format_recall_report(
        [],
        sanity_text="none",
        sanity_is_candidate=False,
        output_dir=tmp_path,
    )
    assert TNS_STATUS_LINE in summary
    assert TNS_STATUS_LINE in score
    assert TNS_STATUS_LINE in report
    assert "Judgeable (a)" in summary
    assert "Judgeable (b)" in summary
    assert "It does not replace (a)." in summary
    assert "Judgeable (a)" in score
    assert "Judgeable (a)" in report
    with pytest.raises(NotImplementedError):
        lookup_classifications(["AT2017gfo"])


def test_wilson_interval_stays_inside_zero_and_one() -> None:
    """Empty, all failures, and all successes have the usual edges."""
    empty_low, empty_high = wilson_interval(0, 0)
    assert empty_low != empty_low and empty_high != empty_high
    low, high = wilson_interval(0, 10)
    assert low == 0.0
    assert 0.0 < high < 1.0
    low, high = wilson_interval(10, 10)
    assert 0.0 < low < 1.0
    assert high == pytest.approx(1.0)
    low, high = wilson_interval(1, 3)
    assert 0.0 <= low <= 1.0 / 3.0 <= high <= 1.0


def test_the_grid_has_the_72_combinations() -> None:
    """Four color cuts, three rises, three windows, two gaps."""
    rows = grid_combinations()
    assert len(rows) == 4 * 3 * 3 * 2
    assert {row["redder_than_g_minus_r"] for row in rows} == {0.0, 0.2, 0.4, 0.6}
    assert {row["min_g_minus_r_rise"] for row in rows} == {0.1, 0.2, 0.3}
    assert {row["reddening_window_days"] for row in rows} == {2.0, 3.0, 5.0}
    assert {row["min_color_separation_days"] for row in rows} == {0.5, 1.0}


def test_pick_keeps_the_band_near_the_best_recall_then_the_strict_order() -> None:
    """Higher g-r, then bigger rise, then larger gap, then shorter window."""
    settings = load_settings()
    rows = [
        _row(0.90, 0.0, 0.1, 0.5, 5.0),
        _row(0.86, 0.4, 0.1, 0.5, 5.0),
        _row(0.86, 0.4, 0.3, 0.5, 5.0),
        _row(0.84, 0.6, 0.3, 1.0, 2.0),
    ]
    chosen = pick_by_recall(rows, settings)
    # 0.84 is more than 5 points under 0.90, so the strictest in-band row
    # is the 0.4 color cut with the bigger rise.
    assert chosen["redder_than_g_minus_r"] == 0.4
    assert chosen["min_g_minus_r_rise"] == 0.3
    assert chosen["best_recall"] == 0.90


def test_a_larger_gap_beats_a_shorter_window_when_the_recall_ties() -> None:
    """Gap is the third key. Window is the fourth, and shorter wins a tie there."""
    settings = load_settings()
    rows = [
        _row(0.90, 0.4, 0.3, 0.5, 2.0),
        _row(0.90, 0.4, 0.3, 1.0, 5.0),
    ]
    chosen = pick_by_recall(rows, settings)
    assert chosen["min_color_separation_days"] == 1.0
    assert chosen["reddening_window_days"] == 5.0
    tied_window = [
        _row(0.90, 0.4, 0.3, 1.0, 5.0),
        _row(0.90, 0.4, 0.3, 1.0, 2.0),
    ]
    shorter = pick_by_recall(tied_window, settings)
    assert shorter["reddening_window_days"] == 2.0


def test_flag_limit_prefers_recall_inside_the_limit_and_says_when_nobody_meets_it() -> (
    None
):
    """At most 10 flags per night, then the recall band, then the order."""
    settings = load_settings()
    rows = [
        {**_row(0.90, 0.0, 0.1, 0.5, 5.0), "flags_per_night": 40.0},
        {**_row(0.86, 0.4, 0.2, 1.0, 3.0), "flags_per_night": 8.0},
        {**_row(0.86, 0.4, 0.1, 0.5, 2.0), "flags_per_night": 8.0},
    ]
    chosen = pick_with_flag_limit(rows, settings)
    assert chosen["limit_met"] is True
    assert chosen["min_g_minus_r_rise"] == 0.2
    none = [
        {**_row(0.90, 0.0, 0.1, 0.5, 5.0), "flags_per_night": 12.0},
        {**_row(0.88, 0.6, 0.3, 1.0, 5.0), "flags_per_night": 11.0},
        {**_row(0.70, 0.6, 0.3, 1.0, 2.0), "flags_per_night": 15.0},
    ]
    fallback = pick_with_flag_limit(none, settings)
    assert fallback["limit_met"] is False
    # 0.70 is outside 5 points of 0.90, so the fewest flags in the band is 11.
    assert fallback["flags_per_night"] == 11.0


def test_sample_size_defaults_and_the_three_stopping_outcomes() -> None:
    """Target 400, cap 20000, floor 50, batches of 500."""
    settings = SampleSizeSettings()
    assert settings.target_judgeable == 400
    assert settings.cap == 20_000
    assert settings.untestable_floor == 50
    assert settings.batch_size == 500
    assert sample_status(500, 400, settings) == "ready"
    assert sample_status(20_000, 40, settings) == "untestable"
    assert sample_status(20_000, 80, settings) == "underpowered"
    assert sample_status(500, 10, settings) == "continue"
    # The last batch shrinks so the cap is not crossed.
    assert next_batch_size(19_700, settings) == 300
    assert next_batch_size(20_000, settings) == 0


def test_tiny_power_prints_as_about_zero() -> None:
    """A chance of 3e-80 is printed as about 0. A real chance keeps its digits."""
    row = {
        "recall_judgeable": 0.10,
        "recall_low": 0.06,
        "recall_high": 0.16,
        "catch_rate": 0.001,
        "catch_low": 0.0,
        "catch_high": 0.01,
        "judgeable_fraction": 0.01,
        "n_reddening_skipped": 0,
        "redder_than_g_minus_r": 0.0,
        "min_g_minus_r_rise": 0.1,
        "reddening_window_days": 2.0,
        "min_color_separation_days": 0.5,
        "power_at_70": 3e-80,
    }
    text = _combo_sentence(row)
    assert "about 0" in text
    assert "3e-80" not in text
    row["power_at_70"] = 0.42
    assert "0.42" in _combo_sentence(row)


def test_power_is_one_when_every_draw_would_clear_the_bar() -> None:
    """Twenty out of twenty has a Wilson lower bound above 70 percent."""
    assert chance_wilson_lower_at_least(20, 1.0, 0.70) == 1.0
    low, _high = wilson_interval(20, 20)
    assert low >= 0.70


def _one_prepared(object_id: str, distance_mpc: float) -> PreparedFake:
    return PreparedFake(
        object_id=object_id,
        distance_mpc=distance_mpc,
        judgeable=True,
        model_failed=False,
        fade_passed=True,
        color_value=0.5,
        rises={(2.0, 0.5): 0.4},
    )


def test_far_distance_bin_ends_at_300_and_includes_300() -> None:
    """300 Mpc is in the 200-300 bin. A fake past 300 is in no bin."""
    combo = {
        "redder_than_g_minus_r": 0.0,
        "min_g_minus_r_rise": 0.1,
        "reddening_window_days": 2.0,
        "min_color_separation_days": 0.5,
    }
    prepared = [
        _one_prepared("near-edge", 100.0),
        _one_prepared("far-edge", 300.0),
        _one_prepared("past-far", 300.1),
    ]
    near = score_prepared(
        prepared,
        combo,
        population="2017-like",
        day=3.0,
        status="ready",
        distance_bin="10-100",
        low_mpc=10.0,
        high_mpc=100.0,
        high_inclusive=False,
    )
    middle = score_prepared(
        prepared,
        combo,
        population="2017-like",
        day=3.0,
        status="ready",
        distance_bin="100-200",
        low_mpc=100.0,
        high_mpc=200.0,
        high_inclusive=False,
    )
    far = score_prepared(
        prepared,
        combo,
        population="2017-like",
        day=3.0,
        status="ready",
        distance_bin="200-300",
        low_mpc=200.0,
        high_mpc=300.0,
        high_inclusive=True,
    )
    assert near["n_total"] == 0
    assert middle["n_total"] == 1
    assert far["n_total"] == 1
    assert far["n_judgeable"] == 1
    assert far["n_judgeable_candidate"] == 0


def test_grid_row_shows_both_judgeable_definitions() -> None:
    """(a) stays the recall count. (b) is reported beside it and is not official."""
    combo = {
        "redder_than_g_minus_r": 0.0,
        "min_g_minus_r_rise": 0.1,
        "reddening_window_days": 2.0,
        "min_color_separation_days": 0.5,
    }
    official = _one_prepared("official-only", 40.0)
    both = _one_prepared("both-definitions", 50.0)
    both.judgeable_candidate = True
    row = score_prepared(
        [official, both],
        combo,
        population="2017-like",
        day=3.0,
        status="ready",
    )
    assert row["n_judgeable"] == 2
    assert row["n_judgeable_candidate"] == 1
    text = _combo_sentence(row)
    assert "Judgeable (a, official): 2" in text
    assert "not adopted" in text
    assert "Judgeable (b, candidate, not adopted): 1" in text


def test_caught_but_not_judgeable_check_includes_day_7() -> None:
    """A day-7 hit is reported even when days 3 and 5 are clean.

    The day-7 fake is a rule mismatch. The checker has to show it.
    The official judgeable rule is not changed to hide it.
    """
    combo = {
        "redder_than_g_minus_r": 0.0,
        "min_g_minus_r_rise": 0.1,
        "reddening_window_days": 2.0,
        "min_color_separation_days": 0.5,
    }
    fake = PreparedFake(
        object_id="realistic-tuning-08304",
        distance_mpc=61.8,
        judgeable=False,
        model_failed=False,
        fade_passed=True,
        color_value=0.5,
        rises={(2.0, 0.5): None},
    )

    def row(day: float, count: int) -> dict:
        return {
            "distance_bin": "all",
            "day": day,
            "n_caught_not_judgeable": count,
            "population": "realistic",
            **combo,
        }

    text = _worst_not_judgeable(
        [row(3.0, 0), row(5.0, 0), row(7.0, 1)],
        {"realistic": ({3.0: [], 5.0: [], 7.0: [fake]}, pd.DataFrame())},
    )
    assert "day 7" in text
    assert "realistic-tuning-08304" in text
    assert "SIGN-OFF" not in text


def test_judgeable_copy_records_a_skipped_reddening_rule() -> None:
    """A judgeable copy with one color records reddening as skipped.

    The official judgeable rule still accepts the copy. The breakdown
    says the reddening rule could not run, which is not a failed rise.
    """
    from kilonova_hunter.grid import (
        exact_copy_rule_breakdown,
        format_exact_copy_breakdown,
    )

    events = pd.DataFrame(
        {
            "object_id": ["at2017gfo-exact-000"],
            "merger_mjd": [60000.0],
        }
    )
    curves = pd.DataFrame(
        {
            "object_id": ["at2017gfo-exact-000"] * 3,
            "time_days": [0.1, 0.1, 1.6],
            "filter": ["g", "r", "g"],
            "magnitude": [18.0, 18.2, 19.0],
            "magnitude_error": [0.05, 0.05, 0.05],
        }
    )
    combo = {
        "redder_than_g_minus_r": 0.0,
        "min_g_minus_r_rise": 0.1,
        "reddening_window_days": 2.0,
        "min_color_separation_days": 0.5,
    }
    report = exact_copy_rule_breakdown(events, curves, combo)
    assert report["n_judgeable"] == 1
    assert report["rows"][0]["reddening"] == "skipped"
    assert report["n_judgeable_reddening_skipped"] == 1
    text = format_exact_copy_breakdown(report)
    assert "Finding:" in text
    assert "official judgeable" in text
    assert "window 2 days" in text


def test_zero_judgeable_copies_are_a_sign_off_blocker() -> None:
    """No judgeable copy is a blocker, not a blank that can be signed off."""
    from kilonova_hunter.grid import _copy_lines

    events = pd.DataFrame(
        {
            "object_id": ["at2017gfo-exact-000"],
            "merger_mjd": [60000.0],
            "ra_deg": [10.0],
            "dec_deg": [-20.0],
            "ebv": [0.01],
        }
    )
    curves = pd.DataFrame(
        {
            "object_id": ["at2017gfo-exact-000"],
            "time_days": [1.0],
            "filter": ["r"],
            "magnitude": [18.0],
            "magnitude_error": [0.05],
        }
    )
    combo = {
        "redder_than_g_minus_r": 0.0,
        "min_g_minus_r_rise": 0.1,
        "reddening_window_days": 2.0,
        "min_color_separation_days": 0.5,
    }
    result = score_exact_copies(events, curves, combo)
    assert result["n_judgeable"] == 0
    assert result["blocker"] is True
    assert "SIGN-OFF BLOCKER" in "\n".join(_copy_lines(result))


def test_exact_copies_are_left_out_of_the_recall_denominator() -> None:
    """A copy mixed into the table does not change n_total."""
    prepared = [
        PreparedFake(
            object_id="realistic-tuning-00000",
            distance_mpc=40.0,
            judgeable=True,
            model_failed=False,
            fade_passed=True,
            color_value=0.5,
            rises={(2.0, 0.5): 0.4},
        )
    ]
    combo = {
        "redder_than_g_minus_r": 0.0,
        "min_g_minus_r_rise": 0.1,
        "reddening_window_days": 2.0,
        "min_color_separation_days": 0.5,
    }
    # The scorer's exclusion happens before prepare_fakes. Score the
    # prepared list, and check a frame that still contains a copy is
    # reduced by the helper the scorer uses.
    row = score_prepared(
        prepared,
        combo,
        population="realistic",
        day=3.0,
        status="ready",
    )
    assert row["n_total"] == 1
    assert row["n_caught"] == 1
    assert row["n_caught_not_judgeable"] == 0
    events = pd.DataFrame(
        {
            "object_id": ["realistic-tuning-00000", f"{EXACT_KIND}-000"],
            "kind": ["realistic", EXACT_KIND],
            "distance_mpc": [40.0, 40.0],
            "merger_mjd": [60000.0, 60000.0],
            "model_failed": [0, 0],
        }
    )
    from kilonova_hunter.grid import _without_exact_copies

    kept = _without_exact_copies(events)
    assert list(kept["object_id"]) == ["realistic-tuning-00000"]


def test_fast_scores_match_the_filter_on_the_demo_curve() -> None:
    """The grid's shortcut and is_kilonova_candidate agree."""
    curve = simulate_kilonova_light_curve(noise_mag=0.0)
    visible = curve.loc[curve["time_days"] <= 3.0]
    combo = {
        "redder_than_g_minus_r": THRESHOLDS.redder_than_g_minus_r,
        "min_g_minus_r_rise": THRESHOLDS.min_g_minus_r_rise,
        "reddening_window_days": THRESHOLDS.reddening_window_days,
        "min_color_separation_days": THRESHOLDS.min_color_separation_days,
    }
    assert fast_verdict_matches_filter(visible, combo) is True


def test_observing_one_fake_does_not_ask_the_filter(monkeypatch) -> None:
    """The batch counter is not allowed to look at the filter."""

    def boom(*_args, **_kwargs):
        raise AssertionError("the filter was called during simulation")

    monkeypatch.setattr("kilonova_hunter.filter.is_kilonova_candidate", boom)
    monkeypatch.setattr("kilonova_hunter.recall.is_kilonova_candidate", boom)
    from kilonova_hunter.recall import SurveySettings, at2017gfo_like_event
    from kilonova_hunter.sample import _simulate_one

    survey = SurveySettings(
        cadence="simple",
        revisit_gap_days=0.5,
        nightly_gap_days=0.0,
        survey_length_days=2.0,
    )
    _simulate_one(
        {
            "population": "stub",
            "event": at2017gfo_like_event(),
            "survey": survey,
            "seed": 1,
        }
    )


def test_mass_is_clipped_and_speed_is_redrawn() -> None:
    """A very wide mass draw piles up on the edge. A wide speed draw does not."""
    rng = np.random.default_rng(4)
    wide_mass = VillarSpread(mass_sigma_dex=8.0, mass_clip_dex=0.5)
    masses = [clipped_lognormal_mass(rng, 0.020, wide_mass) for _ in range(400)]
    log_mass = np.log10(masses)
    center = np.log10(0.020)
    assert np.all(np.abs(log_mass - center) <= 0.5 + 1e-9)
    on_edge = np.mean(np.isclose(np.abs(log_mass - center), 0.5, atol=1e-6))
    assert on_edge > 0.8
    wide_speed = VillarSpread(velocity_sigma_fraction=30.0)
    speeds = [truncated_normal_speed(rng, 0.266, wide_speed) for _ in range(400)]
    assert min(speeds) >= 0.05
    assert max(speeds) <= 0.4
    on_a_limit = np.mean(
        np.isclose(speeds, 0.05, atol=1e-8) | np.isclose(speeds, 0.4, atol=1e-8)
    )
    assert on_a_limit < 0.05


def _row(recall: float, color: float, rise: float, gap: float, window: float) -> dict:
    return {
        "recall_judgeable": recall,
        "redder_than_g_minus_r": color,
        "min_g_minus_r_rise": rise,
        "min_color_separation_days": gap,
        "reddening_window_days": window,
        "untestable": False,
    }

"""Decision 12(22): operational pairing and optional fade measurement."""

from dataclasses import replace

import pandas as pd
import pytest

from kilonova_hunter.config import THRESHOLDS
from kilonova_hunter.features import latest_same_night_color
from kilonova_hunter.filter import evaluate_fade_rule


def curve(rows):
    frame = pd.DataFrame(
        rows, columns=["time_days", "filter", "magnitude", "magnitude_error"]
    )
    frame.attrs["merger_mjd"] = 60000.0
    return frame


def test_latest_cut_snr_and_single_band():
    frame = curve(
        [
            (0.6, "g", 21, 0.1),
            (0.62, "r", 20, 0.1),
            (1.6, "g", 23, 0.1),
            (1.62, "r", 20, 0.1),
            (2.6, "g", 30, 0.1),
            (2.62, "r", 20, 0.3),
            (3.6, "g", 40, 0.1),
            (3.62, "r", 20, 0.1),
        ]
    )
    color = latest_same_night_color(frame, day=3)
    assert color.g_minus_r == 3
    assert color.time_days == 1.62
    assert latest_same_night_color(frame, day=0.61).g_minus_r is None


def test_multiple_points_lexicographic_pair():
    frame = curve(
        [
            (1.6, "g", 21, 0.1),
            (1.7, "g", 23, 0.1),
            (1.62, "r", 20, 0.1),
            (1.8, "r", 19, 0.1),
        ]
    )
    assert latest_same_night_color(frame).g_minus_r == 4


def test_absolute_noon_boundary_and_explicit_ztf_nights():
    frame = curve([(0.49, "g", 21, 0.1), (0.51, "r", 20, 0.1)])
    assert latest_same_night_color(frame).g_minus_r is None
    frame["night"] = [123, 123]
    assert latest_same_night_color(frame).g_minus_r == 1
    frame["night"] = [123, 124]
    assert latest_same_night_color(frame).g_minus_r is None


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [(1, "g", 21, 0.1)],
        [(1, "r", 20, 0.1)],
        [(1, "g", 21, 0.3), (1, "r", 20, 0.1)],
    ],
)
def test_unreadable(rows):
    assert latest_same_night_color(curve(rows)).g_minus_r is None


def test_default_fade_is_exactly_legacy_and_variant_skips():
    frame = curve([(0, "g", 18, 0.1), (2, "g", 18.2, 0.1)])
    legacy = evaluate_fade_rule(frame)
    assert legacy == evaluate_fade_rule(
        frame, replace(THRESHOLDS, fade_requires_two_after_peak=False)
    )
    assert legacy.passed is False
    assert (
        evaluate_fade_rule(
            frame, replace(THRESHOLDS, fade_requires_two_after_peak=True)
        ).passed
        is None
    )


def test_variant_fits_only_two_strictly_after_peak_inside_window():
    settings = replace(THRESHOLDS, fade_requires_two_after_peak=True)
    frame = curve(
        [(0, "g", 18, 0.1), (1, "g", 18.1, 0.1), (2, "g", 18.2, 0.1), (4, "g", 30, 0.1)]
    )
    assert evaluate_fade_rule(frame).passed is False
    assert evaluate_fade_rule(frame, settings) == evaluate_fade_rule(frame)
    frame.loc[2, "magnitude"] = 20
    assert evaluate_fade_rule(frame, settings).passed is True
    frame["time_days"] = frame.time_days.astype(float)
    frame.loc[2, "time_days"] = 3.1
    assert evaluate_fade_rule(frame, settings).passed is None


def test_grid_variant_matches_direct_and_preserves_night_metadata():
    from kilonova_hunter.filter import is_kilonova_candidate
    from kilonova_hunter.grid import (
        caught_by,
        grid_combinations,
        prepare_fakes,
        thresholds_for,
    )

    frame = curve(
        [
            (0.49, "g", 21, 0.1),
            (0.51, "r", 20, 0.1),
            (2.6, "g", 22, 0.1),
            (2.62, "r", 21, 0.1),
        ]
    )
    events = pd.DataFrame(
        [
            dict(
                object_id="one",
                kind="2017-like",
                distance_mpc=40,
                model_failed=False,
                merger_mjd=60000.0,
            )
        ]
    )
    curves = frame.assign(object_id="one")
    for new_fade in (False, True):
        prepared = prepare_fakes(events, curves, 3, new_fade=new_fade)[0]
        for combo in grid_combinations():
            settings = replace(
                thresholds_for(combo), fade_requires_two_after_peak=new_fade
            )
            direct = is_kilonova_candidate(frame, settings)
            assert caught_by(prepared, combo)[0] == direct.is_candidate
            assert prepared.fade_skipped == (direct.fade.passed is None)


def test_new_copy_seed_changes_draw_and_is_recorded(tmp_path, monkeypatch):
    import kilonova_hunter.sample as sample
    from kilonova_hunter.sky import DustSource

    seen = []
    original = sample.draw_exact_copies

    def draw(*args, **kwargs):
        seen.append(kwargs["seed"])
        return original(*args, **kwargs)

    monkeypatch.setattr(sample, "draw_exact_copies", draw)

    # No physical model is needed to verify the seed/cache boundary.
    def jobs(payloads, workers):
        return [(job["event"].object_id, pd.DataFrame(), "") for job in payloads]

    monkeypatch.setattr(sample, "_run_jobs", jobs)
    for seed in (2026110401, 2026110401, 2026110402):
        result = sample.simulate_exact_copies(
            seed=seed,
            count=1,
            dust=DustSource("fallback", "fixture"),
            cache_dir=tmp_path,
        )
        import json

        assert (
            json.loads((result.cache_dir / "manifest.json").read_text())["seed"] == seed
        )
    assert seen == [2026110401, 2026110402]


def test_empty_detection_history_counts_fade_skip_only_in_variant():
    from kilonova_hunter.grid import prepare_fakes

    events = pd.DataFrame(
        [
            dict(
                object_id="empty",
                kind="2017-like",
                distance_mpc=40,
                model_failed=False,
                merger_mjd=60000.0,
            )
        ]
    )
    curves = pd.DataFrame(
        columns=["object_id", "time_days", "filter", "magnitude", "magnitude_error"]
    )
    for variant in (False, True):
        fake = prepare_fakes(events, curves, 3, new_fade=variant)[0]
        assert fake.fade_skipped is variant
        assert fake.color_value is None
        assert fake.judgeable is False


def test_day_d_is_inclusive_for_color():
    frame = curve([(2.98, "g", 21, 0.1), (3.0, "r", 20, 0.1)])
    assert latest_same_night_color(frame, day=3.0).g_minus_r == 1
    assert latest_same_night_color(frame, day=2.999).g_minus_r is None


def test_copy_defaults_are_the_approved_fresh_seed_day_and_target():
    import inspect

    from kilonova_hunter import grid, sample

    # Inspect the actual call defaults, not explicit test-only arguments.
    assert (
        inspect.signature(sample.simulate_exact_copies).parameters["seed"].default
        == 2026110401
    )
    assert (
        inspect.signature(grid.run_tuning_grid).parameters["copy_seed"].default
        == 2026110401
    )
    assert inspect.signature(grid.score_exact_copies).parameters["day"].default == 3.0
    assert (
        inspect.signature(sample.simulate_exact_copies)
        .parameters["target_judgeable"]
        .default
        == 50
    )
    assert (
        inspect.signature(grid.sanity_copy_verdict).parameters["target"].default == 50
    )
    assert grid.sanity_copy_verdict(49, 49)["status"] == "untestable"


def test_tuning_reader_rejects_forged_labels(tmp_path):
    import json

    from kilonova_hunter.alerts import (
        FROZEN_SPLIT_PATH,
        TUNING_ALERTS_FILENAME,
        read_tuning_alerts,
    )

    frozen = json.loads(FROZEN_SPLIT_PATH.read_text())
    tuning = frozen["tuning"]["nights"][0]
    test = frozen["test"]["nights"][0]
    pd.DataFrame({"night": [tuning, test], "night_slice": ["tuning", "tuning"]}).to_csv(
        tmp_path / TUNING_ALERTS_FILENAME, index=False
    )
    assert read_tuning_alerts(tmp_path).night.tolist() == [tuning]


def test_missing_pair_counter_does_not_change_verdict():
    from kilonova_hunter.grid import PreparedFake, grid_combinations, score_prepared

    fake = PreparedFake("no-pair", 40, False, False, True, None, {(2.0, 0.5): None})
    row = score_prepared(
        [fake], grid_combinations()[0], population="2017-like", day=3, status="ready"
    )
    assert row["n_no_same_night_pair"] == 1
    assert row["n_caught"] == 0

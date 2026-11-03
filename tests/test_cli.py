"""Tests for the kilohunt command.

These call ``main`` directly, which is the function the ``kilohunt`` command
runs. The test then does not need the shell to find a ``kilohunt`` program
on ``PATH``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from kilonova_hunter.cli import _format_verdict, main
from kilonova_hunter.filter import RuleResult, is_kilonova_candidate
from kilonova_hunter.inject import simulate_kilonova_light_curve

# The sample curve committed for the README. Resolved from this file so
# the test still finds it when pytest is started from another folder.
_EXAMPLE_CSV = Path(__file__).resolve().parents[1] / "examples" / "demo_kilonova.csv"


def test_help_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` exits successfully and mentions the commands.

    argparse prints the help and then exits the program. From inside a test
    that exit arrives as ``SystemExit``. A code of 0 means "this was help,
    not a failure." ``capsys`` is pytest's copy of what was printed.
    """
    with pytest.raises(SystemExit) as caught:
        main(["--help"])

    assert caught.value.code == 0
    printed = capsys.readouterr().out
    assert "demo" in printed
    assert "check" in printed
    assert "kilonova" in printed.lower()


def test_demo_writes_a_plot(tmp_path: Path) -> None:
    """``demo`` saves the made-up light curve where ``--output`` points."""
    destination = tmp_path / "demo.png"
    main(["demo", "--output", str(destination), "--seed", "42"])
    assert destination.is_file()
    assert destination.read_bytes().startswith(b"\x89PNG")


def test_demo_prints_the_noise_free_verdict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``demo`` prints the filter verdict for the curve with the noise left off.

    The plot still uses ``--seed``. The sentences do not: they are the
    ones ``is_kilonova_candidate`` returns for ``noise_mag=0``, so a
    different seed cannot change the verdict.
    """
    destination = tmp_path / "demo.png"
    main(["demo", "--output", str(destination), "--seed", "7"])
    printed = capsys.readouterr().out

    verdict = is_kilonova_candidate(simulate_kilonova_light_curve(noise_mag=0.0))
    assert verdict.is_candidate
    verdict_text = printed.split("Filter verdict for the noise-free curve:", 1)[1]
    assert verdict_text.strip().splitlines()[0] == "CANDIDATE"
    assert verdict.fade.explanation in verdict_text
    assert verdict.color.explanation in verdict_text
    assert verdict.reddening.explanation in verdict_text
    assert destination.is_file()


def test_demo_save_csv_feeds_check(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``demo --save-csv`` writes the noise-free curve, and check accepts it."""
    plot_path = tmp_path / "demo.png"
    csv_path = tmp_path / "nested" / "demo.csv"
    main(["demo", "--output", str(plot_path), "--save-csv", str(csv_path)])
    assert csv_path.is_file()
    capsys.readouterr()

    main(["check", str(csv_path)])
    printed = capsys.readouterr().out
    assert printed.splitlines()[0] == "CANDIDATE"


def test_check_noise_free_demo_is_a_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The made-up kilonova, with the noise left off, is a CANDIDATE.

    The file is the same table ``simulate_kilonova_light_curve`` returns.
    Every rule's sentence is in the printout, which is how we know the
    command walked the verdict instead of printing only the first line.
    On this curve reddening passes, so the line is PASS, not SKIPPED.
    """
    curve = simulate_kilonova_light_curve(noise_mag=0.0)
    path = tmp_path / "demo.csv"
    curve.to_csv(path, index=False)
    verdict = is_kilonova_candidate(pd.read_csv(path))

    main(["check", str(path)])

    captured = capsys.readouterr()
    printed = captured.out
    assert printed.splitlines()[0] == "CANDIDATE"
    assert f"Fade: PASS. {verdict.fade.explanation}" in printed
    assert f"Color: PASS. {verdict.color.explanation}" in printed
    assert f"Reddening: PASS. {verdict.reddening.explanation}" in printed
    assert captured.err == ""


def test_check_slow_fade_is_not_a_candidate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A curve fading at 0.1 mag/day is NOT A CANDIDATE.

    0.1 is well below the 0.3 mag/day cut. The color is held red, so the
    failure is the fade rule, and the fade rule's own sentence is what
    gets printed. Exit code 0: a valid file is not an error.
    """
    curve = _slow_fading_curve()
    path = tmp_path / "slow.csv"
    curve.to_csv(path, index=False)
    verdict = is_kilonova_candidate(pd.read_csv(path))
    assert not verdict.is_candidate
    assert not verdict.fade.passed

    main(["check", str(path)])

    printed = capsys.readouterr().out
    assert printed.splitlines()[0] == "NOT A CANDIDATE"
    assert f"Fade: FAIL. {verdict.fade.explanation}" in printed
    assert "fade rule fails" in verdict.fade.explanation


def test_check_example_csv_is_a_candidate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CSV in ``examples/`` is the noise-free demo, and check agrees.

    Magnitudes are written with four decimal places, so the file does
    not contain a binary tail such as ``20.900000000000002``.
    """
    text = _EXAMPLE_CSV.read_text(encoding="utf-8")
    assert "20.9000" in text
    assert "000000" not in text

    saved = pd.read_csv(_EXAMPLE_CSV)
    expected = simulate_kilonova_light_curve(noise_mag=0.0)
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False)

    main(["check", str(_EXAMPLE_CSV)])
    printed = capsys.readouterr().out
    assert printed.splitlines()[0] == "CANDIDATE"


def test_check_missing_file_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A path that is not on disk exits nonzero and says the file is missing."""
    missing = tmp_path / "no-such-light-curve.csv"
    with pytest.raises(SystemExit) as caught:
        main(["check", str(missing)])

    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(missing) in captured.err
    assert "does not exist" in captured.err


@pytest.mark.parametrize(
    ("bad_column", "time_days", "magnitude"),
    [
        ("time_days", "noon", "19.5"),
        ("magnitude", "1.0", "bright"),
    ],
)
def test_check_non_numeric_value_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bad_column: str,
    time_days: str,
    magnitude: str,
) -> None:
    """A word in ``time_days`` or ``magnitude`` exits 1 instead of crashing.

    Those two columns are the numbers the filter fits. Text there used
    to raise ``ValueError: could not convert string to float`` from
    inside the fit. The command should stop with a sentence that names
    the column, and it should not print a verdict.
    """
    path = tmp_path / "words.csv"
    path.write_text(
        f"time_days,filter,magnitude,magnitude_error\n{time_days},g,{magnitude},0.05\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as caught:
        main(["check", str(path)])

    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert bad_column in captured.err
    assert "number" in captured.err.lower()


def test_check_missing_columns_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A CSV without the light-curve columns exits nonzero and names them.

    The header is wrong on purpose. The message has to say which columns
    are missing, so a person can fix the file instead of guessing.
    """
    path = tmp_path / "notes.csv"
    path.write_text("day,brightness\n1,19.5\n", encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        main(["check", str(path)])

    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "missing columns" in captured.err
    assert "time_days" in captured.err
    assert "filter" in captured.err
    assert "magnitude" in captured.err
    assert "magnitude_error" in captured.err


def test_verdict_text_lists_every_rule_including_a_skipped_one() -> None:
    """The printer walks every rule, and a skip is ``passed`` set to None.

    There is no separate skipped flag. Reddening uses ``None`` when it
    cannot compare two colors. An extra rule on this stand-in uses
    ``None`` too, so the loop is not a hard-coded list of three names.
    The first line stays ``CANDIDATE`` because ``is_candidate`` is true.
    A skip does not block that.
    """

    @dataclass
    class _VerdictWithExtraRules:
        is_candidate: bool
        fade: RuleResult
        color: RuleResult
        reddening: RuleResult
        not_applied: RuleResult

    verdict = _VerdictWithExtraRules(
        is_candidate=True,
        fade=RuleResult(passed=True, explanation="The fade rule passes."),
        color=RuleResult(passed=True, explanation="The color rule passes."),
        reddening=RuleResult(
            passed=None,
            explanation="Reddening was skipped because there was only one color.",
        ),
        not_applied=RuleResult(
            passed=None,
            explanation="This rule was not applied.",
        ),
    )
    lines = _format_verdict(verdict).splitlines()
    reddening_line = (
        "Reddening: SKIPPED. Reddening was skipped because there was only one color."
    )
    assert lines == [
        "CANDIDATE",
        "Fade: PASS. The fade rule passes.",
        "Color: PASS. The color rule passes.",
        reddening_line,
        "Not applied: SKIPPED. This rule was not applied.",
    ]


def test_check_skipped_reddening_does_not_reject(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One color skips reddening, and the object can still be a candidate.

    r fades quickly, and the only g point is red, so fade and color
    pass. There is no second color, so reddening's ``passed`` is
    ``None``. ``_measured_rules_passed`` leaves that rule out, and the
    printed first line stays CANDIDATE.
    """
    curve = pd.DataFrame(
        {
            "time_days": [0.0, 1.0, 2.0, 3.0, 2.0],
            "filter": ["r", "r", "r", "r", "g"],
            "magnitude": [20.6, 19.6, 20.1, 20.6, 20.5],
            "magnitude_error": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    path = tmp_path / "one_color.csv"
    curve.to_csv(path, index=False)
    verdict = is_kilonova_candidate(pd.read_csv(path))
    assert verdict.fade.passed is True
    assert verdict.color.passed is True
    assert verdict.reddening.passed is None
    assert verdict.is_candidate is True

    main(["check", str(path)])

    printed = capsys.readouterr().out
    assert printed.splitlines()[0] == "CANDIDATE"
    assert f"Reddening: SKIPPED. {verdict.reddening.explanation}" in printed


def _slow_fading_curve() -> pd.DataFrame:
    """A rise, then a fade of 0.1 mag/day, with g minus r held at 0.4.

    The same slope is given to g and r. The color stays red, so a failure
    here is the fade rule and not the color rule. 0.1 mag/day is the slow
    fade of an ordinary supernova, not the fast fade of a kilonova.
    """
    times = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    rows: list[dict[str, float | str]] = []
    for band, color_offset in (("r", 0.0), ("g", 0.4)):
        for time_days in times:
            days_from_peak = time_days - 1.0
            # Before the peak the object is brightening toward day 1.
            # After the peak it fades, slowly, at 0.1 magnitudes per day.
            rate = 1.0 if days_from_peak < 0.0 else 0.1
            rows.append(
                {
                    "time_days": time_days,
                    "filter": band,
                    "magnitude": 19.6 + color_offset + rate * abs(days_from_peak),
                    "magnitude_error": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _frozen_cuts(path: Path, *, frozen: bool) -> Path:
    path.write_text(
        json.dumps(
            {
                "frozen": frozen,
                "redder_than_g_minus_r": 0.0,
                "min_g_minus_r_rise": 0.1,
                "reddening_window_days": 2.0,
                "min_color_separation_days": 0.5,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_score_test_refuses_an_existing_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second score is refused before it can read the cache."""
    output = tmp_path / "results"
    output.mkdir()
    (output / "test_score.md").write_text("already scored\n", encoding="utf-8")
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--output",
                str(output),
                "--cache",
                str(tmp_path / "cache"),
            ]
        )
    assert caught.value.code == 1
    assert "already exists" in capsys.readouterr().err
    assert not (tmp_path / "cache").exists()


def test_score_test_requires_frozen_cuts_and_the_named_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unfrozen cuts are refused, and the cache path is the one --cache named."""
    output = tmp_path / "results"
    output.mkdir()
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=False)
    cache = tmp_path / "not-the-default-cache"
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--output",
                str(output),
                "--cache",
                str(cache),
            ]
        )
    assert caught.value.code == 1
    assert "frozen" in capsys.readouterr().err
    assert not (output / "test_score.md").exists()

    _frozen_cuts(cuts, frozen=True)
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--output",
                str(output),
                "--cache",
                str(cache),
            ]
        )
    assert caught.value.code == 1
    error = capsys.readouterr().err
    assert str(cache) in error
    assert "does not simulate" in error


def test_score_test_requires_sfd_provenance_on_the_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cache that does not name the SFD map cannot score the test set."""
    output = tmp_path / "results"
    output.mkdir()
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    for population in ("2017-like", "realistic"):
        folder = cache / population / "test"
        folder.mkdir(parents=True)
        (folder / "events.csv").write_text("object_id\n", encoding="utf-8")
        (folder / "light_curves.csv").write_text(
            "object_id,time_days,filter,magnitude,magnitude_error\n",
            encoding="utf-8",
        )
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "ready",
                    "dust_mode": "sfd",
                    "cache_version": 3,
                    "diffusion_fix": "arnett-factor-2",
                    "n_simulated": 0,
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--output",
                str(output),
                "--cache",
                str(cache),
            ]
        )
    assert caught.value.code == 1
    assert "SFD provenance" in capsys.readouterr().err
    assert not (output / "test_score.md").exists()


def test_score_test_rejects_a_continue_cache_and_a_second_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unfinished cache is refused, and a second output directory is too."""
    from kilonova_hunter.redback_fix import DIFFUSION_FIX_ID
    from kilonova_hunter.sample import CACHE_VERSION
    from kilonova_hunter.sky import dust_map_label

    output = tmp_path / "results"
    output.mkdir()
    other = tmp_path / "other-results"
    other.mkdir()
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    for population in ("2017-like", "realistic"):
        folder = cache / population / "test"
        folder.mkdir(parents=True)
        (folder / "events.csv").write_text("object_id\n", encoding="utf-8")
        (folder / "light_curves.csv").write_text(
            "object_id,time_days,filter,magnitude,magnitude_error\n",
            encoding="utf-8",
        )
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "continue",
                    "dust_mode": "sfd",
                    "dust_map": dust_map_label(),
                    "cache_version": CACHE_VERSION,
                    "diffusion_fix": DIFFUSION_FIX_ID,
                    "n_simulated": 0,
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--output",
                str(output),
                "--cache",
                str(cache),
            ]
        )
    assert caught.value.code == 1
    assert "continue" in capsys.readouterr().err

    for population in ("2017-like", "realistic"):
        manifest = cache / population / "test" / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["status"] = "ready"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    main(
        [
            "recall",
            "--score-test",
            "--cuts",
            str(cuts),
            "--output",
            str(output),
            "--cache",
            str(cache),
        ]
    )
    assert (cache / "test_scored_once.json").is_file()
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--output",
                str(other),
                "--cache",
                str(cache),
            ]
        )
    assert caught.value.code == 1
    assert "scored once" in capsys.readouterr().err
    assert not (other / "test_score.md").exists()


def _ready_test_cache(cache: Path) -> None:
    """Two empty finished test splits with the current cache identity."""
    from kilonova_hunter.redback_fix import DIFFUSION_FIX_ID
    from kilonova_hunter.sample import CACHE_VERSION
    from kilonova_hunter.sky import dust_map_label

    for population in ("2017-like", "realistic"):
        folder = cache / population / "test"
        folder.mkdir(parents=True)
        (folder / "events.csv").write_text("object_id\n", encoding="utf-8")
        (folder / "light_curves.csv").write_text(
            "object_id,time_days,filter,magnitude,magnitude_error\n",
            encoding="utf-8",
        )
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "ready",
                    "dust_mode": "sfd",
                    "dust_map": dust_map_label(),
                    "cache_version": CACHE_VERSION,
                    "diffusion_fix": DIFFUSION_FIX_ID,
                    "n_simulated": 0,
                }
            ),
            encoding="utf-8",
        )


def _score_argv(argv: list[str]) -> int:
    """Run one score in a child process. Returns the exit code."""
    from kilonova_hunter.cli import main

    try:
        main(argv)
    except SystemExit as exc:
        if exc.code in (None, 0):
            return 0
        return int(exc.code)
    return 0


def test_score_marker_is_exclusive_and_survives_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The marker is created before scoring. A crash still blocks a second run."""
    output = tmp_path / "results"
    output.mkdir()
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    _ready_test_cache(cache)

    def boom(*args, **kwargs):
        raise RuntimeError("scoring crashed")

    monkeypatch.setattr("kilonova_hunter.grid.score_with_frozen_cuts", boom)
    with pytest.raises(RuntimeError, match="scoring crashed"):
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--output",
                str(output),
                "--cache",
                str(cache),
            ]
        )
    marker = cache / "test_scored_once.json"
    assert marker.is_file()
    assert not (output / "test_score.md").exists()
    # A changed cache does not clear the marker.
    manifest = cache / "2017-like" / "test" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["note"] = "cache changed after the crash"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--output",
                str(other),
                "--cache",
                str(cache),
            ]
        )
    assert caught.value.code == 1
    assert "scored once" in capsys.readouterr().err
    assert marker.is_file()


def test_two_concurrent_score_runs_let_one_through(tmp_path: Path) -> None:
    """O_EXCL lets one of two simultaneous scores create the marker."""
    import multiprocessing

    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    shared = [
        "recall",
        "--score-test",
        "--cuts",
        str(cuts),
        "--cache",
        str(cache),
    ]
    argv_a = [*shared, "--output", str(first)]
    argv_b = [*shared, "--output", str(second)]
    context = multiprocessing.get_context("spawn")
    with context.Pool(2) as pool:
        codes = pool.map(_score_argv, [argv_a, argv_b])
    assert sorted(codes) == [0, 1]
    assert (cache / "test_scored_once.json").is_file()
    reports = [first / "test_score.md", second / "test_score.md"]
    assert sum(path.is_file() for path in reports) == 1


def _grid_argv(tmp_path: Path) -> list[str]:
    return [
        "recall",
        "--grid",
        "--output",
        str(tmp_path / "out"),
        "--cache",
        str(tmp_path / "cache"),
    ]


def _patch_grid(monkeypatch: pytest.MonkeyPatch, status: str | None) -> None:
    from kilonova_hunter.grid import GridRun

    monkeypatch.setattr("kilonova_hunter.cli.redback_available", lambda: True)

    def fake_grid(**kwargs):
        return GridRun(
            text="tuning grid\n",
            sign_off_blocker=status is not None,
            copy_status=status,
        )

    monkeypatch.setattr("kilonova_hunter.grid.run_tuning_grid", fake_grid)


def test_grid_exits_2_when_the_copy_verdict_is_untestable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """UNTESTABLE is a sign-off blocker. The grid command must not exit 0."""
    _patch_grid(monkeypatch, "untestable")
    with pytest.raises(SystemExit) as caught:
        main(_grid_argv(tmp_path))
    assert caught.value.code == 2
    assert "UNTESTABLE" in capsys.readouterr().err


def test_grid_exits_2_when_the_copy_verdict_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FAIL is a sign-off blocker. The grid command must not exit 0."""
    _patch_grid(monkeypatch, "fail")
    with pytest.raises(SystemExit) as caught:
        main(_grid_argv(tmp_path))
    assert caught.value.code == 2
    assert "FAIL" in capsys.readouterr().err


def test_grid_exits_0_when_the_copy_verdict_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A passing copy verdict does not change the exit code."""
    _patch_grid(monkeypatch, None)
    main(_grid_argv(tmp_path))
    assert "Sign-off blocker" not in capsys.readouterr().err

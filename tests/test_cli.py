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
    tmp_path: Path,
    scored_test_set_ledger: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unfinished cache is refused, and a second output directory is too."""
    from kilonova_hunter.redback_fix import DIFFUSION_FIX_ID
    from kilonova_hunter.sample import CACHE_VERSION
    from kilonova_hunter.sky import dust_map_label

    assert scored_test_set_ledger == tmp_path / "test_set_scored.json"
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
    assert (tmp_path / "test_set_scored.json").is_file()
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
                    "redback_version": "1.20.0",
                    "seed": 7,
                    "parent_seed": 11,
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
    tmp_path: Path,
    scored_test_set_ledger: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The marker is created before scoring. A crash still blocks a second run."""
    assert scored_test_set_ledger == tmp_path / "test_set_scored.json"
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
    marker = tmp_path / "test_set_scored.json"
    assert marker.is_file()
    assert any((tmp_path / "test_set_claims").iterdir())
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


def test_two_concurrent_score_runs_let_one_through(
    tmp_path: Path, scored_test_set_ledger: Path
) -> None:
    """O_EXCL lets one of two simultaneous scores create the marker.

    The child processes are forked so they inherit the ledger fixture.
    A spawned interpreter would import a fresh module and write the
    repository ledger instead.
    """
    import multiprocessing

    assert scored_test_set_ledger == tmp_path / "test_set_scored.json"
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
    context = multiprocessing.get_context("fork")
    with context.Pool(2) as pool:
        codes = pool.map(_score_argv, [argv_a, argv_b])
    assert sorted(codes) == [0, 1]
    assert (tmp_path / "test_set_scored.json").is_file()
    reports = [first / "test_score.md", second / "test_score.md"]
    assert sum(path.is_file() for path in reports) == 1


def test_a_fresh_cache_does_not_reset_the_score_and_a_new_set_can(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Copying the cache is refused. A different fake list is a new test set."""
    import shutil

    assert scored_test_set_ledger == tmp_path / "test_set_scored.json"
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    output = tmp_path / "out"
    output.mkdir()
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
    copied = tmp_path / "fresh-cache"
    shutil.copytree(cache, copied)
    other = tmp_path / "other-out"
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
                str(copied),
            ]
        )
    assert caught.value.code == 1
    assert "fresh --cache" in capsys.readouterr().err
    manifest_path = copied / "2017-like" / "test" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["parent_seed"] = 999
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    third = tmp_path / "third-out"
    third.mkdir()
    main(
        [
            "recall",
            "--score-test",
            "--cuts",
            str(cuts),
            "--output",
            str(third),
            "--cache",
            str(copied),
        ]
    )
    assert (third / "test_score.md").is_file()


def test_provenance_changes_are_refused_and_a_new_seed_is_allowed(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ruling 12(21): the key is seeds, fake ids, and the population spec.

    cache_version, redback_version, diffusion_fix, and the dust map are
    recorded, not hashed. Changing one of them does not open a rescore.
    A different seed does. There is no override flag.
    """
    import shutil
    import subprocess

    from kilonova_hunter.redback_fix import DIFFUSION_FIX_ID
    from kilonova_hunter.sample import CACHE_VERSION
    from kilonova_hunter.sky import dust_map_label

    assert scored_test_set_ledger == tmp_path / "test_set_scored.json"
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    output = tmp_path / "out"
    output.mkdir()
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
    ledger = json.loads(scored_test_set_ledger.read_text(encoding="utf-8"))
    entry = next(iter(ledger["sets"].values()))[0]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert entry["git_sha"] == head
    assert entry["scored_at"].endswith("Z")
    for population in ("2017-like", "realistic"):
        recorded = entry["provenance"][population]
        assert recorded["cache_version"] == CACHE_VERSION
        assert recorded["redback_version"] == "1.20.0"
        assert recorded["diffusion_fix"] == DIFFUSION_FIX_ID
        assert recorded["dust_map"] == dust_map_label()
    changes = {
        "cache_version": 99,
        "redback_version": "9.9.9",
        "diffusion_fix": "not-the-fix",
        "dust_map": "a different map",
    }
    for field, value in changes.items():
        copied = tmp_path / f"cache-{field}"
        shutil.copytree(cache, copied)
        for population in ("2017-like", "realistic"):
            manifest_path = copied / population / "test" / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload[field] = value
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        refused = tmp_path / f"out-{field}"
        refused.mkdir()
        with pytest.raises(SystemExit) as caught:
            main(
                [
                    "recall",
                    "--score-test",
                    "--cuts",
                    str(cuts),
                    "--output",
                    str(refused),
                    "--cache",
                    str(copied),
                ]
            )
        assert caught.value.code == 1
        err = capsys.readouterr().err
        assert "logged decision" in err
        assert "both scores must be reported" in err
        assert not (refused / "test_score.md").exists()
    seeded = tmp_path / "cache-seed"
    shutil.copytree(cache, seeded)
    for population in ("2017-like", "realistic"):
        manifest_path = seeded / population / "test" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["seed"] = 99
        payload["parent_seed"] = 99
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    allowed = tmp_path / "out-seed"
    allowed.mkdir()
    main(
        [
            "recall",
            "--score-test",
            "--cuts",
            str(cuts),
            "--output",
            str(allowed),
            "--cache",
            str(seeded),
        ]
    )
    assert (allowed / "test_score.md").is_file()
    again = json.loads(scored_test_set_ledger.read_text(encoding="utf-8"))
    assert len(again["sets"]) == 2


def test_an_approved_rescore_keeps_the_first_score(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A decision id appends a second entry. The first score stays."""
    assert scored_test_set_ledger == tmp_path / "test_set_scored.json"
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    output = tmp_path / "out"
    output.mkdir()
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
    ledger = json.loads(scored_test_set_ledger.read_text(encoding="utf-8"))
    test_hash, entries = next(iter(ledger["sets"].items()))
    first = dict(entries[0])
    assert "decision_id" not in first
    approved = _frozen_cuts(tmp_path / "approved.json", frozen=True)
    payload = json.loads(approved.read_text(encoding="utf-8"))
    payload["decision_id"] = "PM-12-21"
    payload["redder_than_g_minus_r"] = 0.2
    approved.write_text(json.dumps(payload), encoding="utf-8")
    second = tmp_path / "second"
    second.mkdir()
    main(
        [
            "recall",
            "--score-test",
            "--cuts",
            str(approved),
            "--output",
            str(second),
            "--cache",
            str(cache),
        ]
    )
    again = json.loads(scored_test_set_ledger.read_text(encoding="utf-8"))
    kept = again["sets"][test_hash]
    assert kept[0] == first
    assert kept[1]["decision_id"] == "PM-12-21"
    assert kept[1]["cuts"]["redder_than_g_minus_r"] == 0.2
    assert (output / "test_score.md").is_file()
    assert (second / "test_score.md").is_file()
    capsys.readouterr()


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


def test_grid_exits_2_when_a_non_chosen_cell_breaks_rule_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A breach outside the placeholder cut exits 2 through ``_run_grid``.

    The population and the exact copies are a one-fake fixture. Scoring
    the 72 cells and the sign-off check are the real functions. The
    placeholder cut, g-r above 0 with a 2-day window, does not catch the
    fake. A wider window does, and that fake is not judgeable.
    """
    from types import SimpleNamespace

    from kilonova_hunter.cli import _run_grid
    from kilonova_hunter.grid import PreparedFake, sanity_copy_verdict
    from kilonova_hunter.sample import SampleResult
    from kilonova_hunter.sky import DustSource

    rises = {
        (window, gap): 0.0 if (window, gap) == (2.0, 0.5) else 0.4
        for window in (2.0, 3.0, 5.0)
        for gap in (0.5, 1.0)
    }
    fake = PreparedFake(
        object_id="wide-window",
        distance_mpc=40.0,
        judgeable=False,
        model_failed=False,
        fade_passed=True,
        color_value=0.5,
        rises=rises,
    )

    def fake_prepare(events, curves, day, settings=None):
        return [fake]

    def fake_population(population, split, **kwargs):
        return SampleResult(
            population=population,
            split=split,
            status="ready",
            n_simulated=1,
            n_judgeable_day3=0,
            n_judgeable_day5=0,
            n_judgeable_day7=0,
            n_model_failed=0,
            dust_mode="dry-run",
            dust_reason="fixture",
            events=pd.DataFrame(columns=["object_id", "kind", "redshift"]),
            light_curves=pd.DataFrame(),
            cache_dir=tmp_path,
        )

    def passing_copies(events, curves, combo, day=3.0):
        official = sanity_copy_verdict(50, 50)
        return {
            "n_copies": 50,
            "n_judgeable": 50,
            "n_judgeable_candidate": 50,
            "n_caught_judgeable": 50,
            "n_caught_candidate": 50,
            "blocker": False,
            "official": official,
            "candidate": official,
            "misses": [],
        }

    monkeypatch.setattr("kilonova_hunter.cli.redback_available", lambda: True)
    monkeypatch.setattr(
        "kilonova_hunter.sky.open_dust_source",
        lambda: DustSource("dry-run", "fixture"),
    )
    monkeypatch.setattr("kilonova_hunter.sample.simulate_population", fake_population)
    monkeypatch.setattr(
        "kilonova_hunter.sample.simulate_exact_copies",
        lambda **kwargs: fake_population("2017-like", "copies"),
    )
    monkeypatch.setattr("kilonova_hunter.grid.prepare_fakes", fake_prepare)
    monkeypatch.setattr("kilonova_hunter.grid.score_exact_copies", passing_copies)
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as caught:
        _run_grid(
            SimpleNamespace(
                output=output,
                cache=tmp_path / "cache",
                target=1,
                cap=1,
                floor=1,
                batch_size=1,
                workers=1,
            )
        )
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "rule 5" in err
    assert "not the chosen cut" in err
    saved = pd.read_csv(output / "tuning_grid.csv")
    chosen = saved[
        (saved["population"] == "2017-like")
        & (saved["day"] == 3.0)
        & (saved["distance_bin"] == "all")
        & (saved["redder_than_g_minus_r"] == 0.0)
        & (saved["min_g_minus_r_rise"] == 0.1)
        & (saved["reddening_window_days"] == 2.0)
        & (saved["min_color_separation_days"] == 0.5)
    ]
    other = saved[
        (saved["population"] == "2017-like")
        & (saved["day"] == 3.0)
        & (saved["distance_bin"] == "all")
        & (saved["reddening_window_days"] == 5.0)
        & (saved["min_g_minus_r_rise"] == 0.1)
    ]
    assert int(chosen["n_caught_not_judgeable"].iloc[0]) == 0
    assert int(other["n_caught_not_judgeable"].max()) > 0


def test_grid_exits_0_when_the_copy_verdict_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A passing copy verdict does not change the exit code."""
    _patch_grid(monkeypatch, None)
    main(_grid_argv(tmp_path))
    assert "Sign-off blocker" not in capsys.readouterr().err


def test_fetch_alerts_refuses_until_cuts_are_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without frozen cuts the command exits and writes nothing."""
    alerts = pd.DataFrame(
        [
            {
                "i:objectId": "a",
                "i:jd": 2450000.5,
                "i:fid": 1,
                "i:magpsf": 20.0,
                "i:sigmapsf": 0.05,
                "mjd": 50000.0,
                "night": 60000,
            },
            {
                "i:objectId": "b",
                "i:jd": 2450001.5,
                "i:fid": 1,
                "i:magpsf": 20.0,
                "i:sigmapsf": 0.05,
                "mjd": 50001.0,
                "night": 60001,
            },
        ]
    )
    monkeypatch.setattr(
        "kilonova_hunter.alerts.fetch_class_between",
        lambda start, stop, **kwargs: alerts,
    )
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--fetch-alerts",
                "--output",
                str(output),
                "--alert-start",
                "2024-06-01",
                "--alert-stop",
                "2024-06-02",
            ]
        )
    assert caught.value.code == 1
    assert "Nothing was written" in capsys.readouterr().err
    assert not (output / "real_alert_slices.json").exists()
    assert not (output / "ztf_proxy_alerts.csv").exists()

    unfrozen = tmp_path / "unfrozen.json"
    unfrozen.write_text(json.dumps({"frozen": False}), encoding="utf-8")
    still = tmp_path / "still"
    still.mkdir()
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--fetch-alerts",
                "--output",
                str(still),
                "--cuts",
                str(unfrozen),
                "--alert-start",
                "2024-06-01",
                "--alert-stop",
                "2024-06-02",
            ]
        )
    assert caught.value.code == 1
    assert not (still / "real_alert_slices.json").exists()
    assert not (still / "ztf_proxy_alerts.csv").exists()

    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    frozen_out = tmp_path / "frozen"
    frozen_out.mkdir()
    main(
        [
            "recall",
            "--fetch-alerts",
            "--output",
            str(frozen_out),
            "--cuts",
            str(cuts),
            "--alert-start",
            "2024-06-01",
            "--alert-stop",
            "2024-06-02",
        ]
    )
    frozen = json.loads(
        (frozen_out / "real_alert_slices.json").read_text(encoding="utf-8")
    )
    assert "n_flagged" in frozen["flag_counts"]["test"]
    assert "refused" not in frozen["flag_counts"]["test"]


def test_cuts_file_errors_are_not_frozen_and_do_not_raise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A list, missing keys, or bad JSON is not frozen, and does not traceback."""
    from kilonova_hunter.cli import _cuts_are_frozen

    missing = _frozen_cuts(tmp_path / "missing-keys.json", frozen=True)
    payload = json.loads(missing.read_text(encoding="utf-8"))
    del payload["min_g_minus_r_rise"]
    missing.write_text(json.dumps(payload), encoding="utf-8")
    listed = tmp_path / "list.json"
    listed.write_text("[1, 2]\n", encoding="utf-8")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    ok, message = _cuts_are_frozen(missing)
    assert ok is False
    assert "missing" in message
    ok, message = _cuts_are_frozen(listed)
    assert ok is False
    assert "JSON object" in message
    ok, message = _cuts_are_frozen(broken)
    assert ok is False
    assert "not valid JSON" in message

    for path, expected in (
        (missing, "missing"),
        (listed, "JSON object"),
        (broken, "not valid JSON"),
    ):
        output = tmp_path / path.stem
        output.mkdir()
        with pytest.raises(SystemExit) as caught:
            main(
                [
                    "recall",
                    "--fetch-alerts",
                    "--output",
                    str(output),
                    "--cuts",
                    str(path),
                    "--alert-start",
                    "2024-06-01",
                    "--alert-stop",
                    "2024-06-02",
                ]
            )
        assert caught.value.code == 1
        assert expected in capsys.readouterr().err
        assert not (output / "real_alert_slices.json").exists()
        assert not (output / "ztf_proxy_alerts.csv").exists()

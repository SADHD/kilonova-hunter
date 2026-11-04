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


def _events_table(*object_ids: str, population: str) -> str:
    """One events row per id, with the columns scoring reads."""
    lines = ["object_id,kind,redshift,model_failed,merger_mjd,distance_mpc"]
    for object_id in object_ids:
        lines.append(f"{object_id},{population},0.01,0,59000.0,40.0")
    return "\n".join(lines) + "\n"


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
        (folder / "events.csv").write_text(
            _events_table(f"{population}-sfd", population=population),
            encoding="utf-8",
        )
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
                    "n_simulated": 1,
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
        (folder / "events.csv").write_text(
            _events_table(f"{population}-1", population=population),
            encoding="utf-8",
        )
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
                    "n_simulated": 1,
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
        (folder / "events.csv").write_text(
            _events_table(f"{population}-1", population=population),
            encoding="utf-8",
        )
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
                    "sample_size": {"target_judgeable": 1},
                    "n_simulated": 1,
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
    # A pending entry counts as scored. The crash consumes the set.
    marker = tmp_path / "test_set_scored.json"
    pending = json.loads(marker.read_text(encoding="utf-8"))
    entry = next(iter(pending["sets"].values()))[0]
    assert entry["status"] == "pending"
    assert entry["object_ids"]["2017-like"] == ["2017-like-1"]
    assert "score_sha256" not in entry
    claim = next((tmp_path / "test_set_claims").iterdir())
    claim_body = json.loads(claim.read_text(encoding="utf-8"))
    assert claim_body["object_ids"]["2017-like"] == ["2017-like-1"]
    assert not (output / "test_score.md").exists()
    assert not (tmp_path / "test_set_scored.json.lock").exists()
    saved = marker.read_text(encoding="utf-8")
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
    assert marker.read_text(encoding="utf-8") == saved
    assert entry["status"] == "pending"


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
    for population in ("2017-like", "realistic"):
        folder = copied / population / "test"
        manifest_path = folder / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["parent_seed"] = 999
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        (folder / "events.csv").write_text(
            _events_table(f"{population}-new", population=population),
            encoding="utf-8",
        )
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
    import hashlib
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
        assert recorded["dust_mode"] == "sfd"
    score_file = output / "test_score.md"
    assert entry["score_path"] == str(score_file)
    assert entry["score_sha256"] == hashlib.sha256(score_file.read_bytes()).hexdigest()
    changes = {
        "cache_version": 99,
        "redback_version": "9.9.9",
        "diffusion_fix": "not-the-fix",
        "dust_map": "a different map",
        "dust_mode": "not-sfd",
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
        assert "logged decision approved by Sage through PM" in err
        assert "both scores must be reported" in err
        assert not (refused / "test_score.md").exists()
    seeded = tmp_path / "cache-seed"
    shutil.copytree(cache, seeded)
    for population in ("2017-like", "realistic"):
        folder = seeded / population / "test"
        manifest_path = folder / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["seed"] = 99
        payload["parent_seed"] = 99
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        (folder / "events.csv").write_text(
            _events_table(f"{population}-other-seed", population=population),
            encoding="utf-8",
        )
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


def _ledger_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def test_overlap_refuses_a_larger_sample_and_a_decision_appends(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same seed and a bigger sample shares ids, so it is not a new set.

    Disjoint ids are allowed, including an id that was scored only in
    the other population. A decision id appends the overlapping set.
    A missing events table or an empty object_id column writes nothing
    and leaves the ledger as it was.
    """
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
    before = _ledger_text(scored_test_set_ledger)
    first = json.loads(before)
    first_hash = next(iter(first["sets"]))
    first_entry = first["sets"][first_hash][0]
    assert first_entry["object_ids"]["2017-like"] == ["2017-like-1"]
    assert "sample_size" not in first_entry["population_spec"]["2017-like"]

    bigger = tmp_path / "bigger"
    shutil.copytree(cache, bigger)
    for population in ("2017-like", "realistic"):
        folder = bigger / population / "test"
        (folder / "events.csv").write_text(
            _events_table(f"{population}-1", f"{population}-2", population=population),
            encoding="utf-8",
        )
        manifest_path = folder / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["sample_size"] = {"target_judgeable": 2}
        payload["n_simulated"] = 2
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    refused = tmp_path / "bigger-out"
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
                str(bigger),
            ]
        )
    assert caught.value.code == 1
    err = capsys.readouterr().err
    assert "2 object ids" in err
    assert "logged decision approved by Sage through PM" in err
    assert not (refused / "test_score.md").exists()
    assert _ledger_text(scored_test_set_ledger) == before

    disjoint = tmp_path / "disjoint"
    shutil.copytree(cache, disjoint)
    # 2017-like gets a new id. realistic reuses the id that was scored
    # only for 2017-like. That id was not scored in the realistic population.
    (disjoint / "2017-like" / "test" / "events.csv").write_text(
        _events_table("2017-like-fresh", population="2017-like"),
        encoding="utf-8",
    )
    (disjoint / "realistic" / "test" / "events.csv").write_text(
        _events_table("2017-like-1", population="realistic"),
        encoding="utf-8",
    )
    disjoint_out = tmp_path / "disjoint-out"
    disjoint_out.mkdir()
    main(
        [
            "recall",
            "--score-test",
            "--cuts",
            str(cuts),
            "--output",
            str(disjoint_out),
            "--cache",
            str(disjoint),
        ]
    )
    assert (disjoint_out / "test_score.md").is_file()
    after_disjoint = json.loads(_ledger_text(scored_test_set_ledger))
    assert first_hash in after_disjoint["sets"]
    assert after_disjoint["sets"][first_hash][0] == first_entry
    assert len(after_disjoint["sets"]) == 2

    approved = _frozen_cuts(tmp_path / "approved.json", frozen=True)
    overlap_out = tmp_path / "overlap-out"
    overlap_out.mkdir()
    main(
        [
            "recall",
            "--score-test",
            "--cuts",
            str(approved),
            "--rescore-decision",
            "PM-overlap",
            "--output",
            str(overlap_out),
            "--cache",
            str(bigger),
        ]
    )
    appended = json.loads(_ledger_text(scored_test_set_ledger))
    assert appended["sets"][first_hash][0] == first_entry
    new_hash = next(
        key
        for key in appended["sets"]
        if key != first_hash and key not in after_disjoint["sets"]
    )
    recorded = appended["sets"][new_hash][-1]
    assert recorded["decision_id"] == "PM-overlap"
    assert "2017-like-1" in recorded["object_ids"]["2017-like"]
    assert (overlap_out / "test_score.md").is_file()

    saved = _ledger_text(scored_test_set_ledger)
    missing = tmp_path / "missing-events"
    shutil.copytree(bigger, missing)
    (missing / "2017-like" / "test" / "events.csv").unlink()
    missing_out = tmp_path / "missing-out"
    missing_out.mkdir()
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--output",
                str(missing_out),
                "--cache",
                str(missing),
            ]
        )
    assert caught.value.code == 1
    assert "events table is missing" in capsys.readouterr().err
    assert not (missing_out / "test_score.md").exists()
    assert _ledger_text(scored_test_set_ledger) == saved

    empty = tmp_path / "empty-ids"
    shutil.copytree(cache, empty)
    for population in ("2017-like", "realistic"):
        folder = empty / population / "test"
        (folder / "events.csv").write_text(
            "object_id,kind,redshift,model_failed,merger_mjd,distance_mpc\n",
            encoding="utf-8",
        )
        manifest_path = folder / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["n_simulated"] = 0
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    empty_out = tmp_path / "empty-out"
    empty_out.mkdir()
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--output",
                str(empty_out),
                "--cache",
                str(empty),
            ]
        )
    assert caught.value.code == 1
    assert "no object_id values" in capsys.readouterr().err
    assert not (empty_out / "test_score.md").exists()
    assert _ledger_text(scored_test_set_ledger) == saved


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
            "--rescore-decision",
            "PM-12-21",
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


def test_a_decision_id_in_the_cuts_file_does_not_authorize_a_rescore(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A decision id stored in the cuts JSON is ignored."""
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
    before = _ledger_text(scored_test_set_ledger)
    payload = json.loads(cuts.read_text(encoding="utf-8"))
    payload["decision_id"] = "PM-in-the-cuts-file"
    cuts.write_text(json.dumps(payload), encoding="utf-8")
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
    err = capsys.readouterr().err
    assert "scored once" in err
    assert "logged decision approved by Sage through PM" in err
    assert not (other / "test_score.md").exists()
    assert _ledger_text(scored_test_set_ledger) == before


def test_a_reused_decision_id_is_refused(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same decision id cannot approve this set twice."""
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    first = tmp_path / "first"
    first.mkdir()
    main(
        [
            "recall",
            "--score-test",
            "--cuts",
            str(cuts),
            "--output",
            str(first),
            "--cache",
            str(cache),
        ]
    )
    second = tmp_path / "second"
    second.mkdir()
    main(
        [
            "recall",
            "--score-test",
            "--cuts",
            str(cuts),
            "--rescore-decision",
            "PM-once",
            "--output",
            str(second),
            "--cache",
            str(cache),
        ]
    )
    saved = _ledger_text(scored_test_set_ledger)
    third = tmp_path / "third"
    third.mkdir()
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--rescore-decision",
                "PM-once",
                "--output",
                str(third),
                "--cache",
                str(cache),
            ]
        )
    assert caught.value.code == 1
    assert "already used" in capsys.readouterr().err
    assert not (third / "test_score.md").exists()
    assert _ledger_text(scored_test_set_ledger) == saved


def test_rescore_decision_on_a_never_scored_set_is_refused(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The flag does not score a set that has no earlier score."""
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    output = tmp_path / "out"
    output.mkdir()
    before = _ledger_text(scored_test_set_ledger)
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--rescore-decision",
                "PM-too-soon",
                "--output",
                str(output),
                "--cache",
                str(cache),
            ]
        )
    assert caught.value.code == 1
    assert "has not been scored" in capsys.readouterr().err
    assert not (output / "test_score.md").exists()
    assert _ledger_text(scored_test_set_ledger) == before
    assert not (tmp_path / "test_set_claims").exists()


def test_score_test_refuses_a_non_numeric_cut(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A cut that is not a number is refused before anything is written."""
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    payload = json.loads(cuts.read_text(encoding="utf-8"))
    payload["min_g_minus_r_rise"] = "red"
    cuts.write_text(json.dumps(payload), encoding="utf-8")
    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    output = tmp_path / "out"
    output.mkdir()
    before = _ledger_text(scored_test_set_ledger)
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
    err = capsys.readouterr().err
    assert "finite number" in err
    assert "Nothing was written" in err
    assert not (output / "test_score.md").exists()
    assert _ledger_text(scored_test_set_ledger) == before
    for bad in ("nan", "inf", "-inf"):
        payload["min_g_minus_r_rise"] = bad
        cuts.write_text(json.dumps(payload), encoding="utf-8")
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
        assert "finite number" in capsys.readouterr().err
        assert _ledger_text(scored_test_set_ledger) == before


def test_a_zero_byte_events_table_is_refused(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty events file is a clean refusal, not a pandas traceback."""
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    events = cache / "2017-like" / "test" / "events.csv"
    events.write_bytes(b"")
    output = tmp_path / "out"
    output.mkdir()
    before = _ledger_text(scored_test_set_ledger)
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
    err = capsys.readouterr().err
    assert "empty" in err
    assert "Nothing was written" in err
    assert "Traceback" not in err
    assert not (output / "test_score.md").exists()
    assert _ledger_text(scored_test_set_ledger) == before


def test_a_crash_after_the_score_file_keeps_the_pending_ids_blocked(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A kill after the score file and before completion still blocks a rescore.

    The pending entry stays. A larger draw of the same seed is refused
    without --rescore-decision, and the ledger bytes do not change.
    An unused decision appends a new entry and leaves the pending one.
    """
    import shutil

    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    output = tmp_path / "out"
    output.mkdir()

    from kilonova_hunter import cli as cli_mod

    real_complete = cli_mod._complete_pending_score

    def killed(test_hash, score_path):
        raise RuntimeError("killed before the ledger was completed")

    monkeypatch.setattr(cli_mod, "_complete_pending_score", killed)
    with pytest.raises(RuntimeError, match="killed before"):
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
    assert (output / "test_score.md").is_file()
    saved = _ledger_text(scored_test_set_ledger)
    pending = json.loads(saved)
    old_hash, old_entries = next(iter(pending["sets"].items()))
    assert old_entries[0]["status"] == "pending"
    assert "score_sha256" not in old_entries[0]
    assert not (tmp_path / "test_set_scored.json.lock").exists()

    bigger = tmp_path / "bigger"
    shutil.copytree(cache, bigger)
    for population in ("2017-like", "realistic"):
        folder = bigger / population / "test"
        (folder / "events.csv").write_text(
            _events_table(f"{population}-1", f"{population}-2", population=population),
            encoding="utf-8",
        )
        manifest_path = folder / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["n_simulated"] = 2
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    refused = tmp_path / "refused"
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
                str(bigger),
            ]
        )
    assert caught.value.code == 1
    assert "object ids" in capsys.readouterr().err
    assert not (refused / "test_score.md").exists()
    assert _ledger_text(scored_test_set_ledger) == saved
    monkeypatch.setattr(cli_mod, "_complete_pending_score", real_complete)

    approved = tmp_path / "approved"
    approved.mkdir()
    main(
        [
            "recall",
            "--score-test",
            "--cuts",
            str(cuts),
            "--rescore-decision",
            "PM-after-crash",
            "--output",
            str(approved),
            "--cache",
            str(bigger),
        ]
    )
    again = json.loads(_ledger_text(scored_test_set_ledger))
    assert again["sets"][old_hash][0] == old_entries[0]
    assert again["sets"][old_hash][0]["status"] == "pending"
    new_entries = next(
        entries for key, entries in again["sets"].items() if key != old_hash
    )
    assert new_entries[-1]["status"] == "complete"
    assert new_entries[-1]["decision_id"] == "PM-after-crash"
    assert "score_sha256" in new_entries[-1]


def test_a_stale_score_lock_is_left_for_a_person_to_remove(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A lock that is already there is not deleted, and the ledger is not touched."""
    from kilonova_hunter.cli import _score_lock_path

    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    output = tmp_path / "out"
    output.mkdir()
    lock = _score_lock_path()
    lock.write_text("stale\n", encoding="utf-8")
    before = _ledger_text(scored_test_set_ledger)
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
    err = capsys.readouterr().err
    assert str(lock) in err
    assert "by hand" in err
    assert lock.read_text(encoding="utf-8") == "stale\n"
    assert _ledger_text(scored_test_set_ledger) == before
    assert not (output / "test_score.md").exists()


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

    def fake_prepare(events, curves, day, settings=None, **kwargs):
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


def _proxy_alerts() -> pd.DataFrame:
    """Two nights. 60461 is the tuning night and 60462 is the other night."""
    return pd.DataFrame(
        [
            {
                "i:objectId": "tuning-object",
                "i:jd": 2450000.5,
                "i:fid": 1,
                "i:magpsf": 20.0,
                "i:sigmapsf": 0.05,
                "mjd": 50000.0,
                "night": 60461,
            },
            {
                "i:objectId": "test-night-object",
                "i:jd": 2450001.5,
                "i:fid": 1,
                "i:magpsf": 20.0,
                "i:sigmapsf": 0.05,
                "mjd": 50001.0,
                "night": 60462,
            },
        ]
    )


def _assert_unfrozen_fetch_is_tuning_only(
    output: Path, printed: str, logged: str
) -> None:
    """No other-night row, count, or total in the files, stdout, or log."""
    text = (output / "real_alert_slices_fetched.json").read_text(encoding="utf-8")
    document = json.loads(text)
    assert "test" not in document
    assert "test" not in document["flag_counts"]
    assert "n_flagged" in document["flag_counts"]["tuning"]
    assert "60462" not in text
    assert "test-night-object" not in text
    tuning_csv = (output / "ztf_proxy_alerts_tuning.csv").read_text(encoding="utf-8")
    assert "tuning-object" in tuning_csv
    assert "60462" not in tuning_csv
    assert "test-night-object" not in tuning_csv
    assert not (output / "ztf_proxy_alerts.csv").exists()
    blob = printed + "\n" + logged
    assert "Test:" not in blob
    assert "60462" not in blob
    assert "test-night-object" not in blob


def test_unfrozen_fetch_writes_no_test_night_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unfrozen fetch stores tuning nights and no other-night total."""
    import logging

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(
        "kilonova_hunter.alerts.fetch_class_between",
        lambda start, stop, **kwargs: _proxy_alerts(),
    )
    output = tmp_path / "out"
    output.mkdir()
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
    captured = capsys.readouterr()
    _assert_unfrozen_fetch_is_tuning_only(
        output, captured.out + captured.err, caplog.text
    )

    unfrozen = tmp_path / "unfrozen.json"
    unfrozen.write_text(json.dumps({"frozen": False}), encoding="utf-8")
    still = tmp_path / "still"
    still.mkdir()
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
    captured = capsys.readouterr()
    _assert_unfrozen_fetch_is_tuning_only(
        still, captured.out + captured.err, caplog.text
    )

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
        (frozen_out / "real_alert_slices_fetched.json").read_text(encoding="utf-8")
    )
    assert "n_flagged" in frozen["flag_counts"]["test"]
    assert "refused" not in frozen["flag_counts"]["test"]
    assert "n_flagged" in frozen["flag_counts"]["tuning"]
    full = pd.read_csv(frozen_out / "ztf_proxy_alerts.csv")
    tuning = pd.read_csv(frozen_out / "ztf_proxy_alerts_tuning.csv")
    assert set(full["i:objectId"]) == {"tuning-object", "test-night-object"}
    assert set(tuning["i:objectId"]) == {"tuning-object"}


def test_cuts_file_errors_are_not_frozen_and_do_not_raise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A list, missing keys, or bad JSON is not frozen, and does not traceback.

    Fetch still writes the tuning nights. It does not store another night's total.
    """
    import logging

    from kilonova_hunter.cli import _cuts_are_frozen

    caplog.set_level(logging.DEBUG)
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

    monkeypatch.setattr(
        "kilonova_hunter.alerts.fetch_class_between",
        lambda start, stop, **kwargs: _proxy_alerts(),
    )
    for path in (missing, listed, broken):
        output = tmp_path / path.stem
        output.mkdir()
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
        captured = capsys.readouterr()
        _assert_unfrozen_fetch_is_tuning_only(
            output, captured.out + captured.err, caplog.text
        )


def _patch_grid_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the light-curve model and keep exact copies to one fake.

    Distances come from the same rng both runs use, so two grids with
    the parent seed write the same parameters. The real Planck18 draw
    is not required for that comparison.
    """
    import numpy as np

    from kilonova_hunter import sample as sample_mod
    from kilonova_hunter.sky import DustSource

    def fake_jobs(jobs, workers):
        return [(job["event"].object_id, pd.DataFrame(), None) for job in jobs]

    def fake_distances(rng, count, **kwargs):
        drawn = rng.random(count)
        return np.full(count, 0.009), 40.0 + 10.0 * np.asarray(drawn)

    real_copies = sample_mod.simulate_exact_copies

    def one_copy(**kwargs):
        return real_copies(**{**kwargs, "count": 1})

    monkeypatch.setattr("kilonova_hunter.cli.redback_available", lambda: True)
    monkeypatch.setattr(
        "kilonova_hunter.sky.open_dust_source",
        lambda: DustSource(mode="dry-run", reason="fixture"),
    )
    monkeypatch.setattr(sample_mod, "_run_jobs", fake_jobs)
    monkeypatch.setattr(sample_mod, "sample_luminosity_distances", fake_distances)
    monkeypatch.setattr(
        sample_mod, "redshift_at_luminosity_distance", lambda _distance: 0.009
    )
    monkeypatch.setattr(sample_mod, "simulate_exact_copies", one_copy)


def _run_grid_fixture(cache: Path, output: Path, *, tuning_only: bool) -> None:
    """One tiny grid. Exact copies are untestable, so the command exits 2."""
    argv = [
        "recall",
        "--grid",
        "--output",
        str(output),
        "--cache",
        str(cache),
        "--target",
        "1",
        "--cap",
        "2",
        "--floor",
        "1",
        "--batch-size",
        "1",
        "--workers",
        "1",
    ]
    if tuning_only:
        argv.append("--tuning-only")
    output.mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit) as caught:
        main(argv)
    assert caught.value.code == 2


def _tuning_events(cache: Path) -> dict[str, bytes]:
    found = {}
    for population in ("2017-like", "realistic"):
        path = cache / population / "tuning" / "events.csv"
        found[population] = path.read_bytes()
    return found


def _test_identity(cache: Path) -> tuple[str, dict[str, list[str]]]:
    from kilonova_hunter.cli import test_set_sha256
    from kilonova_hunter.sample import LIKE_KIND, REALISTIC, _read_cache

    loaded = []
    ids: dict[str, list[str]] = {}
    for population in (LIKE_KIND, REALISTIC):
        cached = _read_cache(cache / population / "test")
        assert cached is not None
        manifest, events, _curves = cached
        loaded.append((population, manifest, events, _curves))
        ids[population] = [str(item) for item in events["object_id"].tolist()]
    return test_set_sha256(loaded), ids


def _paths_with_test_component(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if "test" in path.parts]


def _sha256(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tuning_only_matches_a_full_run_on_the_same_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tuning ids and parameters do not depend on also drawing the test split."""
    _patch_grid_draws(monkeypatch)
    tuning_cache = tmp_path / "tuning-cache"
    full_cache = tmp_path / "full-cache"
    _run_grid_fixture(tuning_cache, tmp_path / "tuning-out", tuning_only=True)
    _run_grid_fixture(full_cache, tmp_path / "full-out", tuning_only=False)
    assert _tuning_events(tuning_cache) == _tuning_events(full_cache)


def test_a_tuning_only_run_does_not_change_the_test_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test-set object ids and the identity hash ignore an earlier tuning-only run."""
    _patch_grid_draws(monkeypatch)
    full_cache = tmp_path / "full-cache"
    _run_grid_fixture(full_cache, tmp_path / "full-out", tuning_only=False)
    followed = tmp_path / "followed-cache"
    _run_grid_fixture(followed, tmp_path / "tuning-out", tuning_only=True)
    _run_grid_fixture(followed, tmp_path / "followed-out", tuning_only=False)
    full_hash, full_ids = _test_identity(full_cache)
    followed_hash, followed_ids = _test_identity(followed)
    assert followed_ids == full_ids
    assert followed_hash == full_hash
    for population in ("2017-like", "realistic"):
        full_events = (full_cache / population / "test" / "events.csv").read_bytes()
        followed_events = (followed / population / "test" / "events.csv").read_bytes()
        assert followed_events == full_events


def test_tuning_only_writes_no_test_split_and_leaves_the_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scored_test_set_ledger: Path,
) -> None:
    """A tuning-only grid does not create a test split or change the ledger."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_ledger = repo_root / "results" / "test_set_scored.json"
    before_repo = _sha256(repo_ledger)
    before_fixture = _sha256(scored_test_set_ledger)
    claims = tmp_path / "test_set_claims"
    assert not claims.exists()
    _patch_grid_draws(monkeypatch)
    cache = tmp_path / "cache"
    output = tmp_path / "out"
    _run_grid_fixture(cache, output, tuning_only=True)
    assert _paths_with_test_component(cache) == []
    assert _paths_with_test_component(output) == []
    assert (cache / "2017-like" / "tuning" / "events.csv").is_file()
    assert (cache / "realistic" / "tuning" / "events.csv").is_file()
    summary = (output / "tuning_grid.md").read_text(encoding="utf-8")
    assert summary.startswith("git ")
    assert "\nredback " in summary
    assert "2017-like day 5, g-r bluer by +0.4 mag:" in summary
    assert "realistic day 7, g-r bluer by +0.3 mag:" in summary
    assert "recall (a)" in summary
    assert "recall (b)" not in summary
    assert _sha256(scored_test_set_ledger) == before_fixture
    assert _sha256(repo_ledger) == before_repo
    assert not claims.exists()


_EXPOSURE_NOTE = (
    "Test-night raw Fink rows were fetched on 2026-11-03 into "
    "results/ztf_proxy_alerts.csv before the cuts freeze. "
    "results/real_alert_slices.json in the repo already shows 9 test nights "
    "and 8,051 test alerts. These totals do not depend on the cuts, and "
    "they reveal nothing about our flags. No test-slice flag counts were "
    "examined."
)


def _pick_rows() -> list[dict]:
    """Two day-3 rows. The flag rate is filled from the tuning alerts."""
    return [
        {
            "recall_judgeable": 0.90,
            "redder_than_g_minus_r": 0.0,
            "min_g_minus_r_rise": 0.1,
            "min_color_separation_days": 0.5,
            "reddening_window_days": 5.0,
            "untestable": False,
        },
        {
            "recall_judgeable": 0.80,
            "redder_than_g_minus_r": 0.4,
            "min_g_minus_r_rise": 0.2,
            "min_color_separation_days": 1.0,
            "reddening_window_days": 2.0,
            "untestable": False,
        },
    ]


def _alerts_with_both_nights() -> pd.DataFrame:
    frame = _proxy_alerts()
    frame["night_slice"] = ["tuning", "test"]
    return frame


def test_tuning_path_never_counts_a_test_night(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grid, pick, and the Fink count ignore rows from the other night.

    The same count comes back when those rows are removed. The only
    alerts file opened is the tuning file.
    """
    from kilonova_hunter.alerts import average_flags_per_night
    from kilonova_hunter.grid import pick_with_tuning_alerts

    mixed = _alerts_with_both_nights()
    tuning_nights = {60461}
    output = tmp_path / "out"
    output.mkdir()
    mixed.to_csv(output / "ztf_proxy_alerts_tuning.csv", index=False)
    mixed.to_csv(output / "ztf_proxy_alerts.csv", index=False)
    seen: list[set[int]] = []
    opened: list[str] = []
    real_flags = average_flags_per_night
    real_read = pd.read_csv

    def spy_flags(alerts, thresholds=None):
        nights = {int(night) for night in alerts["night"].unique()}
        seen.append(nights)
        return real_flags(alerts, thresholds)

    def spy_read(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr("kilonova_hunter.alerts.average_flags_per_night", spy_flags)
    monkeypatch.setattr(pd, "read_csv", spy_read)
    _patch_grid_draws(monkeypatch)
    _run_grid_fixture(tmp_path / "cache", output, tuning_only=True)
    mixed_pick = pick_with_tuning_alerts(_pick_rows(), output)
    summary = (output / "tuning_grid.md").read_text(encoding="utf-8")
    assert "Tuning-night Fink alerts: 1 nights, 0 new candidates." in summary
    clean = mixed.loc[mixed["night_slice"] == "tuning"]
    clean.to_csv(output / "ztf_proxy_alerts_tuning.csv", index=False)
    clean_pick = pick_with_tuning_alerts(_pick_rows(), output)
    assert mixed_pick["flags_per_night"] == clean_pick["flags_per_night"]
    assert mixed_pick["fink_nights"] == clean_pick["fink_nights"] == [60461]
    assert mixed_pick["fink_n_flagged"] == clean_pick["fink_n_flagged"] == 0
    assert mixed_pick["redder_than_g_minus_r"] == clean_pick["redder_than_g_minus_r"]
    assert seen
    for nights in seen:
        assert nights <= tuning_nights
        assert 60462 not in nights
    assert "ztf_proxy_alerts_tuning.csv" in opened
    assert "ztf_proxy_alerts.csv" not in opened


def test_exposure_note_is_recorded_and_is_not_in_the_score_once_key(
    tmp_path: Path,
    scored_test_set_ledger: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Fink exposure sentence is stored. It does not change the hash."""
    from kilonova_hunter.cli import _TEST_SET_IDENTITY_KEYS, test_set_sha256
    from kilonova_hunter.sample import LIKE_KIND, REALISTIC, _read_cache

    repo_root = Path(__file__).resolve().parents[1]
    note_path = repo_root / "results" / "test_set_exposure.json"
    recorded = json.loads(note_path.read_text(encoding="utf-8"))
    assert recorded["exposure_note"] == _EXPOSURE_NOTE
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
    assert entries[0]["exposure_note"] == _EXPOSURE_NOTE
    assert entries[0]["exposure_note_path"] == "results/test_set_exposure.json"
    assert "exposure_note" not in _TEST_SET_IDENTITY_KEYS
    loaded = []
    for population in (LIKE_KIND, REALISTIC):
        manifest, events, curves = _read_cache(cache / population / "test")
        loaded.append((population, manifest, events, curves))
    assert test_set_sha256(loaded) == test_hash
    for _population, manifest, _events, _curves in loaded:
        manifest["exposure_note"] = "a different sentence"
    assert test_set_sha256(loaded) == test_hash
    monkeypatch.setattr(
        "kilonova_hunter.cli._fink_exposure_note",
        lambda: ("a different sentence", "results/test_set_exposure.json"),
    )
    assert test_set_sha256(loaded) == test_hash


@pytest.mark.parametrize("source", ["pending", "claim"])
def test_overlap_layers_work_independently(source, tmp_path, scored_test_set_ledger):
    from kilonova_hunter import cli

    entry = {"status": "pending", "object_ids": {"2017-like": ["old"]}}
    if source == "pending":
        scored_test_set_ledger.write_text(json.dumps({"sets": {"hash": [entry]}}))
    else:
        claims = cli.test_set_claim_dir()
        claims.mkdir(parents=True)
        (claims / "claim.json").write_text(json.dumps(entry))
    assert cli._previously_scored_ids() == {"2017-like": {"old"}}


def test_same_hash_rescore_preserves_pending_history(tmp_path, scored_test_set_ledger):
    from kilonova_hunter import cli
    from kilonova_hunter.sample import _read_cache

    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    loaded = [
        (kind, *_read_cache(cache / kind / "test"))
        for kind in ("2017-like", "realistic")
    ]
    cli._write_pending_score("same", {}, loaded, git_sha="test", scored_at="first")
    old = json.loads(scored_test_set_ledger.read_text())["sets"]["same"][0]
    # The first run crashed; the approved rescore uses exactly the same identity.
    cli._write_pending_score(
        "same", {}, loaded, git_sha="test", scored_at="second", decision_id="approved"
    )
    score = tmp_path / "synthetic-score.md"
    score.write_text("fixture")
    cli._complete_pending_score("same", score)
    entries = json.loads(scored_test_set_ledger.read_text())["sets"]["same"]
    assert [entry["status"] for entry in entries] == ["pending", "complete"]
    assert entries[0] == old


def test_completion_requires_last_entry_pending(tmp_path, scored_test_set_ledger):
    from kilonova_hunter import cli

    payload = {"sets": {"same": [{"status": "pending"}, {"status": "complete"}]}}
    scored_test_set_ledger.write_text(json.dumps(payload))
    before = scored_test_set_ledger.read_bytes()
    score = tmp_path / "synthetic-score.md"
    score.write_text("fixture")
    with pytest.raises(SystemExit) as caught:
        cli._complete_pending_score("same", score)
    assert caught.value.code == 1
    assert scored_test_set_ledger.read_bytes() == before


@pytest.mark.parametrize("bad", [True, False])
def test_boolean_cuts_are_refused(bad, tmp_path, scored_test_set_ledger, capsys):
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    payload = json.loads(cuts.read_text())
    payload["redder_than_g_minus_r"] = bad
    cuts.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--cache",
                str(tmp_path / "absent"),
                "--output",
                str(tmp_path),
            ]
        )
    assert caught.value.code == 1
    assert "finite number" in capsys.readouterr().err
    assert not scored_test_set_ledger.exists()


def test_zero_byte_curves_refused_before_ledger(
    tmp_path, scored_test_set_ledger, capsys
):
    cache = tmp_path / "cache"
    _ready_test_cache(cache)
    (cache / "2017-like" / "test" / "light_curves.csv").write_bytes(b"")
    cuts = _frozen_cuts(tmp_path / "cuts.json", frozen=True)
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "recall",
                "--score-test",
                "--cuts",
                str(cuts),
                "--cache",
                str(cache),
                "--output",
                str(tmp_path),
            ]
        )
    assert caught.value.code == 1
    assert "empty" in capsys.readouterr().err
    assert not scored_test_set_ledger.exists()
    assert not (tmp_path / "test_set_claims").exists()


@pytest.mark.parametrize("variant", ["missing_first", "shifted_start", "unknown"])
def test_fetch_uses_frozen_nights(variant, tmp_path, monkeypatch, capsys):
    from kilonova_hunter import alerts

    frozen = alerts.FROZEN_SPLIT_PATH
    before = frozen.read_bytes()
    split = json.loads(before)
    nights = sorted(split["tuning"]["nights"] + split["test"]["nights"])
    frame = pd.DataFrame([dict(_proxy_alerts().iloc[0], night=n) for n in nights])
    if variant == "unknown":
        frame.loc[len(frame)] = dict(frame.iloc[0], night=99999)
    else:
        frame = frame.loc[frame["night"] != 60461]
    monkeypatch.setattr(alerts, "fetch_class_between", lambda *a, **k: frame.copy())
    output = tmp_path / "out"
    output.mkdir()
    # A sentinel also protects an output-local frozen document from replacement.
    local_split = output / "real_alert_slices.json"
    local_split.write_bytes(before)
    argv = [
        "recall",
        "--fetch-alerts",
        "--output",
        str(output),
        "--alert-start",
        "2024-06-02" if variant == "shifted_start" else "2024-06-01",
        "--alert-stop",
        "2024-06-20",
    ]
    main(argv)
    kept = pd.read_csv(output / "ztf_proxy_alerts_tuning.csv")
    expected = set(split["tuning"]["nights"])
    missing = [] if variant == "unknown" else [60461]
    assert set(kept["night"]) == expected - set(missing)
    assert set(kept["night_slice"]) == {"tuning"}
    assert not set(kept["night"]) & set(split["test"]["nights"])
    assert 99999 not in set(kept["night"])
    summary = (output / "real_alert_slices_fetched.json").read_text()
    report = json.loads(summary)["fetch_report"]
    printed = capsys.readouterr()
    assert report["missing_tuning_nights"] == missing
    assert f"Missing frozen tuning nights: {missing}" in printed.err
    dropped = 1 if variant == "unknown" else 0
    unknown = [99999] if variant == "unknown" else []
    assert report["dropped_unknown_rows"] == dropped
    assert report["dropped_unknown_nights"] == unknown
    assert f"Dropped unknown-night rows: {dropped}; nights: {unknown}" in printed.err
    assert '"test"' not in summary
    assert "Test:" not in printed.out
    for night in split["test"]["nights"]:
        assert str(night) not in summary + printed.out + printed.err
    assert frozen.read_bytes() == before
    assert local_split.read_bytes() == before


def test_missing_frozen_split_refuses_before_writes(tmp_path, monkeypatch, capsys):
    from kilonova_hunter import alerts

    monkeypatch.setattr(alerts, "FROZEN_SPLIT_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(alerts, "fetch_class_between", lambda *a, **k: _proxy_alerts())
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(SystemExit) as caught:
        main(["recall", "--fetch-alerts", "--output", str(output)])
    assert caught.value.code == 1
    assert "frozen night lists" in capsys.readouterr().err
    assert list(output.iterdir()) == []

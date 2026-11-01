"""Tests for the kilohunt command.

These call ``main`` directly, which is the function the ``kilohunt`` command
runs. The test then does not need the shell to find a ``kilohunt`` program
on ``PATH``.
"""

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
    Both rules' sentences are in the printout, which is how we know the
    command walked every rule instead of printing only the first line.
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
    """The CSV in ``examples/`` is the noise-free demo, and check agrees."""
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
    """The printer walks the verdict's rules instead of naming two of them.

    Reddening is not a filter rule yet. This stand-in puts it on the
    verdict anyway, once as ``skipped=True`` and once with ``passed``
    left empty. Both have to print SKIPPED. The overall line stays
    whatever ``is_candidate`` says, even if a skipped rule did not pass.
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
            passed=False,
            skipped=True,
            explanation="Reddening was skipped because the bands were too far apart.",
        ),
        not_applied=RuleResult(
            passed=None,
            explanation="This rule was not applied.",
        ),
    )
    lines = _format_verdict(verdict).splitlines()
    reddening_line = (
        "Reddening: SKIPPED. "
        "Reddening was skipped because the bands were too far apart."
    )
    assert lines == [
        "CANDIDATE",
        "Fade: PASS. The fade rule passes.",
        "Color: PASS. The color rule passes.",
        reddening_line,
        "Not applied: SKIPPED. This rule was not applied.",
    ]


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

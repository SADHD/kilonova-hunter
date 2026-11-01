"""The ``kilohunt`` command.

This is the only module that talks to a person. The other modules take
data in and hand data back, which is what makes them testable without
parsing printed text. A dashboard after version 1.0 can call those same
functions and needs almost no new science code.

``demo`` draws a made-up light curve. ``check`` runs the v1 rules on a
CSV of one light curve and prints the verdict.
"""

import argparse
import sys
from dataclasses import fields
from pathlib import Path
from typing import NoReturn

import pandas as pd

from kilonova_hunter.filter import CandidateVerdict, RuleResult, is_kilonova_candidate
from kilonova_hunter.inject import simulate_kilonova_light_curve
from kilonova_hunter.report import plot_light_curve

# Default place for a plot you make while exploring. The folder is
# gitignored, so trying the demo does not dirty the repository.
_DEFAULT_DEMO_PATH = Path("output") / "demo_kilonova.png"

# The fake kilonova and the plot helper share these columns. check
# refuses a file that is missing any of them and names the missing ones,
# so a typo in a header is obvious instead of a silent wrong answer.
_LIGHT_CURVE_COLUMNS = ("time_days", "filter", "magnitude", "magnitude_error")


def build_parser() -> argparse.ArgumentParser:
    """Build the parser that reads ``kilohunt``'s arguments.

    Kept separate from ``main`` so a test can inspect the help text
    without running a command.
    """
    parser = argparse.ArgumentParser(
        prog="kilohunt",
        description=(
            "Search Rubin Observatory alerts for kilonova candidates "
            "and explain each verdict. "
            "The demo command plots a made-up kilonova and needs no alert download. "
            "The check command runs the v1 rules on one light curve CSV. "
            "Searching a real week of alerts (kilohunt --week) arrives in Sprint 3."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    demo = subparsers.add_parser(
        "demo",
        help="Plot a made-up kilonova light curve and print the filter verdict.",
        description=(
            "Build a kilonova-like light curve (a fast rise, a week-long fade, "
            "and a color that turns red) and save a labeled plot. "
            "The magnitude axis is flipped so brighter is up. "
            "The plot includes a little random noise. The filter verdict is "
            "for the same curve with that noise left off, so the sentences "
            "match the rates built into the toy."
        ),
    )
    demo.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_DEMO_PATH,
        help=f"Where to save the plot. Default: {_DEFAULT_DEMO_PATH.as_posix()}",
    )
    demo.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for the noise in the plot. The same seed redraws "
            "the same curve. The filter verdict does not use this noise. "
            "Default: 42."
        ),
    )
    demo.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help=(
            "Also write the noise-free light curve to this CSV path, so you "
            "can pass that file to kilohunt check. The plot still includes noise."
        ),
    )

    check = subparsers.add_parser(
        "check",
        help="Run the kilonova rules on one light curve CSV.",
        description=(
            "Read one light curve and print whether it looks like a kilonova. "
            "CSV means comma-separated values: the first row names the columns, "
            "and each later row is one observation. The columns are time_days, "
            "filter, magnitude, and magnitude_error, the same ones the demo builds. "
            "filter is the band, such as g, r, or i. "
            "A file that loads exits 0 whether or not it is a candidate. "
            "A missing file, or a table without those columns, exits nonzero."
        ),
    )
    check.add_argument(
        "light_curve",
        type=Path,
        help=("CSV with columns time_days, filter, magnitude, and magnitude_error."),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the kilohunt command.

    With no command, prints help and returns. ``--help`` is handled by
    argparse, which exits after printing it.

    Args:
        argv: Arguments to parse, not including the program name.
            Omit this to use the arguments typed at the terminal.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        _run_demo(args.output, args.seed, args.save_csv)
        return
    if args.command == "check":
        _run_check(args.light_curve)
        return
    # Running `kilohunt` with no command should explain itself, the way
    # `--help` does, rather than failing on a missing subcommand.
    parser.print_help()


def _run_demo(output: Path, seed: int, save_csv: Path | None) -> None:
    """Plot the noisy toy curve, then print the noise-free filter verdict.

    The plot keeps the random scatter from ``seed``, which is the picture
    the demo has always saved. The verdict uses the same shape with that
    scatter turned off. The rates in the sentences are then the ones built
    into the toy, and they do not change when you pick a new seed.
    """
    light_curve = simulate_kilonova_light_curve(seed=seed)
    saved = plot_light_curve(
        light_curve,
        output,
        title="Made-up kilonova light curve",
    )
    print(f"Saved a made-up kilonova light curve to {saved}")

    noise_free = simulate_kilonova_light_curve(noise_mag=0.0)
    if save_csv is not None:
        _write_light_curve_csv(noise_free, save_csv)
        print(f"Saved the noise-free light curve to {save_csv}")

    print("Filter verdict for the noise-free curve:")
    print(_format_verdict(is_kilonova_candidate(noise_free)))


def _run_check(path: Path) -> None:
    """Read one light curve CSV and print the filter verdict.

    A file that loads and has the right columns exits 0 whether or not
    the object is a candidate. The verdict itself is the output. A missing
    file, or a table without those columns, is a problem with the file,
    not a verdict: the command exits nonzero and prints the reason.
    """
    light_curve = _read_light_curve_csv(path)
    print(_format_verdict(is_kilonova_candidate(light_curve)))


def _format_verdict(verdict: CandidateVerdict) -> str:
    """Turn a filter verdict into a few lines of text.

    The first line is the overall answer, ``CANDIDATE`` or
    ``NOT A CANDIDATE``. That word comes from ``verdict.is_candidate``.
    This function does not decide the science again.

    Each line after that is one rule: its name, then PASS, FAIL, or
    SKIPPED, then the sentence. The rules are whatever the verdict
    contains. Fade and color are not named here, so a new rule added
    on the verdict (reddening, for example) is printed too.
    """
    overall = "CANDIDATE" if verdict.is_candidate else "NOT A CANDIDATE"
    lines = [overall]
    for name, result in _iter_rule_results(verdict):
        # "fade" is easier to read as "Fade". Underscores become spaces
        # so a name like "color_rate" would read "Color rate".
        label = name.replace("_", " ").capitalize()
        lines.append(f"{label}: {_rule_status(result)}. {result.explanation}")
    return "\n".join(lines)


def _iter_rule_results(verdict: CandidateVerdict):
    """Yield ``(field_name, rule_result)`` for every rule on a verdict.

    A rule is one field that carries a sentence and a pass/fail flag,
    the way ``RuleResult`` does. ``is_candidate`` is a single yes-or-no
    for the whole object, so this loop leaves it out. The field name is
    the rule's name. Walking the fields, instead of writing "fade" and
    "color" here, means a third rule shows up on its own.
    """
    for field in fields(verdict):
        value = getattr(verdict, field.name)
        if _is_rule_result(value):
            yield field.name, value


def _is_rule_result(value: object) -> bool:
    """True when this value is one rule's answer and its sentence.

    The test is "does it look like a rule result?", not "is it exactly
    the fade field?". A sentence plus a ``passed`` flag is enough. A
    rule that was skipped still counts: it has a sentence too.
    """
    if not isinstance(getattr(value, "explanation", None), str):
        return False
    return hasattr(value, "passed") or bool(getattr(value, "skipped", False))


def _rule_status(result: RuleResult) -> str:
    """Return PASS, FAIL, or SKIPPED for one rule result.

    SKIPPED is for a rule that was not applied. The result says that
    either by setting ``skipped`` to True, or by leaving ``passed``
    empty (``None``). Fade and color always pass or fail. A later rule,
    such as reddening, may be skipped when the curve does not have the
    measurements that rule needs.
    """
    if getattr(result, "skipped", False):
        return "SKIPPED"
    passed = getattr(result, "passed", None)
    if passed is None:
        return "SKIPPED"
    if passed:
        return "PASS"
    return "FAIL"


def _read_light_curve_csv(path: Path) -> pd.DataFrame:
    """Load one light curve, or exit with a message if the file is unusable.

    Args:
        path: The CSV named on the command line.

    Returns:
        The table. The demo's four columns are present. Extra columns
        are left as they are; the filter ignores what it does not use.

    Raises:
        SystemExit: If the path is missing, not a readable CSV, or
            missing a required column. The exit code is 1, and the
            reason is printed to standard error (the channel for
            problems, so it does not get mixed into the verdict).
    """
    if not path.exists():
        _fail(f"Could not check {path}: that file does not exist.")
    if not path.is_file():
        _fail(f"Could not check {path}: that is not a file.")

    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        _fail(
            f"Could not check {path}: the file is empty. "
            "A light curve CSV needs a header and these columns: "
            "time_days, filter, magnitude, magnitude_error."
        )
    except pd.errors.ParserError as error:
        _fail(f"Could not check {path}: this is not a CSV table I can read. {error}")
    except UnicodeDecodeError:
        _fail(
            f"Could not check {path}: this is not a text CSV. "
            "Save the light curve as a plain CSV file and try again."
        )

    missing = [name for name in _LIGHT_CURVE_COLUMNS if name not in table.columns]
    if missing:
        joined = ", ".join(missing)
        _fail(
            f"Could not check {path}: missing columns: {joined}. "
            "A light curve CSV needs time_days, filter, magnitude, and "
            "magnitude_error. time_days is days since a reference time, "
            "filter is the band (g, r, or i), magnitude is the brightness "
            "(a larger number is fainter), and magnitude_error is the "
            "uncertainty on that brightness."
        )
    return table


def _write_light_curve_csv(light_curve: pd.DataFrame, path: Path) -> None:
    """Save a light curve as a CSV that ``check`` can read back.

    ``index=False`` leaves out the row numbers pandas keeps internally.
    Those numbers are not a light-curve column. Writing them would add
    a nameless first column that is easy to mistake for the time.
    Parent folders are created, the same way the plot helper does, so
    ``--save-csv output/curve.csv`` works before ``output/`` exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    light_curve.to_csv(path, index=False)


def _fail(message: str) -> NoReturn:
    """Print ``message`` to standard error and stop the command.

    Exit code 1 means "the file could not be checked". It is not the
    code for "this object is not a kilonova". A real verdict, either
    way, exits 0.
    """
    print(message, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()

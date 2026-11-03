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
from pathlib import Path
from typing import NoReturn

import pandas as pd

from kilonova_hunter.filter import format_verdict as _format_verdict
from kilonova_hunter.filter import is_kilonova_candidate
from kilonova_hunter.inject import simulate_kilonova_light_curve
from kilonova_hunter.recall import redback_available, run_recall
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
            "The recall command scores fake kilonovae and needs the optional "
            "redback library (uv sync --extra sim). "
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

    recall = subparsers.add_parser(
        "recall",
        help="Score fake kilonovae and print how many the filter catches.",
        description=(
            "Build fake neutron-star mergers, sample them with a "
            "Rubin wide-fast-deep cadence (or --cadence simple), and count "
            "how many the v1 rules catch within 3 days of the merger. "
            "Those 3 days start at the merger, not at the first detection. "
            "The official number is the test set. A tuning set is written "
            "beside it so cuts can be tried without using the test set. "
            "The cuts stay the placeholders in config.py. "
            "This needs redback, which is optional: uv sync --extra sim. "
            "The tables go under data/recall/, which git ignores."
        ),
    )
    recall.add_argument(
        "--set-size",
        type=int,
        default=200,
        help="How many fakes in each set. Default: 200.",
    )
    recall.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed. With --which both, the tuning set uses this seed "
            "and the test set uses this seed plus 10000, so the two lists "
            "are different draws. Default: 1100 for tuning and 11100 for test."
        ),
    )
    recall.add_argument(
        "--which",
        choices=("tuning", "test", "both"),
        default="both",
        help="Which set to build. Default: both. The printed recall is the test set.",
    )
    recall.add_argument(
        "--cadence",
        choices=("wfd", "simple"),
        default="wfd",
        help=(
            "How the fake survey visits. wfd (the default) is a wide-fast-deep "
            "style: a field every 2 to 4 nights, two filters about 30 minutes "
            "apart. simple gives each of g, r, and i its own clock, at least "
            "one day between visits of that band."
        ),
    )
    recall.add_argument(
        "--grid",
        action="store_true",
        help=(
            "Simulate the 2017-like and realistic sets, then score the "
            "72-combination tuning grid. Does not pick cuts and does not "
            "score the test set."
        ),
    )
    recall.add_argument(
        "--fetch-alerts",
        action="store_true",
        help=(
            "Download Fink ZTF alerts (class Unknown) and write "
            "results/real_alert_slices.json. The slices are a ZTF proxy for Rubin."
        ),
    )
    recall.add_argument(
        "--score-test",
        action="store_true",
        help=(
            "Score the cached test split once, using the cuts in --cuts. "
            "Does not retune."
        ),
    )
    recall.add_argument(
        "--cuts",
        type=Path,
        default=None,
        help="JSON file of frozen cuts for --score-test.",
    )
    recall.add_argument(
        "--target",
        type=int,
        default=400,
        help=(
            "Stop after a batch that reaches this many day-3 "
            "judgeable fakes. Default: 400."
        ),
    )
    recall.add_argument(
        "--cap",
        type=int,
        default=20000,
        help="Most fakes to simulate in one set and split. Default: 20000.",
    )
    recall.add_argument(
        "--floor",
        type=int,
        default=50,
        help="Below this many day-3 judgeable fakes, day 3 is untestable. Default: 50.",
    )
    recall.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Fakes drawn between judgeable counts. Default: 500.",
    )
    recall.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel light-curve workers for --grid. Default: 1.",
    )
    recall.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Where --grid and --fetch-alerts write. Default: results/.",
    )
    recall.add_argument(
        "--cache",
        type=Path,
        default=Path("data") / "recall" / "cache",
        help="Where simulated light curves are reused. Default: data/recall/cache/.",
    )
    recall.add_argument(
        "--alert-start",
        default="2024-06-01",
        help="First calendar day for --fetch-alerts. Default: 2024-06-01.",
    )
    recall.add_argument(
        "--alert-stop",
        default="2024-06-20",
        help="Last calendar day for --fetch-alerts. Default: 2024-06-20.",
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
    if args.command == "recall":
        if args.fetch_alerts:
            _run_fetch_alerts(args.output, args.alert_start, args.alert_stop)
            return
        if args.grid:
            _run_grid(args)
            return
        if args.score_test:
            _run_score_test(args)
            return
        _run_recall(args.set_size, args.seed, args.which, args.cadence)
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


def _run_fetch_alerts(output: Path, start: str, stop: str) -> None:
    """Download the ZTF proxy slices and write the frozen JSON."""
    from datetime import date

    from kilonova_hunter.alerts import (
        build_slice_document,
        fetch_class_between,
        write_slice_document,
    )

    try:
        start_day = date.fromisoformat(start)
        stop_day = date.fromisoformat(stop)
    except ValueError:
        _fail("alert dates must look like 2024-06-01.")
    if stop_day < start_day:
        _fail("--alert-stop must be on or after --alert-start.")
    alerts = fetch_class_between(start_day, stop_day)
    if alerts.empty:
        _fail("Fink returned no alerts for that date range.")
    document = build_slice_document(alerts, start=start_day, stop=stop_day)
    path = output / "real_alert_slices.json"
    write_slice_document(document, path)
    alerts_path = output / "ztf_proxy_alerts.csv"
    alerts.to_csv(alerts_path, index=False)
    tuning = document["tuning"]
    test = document["test"]
    print(f"Label: {document['label']}")
    print(f"Wrote {path}")
    print(
        f"Tuning: {tuning['n_nights']} nights, {tuning['n_alerts']} alerts. "
        f"Test: {test['n_nights']} nights, {test['n_alerts']} alerts."
    )
    print(document["night_formula"])


def _run_score_test(args) -> None:
    """Score the cached test split once. Refuses to simulate or to retune.

    The command stops unless three things are true: no test score has
    been written yet, the cuts file says it is frozen, and the cache
    manifest names the SFD dust map. ``--cache`` chooses the cache.
    """
    import json

    from kilonova_hunter.grid import format_test_score, score_with_frozen_cuts
    from kilonova_hunter.sample import LIKE_KIND, REALISTIC, _read_cache
    from kilonova_hunter.sky import dust_map_label

    destination = args.output / "test_score.md"
    if destination.is_file():
        _fail(
            f"{destination} already exists. The test set is scored once. "
            "Do not score it again."
        )
    if args.cuts is None:
        _fail("--score-test needs --cuts pointing at a JSON file of frozen cuts.")
    if not args.cuts.is_file():
        _fail(f"--cuts file does not exist: {args.cuts}")
    cuts = json.loads(args.cuts.read_text(encoding="utf-8"))
    if cuts.get("frozen") is not True:
        _fail('The cuts file must contain "frozen": true.')
    needed = (
        "redder_than_g_minus_r",
        "min_g_minus_r_rise",
        "reddening_window_days",
        "min_color_separation_days",
    )
    missing = [name for name in needed if name not in cuts]
    if missing:
        _fail(f"--cuts is missing {', '.join(missing)}.")
    combo = {name: float(cuts[name]) for name in needed}
    cache = Path(args.cache)
    expected_map = dust_map_label()
    by_population = {}
    for population in (LIKE_KIND, REALISTIC):
        folder = cache / population / "test"
        loaded = _read_cache(folder)
        if loaded is None:
            _fail(
                f"No cached test split for {population} in {cache}. "
                "Simulate it before scoring. This command does not simulate."
            )
        manifest, events, curves = loaded
        if manifest.get("status") == "running":
            _fail(
                f"The {population} test cache is incomplete. "
                "Finish the simulation first."
            )
        if manifest.get("dust_map") != expected_map:
            _fail(
                f"The {population} test cache has no recorded SFD provenance. "
                f"dust_map is {manifest.get('dust_map')!r}. "
                f"It has to be {expected_map!r}."
            )
        by_population[population] = score_with_frozen_cuts(
            events,
            curves,
            combo,
            population=population,
            status=str(manifest.get("status", "ready")),
        )
    text = format_test_score(
        by_population,
        dry_run=False,
        provisional=bool(cuts.get("provisional", False)),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + "\n", encoding="utf-8")
    print(text)


def _run_grid(args) -> None:
    """Simulate and score the tuning grid. Does not pick or score the test set."""
    import logging

    from kilonova_hunter.grid import run_tuning_grid
    from kilonova_hunter.sample import SampleSizeSettings

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        size = SampleSizeSettings(
            target_judgeable=args.target,
            cap=args.cap,
            untestable_floor=args.floor,
            batch_size=args.batch_size,
        )
    except ValueError as error:
        _fail(str(error))
    if not redback_available():
        _fail(
            "kilohunt recall --grid needs the redback library. Install it with:\n"
            "    uv sync --extra sim"
        )
    print(
        run_tuning_grid(
            output_dir=args.output,
            cache_dir=args.cache,
            sample_size=size,
            workers=max(1, args.workers),
        )
    )


def _run_recall(set_size: int, seed: int | None, which: str, cadence: str) -> None:
    """Score fake kilonovae, or explain that redback is not installed.

    The science lives in ``recall.py``. This function only checks that
    the optional library is present and prints the report.
    """
    if not redback_available():
        _fail(
            "kilohunt recall needs the redback library. It is optional so "
            "the normal install and the tests stay small. Install it with:\n"
            "    uv sync --extra sim\n"
            "Then run kilohunt recall again."
        )
    if set_size < 1:
        _fail(f"--set-size must be at least 1, got {set_size}.")
    print(run_recall(which=which, set_size=set_size, seed=seed, cadence=cadence))


def _run_check(path: Path) -> None:
    """Read one light curve CSV and print the filter verdict.

    A file that loads and has the right columns exits 0 whether or not
    the object is a candidate. The verdict itself is the output. A missing
    file, or a table without those columns, is a problem with the file,
    not a verdict: the command exits nonzero and prints the reason.
    """
    light_curve = _read_light_curve_csv(path)
    print(_format_verdict(is_kilonova_candidate(light_curve)))


def _read_light_curve_csv(path: Path) -> pd.DataFrame:
    """Load one light curve, or exit with a message if the file is unusable.

    Args:
        path: The CSV named on the command line.

    Returns:
        The table. The demo's four columns are present. Extra columns
        are left as they are; the filter ignores what it does not use.

    Raises:
        SystemExit: If the path is missing, not a readable CSV, missing
            a required column, or has text in ``time_days`` or
            ``magnitude``. The exit code is 1, and the reason is printed
            to standard error (the channel for problems, so it does not
            get mixed into the verdict).
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
    _require_numeric_columns(table, path)
    return table


def _require_numeric_columns(table: pd.DataFrame, path: Path) -> None:
    """Exit if time or brightness cannot be read as a number.

    The filter fits a straight line to ``time_days`` and ``magnitude``.
    A word such as "bright" used to crash that fit with a traceback.
    A blank cell is different: it is a missing measurement, and the
    filter already skips those. Only text that is not a number is
    reported here.
    """
    problems: list[str] = []
    for column in ("time_days", "magnitude"):
        # errors="coerce" turns text into missing, and leaves real
        # numbers alone. A cell that was already blank stays missing
        # on both sides, so it is not counted as text.
        numeric = pd.to_numeric(table[column], errors="coerce")
        bad = table.loc[numeric.isna() & table[column].notna(), column]
        if not bad.empty:
            sample = bad.iloc[0]
            problems.append(f"{column} (for example {sample!r})")
    if not problems:
        return
    if len(problems) == 1:
        joined = f"{problems[0]} must be a number, not text."
    else:
        joined = f"{' and '.join(problems)} must be numbers, not text."
    _fail(
        f"Could not check {path}: {joined} "
        "time_days is days since a reference time, and magnitude is the "
        "brightness (a larger number is fainter). A blank cell is fine. "
        "A word is not."
    )


def _write_light_curve_csv(light_curve: pd.DataFrame, path: Path) -> None:
    """Save a light curve as a CSV that ``check`` can read back.

    ``index=False`` leaves out the row numbers pandas keeps internally.
    Those numbers are not a light-curve column. Writing them would add
    a nameless first column that is easy to mistake for the time.
    Parent folders are created, the same way the plot helper does, so
    ``--save-csv output/curve.csv`` works before ``output/`` exists.

    Numbers are written with four digits after the decimal. That is
    much finer than the 0.01 magnitudes the verdict sentences show, and
    it avoids a long tail of binary rounding such as
    ``20.900000000000002``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    light_curve.to_csv(path, index=False, float_format="%.4f")


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

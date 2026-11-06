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
            "Without --grid this is the legacy scorer, not the official "
            "tuning grid. The test-set number it prints is not the official "
            "result. "
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
        "--tuning-only",
        action="store_true",
        help=(
            "With --grid, simulate and score only the tuning split of "
            "both sets. Does not simulate, read, or write the test "
            "split, and does not touch the test-set ledger."
        ),
    )
    recall.add_argument(
        "--fetch-alerts",
        action="store_true",
        help=(
            "Download Fink ZTF alerts (class Unknown), a ZTF proxy for Rubin. "
            "Writes results/real_alert_slices_fetched.json and "
            "results/ztf_proxy_alerts_tuning.csv. Until the cuts are frozen, "
            "only tuning-night rows are written. A frozen cuts file also "
            "writes results/ztf_proxy_alerts.csv."
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
        "--rebuild-score-cache",
        action="store_true",
        help=(
            "With --score-test, rebuild using the frozen generators into a new "
            "--cache directory, write integrity receipts, and stop WITHOUT scoring."
        ),
    )
    recall.add_argument(
        "--cuts",
        type=Path,
        default=None,
        help=(
            'JSON object of frozen cuts: "frozen": true and the four '
            "cut numbers. --score-test requires it. --fetch-alerts writes "
            "tuning-night rows either way. Rows and counts from the other "
            "nights are written only when this file is frozen."
        ),
    )
    recall.add_argument(
        "--rescore-decision",
        default=None,
        help=(
            "Decision id approved by Sage through PM. Required to score "
            "a test set, or any of its object ids, again."
        ),
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
            _run_fetch_alerts(
                args.output,
                args.alert_start,
                args.alert_stop,
                args.cuts,
            )
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


# The same four numbers --score-test reads. A frozen file has to name them.
_FROZEN_CUT_KEYS = (
    "redder_than_g_minus_r",
    "min_g_minus_r_rise",
    "reddening_window_days",
    "min_color_separation_days",
)


def _cuts_are_frozen(path: Path | None) -> tuple[bool, str]:
    """Whether the cuts file is frozen, and why it is not.

    A frozen file is a JSON object with ``"frozen": true`` and the four
    cut keys ``--score-test`` requires. A missing file, a list, missing
    keys, or invalid JSON is not frozen. This returns that reason. It
    does not raise.
    """
    import json

    if path is None:
        return False, "No cuts file was given."
    if not path.is_file():
        return False, f"Cuts file does not exist: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "Cuts file is not valid JSON."
    if not isinstance(payload, dict):
        return False, "Cuts file must be a JSON object, not a list or other value."
    if payload.get("frozen") is not True:
        return False, 'Cuts file must contain "frozen": true.'
    missing = [name for name in _FROZEN_CUT_KEYS if name not in payload]
    if missing:
        return False, f"Cuts file is missing {', '.join(missing)}."
    return True, ""


def _run_fetch_alerts(output: Path, start: str, stop: str, cuts: Path | None) -> None:
    """Download the ZTF proxy and write the tuning rows.

    The committed night lists label the rows immediately after download.
    The frozen split file is never overwritten. Unless the
    cuts file is frozen, rows from the other nights are dropped before
    any file is written and before anything is printed. A frozen file
    may also write those rows, as the earlier fetch did.
    """
    from datetime import date

    from kilonova_hunter.alerts import (
        FULL_ALERTS_FILENAME,
        TUNING_ALERTS_FILENAME,
        build_slice_document,
        build_tuning_only_document,
        fetch_class_between,
        label_night_slices,
        tuning_only_alerts,
        write_slice_document,
    )

    frozen, _reason = _cuts_are_frozen(cuts)
    try:
        start_day = date.fromisoformat(start)
        stop_day = date.fromisoformat(stop)
    except ValueError:
        _fail("alert dates must look like 2024-06-01.")
    if stop_day < start_day:
        _fail("--alert-stop must be on or after --alert-start.")
    fetched = fetch_class_between(start_day, stop_day)
    if fetched.empty:
        _fail("Fink returned no alerts for that date range.")
    # Label, then drop the other nights, before any write or print.
    fetch_report = {}
    try:
        labeled = label_night_slices(fetched, fetch_report=fetch_report)
    except ValueError as error:
        _fail(f"{error} Nothing was written.")
    print(
        f"Dropped unknown-night rows: {fetch_report['dropped_unknown_rows']}; "
        f"nights: {fetch_report['dropped_unknown_nights']}",
        file=sys.stderr,
    )
    print(
        f"Missing frozen tuning nights: {fetch_report['missing_tuning_nights']}",
        file=sys.stderr,
    )
    tuning_rows = tuning_only_alerts(labeled)
    if tuning_rows.empty:
        _fail("Fink returned no tuning-night alerts for that date range.")
    if frozen:
        document = build_slice_document(
            labeled,
            start=start_day,
            stop=stop_day,
            include_test_flag_counts=True,
        )
    else:
        document = build_tuning_only_document(
            tuning_rows,
            start=start_day,
            stop=stop_day,
        )
    document["fetch_report"] = fetch_report
    path = output / "real_alert_slices_fetched.json"
    write_slice_document(document, path)
    tuning_path = output / TUNING_ALERTS_FILENAME
    tuning_rows.to_csv(tuning_path, index=False)
    if frozen:
        full_path = output / FULL_ALERTS_FILENAME
        labeled.to_csv(full_path, index=False)
    tuning = document["tuning"]
    print(f"Label: {document['label']}")
    print(f"Wrote {path}")
    print(f"Wrote {tuning_path}")
    print(f"Tuning: {tuning['n_nights']} nights, {tuning['n_alerts']} alerts.")
    if frozen:
        test = document["test"]
        print(f"Test: {test['n_nights']} nights, {test['n_alerts']} alerts.")
    print(document["night_formula"])


def _run_score_test(args) -> None:
    """Score an attested cache once, or rebuild separately without scoring.

    The command stops unless the cuts file is frozen and the cache
    manifest names the SFD dust map. A pending ledger entry counts as
    scored: the set is consumed by a crash, and another score is only
    possible with ``--rescore-decision``. ``--cache`` chooses the cache.
    """
    import json

    from kilonova_hunter.grid import format_test_score, score_with_frozen_cuts
    from kilonova_hunter.sample import (
        LIKE_KIND,
        REALISTIC,
        _cache_paths,
        _read_cache,
        cache_refusal_reason,
    )
    from kilonova_hunter.sky import dust_map_label

    destination = args.output / "test_score.md"
    if destination.is_file():
        _fail(
            f"{destination} already exists. The test set is scored once. "
            "Do not score it again."
        )
    if args.cuts is None:
        _fail("--score-test needs --cuts pointing at a JSON file of frozen cuts.")
    frozen, reason = _cuts_are_frozen(args.cuts)
    if not frozen:
        _fail(reason)
    cuts = json.loads(args.cuts.read_text(encoding="utf-8"))
    # A decision id in the cuts file does not approve a rescore.
    raw_decision = getattr(args, "rescore_decision", None)
    decision_id = raw_decision.strip() if isinstance(raw_decision, str) else ""
    decision_id = decision_id or None
    try:
        combo = {}
        for name in _FROZEN_CUT_KEYS:
            if isinstance(cuts[name], bool):
                raise ValueError
            value = float(cuts[name])
            # float("nan") and float("inf") succeed. They are not cuts.
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError
            combo[name] = value
    except (TypeError, ValueError):
        _fail("Each cut in the cuts file must be a finite number. Nothing was written.")
    cache = Path(args.cache)
    if getattr(args, "rebuild_score_cache", False):
        _rebuild_score_cache(cache)
        print("Rebuilt cache with integrity receipts; no scoring or consumption.")
        return
    expected_map = dust_map_label()
    # Preflight ALL populations and sanity before decoding any held tables.
    folders = {
        population: cache / population / "test" for population in (LIKE_KIND, REALISTIC)
    }
    folders["sanity"] = cache / "at2017gfo-exact" / "copies"
    preflight = {}
    for population, folder in folders.items():
        preflight[population] = _preflight_score_cache(folder, expected_map)
    loaded_sets = []
    loaded_hashes = {}
    for population in (LIKE_KIND, REALISTIC):
        folder = cache / population / "test"
        manifest_path, events_path, curves_path = _cache_paths(folder)
        if (
            manifest_path.is_file()
            and curves_path.is_file()
            and not events_path.is_file()
        ):
            _fail(
                f"The {population} events table is missing ({events_path}). "
                "Nothing was written."
            )
        if (
            not manifest_path.is_file()
            or not events_path.is_file()
            or not curves_path.is_file()
        ):
            _fail(
                f"No complete cached test split for {population} in {cache}. "
                "Simulate it before scoring. This command does not simulate."
            )
        try:
            loaded_hashes[population] = preflight[population]["test_set_hashes"]
            loaded = _read_cache(folder)
        except ValueError as error:
            _fail(f"{error} Nothing was written.")
        if loaded is None:
            _fail(
                f"No cached test split for {population} in {cache}. "
                "Simulate it before scoring. This command does not simulate."
            )
        manifest, events, curves = loaded
        expected_n = manifest.get("n_simulated")
        if expected_n is None or int(expected_n) != len(events):
            _fail(
                f"The {population} test cache events do not match n_simulated "
                f"({expected_n!r} vs {len(events)} rows)."
            )
        loaded_sets.append((population, manifest, events, curves))
    for population, _manifest, events, _curves in loaded_sets:
        if not _object_ids(events):
            _fail(
                f"The {population} events table has no object_id values. "
                "Nothing was written."
            )
    # The same fakes are the same test set even when the cache version,
    # redback version, diffusion fix, or dust map differ. An object id
    # already scored in that population is the same set even when the
    # hash differs, for example a larger sample_size on the same seed.
    # A pending ledger entry counts as scored: the set is consumed by a
    # crash, and another score needs --rescore-decision.
    # Provenance preflight precedes loading; overlap is checked under
    # the lock before any pending consumption or numerical scoring.
    identity = test_set_sha256(loaded_sets)
    # Take the lock before any ledger or score file is written. A lock
    # that is already there is left in place. The run that creates it
    # removes it in ``finally``. A pending entry counts as scored.
    lock = _acquire_score_lock()
    try:
        _refuse_unless_score_is_allowed(identity, loaded_sets, decision_id)
        for population, manifest, _events, _curves in loaded_sets:
            reason = cache_refusal_reason(manifest)
            if reason is not None:
                _fail(f"The {population} test cache is refused: {reason}")
            if (
                manifest.get("dust_mode") != "sfd"
                or manifest.get("dust_map") != expected_map
            ):
                _fail(
                    f"The {population} test cache has no recorded SFD provenance. "
                    f"dust_map is {manifest.get('dust_map')!r}. "
                    f"It has to be {expected_map!r}."
                )
        receipts = {
            population: _verify_score_receipt(cache / population / "test")
            for population, *_ in loaded_sets
        }
        for population, receipt in receipts.items():
            if receipt.get("test_set_hashes") != loaded_hashes[population]:
                _fail("Cache changed while loading; nothing was consumed.")
        copies_receipt, copies_loaded = _verified_score_copies(cache)
        if copies_receipt != preflight["sanity"]:
            _fail("Sanity cache changed while loading; nothing was consumed.")
        _write_pending_score(
            identity,
            combo,
            loaded_sets,
            git_sha=_git_sha(),
            scored_at=_scored_at(),
            decision_id=decision_id,
            receipts=receipts,
            copies_receipt=copies_receipt,
        )
        by_population = {}
        sanity = None
        failure = None
        scientific_exit = None
        try:
            _claim_score_once_marker(loaded_sets, combo, decision_id=decision_id)
            for population, manifest, events, curves in loaded_sets:
                by_population[population] = score_with_frozen_cuts(
                    events,
                    curves,
                    combo,
                    population=population,
                    status=str(manifest.get("status", "ready")),
                )
                _record_score_outcome(
                    identity,
                    by_population,
                    sanity,
                    receipts,
                    copies_receipt,
                    status="pending",
                    reason="remaining computations not complete",
                )
            from kilonova_hunter.grid import (
                caught_not_judgeable_blocks,
                copy_sign_off,
                score_exact_copies,
            )

            sanity = score_exact_copies(copies_loaded[1], copies_loaded[2], combo)
            reasons = []
            if any(
                caught_not_judgeable_blocks(rows) for rows in by_population.values()
            ):
                reasons.append("rule 5: caught-but-not-judgeable is nonzero")
            blocked, copy_status = copy_sign_off(sanity)
            if blocked:
                reasons.append(f"sanity {copy_status}")
            # Persist numbers before formatting or writing the report can fail.
            failure = "; ".join(reasons) or None
            _record_score_outcome(
                identity,
                by_population,
                sanity,
                receipts,
                copies_receipt,
                status="FAIL" if failure else "pending",
                reason=failure,
            )
            text = format_test_score(
                by_population,
                dry_run=False,
                provisional=bool(cuts.get("provisional", False)),
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text + "\n", encoding="utf-8")
            _complete_pending_score(identity, destination)
            print(text)
            if failure:
                scientific_exit = SystemExit(2)
                raise scientific_exit
            _record_score_outcome(
                identity,
                by_population,
                sanity,
                receipts,
                copies_receipt,
                status="complete",
                reason=None,
            )
        except BaseException as error:
            reason = f"{type(error).__name__}: {error}"
            _record_score_outcome(
                identity,
                by_population,
                sanity,
                receipts,
                copies_receipt,
                status="FAIL" if error is scientific_exit else "ERROR",
                reason=failure if error is scientific_exit else reason,
                error=None if error is scientific_exit else error,
            )
            if error is scientific_exit:
                raise
            raise SystemExit(1) from error
    finally:
        _release_score_lock(lock)


def _preflight_score_cache(folder: Path, expected_map: str) -> dict:
    """Metadata and raw hashes only: never decode events or light curves."""
    import json

    from kilonova_hunter.sample import cache_refusal_reason

    receipt = _verify_score_receipt(folder)
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
    except (OSError, ValueError) as error:
        _fail(f"Cache manifest refused for {folder}: {error}")
    reason = cache_refusal_reason(manifest)
    if reason is not None:
        _fail(f"The {folder} cache is refused: {reason}")
    if manifest.get("dust_mode") != "sfd" or manifest.get("dust_map") != expected_map:
        _fail(f"The {folder} cache has no recorded SFD provenance.")
    return receipt


def _score_file_hash(path: Path) -> str:
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verified_score_copies(cache: Path):
    from kilonova_hunter.sample import _read_cache

    folder = cache / "at2017gfo-exact" / "copies"
    receipt = _verify_score_receipt(folder)
    loaded = _read_cache(folder)
    if loaded is None:
        _fail("No attested sanity copies; nothing was consumed.")
    if receipt != _verify_score_receipt(folder):
        _fail("Sanity cache changed while loading; nothing was consumed.")
    return receipt, loaded


def _score_map_provenance() -> dict:
    from kilonova_hunter.sky import sfd_directory

    folder = sfd_directory().resolve(strict=True)
    hashes = {
        name: _score_file_hash(folder / name)
        for name in ("SFD_dust_4096_ngp.fits", "SFD_dust_4096_sgp.fits")
    }
    return {"dust_map_path": str(folder), "dust_map_hashes": hashes}


def _score_generator_sha() -> str:
    """A commit id only attests the generator when its source is unchanged."""
    import subprocess

    sha = _git_sha()
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("No generator SHA available")
    result = subprocess.run(
        ["git", "diff", "--quiet", sha, "--", "src/kilonova_hunter"],
        cwd=_repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("Generator source differs from its recorded commit")
    return sha


def _score_source_tree(sha: str) -> str:
    """Hash the committed generator source, independent of non-source commits."""
    import subprocess

    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Invalid generator SHA")
    result = subprocess.run(
        ["git", "rev-parse", f"{sha}:src/kilonova_hunter"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    tree = result.stdout.strip()
    if result.returncode or len(tree) != 40:
        raise ValueError("Generator source tree unavailable")
    return tree


def _score_receipt(folder: Path, maps: dict) -> dict:
    from kilonova_hunter.redback_fix import DIFFUSION_FIX_ID
    from kilonova_hunter.sample import _cache_paths

    sha = _score_generator_sha()
    return {
        "generator_sha": sha,
        "generator_source_tree": _score_source_tree(sha),
        "diffusion_fix": DIFFUSION_FIX_ID,
        "dust_mode": "sfd",
        **maps,
        "test_set_hashes": {p.name: _score_file_hash(p) for p in _cache_paths(folder)},
    }


def _verify_score_receipt(folder: Path) -> dict:
    """Receipts bind a fresh CLI build to its code, maps and exact cache bytes."""
    import json

    try:
        receipt = json.loads((folder / "score_provenance.json").read_text())
        expected = _score_receipt(folder, _score_map_provenance())
        recorded_sha = receipt.get("generator_sha", "")
        if _score_source_tree(recorded_sha) != expected["generator_source_tree"]:
            raise ValueError("Generator source changed")
        expected["generator_sha"] = recorded_sha
        if receipt != expected:
            raise ValueError("missing, stale or mismatched provenance fields/hashes")
    except (OSError, ValueError) as error:
        _fail(f"Cache provenance refused for {folder}: {error}. Nothing consumed.")
    return receipt


def _rebuild_score_cache(cache: Path) -> None:
    """No reuse: frozen generators and defaults, in an exclusively new directory.

    A failed build has no usable receipts. This only builds; scoring requires
    a separate invocation, retaining the ordinary score-once checks.
    """
    import inspect
    import json

    from kilonova_hunter import sky
    from kilonova_hunter.grid import run_tuning_grid
    from kilonova_hunter.sample import (
        LIKE_KIND,
        REALISTIC,
        simulate_exact_copies,
        simulate_population,
    )

    if cache.exists():
        _fail(
            "Rebuild requires a NEW --cache directory; stale caches are never reused."
        )
    maps = _score_map_provenance()
    generator_sha = _score_generator_sha()
    # Bind the query to these maps, rather than an earlier process-global query.
    sky._SFD_QUERY = None
    dust = sky.open_dust_source()
    if dust.mode != "sfd":
        _fail("Rebuild requires working SFD dust; no fallback is allowed.")
    cache.mkdir(parents=True, exist_ok=False)
    folders = []
    for population in (LIKE_KIND, REALISTIC):
        simulate_population(population, "test", dust=dust, cache_dir=cache)
        folders.append(cache / population / "test")
    seed = inspect.signature(run_tuning_grid).parameters["copy_seed"].default
    simulate_exact_copies(dust=dust, cache_dir=cache, seed=seed)
    folders.append(cache / "at2017gfo-exact" / "copies")
    if maps != _score_map_provenance():
        raise ValueError("Dust maps changed during generation; no receipts written")
    if generator_sha != _score_generator_sha():
        raise ValueError("Generator changed during generation; no receipts written")
    for folder in folders:
        (folder / "score_provenance.json").write_text(
            json.dumps(_score_receipt(folder, maps), sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _record_score_outcome(
    identity,
    numbers,
    sanity,
    receipts,
    copies_receipt,
    *,
    status,
    reason,
    error=None,
) -> None:
    """Persist returned results; pending, FAIL and ERROR attempts stay consumed."""
    import traceback

    payload = _ledger_payload()
    entry = payload["sets"][identity][-1]
    entry.update(
        {
            "status": status,
            "consumed": True,
            "numbers": {
                population: numbers.get(
                    population,
                    [{"day": day, "result": None} for day in (3.0, 5.0, 7.0)],
                )
                for population in receipts
            },
            "exception": (
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(traceback.format_exception(error)),
                }
                if error is not None
                else None
            ),
            "sanity": sanity,
            "fail_reason": reason,
            "not_computed": {
                name: "not returned; internal progress unavailable"
                for name in ("2017-like", "realistic", "sanity")
                if (sanity is None if name == "sanity" else name not in numbers)
            },
        }
    )
    for population, receipt in receipts.items():
        entry["provenance"][population].update(receipt)
    entry["sanity_provenance"] = copies_receipt
    _write_ledger(test_set_scored_path(), payload)


# Ruling 12(21): the score-once key is the test set's identity. Software
# and dust provenance are recorded on the ledger entry and are not part
# of the hash. There is no flag that overrides the refusal.
# sample_size is not part of the identity. A larger draw of the same
# seed shares object ids and is refused by the overlap check.
_TEST_SET_IDENTITY_KEYS = (
    "parent_seed",
    "seed",
    "survey",
)
_TEST_SET_RECORDED_KEYS = (
    "cache_version",
    "redback_version",
    "diffusion_fix",
    "dust_mode",
    "dust_map",
)
_RESCORE_NEEDS_A_DECISION = (
    "A rescore needs a logged decision approved by Sage through PM, "
    "and both scores must be reported."
)


def _repo_root() -> Path:
    """Directory that contains ``src`` and ``results``. Not the working directory."""
    root = Path(__file__).resolve().parents[2]
    assert (root / "pyproject.toml").is_file(), (
        f"pyproject.toml was not found in {root}."
    )
    return root


def _fink_exposure_note() -> tuple[str, str]:
    """The committed note that the score-once key does not hash.

    Test-night Fink rows were downloaded before the cuts were frozen.
    The sentence lives in ``results/test_set_exposure.json``. A ledger
    entry copies it. The identity hash does not.
    """
    import json

    relative = Path("results") / "test_set_exposure.json"
    payload = json.loads((_repo_root() / relative).read_text(encoding="utf-8"))
    return str(payload["exposure_note"]), relative.as_posix()


def test_set_scored_path() -> Path:
    """Committed ledger of scored test sets. Not inside the cache folder.

    After a score, commit this file and ``results/test_set_claims/`` with
    the score. A later checkout then refuses the same test set even if
    the cache directory is new.
    """
    return _repo_root() / "results" / "test_set_scored.json"


def test_set_claim_dir() -> Path:
    """Exclusive claim files, one per test-set hash, beside the ledger."""
    return test_set_scored_path().parent / "test_set_claims"


def _object_ids(events) -> list[str]:
    """Sorted object ids. Blank and missing values are not ids."""
    if events is None or getattr(events, "empty", True):
        return []
    if "object_id" not in events.columns:
        return []
    found: list[str] = []
    for item in events["object_id"].tolist():
        if item is None:
            continue
        text = str(item).strip()
        if not text or text.lower() == "nan":
            continue
        found.append(text)
    return sorted(set(found))


def test_set_sha256(loaded_sets: list) -> str:
    """Hash of seeds, the population spec, and sorted fake ids.

    ``sample_size``, ``cache_version``, ``redback_version``,
    ``diffusion_fix``, and the dust map are not in this hash. The cache
    path is not in it either. An empty id list is not hashed.
    """
    import hashlib
    import json

    parts = []
    for population, manifest, events, _curves in loaded_sets:
        object_ids = _object_ids(events)
        if not object_ids:
            raise ValueError(f"The {population} events table has no object_id values.")
        spec = {key: manifest.get(key) for key in _TEST_SET_IDENTITY_KEYS}
        parts.append({"population": population, "spec": spec, "object_ids": object_ids})
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _population_spec(manifest: dict) -> dict:
    """Identity of the population draw, without ``sample_size``."""
    return {key: manifest.get(key) for key in _TEST_SET_IDENTITY_KEYS}


def _ledger_payload() -> dict:
    import json

    path = test_set_scored_path()
    if not path.is_file():
        return {"sets": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("sets", {})
    return payload


def _ids_in_entry(entry: dict, scored: dict[str, set[str]]) -> None:
    for population, ids in (entry.get("object_ids") or {}).items():
        scored.setdefault(str(population), set()).update(str(item) for item in ids)


def _previously_scored_ids() -> dict[str, set[str]]:
    """Object ids already scored, grouped by population name.

    A pending entry counts as scored. The set is consumed by a crash.
    Another score of those ids needs ``--rescore-decision``. Claim files
    are read as well as the ledger, so a claim that has the ids still
    blocks if the ledger write did not finish.
    """
    scored: dict[str, set[str]] = {}
    for entries in _ledger_payload().get("sets", {}).values():
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            # pending and complete are the same for this check.
            _ids_in_entry(entry, scored)
    claim_dir = test_set_claim_dir()
    if claim_dir.is_dir():
        import json

        for path in claim_dir.iterdir():
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                _ids_in_entry(payload, scored)
    return scored


def _overlap_count(loaded_sets: list) -> int:
    """How many new object ids were already scored in that population."""
    previous = _previously_scored_ids()
    count = 0
    for population, _manifest, events, _curves in loaded_sets:
        fresh = set(_object_ids(events))
        count += len(fresh & previous.get(population, set()))
    return count


def _decision_already_used(test_hash: str, loaded_sets: list, decision_id: str) -> bool:
    """True when this decision id already approved this set or these ids."""
    for hash_key, entries in _ledger_payload().get("sets", {}).items():
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            if entry.get("decision_id") != decision_id:
                continue
            if hash_key == test_hash:
                return True
            recorded = entry.get("object_ids") or {}
            for population, _manifest, events, _curves in loaded_sets:
                fresh = set(_object_ids(events))
                prior = {str(item) for item in recorded.get(population, [])}
                if fresh & prior:
                    return True
    return False


def _provenance(loaded_sets: list) -> dict:
    """The four fields ruling 12(21) took out of the score-once key."""
    recorded = {}
    for population, manifest, _events, _curves in loaded_sets:
        recorded[population] = {
            key: manifest.get(key) for key in _TEST_SET_RECORDED_KEYS
        }
    return recorded


def _git_sha() -> str:
    """HEAD of this checkout, or ``unknown`` when git cannot say."""
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _scored_at() -> str:
    """UTC time the ledger entry was written."""
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _test_set_already_claimed(test_hash: str) -> bool:
    claim = test_set_claim_dir() / test_hash
    return test_hash in _ledger_hashes() or claim.is_file()


def _refuse_rescore(test_hash: str, *, fresh_cache: bool) -> None:
    """Stop. The same test set is not scored again from this command."""
    if fresh_cache:
        detail = (
            f"Test set {test_hash[:12]} was scored once "
            f"({test_set_scored_path()}). "
            "A fresh --cache does not allow another score."
        )
    else:
        detail = (
            f"Test set {test_hash[:12]} was scored once. "
            "A different --output or --cache does not allow another score."
        )
    _fail(f"{detail} {_RESCORE_NEEDS_A_DECISION}")


def _ledger_hashes() -> set[str]:
    import json

    path = test_set_scored_path()
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload.get("sets", {}))


def _score_json_values(value):
    """Undefined ledger values are null; measured zero remains zero."""
    import math

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _score_json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_score_json_values(item) for item in value]
    return value


def _write_ledger(path: Path, payload: dict) -> None:
    """Replace the ledger atomically. The temp file sits in the same directory."""
    import json
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(_score_json_values(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _score_lock_path() -> Path:
    """Exclusive lock next to the ledger. A stale one is removed by hand."""
    ledger = test_set_scored_path()
    return ledger.with_name(f"{ledger.name}.lock")


def _acquire_score_lock() -> int:
    """Create the lock, or refuse without deleting a lock that is already there.

    There is no age check and no automatic removal. The message names
    the path so a person can confirm no run is active and delete it.
    """
    import os

    path = _score_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError:
        _fail(
            f"The score lock {path} is already present. "
            "Check that no scoring run is active, then remove that file "
            "by hand. Nothing was written."
        )
    os.write(descriptor, f"pid {os.getpid()}\n".encode())
    return descriptor


def _release_score_lock(descriptor: int) -> None:
    """Remove the lock this run created. Does not run if acquire refused."""
    import os

    path = _score_lock_path()
    os.close(descriptor)
    try:
        os.remove(path)
    except FileNotFoundError:
        return


def _ledger_entries(current: object) -> list:
    if current is None:
        return []
    if isinstance(current, dict):
        return [current]
    return list(current)


def _object_ids_by_population(loaded_sets: list) -> dict[str, list[str]]:
    return {
        population: _object_ids(events)
        for population, _manifest, events, _curves in loaded_sets
    }


def _population_specs(loaded_sets: list) -> dict:
    return {
        population: _population_spec(manifest)
        for population, manifest, _events, _curves in loaded_sets
    }


def _refuse_unless_score_is_allowed(
    identity: str, loaded_sets: list, decision_id: str | None
) -> None:
    """Stop unless this score is the first, or an unused decision approves it."""
    overlap = _overlap_count(loaded_sets)
    already = _test_set_already_claimed(identity)
    if decision_id is None:
        if already:
            _refuse_rescore(identity, fresh_cache=True)
        if overlap:
            _fail(
                f"{overlap} object ids in this test set were already scored "
                f"in the same population. {_RESCORE_NEEDS_A_DECISION}"
            )
        return
    if not already and overlap == 0:
        _fail(
            "--rescore-decision was given, but this test set has not been "
            "scored and none of its object ids overlap a scored population. "
            "Nothing was written."
        )
    if _decision_already_used(identity, loaded_sets, decision_id):
        _fail(
            f"Decision {decision_id} was already used for this test set. "
            f"{_RESCORE_NEEDS_A_DECISION}"
        )


def _write_pending_score(
    test_hash: str,
    combo: dict[str, float],
    loaded_sets: list,
    *,
    git_sha: str,
    scored_at: str,
    decision_id: str | None = None,
    receipts: dict | None = None,
    copies_receipt: dict | None = None,
) -> None:
    """Append a pending entry before any score file is written.

    A pending entry counts as scored. The set is consumed by a crash.
    Rescore is only via ``--rescore-decision``. This function does not
    delete or rewrite an older pending entry. An approved rescore
    appends another entry and leaves the pending one in the history.
    The run that wrote this entry marks it complete after the score
    file exists.
    """
    import hashlib
    import json

    cuts = json.dumps(combo, sort_keys=True, separators=(",", ":"))
    note, note_path = _fink_exposure_note()
    entry = {
        "status": "pending",
        "test_set_sha256": test_hash,
        "cuts_sha256": hashlib.sha256(cuts.encode("utf-8")).hexdigest(),
        "cuts": combo,
        "provenance": _provenance(loaded_sets),
        "population_spec": _population_specs(loaded_sets),
        "object_ids": _object_ids_by_population(loaded_sets),
        "git_sha": git_sha,
        "scored_at": scored_at,
        "exposure_note": note,
        "exposure_note_path": note_path,
    }
    if decision_id:
        entry["decision_id"] = decision_id
    if receipts is not None:
        entry.update(
            {
                "consumed": True,
                "numbers": {
                    population: [
                        {"day": day, "result": None} for day in (3.0, 5.0, 7.0)
                    ]
                    for population in receipts
                },
                "exception": None,
                "sanity": None,
                "not_computed": {
                    name: "not returned; internal progress unavailable"
                    for name in (*receipts, "sanity")
                },
                "fail_reason": "scoring not yet complete",
                "sanity_provenance": copies_receipt,
            }
        )
        for population, receipt in receipts.items():
            entry["provenance"][population].update(receipt)
    payload = _ledger_payload()
    entries = _ledger_entries(payload.setdefault("sets", {}).get(test_hash))
    entries.append(entry)
    payload["sets"][test_hash] = entries
    _write_ledger(test_set_scored_path(), payload)


def _complete_pending_score(test_hash: str, score_path: Path) -> None:
    """Mark the pending entry this run appended as complete.

    Only the last pending entry for this hash is updated. Older pending
    entries stay pending. The score path and its sha256 are recorded
    here, after the score file exists.
    """
    import hashlib

    payload = _ledger_payload()
    entries = _ledger_entries(payload.get("sets", {}).get(test_hash))
    if not entries or entries[-1].get("status") not in ("pending", "FAIL"):
        _fail(
            "The pending score entry is missing, so the score was not "
            "recorded as complete. Nothing else was changed."
        )
    score_bytes = score_path.read_bytes()
    if entries[-1]["status"] != "FAIL":
        entries[-1]["status"] = "complete"
    entries[-1]["score_path"] = str(score_path)
    entries[-1]["score_sha256"] = hashlib.sha256(score_bytes).hexdigest()
    payload["sets"][test_hash] = entries
    _write_ledger(test_set_scored_path(), payload)


def _claim_score_once_marker(
    loaded_sets: list,
    combo: dict[str, float],
    *,
    decision_id: str | None = None,
) -> None:
    """Create an exclusive claim for this test set before scoring.

    The ledger lives at ``results/test_set_scored.json``, not in the
    cache. Copying the cache, or pointing ``--cache`` at a fresh folder
    with the same fakes, does not clear it. A different seed set gets
    its own claim. Changing the cache version, redback version, diffusion
    fix, or dust map does not. A rescore of the same set is appended
    only when ``decision_id`` names the approving decision. The first
    entry stays. The claim stores the population and the object ids.
    This looks at the claim file only. The pending ledger row for this
    run is already written, and it must not block its own claim.
    """
    import json
    import os

    test_hash = test_set_sha256(loaded_sets)
    claim = test_set_claim_dir() / test_hash
    if claim.is_file():
        if not decision_id:
            _refuse_rescore(test_hash, fresh_cache=True)
        return
    git_sha = _git_sha()
    scored_at = _scored_at()
    claim.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(claim, flags, 0o644)
    except FileExistsError:
        _refuse_rescore(test_hash, fresh_cache=False)
    body = (
        json.dumps(
            {
                "test_set_sha256": test_hash,
                "population_spec": _population_specs(loaded_sets),
                "object_ids": _object_ids_by_population(loaded_sets),
                "cuts": combo,
                "provenance": _provenance(loaded_sets),
                "git_sha": git_sha,
                "scored_at": scored_at,
            },
            indent=2,
        )
        + "\n"
    )
    try:
        os.write(descriptor, body.encode("utf-8"))
    finally:
        os.close(descriptor)


def _run_grid(args) -> None:
    """Simulate and score the tuning grid. Does not pick or score the test set."""
    import logging

    from kilonova_hunter.grid import run_tuning_grid
    from kilonova_hunter.sample import SPLITS, SampleSizeSettings

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
    # Missing on a hand-built args object means the historical default:
    # both splits. --tuning-only never asks for the test split.
    splits = ("tuning",) if getattr(args, "tuning_only", False) else SPLITS
    result = run_tuning_grid(
        output_dir=args.output,
        cache_dir=args.cache,
        sample_size=size,
        workers=max(1, args.workers),
        splits=splits,
    )
    print(result.text)
    if result.sign_off_blocker:
        if result.block_reasons:
            detail = "; ".join(result.block_reasons)
        else:
            label = (result.copy_status or "blocker").upper()
            detail = f"the exact-copy verdict is {label}"
        print(f"Sign-off blocker: {detail}.", file=sys.stderr)
        raise SystemExit(2)


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

"""The ``kilohunt`` command.

This is the only module that talks to a person. The other modules take
data in and hand data back, which is what makes them testable without
parsing printed text. A dashboard after version 1.0 can call those same
functions and needs almost no new science code.
"""

import argparse
from pathlib import Path

from kilonova_hunter.inject import simulate_kilonova_light_curve
from kilonova_hunter.report import plot_light_curve

# Default place for a plot you make while exploring. The folder is
# gitignored, so trying the demo does not dirty the repository.
_DEFAULT_DEMO_PATH = Path("output") / "demo_kilonova.png"


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
            "Searching a real week of alerts (kilohunt --week) arrives in Sprint 3."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    demo = subparsers.add_parser(
        "demo",
        help="Plot a made-up kilonova light curve.",
        description=(
            "Build a kilonova-like light curve (a fast rise, a week-long fade, "
            "and a color that turns red) and save a labeled plot. "
            "The magnitude axis is flipped so brighter is up."
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
            "Random seed for the noise. The same seed redraws the same curve. "
            "Default: 42."
        ),
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
        _run_demo(args.output, args.seed)
        return
    # Running `kilohunt` with no command should explain itself, the way
    # `--help` does, rather than failing on a missing subcommand.
    parser.print_help()


def _run_demo(output: Path, seed: int) -> None:
    """Generate the toy light curve and save its plot."""
    light_curve = simulate_kilonova_light_curve(seed=seed)
    saved = plot_light_curve(
        light_curve,
        output,
        title="Made-up kilonova light curve",
    )
    print(f"Saved a made-up kilonova light curve to {saved}")


if __name__ == "__main__":
    main()

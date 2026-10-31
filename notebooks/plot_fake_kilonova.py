"""Plot a made-up kilonova without going through the kilohunt command.

Run it from the project folder:

    uv run python notebooks/plot_fake_kilonova.py

``uv run kilohunt demo`` does the same thing. This script keeps the two
function calls visible in one place while you are learning the pipeline.
"""

from pathlib import Path

from kilonova_hunter.inject import simulate_kilonova_light_curve
from kilonova_hunter.report import plot_light_curve


def main() -> None:
    """Build the toy light curve and save the plot under ``output/``."""
    light_curve = simulate_kilonova_light_curve()
    destination = Path("output") / "demo_kilonova.png"
    saved = plot_light_curve(
        light_curve,
        destination,
        title="Made-up kilonova light curve",
    )
    print(f"Saved {saved}")


if __name__ == "__main__":
    main()

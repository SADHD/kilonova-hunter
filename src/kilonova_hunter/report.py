"""Plots and, later, the written candidate report.

A magnitude is a brightness in which a larger number means a fainter
object. Every plot from this module flips the vertical axis so brighter
is toward the top, which is how most people already read a chart.

``plot_light_curve`` works today. The ranked candidate list, with one
sentence per rule, arrives with the filter in Sprint 3.
"""

from pathlib import Path

import pandas as pd

# Colors astronomers often use for these bands, so g reads as the bluer,
# greener filter and r as the redder one.
_FILTER_COLORS = {"g": "#2ca02c", "r": "#d62728"}

# Held onto after the first plot. Switching Matplotlib's backend has to
# happen before pyplot is imported, and importing it twice would warn.
_pyplot = None

_REQUIRED_COLUMNS = ("time_days", "filter", "magnitude")


def _load_pyplot():
    """Import pyplot set up to draw into a file.

    Matplotlib's usual backend tries to open a window. Tests and GitHub
    Actions have no window to open, so we select Agg, which only writes
    image files. The selection has to happen before pyplot is imported,
    which is why this import lives inside a function.
    """
    global _pyplot
    if _pyplot is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _pyplot = plt
    return _pyplot


def plot_light_curve(
    light_curve: pd.DataFrame,
    output_path: Path | str,
    *,
    title: str = "Light curve",
) -> Path:
    """Save a light curve with one series per filter and brighter at the top.

    Extra columns are ignored, so a table that also carries an object id
    or an error still plots. Pass one transient at a time: every row is
    drawn, and two objects in one table would land on top of each other.

    Args:
        light_curve: Observations with columns ``time_days``, ``filter``,
            and ``magnitude``. ``magnitude_error`` is drawn as error bars
            when it is present.
        output_path: Where to write the image, usually a ``.png``. Parent
            folders are created if they are missing.
        title: Text at the top of the figure.

    Returns:
        The path that was written.

    Raises:
        ValueError: If a required column is missing, the table is empty,
            or it contains more than one ``object_id``.
    """
    missing = [name for name in _REQUIRED_COLUMNS if name not in light_curve.columns]
    if missing:
        raise ValueError(f"light curve is missing columns: {missing}")
    if light_curve.empty:
        raise ValueError("light curve has no rows to plot")
    if "object_id" in light_curve.columns and light_curve["object_id"].nunique() > 1:
        raise ValueError(
            "plot_light_curve draws one transient. "
            "Split the table by object_id and plot each piece."
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt = _load_pyplot()
    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    # sort=True makes the legend order stable (g then r) across machines.
    for filter_name, group in light_curve.groupby("filter", sort=True):
        group = group.sort_values("time_days")
        color = _FILTER_COLORS.get(str(filter_name))
        yerr = group["magnitude_error"] if "magnitude_error" in group.columns else None
        axes.errorbar(
            group["time_days"],
            group["magnitude"],
            yerr=yerr,
            marker="o",
            linestyle="-",
            capsize=2,
            label=f"{filter_name} band",
            color=color,
        )

    # A larger magnitude is fainter, so flip the axis. Brighter points sit higher.
    axes.invert_yaxis()
    axes.set_xlabel("Days since explosion")
    axes.set_ylabel("Magnitude (brighter is up)")
    axes.set_title(title)
    axes.legend(title="Filter")
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=120)
    # Close the figure so a loop of plots does not keep them all in memory.
    plt.close(figure)
    return output


def write_candidate_report(candidates: pd.DataFrame, output_dir: Path | str) -> Path:
    """Write the ranked candidate list and the reason each rule gave.

    Sprint 3 prints this from ``kilohunt`` and saves the plots beside it.
    Sprint 5 adds the weekly scorecard of impostors.

    Args:
        candidates: Scored transients from ``score_candidates``.
        output_dir: Folder for the report and its plots. Created if needed.

    Returns:
        The path of the written report.

    Raises:
        NotImplementedError: Until Sprint 3.
    """
    raise NotImplementedError(
        "Candidate reports arrive in Sprint 3 "
        f"({len(candidates)} candidates would have been written to {output_dir}). "
        "plot_light_curve already saves a single light curve."
    )

"""Tests for saving a light-curve plot."""

from pathlib import Path

import pandas as pd
import pytest

from kilonova_hunter.inject import simulate_kilonova_light_curve
from kilonova_hunter.report import plot_light_curve


def test_plot_writes_a_png(tmp_path: Path) -> None:
    """The plot helper creates missing folders and writes a real PNG file."""
    curve = simulate_kilonova_light_curve()
    destination = tmp_path / "nested" / "curve.png"

    written = plot_light_curve(curve, destination, title="Test curve")

    assert written == destination
    assert destination.is_file()
    # A PNG file always starts with these eight bytes.
    assert destination.read_bytes().startswith(b"\x89PNG")


def test_plot_leaves_the_backend_alone(tmp_path: Path) -> None:
    """Saving a PNG must not switch Matplotlib's backend.

    A notebook draws its inline plots with that backend. ``plot_light_curve``
    writes the file on its own figure, so the notebook backend stays as it was.
    """
    import matplotlib

    # "svg" stands in for a notebook backend: any backend other than the
    # file-only one. The assertion fails if the plot function changes it.
    matplotlib.use("svg")
    plot_light_curve(simulate_kilonova_light_curve(), tmp_path / "curve.png")
    assert matplotlib.get_backend().lower() == "svg"


def test_plot_rejects_a_table_with_two_objects(tmp_path: Path) -> None:
    """Two transients in one table would draw on top of each other, so we refuse."""
    curve = simulate_kilonova_light_curve()
    curve["object_id"] = "one"
    extra = curve.copy()
    extra["object_id"] = "two"
    both = pd.concat([curve, extra], ignore_index=True)

    with pytest.raises(ValueError, match="one transient"):
        plot_light_curve(both, tmp_path / "both.png")

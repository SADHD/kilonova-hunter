"""Compare the three-component model at 40 Mpc with AT2017gfo photometry.

The points are g, r, and i detections from the Open Kilonova Catalog
file ``GW170817.json`` in https://github.com/astrocatalogs/kne-2000-2029
(Villar et al. 2017's public compilation). Model realizations and upper
limits were left out. Magnitudes in the CSV are as published, with no
Milky Way correction. This figure subtracts the Fitzpatrick 1999 law
at E(B-V) = 0.105 and R_V = 3.1, and plots the model with no dust, so
the two can be compared.

The model is the Table 2 center (no scatter) at 40 Mpc. The light
curve goes through the diffusion wrapper, so the Arnett factor of 2
is included. The installed redback package is not edited.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from kilonova_hunter.sky import fitzpatrick_extinction_mag
from kilonova_hunter.villar import EXACT_DISTANCE_MPC, VillarEvent, exact_components

# NGC 4993 / GW170817. The catalog lists the merger at MJD 57982.5.
# time_days in the CSV already uses 57982.528524.
AT2017GFO_EBV = 0.105

_COLORS = {"g": "#2ca02c", "r": "#d62728", "i": "#7b4b9a"}


def _exact_event() -> VillarEvent:
    blue, purple, red = exact_components()
    from kilonova_hunter.sky import redshift_at_luminosity_distance

    return VillarEvent(
        object_id="at2017gfo-exact-plot",
        kind="at2017gfo-exact",
        distance_mpc=EXACT_DISTANCE_MPC,
        redshift=redshift_at_luminosity_distance(EXACT_DISTANCE_MPC),
        ebv=0.0,
        ra_deg=float("nan"),
        dec_deg=float("nan"),
        galactic_b_deg=float("nan"),
        merger_mjd=57982.528524,
        viewing_angle_deg=30.0,
        blue=blue,
        purple=purple,
        red=red,
    )


def load_photometry(path: Path) -> pd.DataFrame:
    """Read the committed g, r, i table. Comment lines start with #."""
    return pd.read_csv(path, comment="#")


def write_brightness_check(
    output: Path,
    photometry: Path,
) -> Path:
    """Save the comparison figure and return its path."""
    import matplotlib

    from kilonova_hunter.villar import villar_magnitudes

    # redback turns LaTeX on at import. This machine has no LaTeX install,
    # so turn it back off before any text is drawn.
    matplotlib.rcParams["text.usetex"] = False
    points = load_photometry(photometry)
    points = points.loc[(points["time_days"] >= 0.0) & (points["time_days"] <= 6.0)]
    event = _exact_event()
    times = np.linspace(0.3, 6.0, 40)
    # The model call switches LaTeX on. Finish every model call before
    # any text is drawn, then switch it off.
    model_curves = {
        band: villar_magnitudes(times, np.array([band] * len(times)), event)
        for band in ("g", "r", "i")
    }
    matplotlib.rcParams["text.usetex"] = False
    figure = Figure(figsize=(7.2, 4.8))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    for band in ("g", "r", "i"):
        magnitudes = model_curves[band]
        axis.plot(
            times,
            magnitudes,
            color=_COLORS[band],
            label=f"model {band}, no dust",
        )
        observed = points.loc[points["band"] == band]
        dimming = fitzpatrick_extinction_mag(AT2017GFO_EBV, band)
        corrected = observed["magnitude"].to_numpy(dtype=float) - dimming
        axis.scatter(
            observed["time_days"],
            corrected,
            s=18,
            color=_COLORS[band],
            label=f"observed {band}",
            zorder=3,
        )
    axis.invert_yaxis()
    axis.set_xlabel("Days after the merger")
    axis.set_ylabel("AB magnitude (brighter is up)")
    axis.set_title("Villar model at 40 Mpc, diffusion fix, vs AT2017gfo")
    axis.legend(fontsize=8, loc="upper right")
    matplotlib.rcParams["text.usetex"] = False
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    return output

"""Model-minus-AT2017gfo g-r at 1.5, 2, 3 d. Model: wrapped redback
three_component, Villar T2, dense grid, LSST bands, 40 Mpc, no dust. Data:
docs/at2017gfo_photometry.csv (AB), MW-dereddened with F99 R_V=3.1 E(B-V)=0.105
at LSST effective wavelengths (same as the PR), each point moved to the LSST
band with the model's own (LSST - instrument-band) term (PR's instrument
mapping), then a local linear fit in time within +-W d of the epoch.
"""

import warnings

import bootstrap

warnings.filterwarnings("ignore")
bootstrap.install()

import astropy.units as u
import numpy as np
import pandas as pd
import redback.transient_models.kilonova_models as km
from astropy.cosmology import Planck18, z_at_value

from kilonova_hunter.redback_fix import corrected_diffusion
from kilonova_hunter.sky import fitzpatrick_extinction_mag

z = float(z_at_value(Planck18.luminosity_distance, 40 * u.Mpc))
COMP = [
    (0.020, 0.266, 674.0, 0.5),
    (0.047, 0.152, 1308.0, 3.0),
    (0.011, 0.137, 3745.0, 10.0),
]
args = [value for component in COMP for value in component]
photometry = pd.read_csv(
    bootstrap.ROOT / "docs" / "at2017gfo_photometry.csv", comment="#"
)
photometry = photometry[photometry.system == "AB"].copy()
residuals = pd.read_csv(bootstrap.ROOT / "docs" / "at2017gfo_bandpass_residuals.csv")


def photometry_key(frame):
    return frame.time_days.round(5).astype(str) + frame.band


photometry["inst_band"] = photometry.assign(k=photometry_key(photometry)).k.map(
    dict(zip(photometry_key(residuals), residuals.inst_band, strict=True))
)
photometry = photometry.dropna(subset=["inst_band"])


def mag(times, bands):
    with corrected_diffusion():
        return np.asarray(
            km.three_component_kilonova_model(
                np.asarray(times, float),
                z,
                *args,
                bands=np.asarray(bands),
                output_format="magnitude",
                cosmology=Planck18,
            ),
            float,
        )


photometry["m_lsst"] = mag(photometry.time_days, "lsst" + photometry.band)
photometry["m_inst"] = mag(photometry.time_days, photometry.inst_band)
photometry["dust"] = [
    fitzpatrick_extinction_mag(0.105, band) for band in photometry.band
]
photometry["obs_lsst"] = (
    photometry.magnitude - photometry.dust + (photometry.m_lsst - photometry.m_inst)
)
photometry["obs_raw"] = photometry.magnitude - photometry.dust


def local(band, t0, col, window):
    chosen = photometry[
        (photometry.band == band) & (abs(photometry.time_days - t0) <= window)
    ]
    if len(chosen) < 2 or chosen.time_days.nunique() < 2:
        return np.nan, len(chosen)
    weights = 1 / np.maximum(chosen.magnitude_error.fillna(0.05), 0.02)
    coefficients = np.polyfit(chosen.time_days - t0, chosen[col], 1, w=weights)
    return coefficients[1], len(chosen)


print(
    " t(d)  W   n_g n_r  data g-r (LSST-equiv)  data g-r (instr. bands)  "
    "model g-r (LSST)  model-minus-data (LSST)  model g  data g(LSST)"
)
for window in (0.3, 0.5):
    for t0 in (1.5, 2.0, 3.0):
        g_lsst, n_g = local("g", t0, "obs_lsst", window)
        r_lsst, n_r = local("r", t0, "obs_lsst", window)
        g_raw, _n_g_raw = local("g", t0, "obs_raw", window)
        r_raw, _n_r_raw = local("r", t0, "obs_raw", window)
        model_g = float(mag([t0], ["lsstg"])[0])
        model_r = float(mag([t0], ["lsstr"])[0])
        print(
            f" {t0:3.1f}  {window:.1f}  {n_g:3d} {n_r:3d}   "
            f"{g_lsst - r_lsst:+.3f}"
            "                 "
            f"{g_raw - r_raw:+.3f}"
            "                   "
            f"{model_g - model_r:+.3f}"
            "            "
            f"{(model_g - model_r) - (g_lsst - r_lsst):+.3f}"
            "                "
            f"{model_g:.2f}   {g_lsst:.2f}"
        )

print(
    "\nRobust version: per-night medians (same-night g and r), "
    "g-r interpolated linearly between nights"
)
photometry["night"] = np.floor(photometry.time_days - 0.4).astype(int)
rows = []
for _night, group in photometry.groupby("night"):
    g_points = group[group.band == "g"]
    r_points = group[group.band == "r"]
    if len(g_points) and len(r_points):
        rows.append(
            (
                group.time_days.median(),
                g_points.obs_lsst.median() - r_points.obs_lsst.median(),
                g_points.obs_raw.median() - r_points.obs_raw.median(),
                len(g_points),
                len(r_points),
            )
        )
nights = pd.DataFrame(rows, columns=["t", "gr_lsst", "gr_raw", "n_g", "n_r"])
print(nights.round(3).to_string(index=False))
print(" t(d)  data g-r(LSST-equiv)  model g-r(LSST)  model-minus-data")
for t0 in (1.5, 2.0, 3.0):
    data_color = np.interp(t0, nights.t, nights.gr_lsst)
    model_color = float(mag([t0], ["lsstg"])[0] - mag([t0], ["lsstr"])[0])
    print(
        f" {t0:3.1f}   {data_color:+.3f}               {model_color:+.3f}           "
        f"{model_color - data_color:+.3f}"
    )

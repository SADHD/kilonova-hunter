"""12(19): wrapped redback three_component_kilonova_model vs MOSFiT (6595248
formulas, reimplemented) in LSST g, r, i (AB, no dust) at 40 Mpc, Planck18,
Villar Table 2, dense grids for both. Also: bolometric on dense grids, and
redback L/T/R pushed through an independent blackbody+bandpass integration
(separates the light-curve engine from the SED/photometry step).
"""

import warnings

import bootstrap

warnings.filterwarnings("ignore")
bootstrap.install()

import astropy.units as u
import numpy as np
import redback.transient_models.kilonova_models as km
import sncosmo
from astropy.cosmology import Planck18, z_at_value
from redback.utils import get_optimal_time_array
from sidebyside_lib import COMP, DAY, KM, C, mosfit_diffusion

from kilonova_hunter.redback_fix import corrected_diffusion

z = float(z_at_value(Planck18.luminosity_distance, 40 * u.Mpc))
dl = (40 * u.Mpc).to(u.cm).value
print(f"z = {z:.6f} (Planck18, d_L = 40 Mpc)")
H, KB, SIG = 6.62607015e-27, 1.380649e-16, 5.670374e-5
T_REP = np.array([0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0])
t_obs = np.unique(np.concatenate([np.geomspace(0.3, 10, 300), T_REP, [1.5, 2.0, 3.0]]))
t_rest = t_obs / (1 + z)
args = []
for _, m, v, k, tf in COMP:
    args += [m, v, tf, k]

# redback, public function, wrapped, as villar.py calls it (no dust).
rb = {}
with corrected_diffusion():
    for band in "gri":
        rb[band] = np.asarray(
            km.three_component_kilonova_model(
                t_obs,
                z,
                *args,
                bands=np.array(["lsst" + band] * len(t_obs)),
                output_format="magnitude",
                cosmology=Planck18,
            ),
            float,
        )


# photometry helper: sum of blackbodies -> AB mag (photon counting)
lam_obs = np.linspace(2500, 12500, 4000)  # Angstrom


def ab_mags(radii, temperatures):
    """Radii and temperatures: one array per component, over rest-frame time."""
    lam_rest_cm = lam_obs[None, :] / (1 + z) * 1e-8
    spectral = 0.0
    for radius, temperature in zip(radii, temperatures, strict=True):
        radius = np.asarray(radius)[:, None]
        temperature = np.asarray(temperature)[:, None]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            planck = (
                2
                * H
                * C**2
                / lam_rest_cm**5
                / np.expm1(H * C / (lam_rest_cm * KB * temperature))
            )
        # erg/s/cm (rest)
        spectral = spectral + np.nan_to_num(4 * np.pi**2 * radius**2 * planck)
    # erg/s/cm2/A observed
    flux = spectral / (4 * np.pi * dl**2 * (1 + z)) * 1e-8
    out = {}
    for band in "gri":
        bandpass = sncosmo.get_bandpass("lsst" + band)
        transmission = bandpass(lam_obs)
        numerator = np.trapezoid(flux * transmission * lam_obs, lam_obs, axis=1)
        denominator = np.trapezoid(
            3631e-23 * 2.99792458e18 / lam_obs**2 * transmission * lam_obs,
            lam_obs,
        )
        out[band] = -2.5 * np.log10(numerator / denominator)
    return out


# MOSFiT: diffusion on MOSFiT's dense grid, temperature_floor photosphere, blackbody.
dense = np.unique(
    np.concatenate(([0.0], np.logspace(-6, np.log10(t_rest.max()), 100), t_rest))
)


def mosfit_run(kappa_gamma):
    luminosities, radii, temperatures = [], [], []
    for _, m, v, k, tf in COMP:
        luminosity, _tau = mosfit_diffusion(
            t_rest, m, v, k, kappa_gamma=kappa_gamma, dense=dense
        )
        # RAD_CONST * v[km/s] * t[d]
        r2 = (v * C / KM * KM * DAY * t_rest) ** 2
        rec = luminosity / (4 * np.pi * SIG * tf**4)
        temperature = np.where(
            r2 < rec, (luminosity / (4 * np.pi * SIG * r2)) ** 0.25, tf
        )
        radius = np.sqrt(np.minimum(r2, rec))
        luminosities.append(luminosity)
        radii.append(radius)
        temperatures.append(temperature)
    return sum(luminosities), ab_mags(radii, temperatures)


mf_Lbol, mf = mosfit_run(None)
mfk_Lbol, mfk = mosfit_run(100.0)

# redback engine (wrapped, own 300-pt grid) -> independent SED step
g = get_optimal_time_array(
    1e-2, 7e6, 300, user_times=t_rest * DAY, time_units="seconds"
)
luminosities, radii, temperatures = [], [], []
with corrected_diffusion():
    for _, m, v, k, tf in COMP:
        luminosity, temperature, radius = km._one_component_kilonova_model(
            g, m, v, k, temperature_floor=tf
        )
        luminosities.append(np.interp(t_rest * DAY, g, luminosity))
        temperatures.append(np.interp(t_rest * DAY, g, temperature))
        radii.append(np.interp(t_rest * DAY, g, radius))
rbL = sum(luminosities)
rbsed = ab_mags(radii, temperatures)

ix = [int(np.argmin(abs(t_obs - epoch))) for epoch in T_REP]
print("\nBolometric (dense grids), wrapped redback / MOSFiT:")
print(" t_obs(d)  rb_x2/MOSFiT(no gamma)  dmag   rb_x2/MOSFiT(kappa_gamma=100)  dmag")
for epoch, index in zip(T_REP, ix, strict=True):
    no_gamma = rbL[index] / mf_Lbol[index]
    with_gamma = rbL[index] / mfk_Lbol[index]
    print(
        f"  {epoch:4.1f}     {no_gamma:.4f}   {-2.5 * np.log10(no_gamma):+.3f}        "
        f"{with_gamma:.4f}   {-2.5 * np.log10(with_gamma):+.3f}"
    )
print(
    "\nBand magnitudes (AB, LSST, 40 Mpc, no dust). Columns: "
    "redback public | MOSFiT | MOSFiT kg=100 | redback-engine+own SED"
)
print(
    " t(d)  band  rb_pub   MOSFiT  d(rb-MF)  "
    "MF_kg100 d(rb-MFkg)  rb_ownSED d(rb_pub-ownSED)"
)
for epoch, index in zip(T_REP, ix, strict=True):
    for band in "gri":
        public_minus_mosfit = rb[band][index] - mf[band][index]
        public_minus_gamma = rb[band][index] - mfk[band][index]
        public_minus_sed = rb[band][index] - rbsed[band][index]
        print(
            f" {epoch:4.1f}   {band}   {rb[band][index]:7.3f}  {mf[band][index]:7.3f}  "
            f"{public_minus_mosfit:+.3f}    {mfk[band][index]:7.3f}  "
            f"{public_minus_gamma:+.3f}      {rbsed[band][index]:7.3f}  "
            f"{public_minus_sed:+.3f}"
        )
mask = t_obs >= 0.5
for band in "gri":
    no_gamma = np.max(abs(rb[band] - mf[band])[mask])
    with_gamma = np.max(abs(rb[band] - mfk[band])[mask])
    own_sed = np.max(abs(rb[band] - rbsed[band])[mask])
    print(
        f"max |rb - MOSFiT| over all dense t >= 0.5 d, {band}: "
        f"no-gamma {no_gamma:.3f}  kg=100 {with_gamma:.3f}  "
        f"rb vs own SED {own_sed:.3f}"
    )


def max_abs_differences():
    """Largest |wrapped redback − MOSFiT| from 0.5 days, in magnitudes.

    Bands use every dense time at or after 0.5 days. Bolometric uses the
    report epochs (0.5, 0.8, 1, 1.5, 2, 3, 5, and 7 days), which is the
    comparison whose largest gap is 0.030 mag.
    """
    epoch = [index for index in ix if t_obs[index] >= 0.5 - 1e-9]
    bolometric = [abs(-2.5 * np.log10(rbL[index] / mf_Lbol[index])) for index in epoch]
    return {
        "g": float(np.max(np.abs(rb["g"] - mf["g"])[mask])),
        "r": float(np.max(np.abs(rb["r"] - mf["r"])[mask])),
        "i": float(np.max(np.abs(rb["i"] - mf["i"])[mask])),
        "bolometric": float(np.max(bolometric)),
    }


def write_external_check(path=None):
    """Write docs/mosfit_external_check.csv from this run, not from typed numbers."""
    from pathlib import Path

    destination = bootstrap.ROOT / "docs" / "mosfit_external_check.csv"
    if path is not None:
        destination = Path(path)
    rows = max_abs_differences()
    source = "checks/mosfit/band_sidebyside.py vs MOSFiT commit 6595248"
    lines = ["quantity,max_abs_difference_mag,limit_mag,passes,source"]
    for name in ("g", "r", "i", "bolometric"):
        value = rows[name]
        passed = "yes" if value <= 0.1 else "no"
        lines.append(f"{name},{value:.6f},0.1,{passed},{source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination, rows


if __name__ == "__main__":
    written, rows = write_external_check()
    print(written.relative_to(bootstrap.ROOT).as_posix())
    print(rows)

"""12(19): wrapped redback three_component_kilonova_model vs MOSFiT (6595248 formulas, reimplemented)
in LSST g, r, i (AB, no dust) at 40 Mpc, Planck18, Villar Table 2, dense grids for both.
Also: bolometric on dense grids, and redback L/T/R pushed through an independent blackbody+bandpass
integration (separates the light-curve engine from the SED/photometry step)."""
import warnings
warnings.filterwarnings("ignore")
import bootstrap
bootstrap.install()
import numpy as np, sncosmo, astropy.units as u
from astropy.cosmology import Planck18, z_at_value
from sidebyside_lib import COMP, DAY, C, KM, mosfit_diffusion
import redback.transient_models.kilonova_models as km
from redback.utils import get_optimal_time_array
from kilonova_hunter.redback_fix import corrected_diffusion

z = float(z_at_value(Planck18.luminosity_distance, 40 * u.Mpc)); dl = (40 * u.Mpc).to(u.cm).value
print(f"z = {z:.6f} (Planck18, d_L = 40 Mpc)")
H, KB, SIG = 6.62607015e-27, 1.380649e-16, 5.670374e-5
T_REP = np.array([0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0])
t_obs = np.unique(np.concatenate([np.geomspace(0.3, 10, 300), T_REP, [1.5, 2.0, 3.0]]))
t_rest = t_obs / (1 + z)
args = []
for _, m, v, k, tf in COMP: args += [m, v, tf, k]

# --- redback, public function, wrapped, as villar.py calls it (no dust) ---
rb = {}
with corrected_diffusion():
    for b in "gri":
        rb[b] = np.asarray(km.three_component_kilonova_model(t_obs, z, *args, bands=np.array(["lsst" + b] * len(t_obs)),
                                                              output_format="magnitude", cosmology=Planck18), float)

# --- photometry helper: sum of blackbodies -> AB mag (photon counting) ---
lam_obs = np.linspace(2500, 12500, 4000)  # Angstrom
def ab_mags(Rs, Ts):
    """Rs, Ts: lists over components of arrays over time (rest-frame photospheres)."""
    lam_rest_cm = lam_obs[None, :] / (1 + z) * 1e-8
    Llam = 0.0
    for R, T in zip(Rs, Ts):
        R = np.asarray(R)[:, None]; T = np.asarray(T)[:, None]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            B = 2 * H * C**2 / lam_rest_cm**5 / np.expm1(H * C / (lam_rest_cm * KB * T))
        Llam = Llam + np.nan_to_num(4 * np.pi**2 * R**2 * B)  # erg/s/cm (rest)
    Flam = Llam / (4 * np.pi * dl**2 * (1 + z)) * 1e-8  # erg/s/cm2/A observed
    out = {}
    for b in "gri":
        bp = sncosmo.get_bandpass("lsst" + b); tr = bp(lam_obs)
        num = np.trapezoid(Flam * tr * lam_obs, lam_obs, axis=1)
        den = np.trapezoid(3631e-23 * 2.99792458e18 / lam_obs**2 * tr * lam_obs, lam_obs)
        out[b] = -2.5 * np.log10(num / den)
    return out

# --- MOSFiT: diffusion on MOSFiT's dense grid, temperature_floor photosphere, blackbody ---
dense = np.unique(np.concatenate(([0.0], np.logspace(-6, np.log10(t_rest.max()), 100), t_rest)))
def mosfit_run(kg):
    Ls, Rs, Ts = [], [], []
    for _, m, v, k, tf in COMP:
        L, _ = mosfit_diffusion(t_rest, m, v, k, kappa_gamma=kg, dense=dense)
        r2 = (v * C / KM * KM * DAY * t_rest) ** 2   # RAD_CONST * v[km/s] * t[d]
        rec = L / (4 * np.pi * SIG * tf**4)
        T = np.where(r2 < rec, (L / (4 * np.pi * SIG * r2)) ** 0.25, tf); R = np.sqrt(np.minimum(r2, rec))
        Ls.append(L); Rs.append(R); Ts.append(T)
    return sum(Ls), ab_mags(Rs, Ts)
mf_Lbol, mf = mosfit_run(None)
mfk_Lbol, mfk = mosfit_run(100.0)

# --- redback engine (wrapped, own 300-pt grid) -> independent SED step ---
g = get_optimal_time_array(1e-2, 7e6, 300, user_times=t_rest * DAY, time_units="seconds")
Ls, Rs, Ts = [], [], []
with corrected_diffusion():
    for _, m, v, k, tf in COMP:
        L, T, R = km._one_component_kilonova_model(g, m, v, k, temperature_floor=tf)
        Ls.append(np.interp(t_rest * DAY, g, L)); Ts.append(np.interp(t_rest * DAY, g, T)); Rs.append(np.interp(t_rest * DAY, g, R))
rbL = sum(Ls); rbsed = ab_mags(Rs, Ts)

ix = [np.argmin(abs(t_obs - t)) for t in T_REP]
print("\nBolometric (dense grids), wrapped redback / MOSFiT:")
print(" t_obs(d)  rb_x2/MOSFiT(no gamma)  dmag   rb_x2/MOSFiT(kappa_gamma=100)  dmag")
for t, i in zip(T_REP, ix):
    r1, r2 = rbL[i] / mf_Lbol[i], rbL[i] / mfk_Lbol[i]
    print(f"  {t:4.1f}     {r1:.4f}   {-2.5*np.log10(r1):+.3f}        {r2:.4f}   {-2.5*np.log10(r2):+.3f}")
print("\nBand magnitudes (AB, LSST, 40 Mpc, no dust). Columns: redback public | MOSFiT | MOSFiT kg=100 | redback-engine+own SED")
print(" t(d)  band  rb_pub   MOSFiT  d(rb-MF)  MF_kg100 d(rb-MFkg)  rb_ownSED d(rb_pub-ownSED)")
worst = {"nokg": 0, "kg": 0}
for t, i in zip(T_REP, ix):
    for b in "gri":
        d1, d2, d3 = rb[b][i] - mf[b][i], rb[b][i] - mfk[b][i], rb[b][i] - rbsed[b][i]
        print(f" {t:4.1f}   {b}   {rb[b][i]:7.3f}  {mf[b][i]:7.3f}  {d1:+.3f}    {mfk[b][i]:7.3f}  {d2:+.3f}      {rbsed[b][i]:7.3f}  {d3:+.3f}")
mask = t_obs >= 0.5
for b in "gri":
    print(f"max |rb - MOSFiT| over all dense t >= 0.5 d, {b}: no-gamma {np.max(abs(rb[b]-mf[b])[mask]):.3f}  kg=100 {np.max(abs(rb[b]-mfk[b])[mask]):.3f}  rb vs own SED {np.max(abs(rb[b]-rbsed[b])[mask]):.3f}")


def max_abs_differences():
    """Largest |wrapped redback − MOSFiT| from 0.5 days, in magnitudes.

    Bands use every dense time at or after 0.5 days. Bolometric uses the
    report epochs (0.5, 0.8, 1, 1.5, 2, 3, 5, and 7 days), which is the
    comparison whose largest gap is 0.030 mag.
    """
    epoch = [index for index in ix if t_obs[index] >= 0.5 - 1e-9]
    bolometric = [
        abs(-2.5 * np.log10(rbL[index] / mf_Lbol[index])) for index in epoch
    ]
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
    print(written)
    print(rows)

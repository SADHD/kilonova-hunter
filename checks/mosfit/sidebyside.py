"""Side-by-side: redback 1.20.0 _one_component_kilonova_model (raw and x2) vs a
direct reimplementation of MOSFiT 6595248 rprocess.py + diffusion.py, Villar 2017
Table 2 parameters. Also constant-Q early-time test and redback grid stats.
"""

import bootstrap

bootstrap.install()

import numpy as np
import redback.transient_models.kilonova_models as km
from redback.utils import get_optimal_time_array
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

DAY = 86400.0
MSUN = 1.9884099e33
C = 2.99792458e10
KM = 1e5
COMP = [
    ("blue", 0.020, 0.266, 0.5, 674.0),
    ("purple", 0.047, 0.152, 3.0, 1308.0),
    ("red", 0.011, 0.137, 10.0, 3745.0),
]
T_REPORT = np.array([0.3, 0.5, 0.8, 1.0, 1.5, 3.0, 5.0, 7.0])

# MOSFiT reimplementation (verbatim formulas from rprocess.py / diffusion.py).
bv = np.array([0.1, 0.2, 0.3, 0.4])
bM = np.array([1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
ba = np.array(
    [
        [2.01, 4.52, 8.16, 16.3],
        [0.81, 1.9, 3.2, 5.0],
        [0.56, 1.31, 2.19, 3.0],
        [0.27, 0.55, 0.95, 2.0],
        [0.20, 0.39, 0.65, 0.9],
    ]
)
bb = np.array(
    [
        [0.28, 0.62, 1.19, 2.4],
        [0.19, 0.28, 0.45, 0.65],
        [0.17, 0.21, 0.31, 0.45],
        [0.10, 0.13, 0.15, 0.17],
        [0.06, 0.11, 0.12, 0.12],
    ]
)
bd = np.array(
    [
        [1.12, 1.39, 1.52, 1.65],
        [0.86, 1.21, 1.39, 1.5],
        [0.74, 1.13, 1.32, 1.4],
        [0.6, 0.9, 1.13, 1.25],
        [0.63, 0.79, 1.04, 1.5],
    ]
)
fa, fb, fd = (
    RegularGridInterpolator((bM, bv), grid, bounds_error=False, fill_value=None)
    for grid in (ba, bb, bd)
)


def mosfit_heating(t_day, m, v):
    a = fa([m, v])[0]
    bx2 = 2 * fb([m, v])[0]
    d = fd([m, v])[0]
    thermal = (0.5 - np.arctan((t_day * DAY - 1.3) / 0.11) / np.pi) ** 1.3
    tail = np.exp(-a * t_day) + np.log1p(bx2 * t_day**d) / (bx2 * t_day**d)
    return m * MSUN * 4e18 * 0.36 * thermal * tail


def mosfit_diffusion(times, m, v, kappa, kappa_gamma=None, dense=None):
    v_kms = v * C / KM
    tau = np.sqrt(2.0 * MSUN / (13.7 * C * KM) * kappa * m / v_kms) / DAY
    if dense is None:
        dense = np.unique(np.concatenate(([0.0], np.logspace(-4, np.log10(100), 3000))))
    dl = np.nan_to_num(mosfit_heating(np.maximum(dense, 1e-12), m, v))
    num = 500
    lsp = np.logspace(np.log10(tau / dense[-1]) - 3, 0, num)
    xm = np.unique(np.concatenate((lsp, 1 - lsp)))
    it = np.clip(0 + times.reshape(-1, 1) * xm, 0, dense[-1])
    te2 = it[:, -1] ** 2
    il = np.interp(it, dense, dl)
    arg = il * it * np.exp((it**2 - te2.reshape(-1, 1)) / tau**2)
    luminosity = np.trapezoid(arg, it) * 2.0 / tau**2
    if kappa_gamma is not None:
        trapped = 3.0 * MSUN / (4 * np.pi * KM**2) * kappa_gamma * m / v_kms**2 / DAY**2
        luminosity *= -np.expm1(-trapped / te2)
    return luminosity, tau


def redback_heating(t_s, m, v):
    av, bvv, dv = km.interpolated_barnes_and_kasen_thermalisation_efficiency(m, v)
    td = t_s / DAY
    efficiency = 0.36 * (
        np.exp(-av * td) + np.log1p(2 * bvv * td**dv) / (2 * bvv * td**dv)
    )
    thermal = (0.5 - np.arctan((t_s - 1.3) / 0.11) / np.pi) ** 1.3
    return 4e18 * m * MSUN * thermal * efficiency


def arnett_quad(t_day, m, v, kappa):
    td = np.sqrt(2 * kappa * m * MSUN / (13.7 * v * C * C))
    t = t_day * DAY

    def integrand(tp):
        return (
            redback_heating(tp, m, v) * 2 * tp / td**2 * np.exp((tp**2 - t**2) / td**2)
        )

    pts = [1e-2, 1.0, 10.0, 1e3, 1e4]
    value = quad(integrand, 1e-6, t, points=[p for p in pts if p < t], limit=500)[0]
    return value


# redback as three_component_kilonova_model calls it. z ~ 0.009 ignored:
# redshift only shifts by (1+z).
user_t = T_REPORT * DAY
grid = get_optimal_time_array(1e-2, 7e6, 300, user_times=user_t, time_units="seconds")
orig = km._kilonova_diffusion_luminosity
print(
    "redback grid: "
    f"n={len(grid)}, first={grid[0]:.3g} s, "
    f"n<1d={(grid < DAY).sum()}, n<0.5d={(grid < 0.5 * DAY).sum()}, "
    f"n<0.3d={(grid < 0.3 * DAY).sum()}"
)
between = grid[(grid > 0.05 * DAY) & (grid < DAY)] / DAY
print("grid points (d) between 0.05 and 1 d:", np.round(between, 4).tolist())

tot = {
    key: np.zeros(len(T_REPORT))
    for key in ("rb_raw", "rb_x2", "mosfit", "mosfit_kg", "quad")
}
for name, m, v, k, tf in COMP:
    luminosity, _temperature, _radius = km._one_component_kilonova_model(
        grid, m, v, k, temperature_floor=tf
    )
    raw = np.interp(user_t, grid, luminosity)
    mosfit, tau = mosfit_diffusion(T_REPORT, m, v, k)
    mosfit_kg, _tau_kg = mosfit_diffusion(T_REPORT, m, v, k, kappa_gamma=10.0)
    quad_l = np.array([arnett_quad(t, m, v, k) for t in T_REPORT])
    tot["rb_raw"] += raw
    tot["rb_x2"] += 2 * raw
    tot["mosfit"] += mosfit
    tot["mosfit_kg"] += mosfit_kg
    tot["quad"] += quad_l
    print(f"\n{name}: t_d = {tau:.3f} d")
    for i, t in enumerate(T_REPORT):
        print(
            f"  t={t:4.1f} d  rb_x2={2 * raw[i]:.3e}  mosfit={mosfit[i]:.3e}  "
            f"quad={quad_l[i]:.3e}  rb_x2/mosfit={2 * raw[i] / mosfit[i]:.4f}"
        )

# Waxman, Ofek, Kushnir, and Gal-Yam 2018, MNRAS 481, 3423, Table B2, L_int.
# The source note is the comment header of the CSV. The Waxman TeX is not in the repo.
_days, _lums = bootstrap.load_waxman_l_int()
_keep = np.array([0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
_index = {day: i for i, day in enumerate(_days)}
wax_t = np.array([day for day in _keep])
wax_L = np.array([_lums[_index[float(day)]] for day in _keep])
print(
    "\nTOTAL (erg/s)  t  rb_raw  rb_x2  mosfit  mosfit(kg=10)  quad  "
    "| rb_x2/mosfit  dmag"
)
for i, t in enumerate(T_REPORT):
    ratio = tot["rb_x2"][i] / tot["mosfit"][i]
    print(
        f"  {t:4.1f}  {tot['rb_raw'][i]:.3e}  {tot['rb_x2'][i]:.3e}  "
        f"{tot['mosfit'][i]:.3e}  {tot['mosfit_kg'][i]:.3e}  {tot['quad'][i]:.3e}  "
        f"| {ratio:.4f}  {-2.5 * np.log10(ratio):+.3f}"
    )

# Waxman ratios reproduced with raw vs x2, to identify what the cloud agent compared.
time_grid = get_optimal_time_array(
    1e-2, 7e6, 300, user_times=wax_t * DAY, time_units="seconds"
)
summed = sum(
    km._one_component_kilonova_model(time_grid, m, v, k, temperature_floor=tf)[0]
    for _, m, v, k, tf in COMP
)
interpolated = np.interp(wax_t * DAY, time_grid, summed)
mosfit_wax = sum(mosfit_diffusion(wax_t, m, v, k)[0] for _, m, v, k, tf in COMP)
print(
    "\nWaxman L_int ratios: t  raw/W  x2/W  mosfit/W   "
    "(PR table: 0.082 0.072 0.207 0.337 0.474 0.657 0.839 0.948 "
    "1.116 1.269 1.461 1.693)"
)
for i, t in enumerate(wax_t):
    print(
        f"  {t:4.1f}  {interpolated[i] / wax_L[i]:.3f}  "
        f"{2 * interpolated[i] / wax_L[i]:.3f}  {mosfit_wax[i] / wax_L[i]:.3f}"
    )

# constant-Q test on the grid redback actually uses for the blue component
print("\nconstant-Q, redback integrator x2, blue t_d, on the redback grid:")
for _name, m, v, k, _tf in COMP[:1]:
    td = np.sqrt(2 * k * m * MSUN / (13.7 * v * C * C))
    samples = np.array([0.1, 0.25, 0.5, 1.0, 2.0]) * td
    fine = get_optimal_time_array(
        1e-2, 7e6, 300, user_times=samples, time_units="seconds"
    )
    corrected = 2 * orig(fine, np.ones_like(fine), td)
    for fraction, sample in zip([0.1, 0.25, 0.5, 1.0, 2.0], samples, strict=True):
        analytic = 1 - np.exp(-(fraction**2))
        got = np.interp(sample, fine, corrected)
        print(
            f"  t/t_d={fraction:4.2f} (t={sample / DAY:.3f} d)  "
            f"got={got:.5f}  analytic={analytic:.5f}  ratio={got / analytic:.4f}"
        )

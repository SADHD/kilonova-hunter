"""Side-by-side: redback 1.20.0 _one_component_kilonova_model (raw and x2) vs a
direct reimplementation of MOSFiT 6595248 rprocess.py + diffusion.py, Villar 2017
Table 2 parameters. Also constant-Q early-time test and redback grid stats."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad
import redback.transient_models.kilonova_models as km
from redback.utils import get_optimal_time_array

DAY = 86400.0; MSUN = 1.9884099e33; C = 2.99792458e10; KM = 1e5
COMP = [("blue", 0.020, 0.266, 0.5, 674.0), ("purple", 0.047, 0.152, 3.0, 1308.0),
        ("red", 0.011, 0.137, 10.0, 3745.0)]
T_REPORT = np.array([0.3, 0.5, 0.8, 1.0, 1.5, 3.0, 5.0, 7.0])

# ---- MOSFiT reimplementation (verbatim formulas from rprocess.py / diffusion.py) ----
bv = np.array([0.1, 0.2, 0.3, 0.4]); bM = np.array([1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
ba = np.array([[2.01,4.52,8.16,16.3],[0.81,1.9,3.2,5.0],[0.56,1.31,2.19,3.0],[.27,.55,.95,2.0],[0.20,0.39,0.65,0.9]])
bb = np.array([[0.28,0.62,1.19,2.4],[0.19,0.28,0.45,0.65],[0.17,0.21,0.31,0.45],[0.10,0.13,0.15,0.17],[0.06,0.11,0.12,0.12]])
bd = np.array([[1.12,1.39,1.52,1.65],[0.86,1.21,1.39,1.5],[0.74,1.13,1.32,1.4],[0.6,0.9,1.13,1.25],[0.63,0.79,1.04,1.5]])
fa, fb, fd = (RegularGridInterpolator((bM, bv), x, bounds_error=False, fill_value=None) for x in (ba, bb, bd))

def mosfit_heating(t_day, m, v):
    a = fa([m, v])[0]; bx2 = 2 * fb([m, v])[0]; d = fd([m, v])[0]
    return m * MSUN * 4e18 * 0.36 * (0.5 - np.arctan((t_day * DAY - 1.3) / 0.11) / np.pi) ** 1.3 * (
        np.exp(-a * t_day) + np.log1p(bx2 * t_day ** d) / (bx2 * t_day ** d))

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
    arg = il * it * np.exp((it ** 2 - te2.reshape(-1, 1)) / tau ** 2)
    L = np.trapezoid(arg, it) * 2.0 / tau ** 2
    if kappa_gamma is not None:
        A = 3.0 * MSUN / (4 * np.pi * KM ** 2) * kappa_gamma * m / v_kms ** 2 / DAY ** 2
        L *= -np.expm1(-A / te2)
    return L, tau

# ---- exact reference: adaptive quad of Arnett with redback's heating ----
def redback_heating(t_s, m, v):
    av, bvv, dv = km.interpolated_barnes_and_kasen_thermalisation_efficiency(m, v)
    td = t_s / DAY
    e = 0.36 * (np.exp(-av * td) + np.log1p(2 * bvv * td ** dv) / (2 * bvv * td ** dv))
    return 4e18 * m * MSUN * (0.5 - np.arctan((t_s - 1.3) / 0.11) / np.pi) ** 1.3 * e

def arnett_quad(t_day, m, v, kappa):
    td = np.sqrt(2 * kappa * m * MSUN / (13.7 * v * C * C))
    t = t_day * DAY
    f = lambda tp: redback_heating(tp, m, v) * 2 * tp / td**2 * np.exp((tp**2 - t**2) / td**2)
    pts = [1e-2, 1.0, 10.0, 1e3, 1e4]
    val = quad(f, 1e-6, t, points=[p for p in pts if p < t], limit=500)[0]
    return val


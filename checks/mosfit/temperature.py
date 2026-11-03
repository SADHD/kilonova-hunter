"""Component temperatures / radii (wrapped x2) vs Waxman Table B2 T_eff, R_phot."""
import bootstrap
bootstrap.install()
import numpy as np
from sidebyside_lib import COMP, DAY
from kilonova_hunter.redback_fix import corrected_diffusion
import redback.transient_models.kilonova_models as km
from redback.utils import get_optimal_time_array
wt = np.array([0.5, 0.8, 1.0, 1.5, 2.5, 3.5, 4.5])
wT = np.array([10266, 7093, 6351, 4967, 3751, 3160, 2836]); wR = np.array([3.7e14, 6.0e14, 6.7e14, 8.9e14, 1.2e15, 1.5e15, 1.7e15])
g = get_optimal_time_array(1e-2, 7e6, 300, user_times=wt * DAY)
sig = 5.670374e-5
with corrected_diffusion():
    out = [km._one_component_kilonova_model(g, m, v, k, temperature_floor=tf) for _, m, v, k, tf in COMP]
print(" t    T_blue T_purp T_red | L-weighted T  | Waxman T_eff | R_blue/Waxman R")
for i, t in enumerate(wt):
    Ls, Ts, Rs = [], [], []
    for L, T, R in out:
        Ls.append(np.interp(t*DAY, g, L)); Ts.append(np.interp(t*DAY, g, T)); Rs.append(np.interp(t*DAY, g, R))
    Teff = (sum(Ls) / (4*np.pi*sig*sum(r**2 for r in Rs)))**0.25
    print(f" {t:3.1f}  {Ts[0]:6.0f} {Ts[1]:6.0f} {Ts[2]:6.0f} | {Teff:6.0f} (area-avg) | {wT[i]:6d}       | {Rs[0]/wR[i]:.2f}")

"""MOSFiT's own dense grid: [0] + logspace(1e-6 d, t_max, 100) + observation times
(densetimes.py L_T_MIN=-6, N_TIMES=100), kappa_gamma=100 (kilonova.json default),
texplosion=0. Totals vs redback x2 at the report epochs."""
import bootstrap
bootstrap.install()
import numpy as np
from sidebyside_lib import *
import redback.transient_models.kilonova_models as km
from redback.utils import get_optimal_time_array
T = T_REPORT
obs = np.unique(np.concatenate((T, np.linspace(0.45, 25, 200))))  # stand-in observation epochs
dense = np.unique(np.concatenate(([0.0], np.logspace(-6, np.log10(obs.max()), 100), obs)))
g = get_optimal_time_array(1e-2, 7e6, 300, user_times=T * DAY, time_units="seconds")
tm = np.zeros(len(T)); tr = np.zeros(len(T)); tq = np.zeros(len(T)); tdirect = np.zeros(len(T))
for name, m, v, k, tf in COMP:
    Lm, _ = mosfit_diffusion(T, m, v, k, kappa_gamma=100.0, dense=dense)
    tm += Lm
    tq += np.array([arnett_quad(t, m, v, k) for t in T])
    tr += 2 * np.interp(T * DAY, g, km._one_component_kilonova_model(g, m, v, k, temperature_floor=tf)[0])
# redback called directly through the public model with output_format='spectra' -> integrate spectrum -> bolometric
print(" t(d)   MOSFiT-grid      redback x2      exact quad    rbx2/MOSFiT   dmag")
for i, t in enumerate(T):
    r = tr[i] / tm[i]
    print(f" {t:4.1f}  {tm[i]:.4e}  {tr[i]:.4e}  {tq[i]:.4e}   {r:.4f}   {-2.5*np.log10(r):+.3f}")

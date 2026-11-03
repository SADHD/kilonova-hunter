"""Blue component, late times: redback x2 on its default grid vs a fine grid vs quad."""
import bootstrap
bootstrap.install()
import numpy as np
from sidebyside_lib import COMP, DAY, arnett_quad
import redback.transient_models.kilonova_models as km
from redback.utils import get_optimal_time_array
_, m, v, k, tf = COMP[0]
T = np.array([3.0, 5.0, 7.0, 10.0])
g = get_optimal_time_array(1e-2, 7e6, 300, user_times=T * DAY)
fine = np.unique(np.concatenate((np.geomspace(1e-2, 7e6, 20000), T * DAY)))
q = np.array([arnett_quad(t, m, v, k) for t in T])
for name, grid in (("default 300", g), ("geom 20000", fine)):
    L = 2 * np.interp(T * DAY, grid, km._one_component_kilonova_model(grid, m, v, k, temperature_floor=tf)[0])
    print(name, "ratio to quad:", np.round(L / q, 4), " grid spacing near 7 d (d):", np.round(np.diff(grid[(grid > 6*DAY) & (grid < 8*DAY)]) / DAY, 3)[:5])

"""How much of MOSFiT's extra early luminosity comes from linear interpolation of the
heating rate between t=0 (where the arctan term is ~1) and the first dense time?"""
import bootstrap
bootstrap.install()
import numpy as np
from sidebyside_lib import *
T = np.array([0.3, 0.5, 0.8, 1.0, 1.5, 3.0])
for name, m, v, k, tf in COMP[:1]:
    q = np.array([arnett_quad(t, m, v, k) for t in T])
    print(name, "exact quad:", " ".join(f"{x:.3e}" for x in q))
    for first in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
        for with0 in (True, False):
            d = np.logspace(np.log10(first), 2, 3000)
            if with0: d = np.concatenate(([0.0], d))
            L, _ = mosfit_diffusion(T, m, v, k, dense=d)
            print(f"  first dense={first:g} d, t=0 point={with0}:  ratio to exact", " ".join(f"{x:.3f}" for x in L / q))
# heating itself: are a,b,d the same?
for name, m, v, k, tf in COMP:
    print(name, "redback abd", km.interpolated_barnes_and_kasen_thermalisation_efficiency(m, v),
          "mosfit abd", fa([m, v])[0], fb([m, v])[0], fd([m, v])[0])
    for t in [0.3, 1.0, 5.0]:
        print(f"   Q({t}) redback={redback_heating(t*DAY, m, v):.4e} mosfit={mosfit_heating(t, m, v):.4e}")

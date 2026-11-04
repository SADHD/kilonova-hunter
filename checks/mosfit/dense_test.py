"""How much of MOSFiT's extra early luminosity comes from linear interpolation of the
heating rate between t=0 (where the arctan term is ~1) and the first dense time?
"""

import bootstrap

bootstrap.install()

import numpy as np
from sidebyside_lib import (
    COMP,
    DAY,
    arnett_quad,
    fa,
    fb,
    fd,
    km,
    mosfit_diffusion,
    mosfit_heating,
    redback_heating,
)

T = np.array([0.3, 0.5, 0.8, 1.0, 1.5, 3.0])
for name, m, v, k, _tf in COMP[:1]:
    quad = np.array([arnett_quad(t, m, v, k) for t in T])
    print(name, "exact quad:", " ".join(f"{x:.3e}" for x in quad))
    for first in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
        for with0 in (True, False):
            dense = np.logspace(np.log10(first), 2, 3000)
            if with0:
                dense = np.concatenate(([0.0], dense))
            luminosity, _tau = mosfit_diffusion(T, m, v, k, dense=dense)
            ratios = " ".join(f"{x:.3f}" for x in luminosity / quad)
            print(
                f"  first dense={first:g} d, t=0 point={with0}:  "
                f"ratio to exact {ratios}"
            )
# heating itself: are a,b,d the same?
for name, m, v, _k, _tf in COMP:
    abd = km.interpolated_barnes_and_kasen_thermalisation_efficiency(m, v)
    print(
        name,
        "redback abd",
        abd,
        "mosfit abd",
        fa([m, v])[0],
        fb([m, v])[0],
        fd([m, v])[0],
    )
    for t in [0.3, 1.0, 5.0]:
        print(
            f"   Q({t}) redback={redback_heating(t * DAY, m, v):.4e} "
            f"mosfit={mosfit_heating(t, m, v):.4e}"
        )

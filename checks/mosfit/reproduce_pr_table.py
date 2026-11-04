"""Try to reproduce the PR's model/Waxman column (0.082 at 0.5 d).

The cases below are plausible ways that column could have been computed.
"""

import bootstrap

bootstrap.install()

import numpy as np
import redback.transient_models.kilonova_models as km
from redback.utils import get_optimal_time_array
from sidebyside_lib import COMP, DAY

_days, _lums = bootstrap.load_waxman_l_int()
_keep = [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
_index = {day: i for i, day in enumerate(_days)}
wt = np.array(_keep)
wL = np.array([_lums[_index[day]] for day in _keep])
pr = np.array(
    [0.082, 0.072, 0.207, 0.337, 0.474, 0.657, 0.839, 0.948, 1.116, 1.269, 1.461, 1.693]
)


def tot(grid_s, at_s, factor=2.0):
    luminosity = sum(
        km._one_component_kilonova_model(grid_s, m, v, k, temperature_floor=tf)[0]
        for _, m, v, k, tf in COMP
    )
    return factor * np.interp(at_s, grid_s, luminosity)


cases = {
    "A redback grid (as three_component), x2": tot(
        get_optimal_time_array(1e-2, 7e6, 300, user_times=wt * DAY), wt * DAY
    ),
    "B redback grid, raw (no wrapper)": tot(
        get_optimal_time_array(1e-2, 7e6, 300, user_times=wt * DAY), wt * DAY, 1.0
    ),
    "C grid = only the Waxman epochs, x2": tot(wt * DAY, wt * DAY),
    "D linspace(0.5,7.5 d, 50), x2": tot(np.linspace(0.5, 7.5, 50) * DAY, wt * DAY),
    "E linspace(0.01,30 d, 300), x2": tot(np.linspace(0.01, 30, 300) * DAY, wt * DAY),
    "F geomspace(0.1,30 d,100), x2": tot(np.geomspace(0.1, 30, 100) * DAY, wt * DAY),
    "G geomspace(0.4,30 d,200), x2": tot(np.geomspace(0.4, 30, 200) * DAY, wt * DAY),
}
print("PR column        ", " ".join(f"{x:6.3f}" for x in pr))
for name, luminosity in cases.items():
    ratio = luminosity / wL
    delta = np.max(np.abs(2.5 * np.log10(ratio / pr)))
    print(
        f"{name:40s}",
        " ".join(f"{x:6.3f}" for x in ratio),
        f"  max|dmag vs PR| = {delta:.2f}",
    )

"""Do the harness's public redback calls depend on how sparse the requested times are?
Calls the same public functions the harness calls (villar.py / recall.py), with the
wrapper on, once at sparse WFD-like times and once with those times embedded in a
dense grid.
"""

import inspect
import warnings

import bootstrap
import numpy as np

warnings.filterwarnings("ignore")
bootstrap.install()

from astropy.cosmology import Planck18
from redback.transient_models import kilonova_models as km

from kilonova_hunter.redback_fix import corrected_diffusion

_PUBLIC = (
    "three_component_kilonova_model",
    "two_component_kilonova_model",
    "two_component_bns_ejecta_relation",
)
for name in _PUBLIC:
    source = inspect.getsource(getattr(km, name))
    calls = [
        fragment
        for fragment in (
            "_one_component_kilonova_model",
            "two_component_kilonova_model(",
        )
        if fragment in source
    ]
    print(
        name,
        "uses get_optimal_time_array:",
        "get_optimal_time_array" in source,
        "| calls:",
        calls,
    )

z = 0.0093
sparse = np.array([0.6, 2.9, 3.0, 5.7, 8.4])
dense = np.sort(np.concatenate([sparse, np.geomspace(0.05, 15, 400)]))
idx = np.searchsorted(dense, sparse)
bands_s = np.array(["lsstg", "lsstr", "lsstg", "lsstr", "lsstg"])


def call(fn, times, bands, *args):
    with corrected_diffusion():
        return np.asarray(
            fn(
                times,
                z,
                *args,
                bands=bands,
                output_format="magnitude",
                cosmology=Planck18,
            ),
            float,
        )


cases = {
    "three_component (Villar T2)": (
        km.three_component_kilonova_model,
        (
            0.020,
            0.266,
            674.0,
            0.5,
            0.047,
            0.152,
            1308.0,
            3.0,
            0.011,
            0.137,
            3745.0,
            10.0,
        ),
    ),
    "two_component (Villar 2-comp)": (
        km.two_component_kilonova_model,
        (0.050, 0.149, 1151.0, 3.65, 0.023, 0.256, 3983.0, 0.5),
    ),
    "two_component_bns_ejecta_relation": (
        km.two_component_bns_ejecta_relation,
        (1.35, 1.35, 400.0, 400.0, 2.17, 0.3, 0.10, 10.0, 0.5, 2500.0, 5000.0),
    ),
}
for label, (fn, args) in cases.items():
    sparse_mags = call(fn, sparse, bands_s, *args)
    dense_bands = np.array(["lsstg"] * len(dense))
    dense_bands[idx] = bands_s
    dense_mags = call(fn, dense, dense_bands, *args)[idx]
    print(
        f"{label:38s} sparse - dense (mag) at {sparse.tolist()}: "
        f"{np.round(sparse_mags - dense_mags, 4).tolist()}"
    )

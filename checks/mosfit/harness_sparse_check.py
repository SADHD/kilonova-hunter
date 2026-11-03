"""Do the harness's public redback calls depend on how sparse the requested times are?
Calls the same public functions the harness calls (villar.py / recall.py), with the wrapper on,
once at sparse WFD-like times and once with those times embedded in a dense grid."""
import inspect, warnings
import numpy as np
warnings.filterwarnings("ignore")
import bootstrap
bootstrap.install()
from astropy.cosmology import Planck18
from redback.transient_models import kilonova_models as km
from kilonova_hunter.redback_fix import corrected_diffusion

for name in ("three_component_kilonova_model", "two_component_kilonova_model", "two_component_bns_ejecta_relation"):
    src = inspect.getsource(getattr(km, name))
    print(name, "uses get_optimal_time_array:", "get_optimal_time_array" in src, "| calls:", [f for f in ("_one_component_kilonova_model", "two_component_kilonova_model(") if f in src])

z = 0.0093
sparse = np.array([0.6, 2.9, 3.0, 5.7, 8.4])
dense = np.sort(np.concatenate([sparse, np.geomspace(0.05, 15, 400)]))
idx = np.searchsorted(dense, sparse)
bands_s = np.array(["lsstg", "lsstr", "lsstg", "lsstr", "lsstg"])
def call(fn, t, b, *args):
    with corrected_diffusion():
        return np.asarray(fn(t, z, *args, bands=b, output_format="magnitude", cosmology=Planck18), float)
cases = {
    "three_component (Villar T2)": (km.three_component_kilonova_model, (0.020,0.266,674.,0.5, 0.047,0.152,1308.,3., 0.011,0.137,3745.,10.)),
    "two_component (Villar 2-comp)": (km.two_component_kilonova_model, (0.050,0.149,1151.,3.65, 0.023,0.256,3983.,0.5)),
    "two_component_bns_ejecta_relation": (km.two_component_bns_ejecta_relation, (1.35,1.35,400.,400.,2.17,0.3,0.10,10.,0.5,2500.,5000.)),
}
for label, (fn, args) in cases.items():
    a = call(fn, sparse, bands_s, *args)
    bd = np.array(["lsstg"]*len(dense)); bd[idx] = bands_s
    b = call(fn, dense, bd, *args)[idx]
    print(f"{label:38s} sparse - dense (mag) at {sparse.tolist()}: {np.round(a-b,4).tolist()}")

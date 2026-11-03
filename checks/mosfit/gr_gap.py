"""Model-minus-AT2017gfo g-r at 1.5, 2, 3 d. Model: wrapped redback three_component, Villar T2,
dense grid, LSST bands, 40 Mpc, no dust. Data: docs/at2017gfo_photometry.csv (AB), MW-dereddened
with F99 R_V=3.1 E(B-V)=0.105 at LSST effective wavelengths (same as the PR), each point moved to the
LSST band with the model's own (LSST - instrument-band) term (PR's instrument mapping), then a local
linear fit in time within +-W d of the epoch."""
import warnings
warnings.filterwarnings("ignore")
import bootstrap
bootstrap.install()
import numpy as np, pandas as pd
import astropy.units as u
from astropy.cosmology import Planck18, z_at_value
import redback.transient_models.kilonova_models as km
from kilonova_hunter.redback_fix import corrected_diffusion
from kilonova_hunter.sky import fitzpatrick_extinction_mag
z = float(z_at_value(Planck18.luminosity_distance, 40 * u.Mpc))
COMP = [(0.020, 0.266, 674.0, 0.5), (0.047, 0.152, 1308.0, 3.0), (0.011, 0.137, 3745.0, 10.0)]
args = [x for c in COMP for x in c]
p = pd.read_csv(bootstrap.ROOT / "docs" / "at2017gfo_photometry.csv", comment="#")
p = p[p.system == "AB"].copy()
r = pd.read_csv(bootstrap.ROOT / "docs" / "at2017gfo_bandpass_residuals.csv")
key = lambda df: df.time_days.round(5).astype(str) + df.band
p["inst_band"] = p.assign(k=key(p)).k.map(dict(zip(key(r), r.inst_band)))
p = p.dropna(subset=["inst_band"])
def mag(times, bands):
    with corrected_diffusion():
        return np.asarray(km.three_component_kilonova_model(np.asarray(times, float), z, *args, bands=np.asarray(bands),
                          output_format="magnitude", cosmology=Planck18), float)
p["m_lsst"] = mag(p.time_days, "lsst" + p.band); p["m_inst"] = mag(p.time_days, p.inst_band)
p["dust"] = [fitzpatrick_extinction_mag(0.105, b) for b in p.band]
p["obs_lsst"] = p.magnitude - p.dust + (p.m_lsst - p.m_inst)
p["obs_raw"] = p.magnitude - p.dust
def local(band, t0, col, W):
    s = p[(p.band == band) & (abs(p.time_days - t0) <= W)]
    if len(s) < 2 or s.time_days.nunique() < 2: return np.nan, len(s)
    c = np.polyfit(s.time_days - t0, s[col], 1, w=1 / np.maximum(s.magnitude_error.fillna(0.05), 0.02))
    return c[1], len(s)
print(" t(d)  W   n_g n_r  data g-r (LSST-equiv)  data g-r (instr. bands)  model g-r (LSST)  model-minus-data (LSST)  model g  data g(LSST)")
for W in (0.3, 0.5):
    for t0 in (1.5, 2.0, 3.0):
        g, ng = local("g", t0, "obs_lsst", W); rr, nr = local("r", t0, "obs_lsst", W)
        g0, _ = local("g", t0, "obs_raw", W); r0, _ = local("r", t0, "obs_raw", W)
        mg = float(mag([t0], ["lsstg"])[0])
        mr = float(mag([t0], ["lsstr"])[0])
        print(f" {t0:3.1f}  {W:.1f}  {ng:3d} {nr:3d}   {g-rr:+.3f}                 {g0-r0:+.3f}                   {mg-mr:+.3f}            {(mg-mr)-(g-rr):+.3f}                {mg:.2f}   {g:.2f}")

print("\nRobust version: per-night medians (same-night g and r), g-r interpolated linearly between nights")
p["night"] = np.floor(p.time_days - 0.4).astype(int)
rows = []
for n, s in p.groupby("night"):
    g = s[s.band == "g"]; rr = s[s.band == "r"]
    if len(g) and len(rr):
        rows.append((s.time_days.median(), g.obs_lsst.median() - rr.obs_lsst.median(), g.obs_raw.median() - rr.obs_raw.median(), len(g), len(rr)))
nights = pd.DataFrame(rows, columns=["t", "gr_lsst", "gr_raw", "n_g", "n_r"])
print(nights.round(3).to_string(index=False))
print(" t(d)  data g-r(LSST-equiv)  model g-r(LSST)  model-minus-data")
for t0 in (1.5, 2.0, 3.0):
    dg = np.interp(t0, nights.t, nights.gr_lsst)
    mgr = float(mag([t0], ["lsstg"])[0] - mag([t0], ["lsstr"])[0])
    print(f" {t0:3.1f}   {dg:+.3f}               {mgr:+.3f}           {mgr-dg:+.3f}")

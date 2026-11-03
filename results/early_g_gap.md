# Model comparison, ruling 12(19)

The bar that blocks a grid rerun is the wrapped redback model versus
MOSFiT, within 0.1 mag in g, r, and i from 0.5 days.

**The bar passes.**

From 0.5 days on, the largest absolute difference is 0.028 mag in g,
0.023 mag in r, 0.020 mag in i, and 0.030 mag in bolometric luminosity.
redback is about 2% brighter. That offset comes from its own diffusion
integral. The comparison is against a line-by-line reimplementation of
the modules in MOSFiT commit 6595248. pip mosfit does not build on
Python 3.10 or newer, so that check was run outside this repository.
The recorded numbers are in `docs/mosfit_external_check.csv`, written
by `scripts/record_mosfit_external_check.py`.

## Reported gap to AT2017gfo

This is not a bar. It does not block the rerun.

The model is too red in g−r by +0.37 mag at 1.5 days, +0.47 mag at
2 days, and +0.41 mag at 3 days, about 0.4 ± 0.05 mag. The sensitivity
hook `grid.recall_with_bluer_color` uses that gap. Its default offset
is +0.4 mag, inside 0.3 to 0.47. The tuning grid does not call it.

A positive residual below means the model is fainter than the
dereddened point. The per-epoch residuals are in
`docs/at2017gfo_bandpass_residuals.csv`. They are a report of the
photometry, not a pass/fail cut.

## Check 1: each point's instrument bandpass, both in AB

216 rows are AB. No row arrived as Vega. One row is dropped and flagged:
day 0.451, i = 17.3, blank magnitude system, blank telescope, blank
instrument (source id 41 in the photometry file).

Where sncosmo has the instrument's bandpass, the model is evaluated in
that bandpass. Otherwise the nearest sncosmo bandpass is used and named.

| instrument | sncosmo band | kind |
| --- | --- | --- |
| DECam | desg, desr, desi | exact |
| Pan-STARRS PS1 | ps1::g, ps1::r, ps1::i | exact |
| Swope | cspg, cspr, cspi | exact. CSP g/r/i are in sncosmo |
| SkyMapper | sdssg, sdssr, sdssi | nearest. The SkyMapper bandpasses cannot be downloaded (the SVO source server returns an error; an earlier check saw HTTP 500, and this environment gets HTTP 403) |
| T80S, LCO Sinistro, LDSS, IMACS, GROND, REM ROS2, OmegaCam, AST3-2, VLT FORS/VIMOS/X-shooter/Danish 1.5 m, Gemini, Prompt, Tripol5, Zadko, and the other rows | sdssg, sdssr, sdssi | nearest. sncosmo has no bandpass for that instrument |

CSP minus SDSS is negative in g: the CSP bandpass is brighter. At the
three Swope g epochs the difference is −0.06 to −0.12 mag. From day 0.4
to day 8 the model difference runs from about −0.01 to −0.12 mag. The
per-epoch Swope differences are in `docs/at2017gfo_csp_minus_sdss.csv`.
The script that writes both residual tables is
`scripts/at2017gfo_bandpass_residuals.py`. The bolometric ratios come
from `scripts/at2017gfo_bolometric.py`.

Dust is Fitzpatrick 1999, R_V = 3.1, E(B-V) = 0.105, at that bandpass's
effective wavelength. The LSST comparison uses lsstg, lsstr, and lssti
and the LSST effective wavelengths, which is the figure
`docs/at2017gfo_brightness_check.png`.

### LSST bandpasses (the figure)

Median residual in each window. Reported, not scored against a bar.

| window (days) | g | r | i |
| --- | --- | --- | --- |
| 0.4–0.8 | +0.65 (n=4) | +0.41 (n=5) | +0.33 (n=6) |
| 0.8–1.5 | +0.20 | −0.06 | −0.29 |
| 1.5–3.0 | +0.46 | +0.17 | +0.10 |
| 3.0–6.0 | +0.26 | +0.00 | +0.10 |
| 6.0 and later | +0.304 | +0.347 | −0.024 |

### Instrument bandpasses

| window (days) | g | r | i |
| --- | --- | --- | --- |
| 0.4–0.8 | +0.68 (n=4) | +0.41 (n=5) | +0.33 (n=6) |
| 0.8–1.5 | +0.304 | −0.046 | −0.286 |
| 1.5–3.0 | +0.580 | +0.191 | +0.098 |
| 3.0–6.0 | +0.333 | −0.011 | +0.060 |
| 6.0 and later | +0.303 | +0.346 | +0.000 |

## Check 2, code reading: three blackbodies, not one photosphere

redback 1.20.0 `three_component_kilonova_model` loops over the three
components (`kilonova_models.py` around lines 1186–1200). Each component
calls `_one_component_kilonova_model`, which returns that component's
own temperature and photospheric radius. Each of those is turned into
its own blackbody spectrum with `blackbody_to_spectrum`. The spectra are
added (`full_spec += spectra`). There is no single combined photosphere.

## Check 3, code reading: exact copies use all three Villar components

`villar.exact_components` returns blue, purple, and red at the Table 2
centers. `villar_magnitudes` calls `three_component_kilonova_model` with
those three components in that order and passes the floors 674 K,
1308 K, and 3745 K as positional arguments (`villar.py`,
`exact_components` and the call in `villar_magnitudes`). The same path
is what the exact-copy simulation uses.

## Check 4: bolometric luminosity against Waxman et al. 2018

The published curve is Waxman, Ofek, Kushnir, and Gal-Yam 2018,
MNRAS 481, 3423, Table B2, column L_int (the trapezoidal bolometric
luminosity they adopt). The numbers are in
`docs/at2017gfo_bolometric_waxman2018.csv`.

An earlier draft called the private `_one_component_kilonova_model` at
the 12 Waxman times only. That starts the diffusion integral at the
first data time and makes the model look about 2.7 mag too faint at
0.5 days. That draft was wrong. The numbers below evaluate the three
Table 2 components on redback's dense grid (`get_optimal_time_array`
from 0.01 seconds to 7e6 seconds, 300 requested points, the Waxman
times included), with the factor-of-2 wrapper, convert source time to
observer time by (1+z), and interpolate to the data times. The grid
that comes back has 298 points and starts at 0.01 seconds. 104 of those
points are before 1 day. The ratios are in
`docs/at2017gfo_bolometric_model.csv`. The script is
`scripts/at2017gfo_bolometric.py`.

A ratio below 1 means the model is fainter than the Waxman integral.
The magnitude column is `-2.5 log10(ratio)`. Positive means the model
is fainter than that integral. This is not the MOSFiT bar above.

| day | model / Waxman L_int | model minus Waxman (mag) |
| --- | --- | --- |
| 0.5 | 0.692 | +0.40 |
| 0.6 | 0.676 | +0.43 |
| 0.8 | 0.810 | +0.23 |
| 1.0 | 0.934 | +0.07 |
| 1.2 | 1.070 | −0.07 |
| 1.5 | 1.213 | −0.21 |
| 2.5 | 1.010 | −0.01 |
| 3.5 | 1.045 | −0.05 |
| 4.5 | 1.184 | −0.18 |
| 5.5 | 1.296 | −0.28 |
| 6.5 | 1.431 | −0.39 |
| 7.5 | 1.584 | −0.50 |

At 0.5, 0.8, 1.5, and 7.5 days the ratio is 0.69, 0.81, 1.21, and
1.58. Against Waxman, the total light is about right: tenths of a
magnitude, not missing heating or missing mass. The g−r gap reported
above is a temperature and color difference relative to AT2017gfo.

The harness is unaffected. `src/` calls the public model functions
(`three_component_kilonova_model`, `two_component_kilonova_model`, and
`two_component_bns_ejecta_relation`). Those functions build this dense
grid themselves. The largest difference between asking them for the
visit times and interpolating a separate dense magnitude grid is
0.0096 mag.

## (i) Heating

redback 1.20.0 `_one_component_kilonova_model` and MOSFiT
`mosfit/modules/engines/rprocess.py` (commit 6595248) use the same
heating expression:
`4.0e18 * mass * (0.5 - arctan((t_seconds - 1.3) / 0.11) / pi)^1.3`
and Barnes & Kasen thermalization with the factor 0.36:
`exp(-a t) + log1p(2 b t^d) / (2 b t^d)`. redback multiplies those as
`lum_in * e_th`. MOSFiT multiplies them in one expression
(`self._lscale = mass * 4.0e18 * 0.36`, then the arctan and the
thermalization factor). The numerical side-by-side is the external
check at the top of this note: bolometric agreement within 0.030 mag,
with redback about 2% brighter from its own diffusion integral.

## (ii) Time zero

The photometry clock and the model both use the GW trigger,
MJD 57982.528524. The catalog merger time 57982.5 is 0.0285 days
earlier. Model g is still brightening there, about −0.51 mag/day, so
that shift is about 0.015 mag.

## (iii) Dust and distance

The catalog magnitudes are not corrected for Milky Way dust. The
residuals above remove Fitzpatrick extinction first. What remains is
the reported gap to AT2017gfo. The model is at 40 Mpc. The catalog
luminosity distance of 43.2 Mpc would make the model 0.17 mag fainter
(`5 log10(43.2/40)`). NGC 4993 at 39.5 Mpc is 0.03 mag from 40.

## (iv) Blue temperature before day 1

With the wrapper, on the same dense grid, the blue component
(0.020 solar masses, 0.266c, opacity 0.5, floor 674 K) is 10594 K at
0.3 source-frame days, 8648 K at 0.5 days, and 6932 K at 0.8 days.
The floor is not binding.

## Opacity prior, not run on the grid

`scripts/opacity_prior_sensitivity.py` redraws the realistic opacities
from redback's Uniform(1, 30) prior on one batch (index 90, 24 fakes).
It is not the full grid, and those numbers stay out of the recall.

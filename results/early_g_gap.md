# Brightness bar, Decision Log 12(15)(b). Not changed by 12(17).

From day 0.8 on, g, r, and i must each be within 0.3 mag of dereddened
AT2017gfo in every window. Before day 0.8 the difference is a model
limit and is not scored against the bar. The bar is not loosened.
The cuts and the model are not changed to close the gap.

**The bar is NOT met.**

A positive residual means the model is fainter than the dereddened
point. The per-epoch residuals are in `docs/at2017gfo_bandpass_residuals.csv`.

## Known model limit, 12(15)(a)

The light curve is radioactive heating only. It has no cocoon shock
cooling (Piro & Kollmeier 2018). The color rule is measured at the
r-band peak plus 1 day, which is past day 0.8, so the early gap does
not affect that rule in either direction. Its only effect is that the
fakes are too faint before day 0.8, which biases day-3 recall low.

## Check 1, done first: each point's instrument bandpass, both in AB

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
The scripts that write both residual tables are
`scripts/at2017gfo_bandpass_residuals.py`. The bolometric ratios come
from `scripts/at2017gfo_bolometric.py`.

Dust is Fitzpatrick 1999, R_V = 3.1, E(B-V) = 0.105, at that bandpass's
effective wavelength. The LSST comparison uses lsstg, lsstr, and lssti
and the LSST effective wavelengths, which is the figure
`docs/at2017gfo_brightness_check.png`.

### LSST bandpasses (the figure)

| window (days) | g | r | i | scored? |
| --- | --- | --- | --- | --- |
| 0.4–0.8 | +0.65 (n=4) | +0.41 (n=5) | +0.33 (n=6) | model limit, not scored |
| 0.8–1.5 | +0.20 PASS | −0.06 PASS | −0.29 PASS | scored |
| 1.5–3.0 | +0.46 FAIL | +0.17 PASS | +0.10 PASS | scored |
| 3.0–6.0 | +0.26 PASS | +0.00 PASS | +0.10 PASS | scored |
| 6.0 and later | +0.304 FAIL | +0.347 FAIL | −0.024 PASS | scored |

g fails at 1.5–3.0 days (+0.46) and again after day 6 (+0.30). r fails
after day 6 (+0.35).

### Instrument bandpasses

| window (days) | g | r | i | scored? |
| --- | --- | --- | --- | --- |
| 0.4–0.8 | +0.68 (n=4) | +0.41 (n=5) | +0.33 (n=6) | model limit, not scored |
| 0.8–1.5 | +0.304 FAIL | −0.046 PASS | −0.286 PASS | scored |
| 1.5–3.0 | +0.580 FAIL | +0.191 PASS | +0.098 PASS | scored |
| 3.0–6.0 | +0.333 FAIL | −0.011 PASS | +0.060 PASS | scored |
| 6.0 and later | +0.303 FAIL | +0.346 FAIL | +0.000 PASS | scored |

The instrument bandpasses do not bring g within 0.3 mag after day 0.8.
g is outside the bar in every scored window. This check did not close
the gap, so the bolometric comparison below was run.

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

## Check 4: bolometric luminosity, because check 1 did not close the gap

The published curve is Waxman, Ofek, Kushnir, and Gal-Yam 2018,
MNRAS 481, 3423, Table B2, column L_int (the trapezoidal bolometric
luminosity they adopt). The numbers are in
`docs/at2017gfo_bolometric_waxman2018.csv`. MOSFiT was not run.

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

A ratio below 1 means the model is fainter. The magnitude column is
`-2.5 log10(ratio)`. Positive means the model is fainter.

| day | model / Waxman L_int | model minus data (mag) |
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

The total light is about right. At 0.5, 0.8, 1.5, and 7.5 days the
ratio is 0.69, 0.81, 1.21, and 1.58. That is tenths of a magnitude,
not missing heating or missing mass. The g shortfall is a
temperature and color problem. The g bar is still not met.

The harness is unaffected. `src/` calls the public model functions
(`three_component_kilonova_model`, `two_component_kilonova_model`, and
`two_component_bns_ejecta_relation`). Those functions build this dense
grid themselves. The largest difference between asking them for the
visit times and interpolating a separate dense magnitude grid is
0.0096 mag.

## (i) Heating, by reading the code, not by running both models

redback 1.20.0 `_one_component_kilonova_model` was read in this
environment. MOSFiT `mosfit/modules/engines/rprocess.py` was read from
the guillochon/MOSFiT master source (lines 70–76). MOSFiT was not
executed.

Both use `4.0e18 * mass * (0.5 - arctan((t_seconds - 1.3) / 0.11) / pi)^1.3`
and Barnes & Kasen thermalization with the factor 0.36:
`exp(-a t) + log1p(2 b t^d) / (2 b t^d)`. redback multiplies those as
`lum_in * e_th`. MOSFiT multiplies them in one expression
(`self._lscale = mass * 4.0e18 * 0.36`, then the arctan and the
thermalization factor). The efficiency changes with time. The formulas
match. This comparison explains about 0 mag of the g gap. It is a
code comparison, not a side-by-side numerical MOSFiT run.

## (ii) Time zero

The photometry clock and the model both use the GW trigger,
MJD 57982.528524. The catalog merger time 57982.5 is 0.0285 days
earlier. Model g is still brightening there, about −0.51 mag/day, so
that shift is about 0.015 mag.

## (iii) Dust and distance

The catalog magnitudes are not corrected for Milky Way dust. The
residuals above remove Fitzpatrick extinction first. The gap that fails
the bar is what remains after that correction. The model is at 40 Mpc.
The catalog luminosity distance of 43.2 Mpc would make the model
0.17 mag fainter (`5 log10(43.2/40)`), which would widen the gap.
NGC 4993 at 39.5 Mpc is 0.03 mag from 40.

## (iv) Blue temperature before day 1

With the wrapper, on the same dense grid, the blue component
(0.020 solar masses, 0.266c, opacity 0.5, floor 674 K) is 10594 K at
0.3 source-frame days, 8648 K at 0.5 days, and 6932 K at 0.8 days.
The floor is not binding.

## Color-shift hook, not run on the grid

`grid.recall_with_bluer_color` shifts the 2017-like g−r bluer by a
configurable offset and recomputes that recall. The default offset is
+0.4 mag. The measured model-minus-AT2017gfo g−r gap in this window is
+0.37 at 1.5 days, +0.47 at 2 days, and +0.41 at 3 days, so the offset
sits in 0.3 to 0.47. The tuning grid does not call the hook.

A separate NOT-COUNTED runner,
`scripts/opacity_prior_sensitivity.py`, redraws the realistic
opacities from redback's Uniform(1, 30) prior on one batch (index 90,
24 fakes). It is not the full grid, and those numbers stay out of the
recall.

## What Science Lead has to rule

The bar is not met. g is 0.46 mag faint at 1.5–3.0 days on the LSST
bandpasses, and the instrument bandpasses make that window 0.58 mag.
The bolometric luminosity is about right, so the g shortfall is a
temperature and color problem, not missing heating or mass. Nothing
here changes the bar, the cuts, or the model.

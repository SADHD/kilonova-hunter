# Early g gap after the diffusion fix

The figure `docs/at2017gfo_brightness_check.png` was regenerated with
the diffusion wrapper. The model is the Villar Table 2 center at 40 Mpc
with no dust. The points are Open Kilonova Catalog g, r, and i, with
Fitzpatrick 1999 extinction removed at E(B-V) = 0.105 and R_V = 3.1.
A positive number means the model is fainter than the corrected point.
Each cell is the median over the points in that window, in magnitudes.

| window (days) | g | r | i | n (g, r, i) |
| --- | --- | --- | --- | --- |
| 0.4–0.8 | +0.65 | +0.41 | +0.34 | 4, 5, 7 |
| 0.8–1.5 | +0.20 | −0.06 | −0.29 | 25, 29, 24 |
| 1.5–3.0 | +0.46 | +0.17 | +0.10 | 11, 23, 21 |
| 3.0–6.0 | +0.26 | +0.00 | +0.10 | 11, 25, 13 |

Early g is still 0.65 mag faint. The ~0.3 mag target is not met there.
Later r is within about 0.2 mag, and r after day 3 is about 0.

Nothing below changes the model. Each line is how much of that 0.65 mag
the check can account for.

## (i) Heating rate and thermalization

redback 1.20.0 `_one_component_kilonova_model` uses

- heating `4.0e18 * M * (0.5 - arctan((t - 1.3) / 0.11) / pi)^1.3`,
  with t, 1.3, and 0.11 in seconds
- thermalization `0.36 * (exp(-a t_days) + log1p(2 b t^d) / (2 b t^d))`,
  with a, b, d from the Barnes and Kasen 2016 grid

MOSFiT `rprocess.py` uses the same heating formula, the same 0.36, and
the same Barnes and Kasen grid. The efficiency changes with time. The
difference explains about 0 mag of the 0.65.

## (ii) Time zero

The photometry clock and the model both use the GW trigger,
MJD 57982.528524. The catalog also lists the merger at MJD 57982.5,
which is 0.0285 days earlier. GRB 170817A is 1.7 seconds after the
merger. Over 0.4–0.8 days the fixed model g changes by about −0.51
mag per day (still getting brighter). A 0.0285-day shift is about
0.015 mag. That is about 0.01 mag of the 0.65, not the gap itself.
This note does not quote a separate MOSFiT fitted explosion time; the
Villar text that was available ties the light curve to the merger.

## (iii) Dust and distance

The catalog magnitudes are not corrected for Milky Way dust. The figure
removes Fitzpatrick extinction before the comparison: A_g = 0.38 mag,
A_r = 0.27 mag, A_i = 0.20 mag at E(B-V) = 0.105. The 0.65 mag gap is
what remains after that correction, so a missed dust correction is not
sitting inside it.

The model is at 40 Mpc. The catalog luminosity distance is 43.2 Mpc,
which would make the model 0.17 mag fainter (`5 log10(43.2/40)`).
Putting the model at 40 instead of 43 makes the faint gap smaller, not
larger. NGC 4993 at 39.5 Mpc is 0.03 mag from 40. Distance contributes
about 0 mag to the model being too faint, and the 43.2 Mpc choice would
add about 0.17 mag to the gap rather than remove it.

## (iv) Blue-component temperature before day 1

With the diffusion wrapper, the blue component (0.020 solar masses,
0.266c, opacity 0.5, floor 674 K) has a temperature of about 7000 K at
0.3 days, 6200 K at 0.5 days, and 5700 K at 0.8 days. The 674 K floor
is not binding. It explains about 0 mag of the 0.65.

## What is left

(i) through (iv) add up to about 0.01 mag. The 0.65 mag early-g gap is
still there after the diffusion fix and after these four checks.

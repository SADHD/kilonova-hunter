# Tuning grid

Label for the alert slices, when they exist: ZTF proxy for Rubin.
This file scores simulated fakes. It does not apply a flags-per-night limit.

Recall among judgeable fakes describes the filter. The catch rate is caught divided by every fake. That is the survey-level number.

Dust: SFD E(B-V) times 0.86, then Fitzpatrick 1999 at R_V = 3.1. Redshift: astropy Planck18. See sky.py.

Contamination limit NOT applied. No cuts were picked from this grid, and the test set was not scored.

## Sample

- 2017-like tuning: 20000 simulated, 144 judgeable at day 3, 546 at day 5, 740 at day 7, status underpowered.
  Underpowered: the cap was hit below the target and at or above the floor.
  Power at the best day-3 recall (10.4% of 144): the chance the 95% Wilson lower bound reaches 70% is about 0.
- realistic tuning: 20000 simulated, 240 judgeable at day 3, 836 at day 5, 1027 at day 7, status underpowered.
  Underpowered: the cap was hit below the target and at or above the floor.
  Power at the best day-3 recall (9.2% of 240): the chance the 95% Wilson lower bound reaches 70% is about 0.
- Test splits were simulated and cached, not scored: 2017-like test 20000 simulated, 156 judgeable at day 3; realistic test 20000 simulated, 222 judgeable at day 3. Both are underpowered.

## Best day-3 recall on judgeable 2017-like tuning fakes

This is the highest recall in the grid. It is not a picked cut.

g-r cut 0.0, rise 0.1, window 2.0 days, gap 0.5 days: recall 10.4% (95% Wilson 6.4% to 16.5%); catch rate 0.1% (95% Wilson 0.0% to 0.1%); judgeable fraction 0.7%; reddening skipped on 14 flagged fakes.

Caught but not judgeable, at that combination and day 3: 0.

## Exact AT2017gfo copies

These copies are not in the recall or the catch rate. The fraction below uses the placeholder cuts in config.py, not a cut picked from the grid.

SIGN-OFF BLOCKER. 1 of 5 judgeable copies caught (20.0% of judgeable copies caught), out of 100 copies.

Missed judgeable copies:
- at2017gfo-exact-005: merger MJD 60000.12646, ra 218.663, dec -24.709, E(B-V) 0.0795. color: Not enough data: g minus r was needed at day 3.70, which is outside the measurements we have for g (from day 0.46 to day 2.68) and r (only at day 2.70).
- at2017gfo-exact-010: merger MJD 60000.96892, ra 342.155, dec -22.473, E(B-V) 0.0228. color: Not enough data: g minus r was needed at day 1.40, which is outside the measurements we have for g (only at day 0.43).
- at2017gfo-exact-012: merger MJD 60000.91444, ra 265.852, dec -67.483, E(B-V) 0.0633. color: Not enough data: g minus r was needed at day 1.19, which is outside the measurements we have for r (only at day 0.19).
- at2017gfo-exact-048: merger MJD 60000.85722, ra 325.848, dec -33.609, E(B-V) 0.0287. color: Not enough data: g minus r was needed at day 1.20, which is outside the measurements we have for g (only at day 2.98).

## Distance bins

Bins are luminosity distance: 10-100, 100-200, and 200 Mpc and farther (the draw stops at 300 Mpc comoving, so the last bin is the 200-300 Mpc bin).

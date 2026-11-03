# Exact-copy rule breakdown

Official judgeable definition only. This file does not change it.
Cuts are the placeholders in config.py, not a cut picked from the grid.
Day 3. Reddening window 2 days, minimum color gap 0.5 days.
5 of 100 copies are judgeable.

Finding: reddening could not run on 5 of 5 copies the official judgeable rule accepted. A skip is not a failed rise. The color points were not there, so the reddening rule was not applied.

Judgeable copies:

- at2017gfo-exact-005: fade passed, g-r cut failed, reddening skipped.
  Fade: It faded 1.53 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: Not enough data: g minus r was needed at day 3.70, which is outside the measurements we have for g (from day 0.46 to day 2.68) and r (only at day 2.70).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-007: fade passed, g-r cut failed, reddening skipped.
  Fade: It faded 1.04 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: Not enough data: g minus r was needed at day 1.15, but the g points 2.82 days apart and r points 2.82 days apart. A color is not read across a gap longer than 1.0 days, so this is no data.
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-010: fade passed, g-r cut failed, reddening skipped.
  Fade: It faded 0.94 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: Not enough data: g minus r was needed at day 1.40, but the r points 2.24 days apart. A color is not read across a gap longer than 1.0 days, so this is no data.
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-012: fade passed, g-r cut failed, reddening skipped.
  Fade: It faded 1.05 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: Not enough data: g minus r was needed at day 1.19, but the g points 2.64 days apart. A color is not read across a gap longer than 1.0 days, so this is no data.
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-048: fade passed, g-r cut failed, reddening skipped.
  Fade: It faded 0.56 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: Not enough data: g minus r was needed at day 1.20, but the r points 2.77 days apart. A color is not read across a gap longer than 1.0 days, so this is no data.
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.

# Exact-copy rule breakdown

Official judgeable definition (a) is unchanged. Candidate (b) also needs a usable g minus r at r-peak + 1 day with the 1.0-day gap limit. (b) is not official. Sage has to approve it before it can replace (a).
Light curves use the diffusion wrapper (arnett-factor-2) on redback 1.20.0. The package itself is not edited.
Cuts are the placeholders in config.py, not a cut picked from the grid.
Day 3. Reddening window 2 days, minimum color gap 0.5 days.
50 of 1200 copies are judgeable under (a). 0 of those are judgeable under (b).

Finding: reddening could not run on 50 of 50 copies the official judgeable rule accepted. A skip is not a failed rise. The color points were not there, so the reddening rule was not applied.

Judgeable copies:

- at2017gfo-exact-00001: definition (b) no, fade failed, no usable g-r: gap 2.44 d > 1.0 d, reddening skipped.
  Fade: The fastest fade was 0.06 mag/day in r, which is not faster than 0.30 mag/day, so the fade rule fails.
  g-r: no usable g-r: gap 2.44 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00003: definition (b) no, fade passed, no usable g-r: gap 2.15 d > 1.0 d, reddening skipped.
  Fade: It faded 1.27 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.15 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00010: definition (b) no, fade passed, no usable g-r: gap 2.28 d > 1.0 d, reddening skipped.
  Fade: It faded 1.33 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.28 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00082: definition (b) no, fade passed, no usable g-r: gap 2.29 d > 1.0 d, reddening skipped.
  Fade: It faded 0.74 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.29 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00126: definition (b) no, fade passed, no usable g-r: gap 2.44 d > 1.0 d, reddening skipped.
  Fade: It faded 0.94 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.44 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00131: definition (b) no, fade passed, no usable g-r: gap 2.31 d > 1.0 d, reddening skipped.
  Fade: It faded 0.75 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.31 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00151: definition (b) no, fade passed, no usable g-r: gap 2.03 d > 1.0 d, reddening skipped.
  Fade: It faded 1.31 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.03 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00152: definition (b) no, fade passed, no usable g-r: Not enough data: g minus r was needed at day 3.81, which is outside the measurements we have for g (from day 0.25 to day 2.83) and r (only at day 2.81)., reddening skipped.
  Fade: It faded 0.91 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: Not enough data: g minus r was needed at day 3.81, which is outside the measurements we have for g (from day 0.25 to day 2.83) and r (only at day 2.81).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00191: definition (b) no, fade passed, no usable g-r: gap 2.31 d > 1.0 d, reddening skipped.
  Fade: It faded 0.92 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.31 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00203: definition (b) no, fade passed, no usable g-r: gap 2.20 d > 1.0 d, reddening skipped.
  Fade: It faded 0.99 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.20 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00214: definition (b) no, fade passed, no usable g-r: gap 2.06 d > 1.0 d, reddening skipped.
  Fade: It faded 1.16 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.06 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00244: definition (b) no, fade passed, no usable g-r: gap 2.02 d > 1.0 d, reddening skipped.
  Fade: It faded 0.49 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.02 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00249: definition (b) no, fade passed, no usable g-r: gap 2.54 d > 1.0 d, reddening skipped.
  Fade: It faded 1.07 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.54 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00321: definition (b) no, fade passed, no usable g-r: gap 2.35 d > 1.0 d, reddening skipped.
  Fade: It faded 1.09 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.35 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00339: definition (b) no, fade passed, no usable g-r: gap 2.72 d > 1.0 d, reddening skipped.
  Fade: It faded 0.43 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.72 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00340: definition (b) no, fade passed, no usable g-r: Not enough data: g minus r was needed at day 3.45, which is outside the measurements we have for g (from day 0.27 to day 2.43) and r (only at day 2.45)., reddening skipped.
  Fade: It faded 0.98 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: Not enough data: g minus r was needed at day 3.45, which is outside the measurements we have for g (from day 0.27 to day 2.43) and r (only at day 2.45).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00404: definition (b) no, fade passed, no usable g-r: gap 2.05 d > 1.0 d, reddening skipped.
  Fade: It faded 1.29 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.05 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00426: definition (b) no, fade failed, no usable g-r: gap 2.51 d > 1.0 d, reddening skipped.
  Fade: The fastest fade was 0.12 mag/day in r, which is not faster than 0.30 mag/day, so the fade rule fails.
  g-r: no usable g-r: gap 2.51 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00449: definition (b) no, fade passed, no usable g-r: gap 2.26 d > 1.0 d, reddening skipped.
  Fade: It faded 1.27 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.26 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00452: definition (b) no, fade passed, no usable g-r: gap 2.06 d > 1.0 d, reddening skipped.
  Fade: It faded 0.94 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.06 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00463: definition (b) no, fade failed, no usable g-r: Not enough data: g minus r was needed at day 3.15, which is outside the measurements we have for g (only at day 0.06) and r (from day 0.08 to day 2.15)., reddening skipped.
  Fade: Not enough data to measure a fade: a fit needs at least 2 points after the peak in g, r, or i, and none of those bands has that many.
  g-r: no usable g-r: Not enough data: g minus r was needed at day 3.15, which is outside the measurements we have for g (only at day 0.06) and r (from day 0.08 to day 2.15).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00467: definition (b) no, fade failed, no usable g-r: gap 2.04 d > 1.0 d, reddening skipped.
  Fade: The fastest fade was 0.09 mag/day in r, which is not faster than 0.30 mag/day, so the fade rule fails.
  g-r: no usable g-r: gap 2.04 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00475: definition (b) no, fade passed, no usable g-r: Not enough data: g minus r was needed at day 3.25, which is outside the measurements we have for g (from day 0.10 to day 2.27) and r (only at day 2.25)., reddening skipped.
  Fade: It faded 0.37 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: Not enough data: g minus r was needed at day 3.25, which is outside the measurements we have for g (from day 0.10 to day 2.27) and r (only at day 2.25).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00507: definition (b) no, fade failed, no usable g-r: gap 2.50 d > 1.0 d, reddening skipped.
  Fade: The fastest fade was 0.23 mag/day in r, which is not faster than 0.30 mag/day, so the fade rule fails.
  g-r: no usable g-r: gap 2.50 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00517: definition (b) no, fade passed, no usable g-r: gap 2.04 d > 1.0 d, reddening skipped.
  Fade: It faded 1.43 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.04 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00566: definition (b) no, fade passed, no usable g-r: gap 2.14 d > 1.0 d, reddening skipped.
  Fade: It faded 0.97 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.14 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00595: definition (b) no, fade failed, no usable g-r: gap 2.60 d > 1.0 d, reddening skipped.
  Fade: The fastest fade was 0.04 mag/day in r, which is not faster than 0.30 mag/day, so the fade rule fails.
  g-r: no usable g-r: gap 2.60 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00601: definition (b) no, fade passed, no usable g-r: Not enough data: g minus r was needed at day 3.71, which is outside the measurements we have for g (from day 0.24 to day 2.69) and r (only at day 2.71)., reddening skipped.
  Fade: It faded 0.91 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: Not enough data: g minus r was needed at day 3.71, which is outside the measurements we have for g (from day 0.24 to day 2.69) and r (only at day 2.71).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00611: definition (b) no, fade passed, no usable g-r: Not enough data: g minus r was needed at day 3.82, which is outside the measurements we have for g (from day 0.12 to day 2.80) and r (only at day 2.82)., reddening skipped.
  Fade: It faded 0.57 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: Not enough data: g minus r was needed at day 3.82, which is outside the measurements we have for g (from day 0.12 to day 2.80) and r (only at day 2.82).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00620: definition (b) no, fade passed, no usable g-r: gap 2.19 d > 1.0 d, reddening skipped.
  Fade: It faded 0.88 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.19 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00624: definition (b) no, fade failed, no usable g-r: gap 2.19 d > 1.0 d, reddening skipped.
  Fade: The fastest fade was 0.05 mag/day in r, which is not faster than 0.30 mag/day, so the fade rule fails.
  g-r: no usable g-r: gap 2.19 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00712: definition (b) no, fade passed, no usable g-r: Not enough data: g minus r was needed at day 3.94, which is outside the measurements we have for g (from day 0.50 to day 2.91) and r (only at day 2.94)., reddening skipped.
  Fade: It faded 1.21 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: Not enough data: g minus r was needed at day 3.94, which is outside the measurements we have for g (from day 0.50 to day 2.91) and r (only at day 2.94).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00747: definition (b) no, fade failed, no usable g-r: gap 2.45 d > 1.0 d, reddening skipped.
  Fade: Not enough data to measure a fade: a fit needs at least 2 points after the peak in g, r, or i, and none of those bands has that many.
  g-r: no usable g-r: gap 2.45 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00800: definition (b) no, fade passed, no usable g-r: gap 2.10 d > 1.0 d, reddening skipped.
  Fade: It faded 0.97 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.10 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00804: definition (b) no, fade failed, no usable g-r: Not enough data: g minus r was needed at day 3.23, which is outside the measurements we have for g (from day 0.07 to day 2.21) and r (from day 0.05 to day 2.23)., reddening skipped.
  Fade: The fastest fade was 0.15 mag/day in g, which is not faster than 0.30 mag/day, so the fade rule fails.
  g-r: no usable g-r: Not enough data: g minus r was needed at day 3.23, which is outside the measurements we have for g (from day 0.07 to day 2.21) and r (from day 0.05 to day 2.23).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00853: definition (b) no, fade passed, no usable g-r: gap 2.11 d > 1.0 d, reddening skipped.
  Fade: It faded 1.42 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.11 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00857: definition (b) no, fade passed, no usable g-r: gap 2.33 d > 1.0 d, reddening skipped.
  Fade: It faded 0.82 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.33 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00881: definition (b) no, fade passed, no usable g-r: gap 2.50 d > 1.0 d, reddening skipped.
  Fade: It faded 1.00 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.50 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00883: definition (b) no, fade passed, no usable g-r: gap 2.21 d > 1.0 d, reddening skipped.
  Fade: It faded 0.88 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.21 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00890: definition (b) no, fade passed, no usable g-r: gap 2.15 d > 1.0 d, reddening skipped.
  Fade: It faded 0.93 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.15 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-00959: definition (b) no, fade passed, no usable g-r: Not enough data: g minus r was needed at day 3.86, which is outside the measurements we have for g (from day 0.18 to day 2.88) and r (only at day 2.86)., reddening skipped.
  Fade: It faded 0.76 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: Not enough data: g minus r was needed at day 3.86, which is outside the measurements we have for g (from day 0.18 to day 2.88) and r (only at day 2.86).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-01013: definition (b) no, fade passed, no usable g-r: gap 2.00 d > 1.0 d, reddening skipped.
  Fade: It faded 1.02 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.00 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-01018: definition (b) no, fade passed, no usable g-r: gap 2.15 d > 1.0 d, reddening skipped.
  Fade: It faded 0.38 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.15 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-01025: definition (b) no, fade passed, no usable g-r: gap 2.29 d > 1.0 d, reddening skipped.
  Fade: It faded 0.52 mag/day in r, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.29 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-01076: definition (b) no, fade failed, no usable g-r: Not enough data: g minus r was needed at day 3.13, which is outside the measurements we have for g (only at day 2.11) and r (from day 0.04 to day 2.13)., reddening skipped.
  Fade: Not enough data to measure a fade: a fit needs at least 2 points after the peak in g, r, or i, and none of those bands has that many.
  g-r: no usable g-r: Not enough data: g minus r was needed at day 3.13, which is outside the measurements we have for g (only at day 2.11) and r (from day 0.04 to day 2.13).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-01097: definition (b) no, fade failed, no usable g-r: Not enough data: g minus r was needed at day 3.39, which is outside the measurements we have for g (only at day 2.37) and r (from day 0.14 to day 2.39)., reddening skipped.
  Fade: Not enough data to measure a fade: a fit needs at least 2 points after the peak in g, r, or i, and none of those bands has that many.
  g-r: no usable g-r: Not enough data: g minus r was needed at day 3.39, which is outside the measurements we have for g (only at day 2.37) and r (from day 0.14 to day 2.39).
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-01116: definition (b) no, fade passed, no usable g-r: gap 2.29 d > 1.0 d, reddening skipped.
  Fade: It faded 1.17 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.29 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-01161: definition (b) no, fade failed, no usable g-r: gap 2.29 d > 1.0 d, reddening skipped.
  Fade: The fastest fade was 0.03 mag/day in g, which is not faster than 0.30 mag/day, so the fade rule fails.
  g-r: no usable g-r: gap 2.29 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-01180: definition (b) no, fade passed, no usable g-r: gap 2.00 d > 1.0 d, reddening skipped.
  Fade: It faded 1.24 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.00 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.
- at2017gfo-exact-01193: definition (b) no, fade passed, no usable g-r: gap 2.04 d > 1.0 d, reddening skipped.
  Fade: It faded 1.37 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
  g-r: no usable g-r: gap 2.04 d > 1.0 d
  Reddening: Skipped: g minus r could not be measured at any time, so we cannot tell whether it keeps getting redder. This rule does not reject the object.

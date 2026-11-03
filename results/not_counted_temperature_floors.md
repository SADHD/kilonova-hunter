# NOT-COUNTED temperature-floor variant

This is not part of the recall, the catch rate, or the grid. The realistic
set still uses redback's default floors: 5000 K for the wind and 2500 K
for the dynamical ejecta.

The variant below changes only those two floors:

- dynamical ejecta, opacity 10: 3745 K. That matches Villar's red
  component, which also has opacity 10.
- wind, opacity 0.5: 1308 K. 1308 K is Villar's purple floor. The purple
  component has opacity 3, and the wind has opacity 0.5, so this match
  is not exact.

One event only, not a population. Masses 1.46 and 1.27 solar masses,
40 Mpc, no dust, the two-component BNS model, with the diffusion
wrapper on. g-band magnitude:

| day | redback defaults 5000/2500 K | NOT-COUNTED 1308/3745 K | difference |
| --- | --- | --- | --- |
| 0.5 | 18.892 | 18.892 | 0.000 |
| 1.0 | 18.493 | 18.475 | -0.019 |
| 2.0 | 19.327 | 19.278 | -0.049 |
| 3.0 | 20.445 | 21.487 | +1.042 |

A positive difference means the variant is fainter. Before day 1 the
floors do not move g. By day 3 this one event is about 1.0 mag fainter
with the variant floors. Do not fold that into any score.

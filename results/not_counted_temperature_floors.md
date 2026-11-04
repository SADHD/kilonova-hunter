# NOT-COUNTED sensitivity. Not part of any recall.

The realistic set keeps this project's assumptions: temperature floors
5000 K (wind) and 2500 K (dynamical), opacities 0.5 (wind) and 10
(dynamical). redback 1.20.0 does not default those numbers. `tf_1` and
`tf_2` are required arguments. redback's shipped prior draws floors
log-uniformly from 100 K to 6000 K and opacities uniformly from 1 to 30.

The full realistic tuning population is 20,000 fakes. Resimulating that
population twice, with the diffusion wrapper, is not cheap without the
grid, so this is a fixed-cut summary of 24 fakes. They are one seeded
realistic tuning batch (batch index 90), SFD dust, the diffusion
wrapper, and the placeholder cuts in `config.py`. The same 24 masses,
distances, and sky positions are used for every row.

## Survey score at day 3

| run | judgeable | caught | caught and judgeable |
| --- | --- | --- | --- |
| project floors 5000/2500 K, opacities 0.5/10 | 0 of 24 | 0 | 0 |
| NOT-COUNTED floors 674 K wind, 3745 K dynamical | 0 of 24 | 0 | 0 |
| NOT-COUNTED opacities drawn Uniform(1, 30) | 0 of 24 | 0 | 0 |

Zero judgeable in 24 is what a roughly 1 percent judgeable fraction
predicts. It does not separate the variants. The magnitudes below do.

## Median g magnitude, no extra cut

| run | g at day 1 | g at day 3 |
| --- | --- | --- |
| project assumptions | 23.00 | 24.96 |
| floors 674 K wind / 3745 K dynamical | 22.98 | 25.87 |
| opacities Uniform(1, 30) | 26.00 | 26.48 |

674 K matches Villar's blue component, which has opacity 0.5, the same
opacity as the wind. 3745 K matches Villar's red component, which has
opacity 10, the same opacity as the dynamical ejecta. At day 1 those
floors barely move g (0.02 mag). At day 3 this sample is 0.91 mag
fainter. Drawing both opacities from redback's Uniform(1, 30) prior
makes the median g about 3.0 mag fainter at day 1.

Do not fold any of these numbers into a recall.

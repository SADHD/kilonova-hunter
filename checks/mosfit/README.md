# MOSFiT check

The bar that blocks a grid rerun is the wrapped redback model versus
MOSFiT, within 0.1 mag in g, r, and i from 0.5 days. The model is the
Villar et al. 2017 Table 2 three-component center, at 40 Mpc, in AB
magnitudes, with no Milky Way dust, on dense time grids. Planck18 sets
the redshift.

pip mosfit does not build on Python 3.10 or newer. `reference/` is a
line-by-line copy of the MOSFiT commit 6595248 modules (rprocess,
diffusion, temperature floor, blackbody, and the transform base). Those
files still import the mosfit package, so they are not executed.
`sidebyside_lib.py` and `band_sidebyside.py` are the runnable
reimplementation of those formulas.

Run from the repository root, with the sim extra installed
(`uv sync --extra sim`):

| Script | What it checks |
| --- | --- |
| `band_sidebyside.py` | Wrapped public redback g, r, i and bolometric luminosity against the MOSFiT reimplementation, from 0.5 days. Writes `docs/mosfit_external_check.csv`. |
| `sidebyside.py` | Component bolometric light curves, redback raw and with the factor-of-2 wrapper, against MOSFiT and an adaptive Arnett integral. Waxman Table B2 L_int is read from `docs/at2017gfo_bolometric_waxman2018.csv` (Waxman, Ofek, Kushnir, and Gal-Yam 2018, MNRAS 481, 3423). |
| `gr_gap.py` | Model minus AT2017gfo g−r at 1.5, 2, and 3 days. This gap is reported. It is not the 0.1 mag bar. |
| `harness_sparse_check.py` | Public harness model calls on a sparse visit list versus the same times inside a dense grid. |
| `reproduce_pr_table.py` | Which time grid reproduces an older, too-faint bolometric column. |
| `dense_test.py` | How the first dense time changes MOSFiT's early luminosity. |
| `mosfit_grid.py` | MOSFiT's own 100-point dense grid, with kappa_gamma = 100. |
| `temperature.py` | Component temperature and radius against the Waxman table. |
| `late_blue.py` | Blue component after a few days, redback versus the Arnett integral. |

`scripts/record_mosfit_external_check.py` calls `band_sidebyside.py` and
writes the same CSV. The saved `.out` files are the outputs of these
scripts from the review that measured max |difference| 0.028 mag in g,
0.023 in r, 0.020 in i, and 0.030 bolometric.

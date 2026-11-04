# Kilonova Hunter

Kilonova Hunter is an open Python tool that searches public alerts from the
Vera C. Rubin Observatory for **kilonovae**: the glow after two neutron stars
collide. Version 1.0 will score each candidate with rules you can read, and
publish how many fake kilonovae those rules catch and which real objects fool them.

This repository is the foundation. The pipeline modules are in place, the
`kilohunt` command runs, and a made-up light curve can be plotted and
checked against the v1 rules today. Talking to the live alert stream comes
in the next sprint.

If you have written a little Java: a **module** (a `.py` file) is a file of
functions, and the package `kilonova_hunter` is the folder that groups them,
like a Java package. `kilohunt` is the program you run, the way a class with
`main` is the program you run.

## What you are looking at

When two neutron stars merge, gravitational-wave detectors can hear the
collision, and the debris glows for days as it forges heavy elements. That
glow brightens within about a day, fades within about a week, and turns red
as it cools. Only one kilonova has been confirmed alongside a gravitational-wave
signal, in 2017. Rubin takes a wide picture of the sky every few nights and
publishes an alert for anything that changed. Most alerts are not kilonovae.
The tool's job is to rank the ones that look like they might be, and to say why.

Astronomers record brightness as a **magnitude**. A larger magnitude is a
fainter object, so a fading kilonova's magnitude climbs day by day. Plots in
this project flip that axis so brighter is up.

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). It
downloads Python and this project's libraries into a `.venv` folder inside
the project, so they stay separate from anything else on your computer.

macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then get the code and install the libraries. uv reads `.python-version` and
installs Python 3.12 if you do not already have it. The project accepts
Python 3.12 or newer, which is the range NumPy, pandas, Astropy, and
Matplotlib support. Pinning 3.12 here means your laptop and GitHub Actions
run the same interpreter.

```bash
git clone https://github.com/SADHD/kilonova-hunter.git
cd kilonova-hunter
uv sync
```

`uv sync` creates `.venv` and installs the libraries listed in `pyproject.toml`.
You do not activate the environment by hand. Put `uv run` in front of a
command and uv uses `.venv` for that command.

## Plot the demo

This draws a made-up kilonova: brighter for about a day, fainter over the
following week, noisier by a fixed random seed, and redder in color as time
goes on. Nothing is downloaded.

```bash
uv run kilohunt demo
```

The picture is saved to `output/demo_kilonova.png`. That folder is listed in
`.gitignore`, so the file stays on your machine. A copy made with the default
seed is already in the repo at
[docs/example_kilonova.png](docs/example_kilonova.png).

The same command also prints the filter verdict for the noise-free version
of that curve. The picture keeps a little random scatter. The verdict
leaves the scatter off, so the sentences match the rates built into the
toy and do not change when you change `--seed`.

To save that noise-free curve as a CSV you can check later:

```bash
uv run kilohunt demo --save-csv output/demo_kilonova.csv
```

`output/` is gitignored. A copy of the noise-free curve is already in the
repo at [examples/demo_kilonova.csv](examples/demo_kilonova.csv).

The same two function calls, written out so you can read them, are in
`notebooks/plot_fake_kilonova.py`:

```bash
uv run python notebooks/plot_fake_kilonova.py
```

See every command:

```bash
uv run kilohunt --help
```

## Check a light curve

`kilohunt check` reads one light curve from a CSV file and runs the v1
rules. CSV means comma-separated values: the first row is the column
names, and each row after that is one observation. The columns are the
same ones the demo builds: `time_days`, `filter`, `magnitude`, and
`magnitude_error`. `filter` is the band, such as `g`, `r`, or `i`. The
table further down says what each column means.

```bash
uv run kilohunt check examples/demo_kilonova.csv
```

The first line is the overall answer, `CANDIDATE` or `NOT A CANDIDATE`.
Each line after that is one rule, marked `PASS`, `FAIL`, or `SKIPPED`,
plus one sentence that says why. The command prints every rule the
verdict carries. It does not have the rule names written out by hand,
so a rule added later shows up on its own. Today the rules are the fade,
the color, and the reddening (whether g minus r keeps climbing). Fade
and color have to pass. Reddening has to pass when two colors can be
compared. If there are not two colors, reddening is `SKIPPED`. A skip
does not reject the object, so the first line can still be `CANDIDATE`.

On the noise-free demo, which does have two colors, the printout is:

```text
CANDIDATE
Fade: PASS. It faded 0.90 mag/day in g, which is faster than 0.30 mag/day, so the fade rule passes.
Color: PASS. 1 day after the r-band peak, g minus r is 0.35, which is redder than 0.00, so the color rule passes.
Reddening: PASS. From day 0.00 to day 2.00, a straight-line fit of g minus r goes from -0.43 to 0.32, a rise of 0.75 mag, which is more than 0.10 mag, so the reddening rule passes. No color had a known uncertainty, so the fit weighted every color equally.
```

A file that loads exits with code 0 even when the answer is not a
candidate. The answer is the printout, not a crash. A missing file, or
a CSV that does not have those columns, exits with a nonzero code and
prints what went wrong.

```bash
uv run kilohunt demo --save-csv output/demo_kilonova.csv
uv run kilohunt check output/demo_kilonova.csv
```

## Score fake kilonovae

`kilohunt recall` asks a published model how a neutron-star merger would
look, then asks the v1 rules whether they would have caught it. Recall
among judgeable fakes describes the filter. The catch rate, caught
divided by every fake, is the survey-level number. The cuts in
`config.py` are still placeholders. The grid does not edit them.

The physical model is [redback](https://github.com/nikhil-sarin/redback).
It is optional, so `uv sync` and the tests stay small:

```bash
uv sync --extra sim
uv run kilohunt recall --set-size 20 --which test
```

`--set-size` is how many fakes (default 200) for that older command.
`--which` is `tuning`, `test`, or `both` (the default). `--cadence wfd`
(the default) visits a field every 2 to 4 nights and takes two filters
about 30 minutes apart. "Within 3 days" is 3 days after the merger.
The depth in each band is the single-visit 5-sigma value from
Ivezić et al. 2019, Table 2.

A fake is **judgeable** by day D when, using only the visits and the
signal-to-noise, it has two detections in the same band at
signal-to-noise of 5 or more and at least 0.5 days apart, and one night
with both a g and an r detection at that signal-to-noise. For the
simulated fakes a night is `floor(MJD - 0.5)`, which changes at noon
UTC. The filter cuts are not part of this. Days 3, 5, and 7 are each
recomputed from the points up to that day.

There are two populations. **Realistic** fakes use redback's
`two_component_bns_ejecta_relation`, whose default ejecta class is
`redback.ejecta_relations.TwoComponentBNS` (Coughlin et al. 2019).
**2017-like** fakes use `three_component_kilonova_model` and the
spherical three-component numbers in Table 2 of Villar et al. 2017
(arXiv:1710.11576). Each component is drawn on its own: mass is
log-normal with sigma 0.2 dex, clipped to plus or minus 0.5 dex, and
speed is a truncated normal with sigma 20 percent of the table value,
kept between 0.05c and 0.4c. Opacity stays fixed. One hundred exact
copies of that table, at 40 Mpc, are reported separately and are not
in any recall number.

Distances are uniform in comoving volume from 10 to 300 Mpc. The
redshift is read from astropy's Planck18 cosmology, and redback applies
it to the clock and to the wavelengths. Milky Way dust for the grid is
the SFD map times 0.86 (Schlafly and Finkbeiner 2011), turned into
magnitudes with the Fitzpatrick 1999 law at R_V = 3.1. Sky positions
are uniform on the sphere inside -70 degrees < dec < +12 degrees and
outside the Galactic plane (|b| > 15 degrees). That strip is a stand-in
for the Rubin footprint, not the official map. The map files live in
`data/dustmaps/sfd/`, which git ignores. If they are missing, the grid
is labeled a dry run and uses the older exponential E(B-V) instead.

The evaluation command draws fakes in batches of 500 until 400 are
judgeable at day 3, and stops at 20,000 even if that target is not met.
It does not look at the filter while deciding to stop. Light curves
are cached under `data/recall/cache/` so the grid does not simulate
them again.

```bash
uv run kilohunt recall --grid --workers 4
```

That writes `results/tuning_grid.csv` and `results/tuning_grid.md`.
It scores the tuning split only. `kilohunt recall --score-test --cuts
<file>` scores the cached test split once and does not retune.

`kilohunt recall --fetch-alerts` downloads a ZTF proxy for Rubin from
Fink's public API (class `Unknown`) and writes
`results/real_alert_slices_fetched.json` and
`results/ztf_proxy_alerts_tuning.csv`. It labels rows using the committed
night lists in `results/real_alert_slices.json`, which it never overwrites.
Missing tuning nights are reported and do not change the split. Unknown-night
rows are dropped; their count and nights are reported to stderr and in the fetched
summary JSON.
Until cuts are frozen, no test-night rows or totals are written or printed.
A ZTF night is `floor(MJD - 20/24)`,
which changes at 20:00 UTC, local noon at Palomar in Pacific Standard
Time. The tuning and test slices do not share a night.

The light-curve model is a sphere. Each fake still gets a random
viewing angle, saved in the table, that does not change the brightness.

## Run the tests and the style check

A test is a function that checks one fact about the code. Pytest finds the
functions in `tests/` whose names start with `test_` and runs them.

```bash
uv run pytest
uv run ruff check
uv run ruff format
```

Ruff flags mistakes and keeps the formatting consistent. `ruff check` reports
problems. `ruff format` rewrites the files to the project's style. GitHub
Actions runs both checks, plus pytest, on every pull request.

## How the code is organized

Each stage does one job and passes a table to the next stage. Only `cli.py`
prints anything.

```text
Rubin alerts (via a broker)
        │
        ▼
     fetch.py          download light curves          Sprint 1
        │
        ▼
     clean.py          drop bad points, fix units     Sprint 2
        │
        ▼
   features.py         rise rate, fade rate, color    Sprint 2
        │
        ▼
    filter.py          kilonova rules and reasons     one light curve works now
        │
        ▼
    report.py          plots and the candidate list   plot works now
```

Two checks sit beside that pipeline:

| Module | Job | When |
| --- | --- | --- |
| `inject.py` | The simple teaching curve the demo plots | Works now. `recall.py` builds the redback fakes |
| `tns.py` | Confirmed types from the Transient Name Server, so we can see what fooled the filter | Sprint 5 |
| `cli.py` | The `kilohunt` command | `demo`, `check`, `recall`, and `--help`. `recall` needs `uv sync --extra sim`. `--week` arrives in Sprint 3 |

A light curve is a pandas table. The demo and the plot helper share these columns:

| Column | Meaning |
| --- | --- |
| `time_days` | Days since a reference time. For the demo, that is the explosion |
| `filter` | Which filter took the measurement, such as `g` or `r` |
| `magnitude` | Brightness. A smaller number is brighter |
| `magnitude_error` | The uncertainty on that magnitude |

Later stages add columns such as `object_id`. `plot_light_curve` ignores
columns it does not need, and it draws one object at a time.

```text
kilonova-hunter/
├── src/kilonova_hunter/    the package (one module per stage)
├── tests/                  pytest tests
├── notebooks/              short exploration scripts
├── examples/               a sample light curve CSV for kilohunt check
├── scoring/                weekly scorecards, once Sprint 5 writes them
├── docs/                   write-ups and the sample plot
├── pyproject.toml          project name, Python version, libraries
└── uv.lock                 exact library versions uv installed
```

Functions that are not written yet raise `NotImplementedError` and name the
sprint that will write them. That is intentional: calling them tells you
what is missing instead of pretending to return an answer.

## Libraries that are not installed yet

These are part of the plan, and they are left out of `pyproject.toml` until
the sprint that uses them:

- **Fink and ALeRCE clients (Sprint 1).** They download Rubin's public alerts.
  Fink is the first choice. ALeRCE is the backup if Fink is unavailable.
- **redback (optional).** Published kilonova models for `kilohunt recall`.
  Install it with `uv sync --extra sim`. The recall number comes from
  those models, not from the made-up curve in the demo. The demo and
  the tests run without it.

NumPy, pandas, Astropy, and Matplotlib are installed now. The demo uses
NumPy, pandas, and Matplotlib. Astropy is here for sky coordinates and times
once real alerts arrive in Sprint 1; the demo does not need it yet.

## Roadmap

| Sprint | Goal | What you can see at the end |
| --- | --- | --- |
| 0 · Foundations | Tools run, and a first plot exists | This demo: a made-up kilonova light curve |
| 1 · See the sky | Real Rubin alerts on screen | Light curves of a few real transients, one color per filter |
| 2 · Measure the change | Every transient gets rise, fade, and color | A table of those features for a week of alerts |
| 3 · Build the filter | A command that flags kilonova-like objects and says why | `kilohunt --week` prints ranked candidates with reasons |
| 4 · Test it with fakes | The fake-kilonova test runs, and the code is tested | A recall number, plus the automated tests |
| 5 · Score and publish | A week of real alerts checked against confirmed types | The impostor scorecard and a public write-up |
| Buffer | Catch up, fix, and release | Version 1.0 |

## Making a change

Work happens on a branch, in small commits, and comes back through a pull
request. GitHub fills in `.github/pull_request_template.md` when you open
one. Describe what changed, how you tested it, and paste a plot when the
change has one. Downloaded data and generated plots stay out of git;
`output/` and `data/` are already ignored.

## License

This project is released under the [MIT License](LICENSE).

### Decision 12(22): bundled v1 revision (Nov 4, 2026)

The color rule uses the latest same-night g+r detection pair available by
judgment day D, with S/N at least 5 in each band. Pair ordering is the later
exposure time, then g time, then r time (and color to break exact duplicates).
Rubin fakes use floor(MJD − 0.5); ZTF uses its frozen Palomar-noon convention.
Reddening retains its separate interpolated color samples. This implements
option (ii), approved by Sage, and reverses Decision 12(3). Judgeability stays
at the frozen filter-independent section 1 definition.

`recall --grid --tuning-only --new-fade` enables the preregistered candidate
fade measurement. It is off by default. A band is measurable only with two
detections strictly after its brightest observed point within three days.
The original straight-line fit, including the brightest point, is preserved.
If no band is measurable, fade is skipped; the grid counts these skips.
The 0.3 mag/day threshold is unchanged. Reporting both variants does not
adopt the candidate variant or freeze cuts. Test scoring remains held.

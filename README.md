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
so a rule added later shows up on its own. Today the rules are the fade
and the color. A rule that was not applied is `SKIPPED`.

A file that loads exits with code 0 even when the answer is not a
candidate. The answer is the printout, not a crash. A missing file, or
a CSV that does not have those columns, exits with a nonzero code and
prints what went wrong.

```bash
uv run kilohunt demo --save-csv output/demo_kilonova.csv
uv run kilohunt check output/demo_kilonova.csv
```

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
| `inject.py` | Fake kilonovae, so we can measure how many true ones the filter would catch | The simple curve works now. Published models arrive in Sprint 4 |
| `tns.py` | Confirmed types from the Transient Name Server, so we can see what fooled the filter | Sprint 5 |
| `cli.py` | The `kilohunt` command | `demo`, `check`, and `--help` work now. `--week` arrives in Sprint 3 |

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
- **redback (Sprint 4).** Published kilonova models for the fake-kilonova
  test. The recall number should come from those models, not from the
  made-up curve in the demo.

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

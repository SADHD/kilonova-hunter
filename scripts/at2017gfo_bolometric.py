"""Bolometric check on redback's dense grid.

``_one_component_kilonova_model`` is private. Calling it only at the
Waxman times drops the early diffusion integral and makes the model
look far too faint. This script evaluates the three Table 2 components
on redback's 300-point grid from 0.01 seconds to 7e6 seconds, with the
factor-of-2 wrapper, converts source time to observer time with (1+z),
and interpolates to the Waxman epochs.

The harness does not do this. ``src/`` calls the public model
functions, and those functions build this dense grid themselves.

    uv run python scripts/at2017gfo_bolometric.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kilonova_hunter.sky import redshift_at_luminosity_distance
from kilonova_hunter.villar import BLUE, EXACT_DISTANCE_MPC, PURPLE, RED

WAXMAN = Path("docs/at2017gfo_bolometric_waxman2018.csv")
OUTPUT = Path("docs/at2017gfo_bolometric_model.csv")
GRID_START_SECONDS = 1e-2
GRID_END_SECONDS = 7e6
GRID_POINTS = 300
SECONDS_PER_DAY = 86400.0


def dense_luminosity(
    redshift: float, user_source_seconds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observer-frame days, total L, and the blue-component temperature.

    ``user_source_seconds`` is passed into redback's grid builder, the
    same way the public model passes the requested times. Without those
    times the day-7.5 ratio sits near 1.70 instead of about 1.58.
    """
    from redback.transient_models.kilonova_models import _one_component_kilonova_model
    from redback.utils import get_optimal_time_array

    from kilonova_hunter.redback_fix import corrected_diffusion

    source_seconds = get_optimal_time_array(
        GRID_START_SECONDS,
        GRID_END_SECONDS,
        GRID_POINTS,
        user_times=np.asarray(user_source_seconds, dtype=float),
    )
    total = np.zeros(len(source_seconds), dtype=float)
    blue_temperature = None
    with corrected_diffusion():
        for component in (BLUE, PURPLE, RED):
            luminosity, temperature, _radius = _one_component_kilonova_model(
                source_seconds,
                component["mej"],
                component["vej"],
                component["kappa"],
                temperature_floor=component["temperature_floor"],
            )
            total += np.asarray(luminosity, dtype=float)
            if component is BLUE:
                blue_temperature = np.asarray(temperature, dtype=float)
    observer_days = source_seconds / SECONDS_PER_DAY * (1.0 + redshift)
    return observer_days, total, blue_temperature


def main() -> None:
    import matplotlib

    matplotlib.rcParams["text.usetex"] = False
    redshift = redshift_at_luminosity_distance(EXACT_DISTANCE_MPC)
    waxman = pd.read_csv(WAXMAN, comment="#")
    user_source_seconds = (
        waxman["time_days"].to_numpy(dtype=float) * SECONDS_PER_DAY / (1.0 + redshift)
    )
    observer_days, luminosity, blue_temperature = dense_luminosity(
        redshift, user_source_seconds
    )
    model = np.interp(
        waxman["time_days"].to_numpy(dtype=float),
        observer_days,
        luminosity,
        left=np.nan,
        right=np.nan,
    )
    ratio = model / waxman["l_int_erg_s"].to_numpy(dtype=float)
    table = pd.DataFrame(
        {
            "time_days": waxman["time_days"],
            "waxman_l_int_erg_s": waxman["l_int_erg_s"],
            "model_l_erg_s": model,
            "model_over_waxman": ratio,
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    source_seconds = observer_days / (1.0 + redshift) * SECONDS_PER_DAY
    before_one_day = int(np.sum(source_seconds < SECONDS_PER_DAY))
    print(f"redshift {redshift:.6f}")
    print(
        f"grid start {source_seconds[0]:.4g} s, "
        f"{len(source_seconds)} points, {before_one_day} before 1 day"
    )
    for day in (0.3, 0.5, 0.8):
        temperature = float(
            np.interp(day * SECONDS_PER_DAY, source_seconds, blue_temperature)
        )
        print(f"blue temperature at source-frame {day} d: {temperature:.1f} K")
    for day in (0.5, 0.8, 1.5, 7.5):
        row = table.loc[np.isclose(table["time_days"], day)].iloc[0]
        print(
            f"day {day}: model/L_int {row['model_over_waxman']:.3f} "
            f"({-2.5 * np.log10(row['model_over_waxman']):+.3f} mag)"
        )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

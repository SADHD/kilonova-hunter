"""NOT-COUNTED opacity-prior sensitivity. Not part of the recall.

Draws both realistic opacities from redback's Uniform(1, 30) prior and
compares day-1 and day-3 g with the project opacities 10 and 0.5, on
one realistic tuning batch. The batch in the written note is index 90
with 24 fakes. This is not the 20,000-fake grid. Do not fold the
numbers into a recall.

Run it by hand:

    uv run python scripts/opacity_prior_sensitivity.py
"""

from __future__ import annotations

import numpy as np

from kilonova_hunter.sky import DustSource
from kilonova_hunter.villar import (
    NOT_COUNTED_KAPPA_MAX,
    NOT_COUNTED_KAPPA_MIN,
    realistic_magnitudes,
    with_opacity_prior,
)

# The written sensitivity used this batch, not a new draw of the grid.
BATCH_INDEX = 90
BATCH_COUNT = 24
SPLIT = "tuning"


def opacity_prior_batch(
    dust: DustSource,
    *,
    batch_index: int = BATCH_INDEX,
    count: int = BATCH_COUNT,
    seed: int = BATCH_INDEX,
):
    """The project events, and the same events with prior opacities."""
    from kilonova_hunter.sample import draw_realistic_batch

    events, _seeds = draw_realistic_batch(
        SPLIT, batch_index, count, start_index=0, dust=dust
    )
    rng = np.random.default_rng(seed)
    varied = [with_opacity_prior(event, rng) for event in events]
    return events, varied


def g_at(event, day: float) -> float:
    """Dust-free g magnitude at one observer-frame day."""
    magnitude = realistic_magnitudes(
        np.array([day]),
        np.array(["g"]),
        event,
    )
    return float(magnitude[0])


def main() -> None:
    """Print median g for the project opacities and the prior draw."""
    dust = DustSource(mode="dry-run", reason="opacity prior script, not the grid")
    project, varied = opacity_prior_batch(dust)
    for label, events in (("project 10/0.5", project), ("Uniform(1, 30)", varied)):
        day_1 = np.array([g_at(event, 1.0) for event in events])
        day_3 = np.array([g_at(event, 3.0) for event in events])
        print(
            f"{label}: n={len(events)} "
            f"median g day 1 {np.nanmedian(day_1):.2f} "
            f"day 3 {np.nanmedian(day_3):.2f}"
        )
    print(
        "NOT-COUNTED. Opacities are uniform from "
        f"{NOT_COUNTED_KAPPA_MIN:g} to {NOT_COUNTED_KAPPA_MAX:g}. "
        "Do not fold this into a recall."
    )


if __name__ == "__main__":
    main()

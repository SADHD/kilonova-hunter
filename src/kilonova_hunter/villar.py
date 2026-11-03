"""The 2017-like fakes, built from Villar et al. 2017's three components.

The numbers are the spherical three-component row of Table 2 in
Villar et al. 2017, ApJ 851 L21, arXiv:1710.11576:

- blue:   mass 0.020 solar masses, speed 0.266c, opacity 0.5, floor 674 K
- purple: mass 0.047 solar masses, speed 0.152c, opacity 3,   floor 1308 K
- red:    mass 0.011 solar masses, speed 0.137c, opacity 10,  floor 3745 K

Opacities are in cm^2/g. Speeds are a fraction of the speed of light.
The temperature floors are the table medians. The decision log fixed
the masses, speeds, and opacities and did not list temperatures, so
the floors stay at the table values and are not drawn.

Each component is drawn on its own. Mass is log-normal with a standard
deviation of 0.2 dex, then clipped to plus or minus 0.5 dex. Clipped
means a draw past the edge is moved back to the edge, so the edge can
hold a pile of fakes. Speed is a normal draw with a standard deviation
of 20 percent of the table value, and draws outside 0.05c to 0.4c are
thrown away and tried again. That is a truncated normal: the edges do
not get a pile. Opacity does not move.

The light curve is redback's ``three_component_kilonova_model``
(``redback.transient_models.kilonova_models``). It sums three
blackbodies, and it takes mass, speed, opacity, and a temperature
floor for each one, which is the same list as Table 2. redback also
has ``mosfit_kilonova``, whose docstring says it should match Villar
or MOSFiT, but that function adds a gamma-ray opacity the table does
not fit. The three-component model is the closer one.

The model is a sphere. A viewing angle is saved and not passed in.
redback applies the Planck18 redshift to the clock and to the
wavelengths; see ``sky.REDSHIFT_APPLIED_IN_REDBACK``. Milky Way dust
is added after redback, with the Fitzpatrick 1999 law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kilonova_hunter.sky import extinction_for_bands

# Table 2, spherical 3-comp row. Names match the paper's order.
BLUE = {"mej": 0.020, "vej": 0.266, "kappa": 0.5, "temperature_floor": 674.0}
PURPLE = {"mej": 0.047, "vej": 0.152, "kappa": 3.0, "temperature_floor": 1308.0}
RED = {"mej": 0.011, "vej": 0.137, "kappa": 10.0, "temperature_floor": 3745.0}

# How far a mass draw may sit from the table value, in dex.
# 0.5 dex is a factor of about 3.16.
MASS_SIGMA_DEX = 0.2
MASS_CLIP_DEX = 0.5

# Speed: 20 percent of the table value, then keep only 0.05c to 0.4c.
VELOCITY_SIGMA_FRACTION = 0.20
VELOCITY_MIN_C = 0.05
VELOCITY_MAX_C = 0.40

EXACT_KIND = "at2017gfo-exact"
LIKE_KIND = "2017-like"

# The 100 exact copies use this seed, not the tuning or test seed.
EXACT_SEED = 20170817
EXACT_COUNT = 100
EXACT_DISTANCE_MPC = 40.0


@dataclass(frozen=True)
class VillarSpread:
    """The scatter around the Table 2 centers. These are the ruled values.

    ``mass_sigma_dex`` is the standard deviation of log10(mass).
    ``mass_clip_dex`` is how far log10(mass) may sit from the center
    before it is pulled back. ``velocity_sigma_fraction`` is the
    standard deviation as a fraction of the center speed. Speeds
    outside ``velocity_min_c`` to ``velocity_max_c`` are redrawn.
    """

    mass_sigma_dex: float = MASS_SIGMA_DEX
    mass_clip_dex: float = MASS_CLIP_DEX
    velocity_sigma_fraction: float = VELOCITY_SIGMA_FRACTION
    velocity_min_c: float = VELOCITY_MIN_C
    velocity_max_c: float = VELOCITY_MAX_C

    def __post_init__(self) -> None:
        if self.mass_sigma_dex <= 0 or self.mass_clip_dex <= 0:
            raise ValueError("mass sigma and clip must be positive.")
        if self.velocity_sigma_fraction <= 0:
            raise ValueError("velocity sigma must be positive.")
        if not 0.0 < self.velocity_min_c < self.velocity_max_c:
            raise ValueError("velocity limits must be positive and ordered.")


@dataclass(frozen=True)
class Component:
    """One ejecta component: blue, purple, or red."""

    name: str
    mej: float
    vej: float
    kappa: float
    temperature_floor: float


@dataclass(frozen=True)
class VillarEvent:
    """One 2017-like merger, or one exact AT2017gfo copy.

    ``kind`` is ``"2017-like"`` or ``"at2017gfo-exact"``. Exact copies
    are reported on their own and are not part of any recall denominator.
    """

    object_id: str
    kind: str
    distance_mpc: float
    redshift: float
    ebv: float
    ra_deg: float
    dec_deg: float
    galactic_b_deg: float
    merger_mjd: float
    viewing_angle_deg: float
    blue: Component
    purple: Component
    red: Component


def clipped_lognormal_mass(
    rng: np.random.Generator,
    center: float,
    spread: VillarSpread,
) -> float:
    """Log-normal mass, pulled back inside plus or minus ``mass_clip_dex``.

    Clipping piles draws up on the edge. That is what was asked for
    the mass, and it is not what the speed draw does.
    """
    log_center = np.log10(center)
    draw = float(rng.normal(log_center, spread.mass_sigma_dex))
    low = log_center - spread.mass_clip_dex
    high = log_center + spread.mass_clip_dex
    clipped = min(max(draw, low), high)
    return float(10.0**clipped)


def truncated_normal_speed(
    rng: np.random.Generator,
    center: float,
    spread: VillarSpread,
) -> float:
    """Normal speed, redrawn until it sits between 0.05c and 0.4c.

    Redrawing, rather than pulling the tails back to the edge, is a
    truncated normal. The edges do not collect a pile of fakes.
    """
    sigma = spread.velocity_sigma_fraction * center
    low = spread.velocity_min_c
    high = spread.velocity_max_c
    for _ in range(100_000):
        draw = float(rng.normal(center, sigma))
        if low <= draw <= high:
            return draw
    raise RuntimeError(
        f"could not draw a speed near {center}c inside {low}c to {high}c."
    )


def draw_component(
    rng: np.random.Generator,
    name: str,
    center: dict[str, float],
    spread: VillarSpread,
) -> Component:
    """One component, scattered on its own. Opacity and temperature stay put."""
    return Component(
        name=name,
        mej=clipped_lognormal_mass(rng, center["mej"], spread),
        vej=truncated_normal_speed(rng, center["vej"], spread),
        kappa=center["kappa"],
        temperature_floor=center["temperature_floor"],
    )


def table_component(name: str, center: dict[str, float]) -> Component:
    """The Table 2 center, with no scatter. Used for the exact copies."""
    return Component(
        name=name,
        mej=center["mej"],
        vej=center["vej"],
        kappa=center["kappa"],
        temperature_floor=center["temperature_floor"],
    )


def exact_components() -> tuple[Component, Component, Component]:
    """Blue, purple, and red at the Table 2 medians."""
    return (
        table_component("blue", BLUE),
        table_component("purple", PURPLE),
        table_component("red", RED),
    )


def draw_components(
    rng: np.random.Generator,
    spread: VillarSpread | None = None,
) -> tuple[Component, Component, Component]:
    """Blue, then purple, then red. Each draw is independent.

    The order is fixed so a given seed always rebuilds the same fake.
    """
    settings = VillarSpread() if spread is None else spread
    return (
        draw_component(rng, "blue", BLUE, settings),
        draw_component(rng, "purple", PURPLE, settings),
        draw_component(rng, "red", RED, settings),
    )


def villar_magnitudes(
    times: np.ndarray,
    bands: np.ndarray,
    event: VillarEvent,
) -> np.ndarray:
    """Observer-frame AB magnitudes, then Milky Way dust.

    Component order in the redback call is blue, purple, red, the
    same order as Table 2. Dust is the Fitzpatrick 1999 law at the
    event's E(B-V), added after redback so the model itself stays
    dust-free. The cosmology passed in is Planck18, and the redshift
    on the event came from that same cosmology.
    """
    from astropy.cosmology import Planck18
    from redback.transient_models.kilonova_models import (
        three_component_kilonova_model,
    )

    from kilonova_hunter.recall import _SNCOSMO_BANDS
    from kilonova_hunter.redback_fix import corrected_diffusion

    sn_bands = np.array([_SNCOSMO_BANDS[str(band)] for band in bands])
    # The temperature floors are positional on purpose. redback's default
    # floor is 4000 K, which is not the Villar table.
    with corrected_diffusion():
        raw = three_component_kilonova_model(
            np.asarray(times, dtype=float),
            event.redshift,
            event.blue.mej,
            event.blue.vej,
            event.blue.temperature_floor,
            event.blue.kappa,
            event.purple.mej,
            event.purple.vej,
            event.purple.temperature_floor,
            event.purple.kappa,
            event.red.mej,
            event.red.vej,
            event.red.temperature_floor,
            event.red.kappa,
            bands=sn_bands,
            output_format="magnitude",
            cosmology=Planck18,
        )
    dust = extinction_for_bands(event.ebv, bands)
    return np.asarray(raw, dtype=float) + dust


def realistic_magnitudes(
    times: np.ndarray,
    bands: np.ndarray,
    event,
) -> np.ndarray:
    """Realistic-population magnitudes, with Fitzpatrick dust on top.

    The ejecta come from ``two_component_bns_ejecta_relation``. In
    redback 1.20.0 that function's default relation class is
    ``redback.ejecta_relations.TwoComponentBNS``. The class docstring
    says the relations are Coughlin et al. 2019, and ``reference``
    points at 2019MNRAS.489L..91C. The module path of the function we
    call is ``redback.transient_models.kilonova_models``.

    This is the same model the older ``redback_magnitudes`` uses. The
    difference is the dust: Fitzpatrick 1999 at R_V = 3.1, not the
    fixed Schlafly and Finkbeiner band coefficients.

    The temperature floors on the event are redback's defaults when the
    recall built the event: 5000 K for the wind and 2500 K for the
    dynamical ejecta. Those defaults stay. A NOT-COUNTED variant uses
    different floors; that copy is ``not_counted_realistic_event`` and
    it is not part of the recall.
    """
    from astropy.cosmology import Planck18
    from redback.transient_models.kilonova_models import (
        two_component_bns_ejecta_relation,
    )

    from kilonova_hunter.recall import _SNCOSMO_BANDS
    from kilonova_hunter.redback_fix import corrected_diffusion

    sn_bands = np.array([_SNCOSMO_BANDS[str(band)] for band in bands])
    with corrected_diffusion():
        raw = two_component_bns_ejecta_relation(
            np.asarray(times, dtype=float),
            event.redshift,
            event.mass_1,
            event.mass_2,
            event.lambda_1,
            event.lambda_2,
            event.mtov,
            event.zeta,
            event.vej_wind,
            event.kappa_dynamical,
            event.kappa_wind,
            event.tf_dynamical,
            event.tf_wind,
            bands=sn_bands,
            output_format="magnitude",
            cosmology=Planck18,
        )
    dust = extinction_for_bands(event.ebv, bands)
    return np.asarray(raw, dtype=float) + dust


# NOT-COUNTED. These floors are not used by the recall or the grid.
# 3745 K is Villar's red floor. The realistic dynamical ejecta use
# opacity 10, the same opacity as that red component.
# 1308 K is Villar's purple floor. The realistic wind uses opacity 0.5,
# while the purple component uses opacity 3, so that match is not exact.
NOT_COUNTED_TF_DYNAMICAL_K = 3745.0
NOT_COUNTED_TF_WIND_K = 1308.0


def not_counted_realistic_event(event):
    """Copy one realistic event onto the NOT-COUNTED temperature floors.

    Do not put this copy into a recall, a catch rate, or the grid. It
    exists so a report can show the same ejecta with floors matched to
    Villar by opacity, beside the redback-default floors the recall uses.
    """
    from dataclasses import replace

    return replace(
        event,
        tf_dynamical=NOT_COUNTED_TF_DYNAMICAL_K,
        tf_wind=NOT_COUNTED_TF_WIND_K,
    )

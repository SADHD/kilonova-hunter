"""Measure how many fake kilonovae the filter catches.

Recall is the fraction of true kilonovae the filter keeps. A recall of
0.70 would mean 70 out of 100 fakes were caught. This module builds the
fakes, samples them the way a survey would, and counts the catches.

The physical light curves come from redback, which is optional. Install
it with ``uv sync --extra sim``. The tests do not need it: they pass a
small stand-in model into the same counting code.

Two lists are written under ``data/`` (that folder is gitignored):

- the tuning set, for the science lead to try cuts on
- the test set, which is the only list the recall number is allowed to
  use

The cuts themselves stay the placeholders in ``config.py``. This module
does not change them.

Assumptions, written here so they are not hidden in the middle of a
function:

- The light-curve model for the random fakes is redback's
  ``two_component_bns_ejecta_relation``. That is the two-component
  neutron-star merger model: dynamical ejecta plus a disk wind, with
  the masses of those ejecta taken from the Coughlin et al. 2019
  relations. It is a sphere. A random viewing angle is drawn for every
  fake and saved in the table, but the sphere does not get brighter or
  fainter with angle.
- Neutron-star masses are drawn uniformly between 1.1 and 1.6 solar
  masses. The heavier star is called mass 1.
- Distances are uniform in comoving volume, from 10 Mpc out to 300 Mpc.
  300 Mpc is a round number near the LIGO O5 binary-neutron-star range
  (about 330 Mpc). O4 is nearer, roughly 160 to 190 Mpc. The maximum is
  a setting. The distance stored in the table is the luminosity
  distance, which is the distance that sets the brightness. The draw
  itself is uniform in comoving volume. The distances are computed from
  a flat Planck 2018 universe with numpy, so this does not need scipy.
- The other model knobs (tidal deformability, maximum neutron-star
  mass, wind speed, opacities, temperature floors) are fixed. They are
  named on ``PopulationSettings`` and copied into the saved table.
- Milky Way dust is a random E(B-V), drawn from an exponential
  distribution with a mean of 0.05 mag and a cap of 0.5 mag. That is a
  simple stand-in for a random sightline through the Galaxy, not a dust
  map. Bluer bands are dimmed more, using the Schlafly and Finkbeiner
  2011 coefficients for SDSS g, r, and i as a stand-in for Rubin.
- The survey is not Rubin's real scheduler. Each of g, r, and i has its
  own clock. Visits of one band are separated by a configurable gap
  plus a random extra wait, so some nights are skipped. It is not the
  smooth half-day curve the demo uses.
- A point fainter than that band's limiting magnitude is a
  non-detection and is dropped. The limits are 24.5 in g, 24.3 in r,
  and 24.1 in i, inside the 24 to 24.5 single-visit range. The
  magnitude error comes from that depth, and Gaussian noise of that
  size is added to the points we keep.
- "Within 3 days" is counted from the merger. The filter only sees
  detections with time at most 3 days. The fake is caught when
  ``is_kilonova_candidate`` is true on that shortened table.
- A fake with no detection at any simulated time is "never detected".
  That count is printed on its own and is also included in the
  denominator. A merger the survey never saw is a miss.
- The AT2017gfo-like fake is a sanity check, not a member of either
  set, so it does not move the recall number.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from kilonova_hunter.config import THRESHOLDS
from kilonova_hunter.filter import RuleResult, is_kilonova_candidate

# The official recall number uses the test set. The tuning set is written
# so the science lead can try cuts without looking at the test set.
TUNING_SET = "tuning"
TEST_SET = "test"

# Default seeds differ by much more than one, so the two lists are not
# neighbors of the same random stream.
TUNING_SEED = 1100
TEST_SEED_OFFSET = 10000

# How long after the merger a detection is allowed to count. The filter
# is not shown anything later than this.
CATCH_WITHIN_DAYS = 3.0

# Rubin-like single-visit depths, in magnitudes. A larger number is a
# fainter limit. All three sit inside the 24 to 24.5 range.
LIMITING_MAGNITUDE = {"g": 24.5, "r": 24.3, "i": 24.1}

# Schlafly and Finkbeiner 2011, SDSS, R_V = 3.1. How many magnitudes of
# dimming one magnitude of E(B-V) causes. Used as a stand-in for Rubin.
EXTINCTION_PER_EBV = {"g": 3.303, "r": 2.285, "i": 1.698}

# sncosmo names for the Rubin bands redback understands.
_REDSHIFT_BAND = {"g": "lsstg", "r": "lsstr", "i": "lssti"}

# Distance bins for the breakdown, in Mpc of luminosity distance.
# The last edge is above the default maximum so the far bin has a home.
DISTANCE_BIN_EDGES_MPC = (0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 1000.0)

# 95% of a standard normal. The Wilson interval uses this for "about
# 95 times out of 100".
_WILSON_Z = 1.959963984540054

# A positive magnitude error smaller than this is lifted before 1/SNR
# is inverted, so a numerical zero from the formula does not explode.
_ERROR_FLOOR_MAG = 0.005


@dataclass(frozen=True)
class PopulationSettings:
    """How the random mergers are drawn. These are not the filter cuts.

    Attributes:
        count: How many fakes to build.
        seed: Starting point for the random draws. The same seed rebuilds
            the same list.
        mass_min_solar: Lightest neutron star, in solar masses.
        mass_max_solar: Heaviest neutron star, in solar masses.
        distance_min_mpc: Near edge of the volume, in Mpc.
        distance_max_mpc: Far edge of the volume, in Mpc. 300 is near
            the LIGO O5 binary-neutron-star range and above the O4 range.
        ebv_mean: Mean of the exponential E(B-V) draw, in magnitudes.
        ebv_cap: Largest E(B-V) we keep. A draw above this is clipped.
        lambda_1: Tidal deformability of the heavier star. Fixed.
            400 is a round value near the GW170817 constraint.
        lambda_2: Tidal deformability of the lighter star. Fixed.
        mtov: Maximum neutron-star mass the relation uses, in solar
            masses. Fixed. About the heaviest known pulsars.
        zeta: Fraction of the disk that becomes unbound. Fixed.
        vej_wind: Disk-wind speed, as a fraction of the speed of light.
        kappa_dynamical: Opacity of the dynamical (redder) ejecta.
        kappa_wind: Opacity of the disk wind (bluer ejecta).
        tf_dynamical: Temperature floor of the dynamical ejecta, in kelvin.
        tf_wind: Temperature floor of the disk wind, in kelvin.
    """

    count: int = 200
    seed: int = TUNING_SEED
    mass_min_solar: float = 1.1
    mass_max_solar: float = 1.6
    distance_min_mpc: float = 10.0
    distance_max_mpc: float = 300.0
    ebv_mean: float = 0.05
    ebv_cap: float = 0.5
    lambda_1: float = 400.0
    lambda_2: float = 400.0
    mtov: float = 2.17
    zeta: float = 0.15
    vej_wind: float = 0.10
    kappa_dynamical: float = 10.0
    kappa_wind: float = 0.5
    tf_dynamical: float = 2500.0
    tf_wind: float = 5000.0

    def __post_init__(self) -> None:
        """Reject a draw that cannot describe a population."""
        if self.count < 1:
            raise ValueError(f"count must be at least 1, got {self.count}.")
        if self.mass_min_solar <= 0 or self.mass_max_solar < self.mass_min_solar:
            raise ValueError(
                "neutron-star masses must be positive and the maximum must "
                f"be at least the minimum, got {self.mass_min_solar} to "
                f"{self.mass_max_solar}."
            )
        if self.distance_min_mpc <= 0 or self.distance_max_mpc <= self.distance_min_mpc:
            raise ValueError(
                "distances must be positive and the maximum must be larger "
                f"than the minimum, got {self.distance_min_mpc} to "
                f"{self.distance_max_mpc} Mpc."
            )
        if self.ebv_mean <= 0 or self.ebv_cap <= 0:
            raise ValueError("E(B-V) mean and cap must be positive.")


@dataclass(frozen=True)
class SurveySettings:
    """How the fake survey visits the sky. Not the filter cuts.

    Attributes:
        bands: Filters that get their own visit clock.
        revisit_gap_days: The minimum wait between two visits of the
            same band.
        nightly_gap_days: An extra random wait, uniform from 0 up to
            this many days, added on top of the revisit gap. That is
            the skipped night.
        survey_length_days: How long to keep generating visits. The
            filter still only sees the first ``catch_within_days``.
        catch_within_days: Detections later than this are hidden from
            the filter.
        limits: Faintest magnitude that still counts as a detection,
            per band.
        error_floor_mag: Smallest magnitude error we will store. A very
            bright source would otherwise get an error of nearly zero.
    """

    bands: tuple[str, ...] = ("g", "r", "i")
    revisit_gap_days: float = 1.0
    nightly_gap_days: float = 1.0
    survey_length_days: float = 6.0
    catch_within_days: float = CATCH_WITHIN_DAYS
    limits: dict[str, float] | None = None
    error_floor_mag: float = _ERROR_FLOOR_MAG

    def __post_init__(self) -> None:
        """Fill the default limits and reject a survey that cannot visit."""
        if self.limits is None:
            object.__setattr__(self, "limits", dict(LIMITING_MAGNITUDE))
        if self.revisit_gap_days <= 0:
            raise ValueError(
                f"revisit_gap_days must be positive, got {self.revisit_gap_days}."
            )
        if self.nightly_gap_days < 0:
            raise ValueError(
                f"nightly_gap_days cannot be negative, got {self.nightly_gap_days}."
            )
        if self.survey_length_days <= 0 or self.catch_within_days <= 0:
            raise ValueError("survey length and the catch window must be positive.")
        if not self.bands:
            raise ValueError("the survey needs at least one band.")

    def limit_for(self, band: str) -> float:
        """The detection limit for one band."""
        assert self.limits is not None
        try:
            return float(self.limits[band])
        except KeyError as error:
            raise KeyError(
                f"no limiting magnitude is set for band {band!r}."
            ) from error


@dataclass(frozen=True)
class FakeEvent:
    """One fake merger and the knobs that built its light curve.

    ``kind`` is ``"population"`` for a random draw, or ``"at2017gfo"``
    for the sanity check. The sanity check uses published ejecta
    numbers instead of the mass relation. Those numbers live in
    ``redback_magnitudes``. The masses stored here for that one fake
    are the GW170817 estimates, so the row still has a pair of neutron
    stars, but they are not what the light curve is computed from.
    """

    object_id: str
    kind: str
    mass_1: float
    mass_2: float
    distance_mpc: float
    redshift: float
    viewing_angle_deg: float
    ebv: float
    lambda_1: float
    lambda_2: float
    mtov: float
    zeta: float
    vej_wind: float
    kappa_dynamical: float
    kappa_wind: float
    tf_dynamical: float
    tf_wind: float


@dataclass(frozen=True)
class DistanceBin:
    """Recall inside one slice of distance.

    ``n_never_detected`` is included in ``n_total``, same choice as the
    headline number.
    """

    low_mpc: float
    high_mpc: float
    n_total: int
    n_caught: int
    n_never_detected: int

    @property
    def recall(self) -> float:
        """Caught divided by every fake in the bin, including never detected."""
        if self.n_total == 0:
            return float("nan")
        return self.n_caught / self.n_total


@dataclass(frozen=True)
class RecallNumbers:
    """The score for one set.

    ``n_never_detected`` is also inside ``n_total``. A fake the survey
    never saw counts as not caught.
    """

    set_name: str
    n_total: int
    n_caught: int
    n_never_detected: int
    n_model_failed: int
    recall: float
    ci_low: float
    ci_high: float
    by_distance: tuple[DistanceBin, ...]


def redback_available() -> bool:
    """True when the optional redback library can be imported."""
    return importlib.util.find_spec("redback") is not None


def dust_dimming_mag(band: str, ebv: float) -> float:
    """How many magnitudes fainter ``band`` becomes at this E(B-V).

    A larger magnitude is fainter, so the caller adds this number to
    the model's magnitude. g is dimmed more than r, and r more than i.
    """
    try:
        per_ebv = EXTINCTION_PER_EBV[band]
    except KeyError as error:
        raise KeyError(f"no dust coefficient is set for band {band!r}.") from error
    return float(per_ebv * ebv)


def magnitude_uncertainty(
    magnitude: float,
    limiting_magnitude: float,
    *,
    floor: float = _ERROR_FLOOR_MAG,
) -> float:
    """Magnitude error implied by a 5-sigma limiting magnitude.

    At the limit, the signal-to-noise ratio is 5. A source brighter by
    ``dm`` magnitudes has a signal-to-noise ratio of
    ``5 * 10**(0.4 * dm)``. The error in magnitudes is about
    ``1.0857 / signal-to-noise``. ``1.0857`` is ``2.5 / ln(10)``, the
    conversion from a fractional flux error into magnitudes.

    The error is computed from the true magnitude, before noise is
    added. A floor keeps a very bright source from claiming a perfect
    measurement.
    """
    signal_to_noise = 5.0 * 10 ** (-0.4 * (magnitude - limiting_magnitude))
    if not np.isfinite(signal_to_noise) or signal_to_noise <= 0:
        return float(floor)
    sigma = (2.5 / np.log(10.0)) / signal_to_noise
    return float(max(sigma, floor))


def sample_masses(
    rng: np.random.Generator,
    settings: PopulationSettings,
) -> tuple[float, float]:
    """Draw two neutron-star masses and put the heavier one first."""
    draws = rng.uniform(settings.mass_min_solar, settings.mass_max_solar, size=2)
    heavier = float(max(draws))
    lighter = float(min(draws))
    return heavier, lighter


def sample_viewing_angle_deg(rng: np.random.Generator) -> float:
    """Draw a viewing angle that is equally likely in every direction.

    Isotropic means the cosine of the angle is uniform between -1 and 1,
    not the angle itself. Face-on is 0 degrees. Edge-on is 90. The far
    side, looking at the other pole, goes out to 180.

    The two-component model is a sphere and does not use this number.
    It is saved so a later model can.
    """
    cosine = float(rng.uniform(-1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def sample_ebv(rng: np.random.Generator, settings: PopulationSettings) -> float:
    """Draw Milky Way E(B-V) from an exponential, then clip the tail.

    An exponential with mean ``ebv_mean`` is a simple stand-in for a
    random sightline: most lines of sight are fairly clear, and a few
    are dustier. Values above ``ebv_cap`` are pulled back to the cap.
    """
    draw = float(rng.exponential(settings.ebv_mean))
    return float(min(draw, settings.ebv_cap))


# Planck 2018 flat-universe numbers (Planck Collaboration 2020).
# Radiation is included so the distances match the usual Planck18
# cosmology. We integrate with numpy instead of astropy, because
# astropy's distance functions need scipy, and scipy is not part of
# the light install.
_HUBBLE_KM_S_MPC = 67.66
_OMEGA_MATTER = 0.30966
_OMEGA_LAMBDA = 0.6888463055445441
_OMEGA_RADIATION = 5.402015137139353e-05
_SPEED_OF_LIGHT_KM_S = 299792.458

# Filled the first time a distance is converted.
_COSMOLOGY_TABLE: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None


def _cosmology_table() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Redshift, comoving distance, comoving volume, and luminosity distance.

    Distances are in Mpc. Volume is in cubic Mpc. The rows are evenly
    spaced in redshift from 0 to 2, which reaches far beyond the default
    300 Mpc. Reading a distance back off the table is ``numpy.interp``:
    find where the distance sits between two rows and take the redshift
    in the same proportion.

    The comoving distance is ``(c / H0)`` times the integral of
    ``1 / E(z)``. ``E(z)`` is how much faster the universe was expanding
    at that redshift. A flat universe then has luminosity distance
    ``(1 + z)`` times the comoving distance, and comoving volume
    ``4/3 pi`` times the comoving distance cubed.
    """
    global _COSMOLOGY_TABLE
    if _COSMOLOGY_TABLE is None:
        redshift = np.linspace(0.0, 2.0, 8001)
        one_plus = 1.0 + redshift
        e_z = np.sqrt(
            _OMEGA_MATTER * one_plus**3 + _OMEGA_RADIATION * one_plus**4 + _OMEGA_LAMBDA
        )
        dz = float(redshift[1] - redshift[0])
        integrand = 1.0 / e_z
        cumulative = np.zeros_like(redshift)
        steps = (integrand[:-1] + integrand[1:]) * 0.5 * dz
        cumulative[1:] = np.cumsum(steps)
        comoving = (_SPEED_OF_LIGHT_KM_S / _HUBBLE_KM_S_MPC) * cumulative
        luminosity = one_plus * comoving
        volume = (4.0 / 3.0) * np.pi * comoving**3
        _COSMOLOGY_TABLE = (redshift, comoving, volume, luminosity)
    return _COSMOLOGY_TABLE


def comoving_volume_mpc3(redshift: np.ndarray) -> np.ndarray:
    """Comoving volume, in cubic Mpc, at each redshift."""
    grid_z, _comoving, volume, _luminosity = _cosmology_table()
    return np.interp(np.asarray(redshift, dtype=float), grid_z, volume)


def redshift_at_comoving_distance(distance_mpc: float) -> float:
    """Redshift whose comoving distance is ``distance_mpc``."""
    redshift, comoving, _volume, _luminosity = _cosmology_table()
    return float(np.interp(distance_mpc, comoving, redshift))


def redshift_at_luminosity_distance(distance_mpc: float) -> float:
    """Redshift whose luminosity distance is ``distance_mpc``."""
    redshift, _comoving, _volume, luminosity = _cosmology_table()
    return float(np.interp(distance_mpc, luminosity, redshift))


def sample_redshifts_and_distances(
    rng: np.random.Generator,
    count: int,
    settings: PopulationSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw distances that are uniform in comoving volume.

    Returns ``(redshift, luminosity_distance_mpc)``. The edges of the
    range are comoving distances. The number we store and print is the
    luminosity distance, because that is what makes a source faint.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}.")
    redshift, _comoving, volume, luminosity = _cosmology_table()
    z_near = redshift_at_comoving_distance(settings.distance_min_mpc)
    z_far = redshift_at_comoving_distance(settings.distance_max_mpc)
    volume_near = float(np.interp(z_near, redshift, volume))
    volume_far = float(np.interp(z_far, redshift, volume))
    drawn = rng.uniform(volume_near, volume_far, size=count)
    redshifts = np.interp(drawn, volume, redshift)
    distances = np.interp(redshifts, redshift, luminosity)
    return np.asarray(redshifts, dtype=float), np.asarray(distances, dtype=float)


def visit_times_for_one_band(
    rng: np.random.Generator,
    *,
    survey: SurveySettings,
) -> np.ndarray:
    """Visit times for one band, in days after the merger.

    The first visit waits a random fraction of the revisit gap, so the
    merger is not always observed immediately. Every later visit waits
    the revisit gap plus a random nightly gap. The gaps are not equal,
    and they are not the demo's smooth half-day steps.
    """
    times: list[float] = []
    clock = float(rng.uniform(0.0, survey.revisit_gap_days))
    while clock <= survey.survey_length_days:
        times.append(clock)
        extra = float(rng.uniform(0.0, survey.nightly_gap_days))
        clock += survey.revisit_gap_days + extra
    return np.array(times, dtype=float)


def survey_visits(
    rng: np.random.Generator,
    survey: SurveySettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Every visit of every band, as parallel time and band arrays."""
    times: list[float] = []
    bands: list[str] = []
    for band in survey.bands:
        for time_days in visit_times_for_one_band(rng, survey=survey):
            times.append(float(time_days))
            bands.append(band)
    order = np.argsort(times, kind="stable")
    return np.array(times, dtype=float)[order], np.array(bands, dtype=str)[order]


def observe_visits(
    times: np.ndarray,
    bands: np.ndarray,
    true_magnitudes: np.ndarray,
    *,
    survey: SurveySettings,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Turn true magnitudes into the detections a survey would keep.

    A point fainter than that band's limit is dropped. That comparison
    uses the true magnitude, before noise, so the cut does not depend
    on one lucky scatter. Points we keep get Gaussian noise with the
    width from ``magnitude_uncertainty``. The stored ``magnitude_error``
    is that width, not a new estimate from the noisy point.

    A non-finite magnitude (the model had nothing to say) is dropped.
    """
    columns = ["time_days", "filter", "magnitude", "magnitude_error"]
    rows: list[dict[str, float | str]] = []
    for time_days, band, true_magnitude in zip(
        times, bands, true_magnitudes, strict=True
    ):
        if not np.isfinite(true_magnitude):
            continue
        limit = survey.limit_for(str(band))
        # A larger magnitude is fainter than the survey can see.
        if float(true_magnitude) > limit:
            continue
        sigma = magnitude_uncertainty(
            float(true_magnitude),
            limit,
            floor=survey.error_floor_mag,
        )
        noisy = float(true_magnitude) + float(rng.normal(0.0, sigma))
        rows.append(
            {
                "time_days": float(time_days),
                "filter": str(band),
                "magnitude": noisy,
                "magnitude_error": sigma,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def light_curve_for_the_filter(
    observed: pd.DataFrame,
    within_days: float = CATCH_WITHIN_DAYS,
) -> pd.DataFrame:
    """The rows the filter is allowed to see: time at most ``within_days``.

    A point exactly at the limit is kept. A point later than the limit
    is hidden, even if the survey detected it.
    """
    if observed.empty:
        return observed.copy()
    kept = observed.loc[observed["time_days"] <= within_days + 1e-9]
    return kept.reset_index(drop=True)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """A 95% Wilson score interval for ``successes`` out of ``total``.

    This is a range that would contain the true recall about 95 times
    out of 100 if the whole experiment were repeated. It stays inside
    0 and 1, including when the recall is near 0 or 1 or the list is
    short. Returns ``(nan, nan)`` when ``total`` is 0.
    """
    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    z2 = _WILSON_Z * _WILSON_Z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half = (
        _WILSON_Z
        * np.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
        / denominator
    )
    low = float(max(0.0, center - half))
    high = float(min(1.0, center + half))
    return low, high


def stub_magnitudes(
    times: np.ndarray,
    bands: np.ndarray,
    event: FakeEvent,
) -> np.ndarray:
    """A tiny light curve that does not need redback.

    The absolute magnitude is the brightness the object would have at
    10 parsecs. The distance modulus moves that to the fake's real
    distance: farther means a larger (fainter) magnitude. g fades faster
    than r, so g minus r climbs. Dust is added on top.

    The viewing angle is ignored, on purpose, for the same reason the
    redback sphere ignores it. The peak is at the merger. This is a
    teaching curve for the counting code, not a kilonova model.
    """
    absolute = {"g": -15.5, "r": -16.0, "i": -16.1}
    fade_per_day = {"g": 1.0, "r": 0.45, "i": 0.40}
    # 5 log10(d_Mpc) + 25 is the distance modulus, with d in megaparsecs.
    distance_modulus = 5.0 * np.log10(event.distance_mpc) + 25.0
    magnitudes = np.empty(len(times), dtype=float)
    for index, (time_days, band) in enumerate(zip(times, bands, strict=True)):
        magnitudes[index] = (
            absolute[str(band)]
            + distance_modulus
            + fade_per_day[str(band)] * float(time_days)
            + dust_dimming_mag(str(band), event.ebv)
        )
    return magnitudes


def at2017gfo_like_event() -> FakeEvent:
    """One fake with published AT2017gfo-like numbers, at about 40 Mpc.

    The light curve, when redback is installed, uses the spherical
    two-component fit from Villar et al. 2017 (arXiv:1710.11576):

    - blue component: ejecta mass 0.023 solar masses, speed 0.256c,
      opacity 0.5 cm^2/g (they held this fixed), temperature floor 3983 K
    - red component: ejecta mass 0.050 solar masses, speed 0.149c,
      opacity 3.65 cm^2/g, temperature floor 1151 K

    Those ejecta numbers are passed to redback's
    ``two_component_kilonova_model``, which is the light-curve piece
    inside the two-component BNS relation. The relation itself is not
    used for this one fake, because a published fit quotes ejecta, not
    a fresh draw from neutron-star masses.

    The distance is 40 Mpc. NGC 4993, the host, is usually quoted near
    39.5 to 40 Mpc. Galactic E(B-V) is 0.105, the Schlafly and
    Finkbeiner value used for this event. The viewing angle is 30
    degrees, a round number in the 20 to 30 degree range often quoted
    for GW170817. The spherical model does not use it.

    The masses 1.46 and 1.27 solar masses are a common GW170817 point
    estimate. They are labels on the row. They are not fed to the
    ejecta relation for this fake.
    """
    distance_mpc = 40.0
    redshift = redshift_at_luminosity_distance(distance_mpc)
    blanks = PopulationSettings()
    return FakeEvent(
        object_id="at2017gfo-like",
        kind="at2017gfo",
        mass_1=1.46,
        mass_2=1.27,
        distance_mpc=distance_mpc,
        redshift=redshift,
        viewing_angle_deg=30.0,
        ebv=0.105,
        lambda_1=blanks.lambda_1,
        lambda_2=blanks.lambda_2,
        mtov=blanks.mtov,
        zeta=blanks.zeta,
        vej_wind=blanks.vej_wind,
        kappa_dynamical=blanks.kappa_dynamical,
        kappa_wind=blanks.kappa_wind,
        tf_dynamical=blanks.tf_dynamical,
        tf_wind=blanks.tf_wind,
    )


def redback_magnitudes(
    times: np.ndarray,
    bands: np.ndarray,
    event: FakeEvent,
) -> np.ndarray:
    """True magnitudes from redback, plus our Milky Way dust.

    Random fakes use ``two_component_bns_ejecta_relation``. The
    AT2017gfo-like fake uses ``two_component_kilonova_model`` with the
    Villar et al. 2017 numbers documented on ``at2017gfo_like_event``.
    Dust is added here, after redback, with the same coefficients the
    stub uses, so the tests and the real run dim the bands the same way.

    The viewing angle is not passed in. The model is a sphere.
    """
    from astropy.cosmology import Planck18
    from redback.transient_models.kilonova_models import (
        two_component_bns_ejecta_relation,
        two_component_kilonova_model,
    )

    sn_bands = np.array([_REDSHIFT_BAND[str(band)] for band in bands])
    shared = {
        "bands": sn_bands,
        "output_format": "magnitude",
        "cosmology": Planck18,
    }
    time_array = np.asarray(times, dtype=float)
    if event.kind == "at2017gfo":
        # Red component first, then blue. The order matches the
        # population model, where component 1 is the redder dynamical
        # ejecta and component 2 is the bluer wind. The numbers are the
        # Villar et al. 2017 two-component best fit.
        raw = two_component_kilonova_model(
            time_array,
            event.redshift,
            0.050,
            0.149,
            1151.0,
            3.65,
            0.023,
            0.256,
            3983.0,
            0.5,
            **shared,
        )
    else:
        raw = two_component_bns_ejecta_relation(
            time_array,
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
            **shared,
        )
    dust = np.array([dust_dimming_mag(str(band), event.ebv) for band in bands])
    return np.asarray(raw, dtype=float) + dust


def draw_population(
    settings: PopulationSettings,
    rng: np.random.Generator,
    *,
    set_name: str,
) -> list[FakeEvent]:
    """Draw ``settings.count`` random mergers. No light curves yet."""
    redshifts, distances = sample_redshifts_and_distances(rng, settings.count, settings)
    events: list[FakeEvent] = []
    for index in range(settings.count):
        mass_1, mass_2 = sample_masses(rng, settings)
        events.append(
            FakeEvent(
                object_id=f"{set_name}-{index:04d}",
                kind="population",
                mass_1=mass_1,
                mass_2=mass_2,
                distance_mpc=float(distances[index]),
                redshift=float(redshifts[index]),
                viewing_angle_deg=sample_viewing_angle_deg(rng),
                ebv=sample_ebv(rng, settings),
                lambda_1=settings.lambda_1,
                lambda_2=settings.lambda_2,
                mtov=settings.mtov,
                zeta=settings.zeta,
                vej_wind=settings.vej_wind,
                kappa_dynamical=settings.kappa_dynamical,
                kappa_wind=settings.kappa_wind,
                tf_dynamical=settings.tf_dynamical,
                tf_wind=settings.tf_wind,
            )
        )
    return events


def simulate_event(
    event: FakeEvent,
    survey: SurveySettings,
    rng: np.random.Generator,
    model,
) -> tuple[pd.DataFrame, bool]:
    """Sample one fake and return ``(detections, model_failed)``.

    ``model`` is a function ``(times, bands, event) -> magnitudes``.
    The stub in the tests is one. ``redback_magnitudes`` is the other.
    A model that raises is recorded as a failure and produces no
    detections, so the fake counts as never detected instead of
    aborting the whole list.
    """
    times, bands = survey_visits(rng, survey)
    if len(times) == 0:
        return _empty_curve(), False
    try:
        true_magnitudes = np.asarray(model(times, bands, event), dtype=float)
    except Exception:
        return _empty_curve(), True
    if len(true_magnitudes) != len(times):
        return _empty_curve(), True
    observed = observe_visits(
        times,
        bands,
        true_magnitudes,
        survey=survey,
        rng=rng,
    )
    return observed, False


def score_observed_curve(
    observed: pd.DataFrame,
    *,
    within_days: float = CATCH_WITHIN_DAYS,
) -> tuple[bool, bool, int]:
    """Score one observed table.

    Returns ``(never_detected, is_candidate, n_visible)``.

    ``never_detected`` is true only when the survey kept no point at
    any time. ``is_candidate`` is the filter's answer on the points
    with time at most ``within_days``. An empty window is not a
    candidate. ``n_visible`` is how many points the filter was shown.
    """
    never_detected = observed.empty
    visible = light_curve_for_the_filter(observed, within_days)
    n_visible = int(len(visible))
    if n_visible == 0:
        return never_detected, False, 0
    caught = bool(is_kilonova_candidate(visible).is_candidate)
    return never_detected, caught, n_visible


def summarize_recall(scored: pd.DataFrame, *, set_name: str) -> RecallNumbers:
    """Turn one row per fake into a recall, an interval, and distance bins.

    Every row is in the denominator, including fakes that were never
    detected. Those are also counted on their own.
    """
    total = int(len(scored))
    caught = int(scored["is_candidate"].sum()) if total else 0
    never = int(scored["never_detected"].sum()) if total else 0
    failed = int(scored["model_failed"].sum()) if total else 0
    low, high = wilson_interval(caught, total)
    recall = float(caught / total) if total else float("nan")
    return RecallNumbers(
        set_name=set_name,
        n_total=total,
        n_caught=caught,
        n_never_detected=never,
        n_model_failed=failed,
        recall=recall,
        ci_low=low,
        ci_high=high,
        by_distance=tuple(_distance_bins(scored)),
    )


def build_recall_set(
    *,
    set_name: str,
    population: PopulationSettings,
    survey: SurveySettings,
    model,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, RecallNumbers]:
    """Draw a set, observe it, and score it.

    Returns ``(events, light_curves, numbers)``. ``events`` has one row
    per fake, including the score. ``light_curves`` has one row per
    detection that passed the depth cut, including visits after the
    3-day window. The column ``within_catch_window`` says which of
    those rows the filter was shown.
    """
    draw = np.random.default_rng(population.seed) if rng is None else rng
    events = draw_population(population, draw, set_name=set_name)
    event_rows: list[dict[str, object]] = []
    curve_frames: list[pd.DataFrame] = []
    for event in events:
        # A fresh stream per fake, started from the population stream,
        # so one event's cadence does not depend on how long the
        # previous model call took.
        event_rng = np.random.default_rng(int(draw.integers(0, 2**31 - 1)))
        observed, model_failed = simulate_event(event, survey, event_rng, model)
        never, caught, n_visible = score_observed_curve(
            observed,
            within_days=survey.catch_within_days,
        )
        if model_failed:
            never = True
            caught = False
        event_rows.append(
            _event_row(
                event,
                never_detected=never,
                is_candidate=caught,
                n_visible=n_visible,
                model_failed=model_failed,
            )
        )
        if not observed.empty:
            frame = observed.copy()
            frame.insert(0, "object_id", event.object_id)
            frame["within_catch_window"] = frame["time_days"] <= (
                survey.catch_within_days + 1e-9
            )
            curve_frames.append(frame)
    events_table = pd.DataFrame(event_rows)
    if curve_frames:
        curves = pd.concat(curve_frames, ignore_index=True)
    else:
        curves = pd.DataFrame(
            columns=[
                "object_id",
                "time_days",
                "filter",
                "magnitude",
                "magnitude_error",
                "within_catch_window",
            ]
        )
    numbers = summarize_recall(events_table, set_name=set_name)
    return events_table, curves, numbers


def write_recall_tables(
    directory: Path,
    events: pd.DataFrame,
    light_curves: pd.DataFrame,
) -> None:
    """Write one set as two CSV files. Parent folders are created."""
    directory.mkdir(parents=True, exist_ok=True)
    events.to_csv(directory / "events.csv", index=False)
    light_curves.to_csv(directory / "light_curves.csv", index=False)


def run_recall(
    *,
    which: str = "both",
    set_size: int = 200,
    seed: int | None = None,
    output_dir: Path | str = Path("data") / "recall",
    model=None,
    survey: SurveySettings | None = None,
) -> str:
    """Build the requested sets, the sanity fake, and the plain-English report.

    ``model`` defaults to redback. Tests pass ``stub_magnitudes`` instead.
    The report's headline number is the test set whenever that set was
    built. The tuning set is labeled as not the official number.

    ``seed`` replaces the seed of the set you asked for. With
    ``which="both"``, the tuning set uses ``seed`` and the test set uses
    ``seed + 10000``, so the two lists are different draws.
    """
    if which not in {TUNING_SET, TEST_SET, "both"}:
        raise ValueError(f"which must be tuning, test, or both, got {which!r}.")
    if model is None:
        model = redback_magnitudes
    sky = survey if survey is not None else SurveySettings()
    destination = Path(output_dir)
    set_reports: list[RecallNumbers] = []
    for set_name, set_seed in _seeds_for(which, seed):
        population = PopulationSettings(count=set_size, seed=set_seed)
        events, curves, numbers = build_recall_set(
            set_name=set_name,
            population=population,
            survey=sky,
            model=model,
        )
        write_recall_tables(destination / set_name, events, curves)
        set_reports.append(numbers)

    sanity_text, sanity_candidate = _write_sanity_fake(destination, sky, model)
    text = format_recall_report(
        set_reports,
        sanity_text=sanity_text,
        sanity_is_candidate=sanity_candidate,
        output_dir=destination,
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.txt").write_text(text + "\n", encoding="utf-8")
    return text


def format_recall_report(
    sets: list[RecallNumbers],
    *,
    sanity_text: str,
    sanity_is_candidate: bool,
    output_dir: Path,
) -> str:
    """Turn the scores into the paragraphs ``kilohunt recall`` prints."""
    by_name = {item.set_name: item for item in sets}
    lines: list[str] = []
    lines.append(
        "Recall uses the placeholder cuts in config.py. They have not been tuned."
    )
    lines.append(
        "Fade faster than "
        f"{THRESHOLDS.fade_faster_than_mag_per_day:.2f} mag/day, "
        f"g minus r redder than {THRESHOLDS.redder_than_g_minus_r:.2f}, "
        "and a reddening rise of more than "
        f"{THRESHOLDS.min_g_minus_r_rise:.2f} mag."
    )
    lines.append(
        "The official number is the test set. The tuning set is written "
        "so cuts can be tried without looking at the test set."
    )
    lines.append(f"Tables are in {output_dir}.")
    lines.append("")
    if TEST_SET in by_name:
        lines.extend(_set_paragraphs(by_name[TEST_SET], official=True))
    if TUNING_SET in by_name:
        if TEST_SET in by_name:
            lines.append("")
        lines.extend(
            _set_paragraphs(by_name[TUNING_SET], official=TEST_SET not in by_name)
        )
    lines.append("")
    verdict = "CANDIDATE" if sanity_is_candidate else "NOT A CANDIDATE"
    lines.append(f"AT2017gfo-like sanity check: {verdict}")
    lines.append(
        "This fake is about 40 Mpc, with the Villar et al. 2017 two-component "
        "ejecta numbers and Galactic E(B-V) of 0.105. It is not part of "
        "either set, so it is not in the recall denominator."
    )
    lines.append(sanity_text)
    return "\n".join(lines)


def _set_paragraphs(numbers: RecallNumbers, *, official: bool) -> list[str]:
    """The sentences for one set."""
    label = "Test set" if numbers.set_name == TEST_SET else "Tuning set"
    if official and numbers.set_name == TEST_SET:
        heading = "Test set (this is the recall number):"
    elif official:
        heading = f"{label} only. This is not the test-set number the recall goal uses:"
    else:
        heading = f"{label}, not the official number:"
    lines = [heading]
    if numbers.n_total == 0:
        lines.append("No fakes were scored.")
        return lines
    percent = 100.0 * numbers.recall
    lines.append(f"Caught {numbers.n_caught} of {numbers.n_total} ({percent:.1f}%).")
    lines.append(
        "95% confidence interval: "
        f"{100.0 * numbers.ci_low:.1f}% to {100.0 * numbers.ci_high:.1f}%."
    )
    lines.append(
        "The interval is a Wilson score interval: a range that would "
        "contain the true recall about 95 times out of 100 if this "
        "experiment were repeated."
    )
    lines.append(
        f"Never detected at all: {numbers.n_never_detected}. "
        "That count is also included in the "
        f"{numbers.n_total} above. A merger the survey never saw is a miss, "
        "and it is listed on its own so a miss from 'no detection' can be "
        "told apart from a miss from the rules."
    )
    if numbers.n_model_failed:
        lines.append(
            f"The light-curve model failed for {numbers.n_model_failed} "
            "fake(s). Those are included in 'never detected'."
        )
    lines.append(
        "Recall by luminosity-distance bin (never detected stay in the denominator)."
    )
    lines.append(
        "The random draw is uniform in comoving volume. These bins use "
        "luminosity distance, which runs a little farther than the comoving "
        "edge, so a few fakes can sit past the configured maximum."
    )
    for bin_ in numbers.by_distance:
        if bin_.n_total == 0:
            caught_text = "none"
        else:
            caught_text = (
                f"{bin_.n_caught}/{bin_.n_total} "
                f"({100.0 * bin_.recall:.1f}%), "
                f"{bin_.n_never_detected} never detected"
            )
        lines.append(f"  {bin_.low_mpc:.0f} to {bin_.high_mpc:.0f} Mpc: {caught_text}")
    return lines


def _write_sanity_fake(
    destination: Path, survey: SurveySettings, model
) -> tuple[str, bool]:
    """Observe the AT2017gfo-like fake and return its verdict text."""
    event = at2017gfo_like_event()
    # 2017 is fixed so this one object does not change when the set seed changes.
    rng = np.random.default_rng(2017)
    observed, model_failed = simulate_event(event, survey, rng, model)
    visible = light_curve_for_the_filter(observed, survey.catch_within_days)
    if model_failed:
        text = "The light-curve model failed, so there is no verdict."
        candidate = False
    elif visible.empty:
        text = (
            "No detection within "
            f"{survey.catch_within_days:.0f} days, so the filter was not "
            "shown a light curve."
        )
        candidate = False
    else:
        verdict = is_kilonova_candidate(visible)
        candidate = bool(verdict.is_candidate)
        text = _format_verdict(verdict)
    frame = observed.copy()
    if not frame.empty:
        frame.insert(0, "object_id", event.object_id)
        frame["within_catch_window"] = (
            frame["time_days"] <= survey.catch_within_days + 1e-9
        )
    else:
        frame = pd.DataFrame(
            columns=[
                "object_id",
                "time_days",
                "filter",
                "magnitude",
                "magnitude_error",
                "within_catch_window",
            ]
        )
    row = _event_row(
        event,
        never_detected=observed.empty or model_failed,
        is_candidate=candidate,
        n_visible=int(len(visible)),
        model_failed=model_failed,
    )
    write_recall_tables(destination / "at2017gfo", pd.DataFrame([row]), frame)
    return text, candidate


def _seeds_for(which: str, seed: int | None) -> list[tuple[str, int]]:
    """Which sets to build, and the seed for each.

    With both sets, the test seed is the tuning seed plus
    ``TEST_SEED_OFFSET``. Passing one seed cannot make the two lists
    the same draw.
    """
    if which == TUNING_SET:
        return [(TUNING_SET, TUNING_SEED if seed is None else seed)]
    if which == TEST_SET:
        default = TUNING_SEED + TEST_SEED_OFFSET
        return [(TEST_SET, default if seed is None else seed)]
    tuning = TUNING_SEED if seed is None else seed
    return [
        (TUNING_SET, tuning),
        (TEST_SET, tuning + TEST_SEED_OFFSET),
    ]


def _distance_bins(scored: pd.DataFrame) -> list[DistanceBin]:
    """One recall for each distance slice, including empty slices."""
    edges = DISTANCE_BIN_EDGES_MPC
    bins: list[DistanceBin] = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        if scored.empty:
            chosen = scored
        else:
            chosen = scored.loc[
                (scored["distance_mpc"] >= low) & (scored["distance_mpc"] < high)
            ]
        total = int(len(chosen))
        bins.append(
            DistanceBin(
                low_mpc=low,
                high_mpc=high,
                n_total=total,
                n_caught=int(chosen["is_candidate"].sum()) if total else 0,
                n_never_detected=int(chosen["never_detected"].sum()) if total else 0,
            )
        )
    return bins


def _event_row(
    event: FakeEvent,
    *,
    never_detected: bool,
    is_candidate: bool,
    n_visible: int,
    model_failed: bool,
) -> dict[str, object]:
    """One fake as a flat row, including the score columns."""
    row: dict[str, object] = {
        field.name: getattr(event, field.name) for field in fields(event)
    }
    row["never_detected"] = never_detected
    row["is_candidate"] = is_candidate
    row["n_detections_within_3_days"] = n_visible
    row["model_failed"] = model_failed
    return row


def _empty_curve() -> pd.DataFrame:
    """A detection table with the right columns and no rows."""
    return pd.DataFrame(columns=["time_days", "filter", "magnitude", "magnitude_error"])


def _format_verdict(verdict) -> str:
    """The same PASS / FAIL / SKIPPED lines ``kilohunt check`` prints."""
    overall = "CANDIDATE" if verdict.is_candidate else "NOT A CANDIDATE"
    lines = [overall]
    for field in fields(verdict):
        value = getattr(verdict, field.name)
        if not isinstance(value, RuleResult):
            continue
        label = field.name.replace("_", " ").capitalize()
        if value.passed is None:
            status = "SKIPPED"
        elif value.passed:
            status = "PASS"
        else:
            status = "FAIL"
        lines.append(f"{label}: {status}. {value.explanation}")
    return "\n".join(lines)

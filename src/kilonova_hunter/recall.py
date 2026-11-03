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
  relations. The model is a sphere. A random viewing angle is drawn
  for every fake and saved in the table, but it does not change the
  light curve. That differs from the v1 validation plan, which asked
  for the angle to matter. The angle is kept so a later model can use it.
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
- The default survey is a simple wide-fast-deep pattern, not Rubin's
  real scheduler. A field is revisited every 2 to 4 nights. Each visit
  is two exposures about 30 minutes apart, in two of g, r, and i
  (pairs such as g+r, r+i, or g+i). The sequence is already running
  before the merger. The merger is dropped in at a random time, and
  only the visits after it are kept.
  ``--cadence simple`` keeps the older pattern: each band on its own
  clock, at least one day between visits of that band.
- A point fainter than that band's limiting magnitude is a
  non-detection and is dropped. The limits are the single-visit
  5-sigma depths in Ivezić et al. 2019, Table 2: 24.81 in g, 24.35 in
  r, and 23.92 in i. The magnitude error comes from that depth, and
  Gaussian noise of that size is added to the points we keep.
- "Within 3 days" is counted from the merger, not from the first
  detection. The filter only sees detections with time at most 3 days
  after the merger. The fake is caught when ``is_kilonova_candidate``
  is true on that shortened table.
- A fake with no detection at any simulated time is "never detected".
  That count is printed on its own and is also included in the
  denominator. A merger the survey never saw is a miss.
- The AT2017gfo-like fake is a sanity check, not a member of either
  set, so it does not move the recall number.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from kilonova_hunter.config import THRESHOLDS
from kilonova_hunter.features import (
    fade_rates,
    g_minus_r_at_time,
    g_minus_r_rise,
    peak_time_days,
)
from kilonova_hunter.filter import format_verdict, is_kilonova_candidate
from kilonova_hunter.judge import is_judgeable
from kilonova_hunter.tns import TNS_STATUS_LINE

logger = logging.getLogger(__name__)

# Printed on the recall, grid, and test-set reports. Decision Log 12(15).
# The bar is not loosened and the model is not tuned to close the gap.
EARLY_GAP_MODEL_LIMIT = (
    "Known model limit: radioactive heating only, no cocoon shock cooling "
    "(Piro & Kollmeier 2018). The color rule is measured at the r-band peak "
    "plus 1 day, which is past day 0.8, so the early gap does not affect "
    "that rule in either direction. Its only effect is that the fakes are "
    "too faint before day 0.8, which biases day-3 recall low."
)

# The official recall number uses the test set. The tuning set is written
# so the science lead can try cuts without looking at the test set.
TUNING_SET = "tuning"
TEST_SET = "test"

# Default seeds differ by much more than one, so the two lists are not
# neighbors of the same random stream.
TUNING_SEED = 1100
TEST_SEED_OFFSET = 10000

# How long after the merger a detection is allowed to count. The filter
# is not shown anything later than this. The clock is the merger, not
# the first detection.
CATCH_WITHIN_DAYS = 3.0

# A wider window used only as context. It is not the official recall.
CONTEXT_WINDOW_DAYS = 5.0

# Stop the run if the light-curve model fails for more than this
# fraction of the fakes. A broken model must not look like a low recall.
MODEL_ERROR_ABORT_FRACTION = 0.20

# Two visit times closer than this are one epoch. Thirty minutes is
# much longer, so the two exposures in one wide-fast-deep visit count
# as two epochs.
_SAME_EPOCH_DAYS = 1e-4

# A point that lands exactly on the window edge is kept when rounding
# nudges it by less than this.
_WINDOW_EDGE_DAYS = 1e-9

# The default wide-fast-deep cadence, and the older per-band clock.
CADENCE_WFD = "wfd"
CADENCE_SIMPLE = "simple"

# Single-visit 5-sigma point-source depths at zenith, AB magnitudes.
# Ivezić et al. 2019, ApJ 873 111, arXiv:0805.2366, Table 2, row m5
# (the parameters of their equations 5 and 6, for a 2 x 15 second
# visit). These are the expected depths. The rounder design goals in
# their Table 1 are g 25.0, r 24.7, and i 24.0. A larger number is a
# fainter limit.
LIMITING_MAGNITUDE = {"g": 24.81, "r": 24.35, "i": 23.92}

# Schlafly and Finkbeiner 2011, SDSS, R_V = 3.1. How many magnitudes of
# dimming one magnitude of E(B-V) causes. Used as a stand-in for Rubin.
EXTINCTION_PER_EBV = {"g": 3.303, "r": 2.285, "i": 1.698}

# sncosmo names for the Rubin bands redback understands.
_SNCOSMO_BANDS = {"g": "lsstg", "r": "lsstr", "i": "lssti"}

# Wide-fast-deep visit pairs. Each night uses one pair, in either order.
_WFD_PAIRS = (("g", "r"), ("r", "i"), ("g", "i"))

# 30 minutes, in days. The two exposures in one visit are this far apart.
_THIRTY_MINUTES_DAYS = 30.0 / (24.0 * 60.0)

# Failures a kilonova model is allowed to raise for one unlucky draw.
# Anything else is a bug and should stop the run with a traceback.
_MODEL_FAILURES = (
    ArithmeticError,
    FloatingPointError,
    OverflowError,
    RuntimeError,
    ValueError,
    np.linalg.LinAlgError,
)

# A function (times, bands, event) -> one magnitude per visit.
LightCurveModel = Callable[[np.ndarray, np.ndarray, "FakeEvent"], np.ndarray]

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
        kappa_dynamical: Opacity of the dynamical ejecta, in cm^2/g.
            10 is this project's assumption. redback's prior draws
            opacities uniformly from 1 to 30.
        kappa_wind: Opacity of the disk wind, in cm^2/g. 0.5 is this
            project's assumption, not a redback default.
        tf_dynamical: Temperature floor of the dynamical ejecta, in kelvin.
            2500 K is this project's assumption. redback 1.20.0 does not
            supply a default. ``tf_1`` and ``tf_2`` are required, and the
            shipped prior is log-uniform from 100 K to 6000 K.
        tf_wind: Temperature floor of the disk wind, in kelvin. 5000 K
            is this project's assumption, same as ``tf_dynamical``.
            A NOT-COUNTED sensitivity uses 3745 K on the dynamical
            ejecta and 674 K on the wind. That run is not this population.
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
    # Project assumptions, not redback defaults. redback 1.20.0 requires
    # tf_1 and tf_2. Its prior is log-uniform from 100 K to 6000 K, and
    # its opacities are uniform from 1 to 30. Kept on purpose.
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
        cadence: ``"wfd"`` (the default) or ``"simple"``. Wide-fast-deep
            revisits a field every few nights and takes two filters
            half an hour apart. Simple gives each band its own clock.
        wfd_gap_min_days: Shortest wait between wide-fast-deep visits.
        wfd_gap_max_days: Longest wait between those visits. The same
            range is used before and after the merger.
        revisit_gap_days: For the simple cadence, the minimum wait
            between two visits of the same band.
        nightly_gap_days: For the simple cadence, an extra random wait
            on top of the revisit gap.
        survey_length_days: How long to keep generating visits. Eight
            days leaves room for a 5-day context window. The official
            filter still only sees ``catch_within_days``.
        catch_within_days: Detections later than this, counted from the
            merger, are hidden from the filter.
        limits: Faintest magnitude that still counts as a detection,
            per band.
        error_floor_mag: Smallest magnitude error we will store. A very
            bright source would otherwise get an error of nearly zero.
        model_error_abort_fraction: Stop the run if the light-curve
            model fails for more than this fraction of the fakes.
    """

    cadence: str = CADENCE_WFD
    wfd_gap_min_days: float = 2.0
    wfd_gap_max_days: float = 4.0
    bands: tuple[str, ...] = ("g", "r", "i")
    revisit_gap_days: float = 1.0
    nightly_gap_days: float = 1.0
    survey_length_days: float = 8.0
    catch_within_days: float = CATCH_WITHIN_DAYS
    limits: dict[str, float] | None = None
    error_floor_mag: float = _ERROR_FLOOR_MAG
    model_error_abort_fraction: float = MODEL_ERROR_ABORT_FRACTION

    def __post_init__(self) -> None:
        """Fill the default limits and reject a survey that cannot visit."""
        if self.limits is None:
            object.__setattr__(self, "limits", dict(LIMITING_MAGNITUDE))
        if self.cadence not in {CADENCE_WFD, CADENCE_SIMPLE}:
            raise ValueError(
                f"cadence must be {CADENCE_WFD!r} or {CADENCE_SIMPLE!r}, "
                f"got {self.cadence!r}."
            )
        if self.wfd_gap_min_days <= 0 or self.wfd_gap_max_days < self.wfd_gap_min_days:
            raise ValueError(
                "the wide-fast-deep gap must be positive and the maximum "
                f"must be at least the minimum, got {self.wfd_gap_min_days} "
                f"to {self.wfd_gap_max_days} days."
            )
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
        if not 0.0 <= self.model_error_abort_fraction <= 1.0:
            raise ValueError(
                "model_error_abort_fraction must be between 0 and 1, "
                f"got {self.model_error_abort_fraction}."
            )

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
    # Sky position and merger time. The older draw leaves these at the
    # defaults. The evaluation harness fills them in: the night index
    # is floor(merger_mjd + time_days - 0.5), and the dust map is read
    # at ra_deg, dec_deg.
    ra_deg: float = float("nan")
    dec_deg: float = float("nan")
    galactic_b_deg: float = float("nan")
    merger_mjd: float = 60000.0


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


def times_in_window(times: np.ndarray, window_days: float) -> np.ndarray:
    """True for each time that is at most ``window_days`` after the merger.

    The times are already days since the merger. The window does not
    start at the first detection. A tiny margin keeps a point that lands
    exactly on the edge when floating-point rounding nudges it.
    """
    return np.asarray(times, dtype=float) <= window_days + _WINDOW_EDGE_DAYS


def time_is_in_window(time_days: float, window_days: float) -> bool:
    """True when one time is inside the merger window. See ``times_in_window``."""
    return bool(times_in_window(np.array([time_days]), window_days)[0])


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
    """Every visit, as parallel time and band arrays.

    ``survey.cadence`` picks the pattern. ``"wfd"`` is the default
    wide-fast-deep pattern. ``"simple"`` is one clock per band.
    """
    if survey.cadence == CADENCE_WFD:
        return _wfd_visits(rng, survey)
    return _simple_visits(rng, survey)


# The wide-fast-deep sequence is already running when the merger
# happens. Thirty days is several revisits, so the merger is not a
# special first night and there is no separate 0-to-4 day first wait.
_WFD_PRE_MERGER_DAYS = 30.0


def _wfd_pair_starts(
    rng: np.random.Generator,
    survey: SurveySettings,
) -> np.ndarray:
    """Times when each visit pair starts, in days after the merger.

    The survey is built first, as a continuous sequence of 2 to 4 day
    gaps. The merger is then placed at a uniform random time on that
    sequence. A uniform instant lands in a longer gap more often, which
    is the length weighting a running survey actually has. Drawing the
    gap first and then a uniform time inside it would miss that, and
    the mean wait until the next visit would be 1.5 days instead of
    14/9 days. Times before the merger are negative. The earliest pair
    is at least ``_WFD_PRE_MERGER_DAYS`` before the merger.
    """
    low = survey.wfd_gap_min_days
    high = survey.wfd_gap_max_days
    # Long enough that the merger is not stuck on the first few gaps,
    # where a sequence that starts on a visit has not settled.
    target = max(_WFD_PRE_MERGER_DAYS + survey.survey_length_days + high, 120.0)
    starts = [0.0]
    clock = 0.0
    while clock < target:
        clock += float(rng.uniform(low, high))
        starts.append(clock)
    earliest = _WFD_PRE_MERGER_DAYS
    latest = clock - survey.survey_length_days
    merger = float(rng.uniform(earliest, latest))
    return np.asarray(starts, dtype=float) - merger


def _wfd_visits(
    rng: np.random.Generator,
    survey: SurveySettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Wide-fast-deep visits: two filters, then a gap of a few nights.

    The survey is already visiting the field before the merger. Only
    visits at or after the merger are returned. A pair that started
    before the merger is dropped whole, so one filter of that pair is
    never kept on its own. On each night the survey takes two exposures
    about 30 minutes apart, in a random pair drawn from g+r, r+i, and
    g+i, in either order.
    """
    times: list[float] = []
    bands: list[str] = []
    for clock in _wfd_pair_starts(rng, survey):
        # A pair that started before the merger stays before it.
        if float(clock) < 0.0 or float(clock) > survey.survey_length_days:
            continue
        pair = _WFD_PAIRS[int(rng.integers(0, len(_WFD_PAIRS)))]
        if float(rng.random()) < 0.5:
            pair = (pair[1], pair[0])
        for offset, band in ((0.0, pair[0]), (_THIRTY_MINUTES_DAYS, pair[1])):
            when = float(clock) + offset
            if when <= survey.survey_length_days:
                times.append(when)
                bands.append(band)
    if not times:
        return np.array([], dtype=float), np.array([], dtype=str)
    order = np.argsort(times, kind="stable")
    return np.array(times, dtype=float)[order], np.array(bands, dtype=str)[order]


def _simple_visits(
    rng: np.random.Generator,
    survey: SurveySettings,
) -> tuple[np.ndarray, np.ndarray]:
    """One independent clock per band. This is ``--cadence simple``."""
    times: list[float] = []
    bands: list[str] = []
    for band in survey.bands:
        for time_days in visit_times_for_one_band(rng, survey=survey):
            times.append(float(time_days))
            bands.append(band)
    if not times:
        return np.array([], dtype=float), np.array([], dtype=str)
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
    """Rows the filter may see: at most ``within_days`` after the merger.

    The clock is the merger. A first detection on day 4 is not "within
    3 days" just because nothing came before it. A point exactly on the
    edge is kept. A later point is hidden even if the survey detected it.
    """
    if observed.empty:
        return observed.copy()
    kept = observed.loc[times_in_window(observed["time_days"].to_numpy(), within_days)]
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

    The viewing angle is not passed in. The model is a sphere, so the
    angle saved on the event does not change these magnitudes. That
    differs from the v1 validation plan.

    Pipeline, checked against a units bug:

    - ``output_format`` is ``magnitude``. Bands are the sncosmo Rubin
      names (``lsstg``, ``lsstr``, ``lssti``). sncosmo returns AB
      magnitudes.
    - We pass a redshift, not a distance. redback turns that redshift
      into a luminosity distance in centimeters with Planck18. We do
      not convert units ourselves.
    - redback applies time dilation once, inside the model, by
      dividing the observer-frame time by ``1 + z``. We do not do it
      again. At the AT2017gfo redshift that factor is about 1 percent.
    - Milky Way dust is added once, here, after redback. The redback
      call itself does not apply extinction.
    - Time zero is the merger. ``times`` are observer-frame days since
      then, which is what redback expects.
    """
    from astropy.cosmology import Planck18
    from redback.transient_models.kilonova_models import (
        two_component_bns_ejecta_relation,
        two_component_kilonova_model,
    )

    from kilonova_hunter.redback_fix import corrected_diffusion

    sn_bands = np.array([_SNCOSMO_BANDS[str(band)] for band in bands])
    shared = {
        "bands": sn_bands,
        "output_format": "magnitude",
        "cosmology": Planck18,
    }
    time_array = np.asarray(times, dtype=float)
    with corrected_diffusion():
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


def ejecta_masses_and_speeds(event: FakeEvent) -> dict[str, float]:
    """Ejecta mass and speed that go into this fake's light curve.

    For the AT2017gfo-like fake these are the Villar et al. 2017
    two-component numbers (red 0.050 solar masses at 0.149c, blue 0.023
    solar masses at 0.256c). Their sum is about 0.07 solar masses.

    For a random fake they come from the Coughlin et al. 2019
    two-component BNS relation, the same formulas as
    ``redback.ejecta_relations.TwoComponentBNS``. The dynamical piece is
    the redder one (opacity 10 in this run). The disk wind is the bluer
    one (opacity 0.5). The wind speed is the fixed 0.1c on the event.
    The relation derives the dynamical speed. Those random draws are
    usually well below 0.02 solar masses in total, which is why the
    population is fainter than AT2017gfo scaled to the same distance.
    """
    if event.kind == "at2017gfo":
        return {
            "mej_dynamical_msun": 0.050,
            "vej_dynamical_c": 0.149,
            "mej_disk_wind_msun": 0.023,
            "vej_disk_wind_c": 0.256,
        }
    return _coughlin_two_component_ejecta(event)


def _coughlin_two_component_ejecta(event: FakeEvent) -> dict[str, float]:
    """Dynamical and disk-wind ejecta from Coughlin et al. 2019.

    The algebra is the same as redback's ``TwoComponentBNS``. Compactness
    comes from the tidal deformability the way that class does.
    """
    compactness_1 = _compactness_from_lambda(event.lambda_1)
    compactness_2 = _compactness_from_lambda(event.lambda_2)
    dynamical = _dynamical_ejecta_mass(
        event.mass_1, event.mass_2, compactness_1, compactness_2
    )
    wind = _disk_wind_ejecta_mass(
        event.mass_1,
        event.mass_2,
        event.lambda_1,
        event.lambda_2,
        event.mtov,
        event.zeta,
    )
    speed = _dynamical_ejecta_speed(
        event.mass_1, event.mass_2, compactness_1, compactness_2
    )
    return {
        "mej_dynamical_msun": float(dynamical),
        "vej_dynamical_c": float(speed),
        "mej_disk_wind_msun": float(wind),
        "vej_disk_wind_c": float(event.vej_wind),
    }


def _compactness_from_lambda(tidal_deformability: float) -> float:
    """Neutron-star compactness from the dimensionless tidal deformability.

    Same fit as ``redback.ejecta_relations.calc_compactness_from_lambda``.
    """
    log_lambda = np.log(tidal_deformability)
    return float(0.371 - 0.0391 * log_lambda + 0.001056 * log_lambda**2)


def _dynamical_ejecta_mass(
    mass_1: float,
    mass_2: float,
    compactness_1: float,
    compactness_2: float,
) -> float:
    """Dynamical ejecta mass in solar masses. Coughlin et al. 2019."""
    log10_mej = (
        -0.0719
        * (
            mass_1 * (1.0 - 2.0 * compactness_1) / compactness_1
            + mass_2 * (1.0 - 2.0 * compactness_2) / compactness_2
        )
        + 0.2116
        * (mass_1 * (mass_2 / mass_1) ** -2.905 + mass_2 * (mass_1 / mass_2) ** -2.905)
        - 2.42
    )
    return float(10.0**log10_mej)


def _dynamical_ejecta_speed(
    mass_1: float,
    mass_2: float,
    compactness_1: float,
    compactness_2: float,
) -> float:
    """Dynamical ejecta speed in units of c. Coughlin et al. 2019."""
    return float(
        -0.3090 * (mass_1 / mass_2) * (1.0 + -1.879 * compactness_1)
        + -0.3090 * (mass_2 / mass_1) * (1.0 + -1.879 * compactness_2)
        + 0.657
    )


def _disk_wind_ejecta_mass(
    mass_1: float,
    mass_2: float,
    lambda_1: float,
    lambda_2: float,
    mtov: float,
    zeta: float,
) -> float:
    """Disk-wind ejecta mass in solar masses. Coughlin et al. 2019.

    ``zeta`` is the fraction of the disk that becomes unbound.
    """
    mass_ratio = mass_1 / mass_2
    lambda_tilde = (
        (16.0 / 13.0)
        * (
            lambda_2
            + lambda_1 * mass_ratio**5
            + 12.0 * lambda_1 * mass_ratio**4
            + 12.0 * lambda_2 * mass_ratio
        )
        / (mass_ratio + 1.0) ** 5
    )
    chirp = (mass_1 * mass_2) ** 0.6 * (mass_1 + mass_2) ** -0.2
    radius_16 = chirp * (lambda_tilde / 0.0042) ** (1.0 / 6.0)
    threshold = (2.38 - 3.606 * mtov / radius_16) * mtov
    log10_disk = -31.335 * (
        1.0 + -0.9760 * np.tanh((1.0474 - (mass_1 + mass_2) / threshold) / 0.05957)
    )
    return float(zeta * 10.0**log10_disk)


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


def _peak_magnitudes(bands: np.ndarray, magnitudes: np.ndarray) -> dict[str, float]:
    """Brightest true magnitude in each band. Smallest magnitude is brightest."""
    peaks: dict[str, float] = {}
    for band in ("g", "r", "i"):
        chosen = magnitudes[np.asarray(bands) == band]
        finite = chosen[np.isfinite(chosen)]
        peaks[band] = float(np.min(finite)) if len(finite) else float("nan")
    return peaks


def _epoch_count(times: np.ndarray) -> int:
    """How many distinct visit times are in ``times``.

    Two rows at the same time are one epoch. Two exposures 30 minutes
    apart are two epochs.
    """
    if len(times) == 0:
        return 0
    ordered = np.sort(np.asarray(times, dtype=float))
    count = 1
    previous = float(ordered[0])
    for time_days in ordered[1:]:
        if float(time_days) - previous > _SAME_EPOCH_DAYS:
            count += 1
            previous = float(time_days)
    return count


def _blank_diagnosis(window_days: float) -> dict[str, object]:
    """Score columns for a fake the filter was not shown."""
    return {
        "n_detections_in_window": 0,
        "window_days": window_days,
        "judgeable": False,
        "is_candidate": False,
        "fade_result": "",
        "color_result": "",
        "reddening_result": "",
        "fade_rate_mag_per_day": float("nan"),
        "g_minus_r": float("nan"),
        "reddening_rise_mag": float("nan"),
    }


def diagnose_catch_window(
    observed: pd.DataFrame,
    window_days: float,
    merger_mjd: float = 60000.0,
) -> dict[str, object]:
    """Measure the rules on detections inside ``window_days`` of the merger.

    The window starts at the merger. ``judgeable`` uses the frozen rule
    in ``judge.py``: two detections in one band at signal-to-noise of
    5 or more and at least 0.5 days apart, plus one night with both g
    and r at that signal-to-noise. The filter cuts are not part of it.
    ``merger_mjd`` sets which visits share a night via
    ``floor(MJD - 0.5)``. The default is only for a table that has no
    merger time of its own.

    A reddening result of ``skipped`` means there were not two colors
    to fit. That does not reject the object. Fade and color use
    ``not_enough_data`` instead of skipping: too little data fails
    those two rules. ``below_cut`` means the number was measured and
    it was on the wrong side of the placeholder cut.
    """
    visible = light_curve_for_the_filter(observed, window_days)
    if visible.empty:
        return _blank_diagnosis(window_days)

    rates = fade_rates(visible)
    measured_rates = [
        rate.mag_per_day for rate in rates.values() if rate.mag_per_day is not None
    ]
    fade_rate = float(max(measured_rates)) if measured_rates else float("nan")
    if not np.isfinite(fade_rate):
        fade_result = "not_enough_data"
    elif fade_rate > THRESHOLDS.fade_faster_than_mag_per_day:
        fade_result = "pass"
    else:
        fade_result = "below_cut"

    peak = peak_time_days(visible, "r")
    color_value = float("nan")
    if peak is not None:
        color = g_minus_r_at_time(
            visible,
            peak + THRESHOLDS.color_days_after_peak,
            max_gap_days=THRESHOLDS.max_color_interpolation_gap_days,
        )
        if color.g_minus_r is not None:
            color_value = float(color.g_minus_r)
    if not np.isfinite(color_value):
        color_result = "not_enough_data"
    elif color_value > THRESHOLDS.redder_than_g_minus_r:
        color_result = "pass"
    else:
        color_result = "below_cut"

    change = g_minus_r_rise(visible)
    rise_value = float(change.rise) if change.rise is not None else float("nan")
    if not np.isfinite(rise_value):
        reddening_result = "skipped"
    elif rise_value > THRESHOLDS.min_g_minus_r_rise:
        reddening_result = "pass"
    else:
        reddening_result = "below_cut"

    # The full table, not the already-cut window: is_judgeable drops
    # every row after `window_days` itself, including non-detections.
    judgeable = is_judgeable(observed, window_days, merger_mjd=merger_mjd)
    verdict = is_kilonova_candidate(visible)
    return {
        "n_detections_in_window": int(len(visible)),
        "window_days": window_days,
        "judgeable": judgeable,
        "is_candidate": bool(verdict.is_candidate),
        "fade_result": fade_result,
        "color_result": color_result,
        "reddening_result": reddening_result,
        "fade_rate_mag_per_day": fade_rate,
        "g_minus_r": color_value,
        "reddening_rise_mag": rise_value,
    }


def _candidate_in_window(observed: pd.DataFrame, window_days: float) -> bool:
    """Whether the filter would catch the fake inside a different window.

    Used for the 5-day context number. It is not the official recall.
    The window is still counted from the merger.
    """
    visible = light_curve_for_the_filter(observed, window_days)
    if visible.empty:
        return False
    return bool(is_kilonova_candidate(visible).is_candidate)


def _abort_if_model_is_broken(n_failed: int, n_total: int, limit: float) -> None:
    """Stop if the model failed for more than ``limit`` of the fakes.

    A broken model must not be reported as a low recall. Each failure
    has already been written to the log with its reason.
    """
    if n_total <= 0:
        return
    fraction = n_failed / n_total
    if fraction > limit:
        raise RuntimeError(
            "The light-curve model failed for "
            f"{n_failed} of {n_total} fakes "
            f"({100.0 * fraction:.0f}%), which is more than "
            f"{100.0 * limit:.0f}%. "
            "That is a broken model, not a survey that missed them. "
            "The recall was not computed. "
            "Each failure, and its reason, is in the log just above."
        )


def simulate_event(
    event: FakeEvent,
    survey: SurveySettings,
    rng: np.random.Generator,
    model: LightCurveModel,
) -> tuple[pd.DataFrame, str | None, dict[str, float]]:
    """Sample one fake.

    Returns ``(detections, error_reason, peak_magnitude)``.

    ``model`` is ``(times, bands, event) -> magnitudes``. The stub in
    the tests is one. ``redback_magnitudes`` is the other.

    ``error_reason`` is ``None`` when the model ran. A model that raises
    one of the expected failures, or that returns the wrong number of
    magnitudes, is logged and returned as an error reason. That fake is
    not called "never detected": the survey was never asked. If most
    fakes fail this way, ``build_recall_set`` stops the run.

    ``peak_magnitude`` is the brightest true magnitude at a visit in
    each band, before the depth cut and before noise. Brightest means
    the smallest magnitude. It is NaN when that band was not visited or
    the model failed.
    """
    empty_peaks = {band: float("nan") for band in ("g", "r", "i")}
    times, bands = survey_visits(rng, survey)
    if len(times) == 0:
        return _empty_curve(), None, empty_peaks
    try:
        true_magnitudes = np.asarray(model(times, bands, event), dtype=float)
    except _MODEL_FAILURES as error:
        reason = f"{type(error).__name__}: {error}"
        logger.warning(
            "Fake %s: the light-curve model failed (%s).",
            event.object_id,
            reason,
        )
        return _empty_curve(), reason, empty_peaks
    if true_magnitudes.shape != (len(times),):
        reason = (
            f"the model returned {true_magnitudes.shape} values for {len(times)} visits"
        )
        logger.warning(
            "Fake %s: the light-curve model failed (%s).",
            event.object_id,
            reason,
        )
        return _empty_curve(), reason, empty_peaks
    observed = observe_visits(
        times,
        bands,
        true_magnitudes,
        survey=survey,
        rng=rng,
    )
    return observed, None, _peak_magnitudes(bands, true_magnitudes)


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
    model: LightCurveModel,
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
        observed, error_reason, peaks = simulate_event(event, survey, event_rng, model)
        model_failed = error_reason is not None
        if model_failed:
            # A broken model is not "the survey never saw it".
            never = False
            diagnosis = _blank_diagnosis(survey.catch_within_days)
            caught_later = False
        else:
            never = observed.empty
            diagnosis = diagnose_catch_window(
                observed,
                survey.catch_within_days,
                merger_mjd=event.merger_mjd,
            )
            caught_later = _candidate_in_window(observed, CONTEXT_WINDOW_DAYS)
        event_rows.append(
            _event_row(
                event,
                never_detected=never,
                is_candidate=bool(diagnosis["is_candidate"]),
                n_detections_in_window=int(diagnosis["n_detections_in_window"]),
                window_days=float(diagnosis["window_days"]),
                model_failed=model_failed,
                model_error_reason="" if error_reason is None else error_reason,
                judgeable=bool(diagnosis["judgeable"]),
                fade_result=str(diagnosis["fade_result"]),
                color_result=str(diagnosis["color_result"]),
                reddening_result=str(diagnosis["reddening_result"]),
                fade_rate_mag_per_day=diagnosis["fade_rate_mag_per_day"],
                g_minus_r=diagnosis["g_minus_r"],
                reddening_rise_mag=diagnosis["reddening_rise_mag"],
                is_candidate_5day=caught_later,
                peak_mag_g=peaks["g"],
                peak_mag_r=peaks["r"],
                peak_mag_i=peaks["i"],
            )
        )
        if not observed.empty:
            frame = observed.copy()
            frame.insert(0, "object_id", event.object_id)
            frame["within_catch_window"] = times_in_window(
                frame["time_days"].to_numpy(),
                survey.catch_within_days,
            )
            curve_frames.append(frame)
    events_table = pd.DataFrame(event_rows)
    _abort_if_model_is_broken(
        int(events_table["model_failed"].sum()),
        len(events_table),
        survey.model_error_abort_fraction,
    )
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
    model: LightCurveModel | None = None,
    survey: SurveySettings | None = None,
    cadence: str = CADENCE_WFD,
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
    sky = survey if survey is not None else SurveySettings(cadence=cadence)
    destination = Path(output_dir)
    set_reports: list[tuple[RecallNumbers, pd.DataFrame]] = []
    for set_name, set_seed in _seeds_for(which, seed):
        population = PopulationSettings(count=set_size, seed=set_seed)
        events, curves, numbers = build_recall_set(
            set_name=set_name,
            population=population,
            survey=sky,
            model=model,
        )
        write_recall_tables(destination / set_name, events, curves)
        set_reports.append((numbers, events))

    sanity_text, sanity_candidate = _write_sanity_fake(destination, sky, model)
    text = format_recall_report(
        set_reports,
        sanity_text=sanity_text,
        sanity_is_candidate=sanity_candidate,
        output_dir=destination,
        cadence=sky.cadence,
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.txt").write_text(text + "\n", encoding="utf-8")
    return text


def format_recall_report(
    sets: list[tuple[RecallNumbers, pd.DataFrame]],
    *,
    sanity_text: str,
    sanity_is_candidate: bool,
    output_dir: Path,
    cadence: str = CADENCE_WFD,
) -> str:
    """Turn the scores into the paragraphs ``kilohunt recall`` prints."""
    by_name = {item.set_name: (item, events) for item, events in sets}
    lines: list[str] = []
    lines.append(
        "Recall uses the placeholder cuts in config.py. They have not been tuned."
    )
    lines.append(TNS_STATUS_LINE)
    lines.append(EARLY_GAP_MODEL_LIMIT)
    lines.append(
        "Judgeable (a) is the official section 1 rule. "
        "Judgeable (b) also needs a usable g minus r at r-peak + 1 day, "
        "using the 1.0-day interpolation limit. (b) is not official. "
        "Sage has to approve it before it can replace (a)."
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
    lines.append(_cadence_sentence(cadence))
    lines.append(
        "Detection limits are the single-visit 5-sigma depths from "
        "Ivezić et al. 2019, Table 2: "
        f"g {LIMITING_MAGNITUDE['g']:.2f}, "
        f"r {LIMITING_MAGNITUDE['r']:.2f}, "
        f"i {LIMITING_MAGNITUDE['i']:.2f}."
    )
    lines.append(
        "Within 3 days is counted from the merger, not from the first "
        "detection. A point on day 4 is too late even if the survey had "
        "not seen the object before."
    )
    lines.append("")
    if TEST_SET in by_name:
        numbers, events = by_name[TEST_SET]
        lines.extend(_set_paragraphs(numbers, official=True, events=events))
    if TUNING_SET in by_name:
        if TEST_SET in by_name:
            lines.append("")
        numbers, events = by_name[TUNING_SET]
        lines.extend(
            _set_paragraphs(
                numbers,
                official=TEST_SET not in by_name,
                events=events,
            )
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


def _set_paragraphs(
    numbers: RecallNumbers,
    *,
    official: bool,
    events: pd.DataFrame | None = None,
) -> list[str]:
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
            f"Light-curve model errors: {numbers.n_model_failed}. "
            "These are not counted as never detected. Each one was logged "
            "with its reason. The run stops if more than "
            f"{100.0 * MODEL_ERROR_ABORT_FRACTION:.0f}% of a set fails this way."
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
    if events is not None:
        lines.extend(_diagnostic_lines(events))
    return lines


def _cadence_sentence(cadence: str) -> str:
    """One sentence naming which visit pattern this report used."""
    if cadence == CADENCE_WFD:
        return (
            "Cadence: wide-fast-deep style, which is the default. "
            "A field is visited every 2 to 4 nights. Each visit is two "
            "exposures about 30 minutes apart, in two of g, r, and i. "
            "That sequence starts before the merger. The merger is dropped "
            "in at a random time, and only later visits are kept."
        )
    return (
        "Cadence: simple, not the default. Each of g, r, and i has its "
        "own clock, with at least one day between visits of that band "
        "plus a random extra wait."
    )


def _diagnostic_lines(events: pd.DataFrame) -> list[str]:
    """The breakdown Science Lead asked for. Empty if the columns are absent."""
    if "judgeable" not in events.columns or events.empty:
        return []
    lines: list[str] = [""]
    total = int(len(events))
    judgeable = events["judgeable"].astype(bool)
    n_judge = int(judgeable.sum())
    caught_judge = int((judgeable & events["is_candidate"].astype(bool)).sum())
    lines.append(
        f"Judgeable fakes: {n_judge} of {total}. "
        "A fake is judgeable by day 3 when it has two detections in the "
        "same band at signal-to-noise of 5 or more, at least 0.5 days "
        "apart, and one night with both g and r at that signal-to-noise. "
        "The filter cuts are not part of this. A night is floor(MJD - 0.5), "
        "which changes at noon UTC."
    )
    if n_judge:
        low, high = wilson_interval(caught_judge, n_judge)
        percent = 100.0 * caught_judge / n_judge
        lines.append(
            f"Judgeable recall: {caught_judge} of {n_judge} ({percent:.1f}%). "
            "95% confidence interval: "
            f"{100.0 * low:.1f}% to {100.0 * high:.1f}%."
        )
    else:
        lines.append("Judgeable recall: no fake had enough visits to be judged.")
    lines.append(
        "The official recall above counts every fake, including ones the "
        "survey barely visited. If the judgeable recall is high and the "
        "official recall is low, the visit pattern is the limit, not the "
        "filter cuts."
    )
    lines.append(
        "How the rules came out, for fakes with at least one detection "
        "in the 3 days after the merger:"
    )
    shown = events["n_detections_in_window"] > 0
    n_shown = int(shown.sum())
    lines.append(f"  Shown to the filter: {n_shown} of {total}.")
    for name, column in (
        ("Fade", "fade_result"),
        ("Color", "color_result"),
        ("Reddening", "reddening_result"),
    ):
        lines.append(f"  {name}: {_rule_tally(events.loc[shown, column])}")
    lines.append(
        "  not_enough_data means the rule failed because it could not be "
        "measured. below_cut means it was measured and landed on the wrong "
        "side of the placeholder cut. skipped (reddening only) means there "
        "were not two colors to fit, which does not by itself reject the object."
    )
    if "is_candidate_5day" in events.columns and total:
        caught_later = int(events["is_candidate_5day"].astype(bool).sum())
        low, high = wilson_interval(caught_later, total)
        percent = 100.0 * caught_later / total
        lines.append(
            "If the window were 5 days after the merger instead of 3, "
            f"the filter would catch {caught_later} of {total} ({percent:.1f}%). "
            "95% confidence interval: "
            f"{100.0 * low:.1f}% to {100.0 * high:.1f}%. "
            "That is context only. It is not the official recall."
        )
    lines.extend(_quantile_lines(events))
    lines.extend(_detection_bin_lines(events))
    return lines


def _rule_tally(results: pd.Series) -> str:
    """One line of pass / not enough data / below the cut / skipped."""

    def count(label: str) -> int:
        return int((results == label).sum())

    return (
        f"{count('pass')} pass, "
        f"{count('not_enough_data')} not enough data, "
        f"{count('below_cut')} below the cut, "
        f"{count('skipped')} skipped"
    )


def _quantile_lines(events: pd.DataFrame) -> list[str]:
    """10/50/90 percent points for the measured numbers and the peaks."""
    lines = ["Measured numbers, as the 10th, 50th, and 90th percent points:"]
    usable = ~events["model_failed"].astype(bool)
    detections = events.loc[usable, "n_detections_in_window"].to_numpy(dtype=float)
    lines.append(
        "  Detections per fake in the first 3 days after the merger "
        f"(including zeros): {_format_quantiles(detections)}."
    )
    lines.append(
        "  Fastest fade, in mag/day, where a fade could be measured: "
        f"{_format_quantiles(events['fade_rate_mag_per_day'].to_numpy(dtype=float))}."
    )
    lines.append(
        "  g minus r one day after the r-band peak, where it could be measured: "
        f"{_format_quantiles(events['g_minus_r'].to_numpy(dtype=float))}."
    )
    lines.append(
        "True peak magnitude at a visit, before faint points are dropped "
        "(a smaller number is brighter):"
    )
    for band in ("g", "r", "i"):
        column = events[f"peak_mag_{band}"].to_numpy(dtype=float)
        lines.append(f"  {band}: {_format_quantiles(column)}.")
    near = events.loc[
        usable & (events["distance_mpc"] >= 250.0) & (events["distance_mpc"] < 350.0)
    ]
    lines.append(
        "  Same peaks for fakes with luminosity distance 250 to 350 Mpc "
        "(a 2017-like kilonova at 300 Mpc is expected near 21.5 to 22):"
    )
    if near.empty:
        lines.append("    none in this distance slice.")
    else:
        for band in ("g", "r", "i"):
            lines.append(
                f"    {band}: "
                f"{_format_quantiles(near[f'peak_mag_{band}'].to_numpy(dtype=float))}."
            )
    return lines


def _format_quantiles(values: np.ndarray) -> str:
    """``10% / 50% / 90%`` or a short note when there is nothing to summarize."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return "not measured"
    low, middle, high = np.quantile(clean, [0.1, 0.5, 0.9])
    return f"{low:.2f} / {middle:.2f} / {high:.2f} (n={len(clean)})"


def _detection_bin_lines(events: pd.DataFrame) -> list[str]:
    """Recall by how many detections fell inside the 3-day window."""
    usable = events.loc[~events["model_failed"].astype(bool)]
    lines = [
        "Recall by number of detections in the first 3 days after the merger "
        "(never detected stay in the 0 bin; model errors are left out):"
    ]
    order = ("0", "1", "2", "3-4", "5+")
    labels = usable["n_detections_in_window"].map(_detection_bin_label)
    for label in order:
        chosen = usable.loc[labels == label]
        total = int(len(chosen))
        if total == 0:
            lines.append(f"  {label} detections: none")
            continue
        caught = int(chosen["is_candidate"].astype(bool).sum())
        percent = 100.0 * caught / total
        noun = "detection" if label == "1" else "detections"
        lines.append(f"  {label} {noun}: {caught}/{total} ({percent:.1f}%)")
    return lines


def _detection_bin_label(count: int) -> str:
    """Bucket a detection count as 0, 1, 2, 3-4, or 5+."""
    if count <= 2:
        return str(int(count))
    if count <= 4:
        return "3-4"
    return "5+"


def _write_sanity_fake(
    destination: Path,
    survey: SurveySettings,
    model: LightCurveModel,
) -> tuple[str, bool]:
    """Observe the AT2017gfo-like fake and return its verdict text."""
    event = at2017gfo_like_event()
    # 2017 is fixed so this one object does not change when the set seed changes.
    rng = np.random.default_rng(2017)
    observed, error_reason, peaks = simulate_event(event, survey, rng, model)
    model_failed = error_reason is not None
    if model_failed:
        text = f"The light-curve model failed ({error_reason}), so there is no verdict."
        diagnosis = _blank_diagnosis(survey.catch_within_days)
        never = False
        caught_later = False
    else:
        diagnosis = diagnose_catch_window(
            observed,
            survey.catch_within_days,
            merger_mjd=event.merger_mjd,
        )
        caught_later = _candidate_in_window(observed, CONTEXT_WINDOW_DAYS)
        never = observed.empty
        if int(diagnosis["n_detections_in_window"]) == 0:
            text = (
                "No detection within "
                f"{survey.catch_within_days:.0f} days of the merger, "
                "so the filter was not shown a light curve."
            )
        else:
            visible = light_curve_for_the_filter(observed, survey.catch_within_days)
            text = format_verdict(is_kilonova_candidate(visible))
    frame = observed.copy()
    if not frame.empty:
        frame.insert(0, "object_id", event.object_id)
        frame["within_catch_window"] = times_in_window(
            frame["time_days"].to_numpy(),
            survey.catch_within_days,
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
        never_detected=never,
        is_candidate=bool(diagnosis["is_candidate"]),
        n_detections_in_window=int(diagnosis["n_detections_in_window"]),
        window_days=float(diagnosis["window_days"]),
        model_failed=model_failed,
        model_error_reason="" if error_reason is None else error_reason,
        judgeable=bool(diagnosis["judgeable"]),
        fade_result=str(diagnosis["fade_result"]),
        color_result=str(diagnosis["color_result"]),
        reddening_result=str(diagnosis["reddening_result"]),
        fade_rate_mag_per_day=diagnosis["fade_rate_mag_per_day"],
        g_minus_r=diagnosis["g_minus_r"],
        reddening_rise_mag=diagnosis["reddening_rise_mag"],
        is_candidate_5day=caught_later,
        peak_mag_g=peaks["g"],
        peak_mag_r=peaks["r"],
        peak_mag_i=peaks["i"],
    )
    write_recall_tables(destination / "at2017gfo", pd.DataFrame([row]), frame)
    return text, bool(diagnosis["is_candidate"])


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


def _event_row(event: FakeEvent, **scores: object) -> dict[str, object]:
    """One fake as a flat row, plus the score columns in ``scores``."""
    row: dict[str, object] = {
        field.name: getattr(event, field.name) for field in fields(event)
    }
    # Ejecta first, then the score, so a score column cannot be overwritten
    # by a mass. The masses are what the light curve was built from.
    row.update(ejecta_masses_and_speeds(event))
    row.update(scores)
    return row


def _empty_curve() -> pd.DataFrame:
    """A detection table with the right columns and no rows."""
    return pd.DataFrame(columns=["time_days", "filter", "magnitude", "magnitude_error"])

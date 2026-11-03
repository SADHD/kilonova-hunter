"""Distance, sky position, redshift, and Milky Way dust for one fake.

Two things changed from the first recall harness.

Redshift used to come from a small numpy integral written to match
Planck 2018. The evaluation harness now asks ``astropy.cosmology.Planck18``
for the luminosity distance, the comoving distance, and the comoving
volume, and reads the redshift back off that grid. redback then applies
that redshift to the clock and to the wavelengths. The lines that do
so are listed on ``REDSHIFT_APPLIED_IN_REDBACK``.

Dust used to be an exponential E(B-V) with mean 0.05, capped at 0.5,
times the Schlafly and Finkbeiner 2011 coefficients (g 3.303, r 2.285,
i 1.698). That helper is still ``recall.sample_ebv`` and
``recall.dust_dimming_mag``, and the older ``run_recall`` path still
uses it so the offline tests do not download a map. The evaluation
harness does not. It reads the SFD dust map at a random sky position,
multiplies by 0.86 (Schlafly and Finkbeiner 2011), and turns that
E(B-V) into magnitudes with the Fitzpatrick 1999 law at R_V = 3.1.

Sky positions are uniform on the sphere inside the declination strip
-70 degrees < dec < +12 degrees, and outside the Galactic plane
(|b| > 15 degrees). That is a simple stand-in for the Rubin
wide-fast-deep footprint. It is not the official footprint map.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18

# Where redback 1.20.0 applies the redshift we pass in.
# kilonova_models.three_component_kilonova_model:
#   line 1141  luminosity distance from the cosmology (Planck18 when we pass it)
#   line 1144  source-frame time = observer time / (1 + z)   (time dilation)
# utils.calc_kcorrected_properties:
#   lines 388-389  time / (1 + z) again on the spectrum grid, and
#                  frequency * (1 + z) so the bandpass is shifted into
#                  the source frame (the wavelength redshift)
# The same pattern is in two_component_bns_ejecta_relation, which calls
# two_component_kilonova_model. File:
# redback/transient_models/kilonova_models.py and redback/utils.py,
# version 1.20.0.
REDSHIFT_APPLIED_IN_REDBACK = (
    "redback/transient_models/kilonova_models.py:1141",
    "redback/transient_models/kilonova_models.py:1144",
    "redback/utils.py:388",
    "redback/utils.py:389",
)

# Fitzpatrick 1999, the value the science lead required.
R_V = 3.1

# Schlafly and Finkbeiner 2011: multiply the raw SFD E(B-V) by 0.86.
SFD_RECALIBRATION = 0.86

# sncosmo lsstg / lsstr / lssti effective wavelengths, in angstroms.
# Read from sncosmo 2.13.1 with the LSST total-throughput bandpasses.
# Dust is computed at this one wavelength per band.
BAND_WAVELENGTH_AA = {"g": 4826.8523, "r": 6223.2404, "i": 7545.9752}

# Declination strip used as a stand-in for the Rubin footprint.
DEC_MIN_DEG = -70.0
DEC_MAX_DEG = 12.0
# Keep sightlines out of the Galactic plane.
MIN_ABS_GALACTIC_B_DEG = 15.0

# A merger time is drawn uniformly inside this one-day window.
# Only the fractional day matters for which visits fall before noon UTC.
# 60000 is an arbitrary Modified Julian Date near the present, used so
# the stored number looks like a real MJD.
MERGER_MJD_ORIGIN = 60000.0

_PLANCK_GRID: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
_SFD_QUERY = None


def _planck_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Redshift, comoving distance, comoving volume, luminosity distance.

    Distances are megaparsecs. Volume is cubic megaparsecs. The rows
    come from ``Planck18`` itself. Reading a distance back is
    ``numpy.interp``: find where the distance sits between two rows
    and take the redshift in the same proportion. 300 Mpc is near
    redshift 0.07, so a grid out to redshift 0.2 is enough.
    """
    global _PLANCK_GRID
    if _PLANCK_GRID is None:
        redshift = np.linspace(0.0, 0.2, 4001)
        comoving = Planck18.comoving_distance(redshift).to(u.Mpc).value
        # A flat universe has comoving volume (4/3) pi times distance cubed.
        # Planck18.comoving_volume agrees with that. Using the distance
        # we just read keeps this grid on one cosmology.
        volume = (4.0 / 3.0) * np.pi * comoving**3
        luminosity = Planck18.luminosity_distance(redshift).to(u.Mpc).value
        _PLANCK_GRID = (
            np.asarray(redshift, dtype=float),
            np.asarray(comoving, dtype=float),
            np.asarray(volume, dtype=float),
            np.asarray(luminosity, dtype=float),
        )
    return _PLANCK_GRID


def redshift_and_luminosity_distance(
    comoving_distance_mpc: float,
) -> tuple[float, float]:
    """Redshift and luminosity distance for one comoving distance.

    Both numbers come from the Planck18 grid.
    """
    redshift, comoving, _volume, luminosity = _planck_grid()
    z = float(np.interp(comoving_distance_mpc, comoving, redshift))
    distance = float(np.interp(z, redshift, luminosity))
    return z, distance


def redshift_at_luminosity_distance(distance_mpc: float) -> float:
    """Planck18 redshift whose luminosity distance is ``distance_mpc``."""
    redshift, _comoving, _volume, luminosity = _planck_grid()
    return float(np.interp(distance_mpc, luminosity, redshift))


def sample_luminosity_distances(
    rng: np.random.Generator,
    count: int,
    *,
    near_mpc: float = 10.0,
    far_mpc: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw distances uniform in Planck18 comoving volume.

    ``near_mpc`` and ``far_mpc`` are luminosity distances, not comoving
    ones. The shell is the comoving volume between the redshifts that
    have those luminosity distances, so every fake we keep sits between
    10 and 300 Mpc of luminosity distance. A draw that used comoving
    edges of 10 and 300 would let the luminosity distance run a little
    past 300, and the far bin is not allowed to do that.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}.")
    redshift, _comoving, volume, luminosity = _planck_grid()
    z_near = redshift_at_luminosity_distance(near_mpc)
    z_far = redshift_at_luminosity_distance(far_mpc)
    volume_near = float(np.interp(z_near, redshift, volume))
    volume_far = float(np.interp(z_far, redshift, volume))
    drawn = rng.uniform(volume_near, volume_far, size=count)
    redshifts = np.interp(drawn, volume, redshift)
    distances = np.interp(redshifts, redshift, luminosity)
    return np.asarray(redshifts, dtype=float), np.asarray(distances, dtype=float)


def draw_merger_mjd(rng: np.random.Generator, count: int = 1) -> np.ndarray:
    """Merger times, uniform across one UTC day.

    The fractional day moves noon relative to the merger, so a
    wide-fast-deep pair is not always on the same side of the night
    boundary. The integer day is ``MERGER_MJD_ORIGIN`` plus that fraction.
    """
    fraction = rng.uniform(0.0, 1.0, size=count)
    return MERGER_MJD_ORIGIN + fraction


def draw_footprint_positions(
    rng: np.random.Generator,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right ascension, declination, and Galactic latitude, in degrees.

    Declination is drawn so the sky is uniform inside the strip: the
    sine of declination is uniform between sin(-70 deg) and sin(+12 deg).
    Positions with Galactic latitude between -15 and +15 degrees are
    thrown out and drawn again. A batch that somehow cannot fill
    ``count`` slots raises, which would mean the cuts reject everything.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}.")
    sin_low = np.sin(np.deg2rad(DEC_MIN_DEG))
    sin_high = np.sin(np.deg2rad(DEC_MAX_DEG))
    ra_kept: list[float] = []
    dec_kept: list[float] = []
    b_kept: list[float] = []
    # A hard stop so a bug in the cut cannot spin forever.
    for _ in range(1000):
        need = count - len(ra_kept)
        if need <= 0:
            break
        batch = max(need * 4, 32)
        ra = rng.uniform(0.0, 360.0, size=batch)
        dec = np.rad2deg(np.arcsin(rng.uniform(sin_low, sin_high, size=batch)))
        coordinates = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        galactic_b = coordinates.galactic.b.deg
        keep = np.abs(galactic_b) > MIN_ABS_GALACTIC_B_DEG
        ra_kept.extend(ra[keep].tolist())
        dec_kept.extend(dec[keep].tolist())
        b_kept.extend(galactic_b[keep].tolist())
    if len(ra_kept) < count:
        raise RuntimeError(
            "Could not draw sky positions inside the declination strip "
            "and outside the Galactic plane."
        )
    return (
        np.asarray(ra_kept[:count], dtype=float),
        np.asarray(dec_kept[:count], dtype=float),
        np.asarray(b_kept[:count], dtype=float),
    )


def sfd_directory() -> Path:
    """Directory that holds ``SFD_dust_4096_ngp.fits`` and the south file.

    ``KILONOVA_SFD_DIR`` overrides the default, which is ``data/dustmaps/sfd``
    inside the repository. That folder is gitignored.
    """
    override = os.environ.get("KILONOVA_SFD_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "dustmaps" / "sfd"


def sfd_map_is_available() -> bool:
    """True when both SFD pole files are on disk."""
    folder = sfd_directory()
    return (folder / "SFD_dust_4096_ngp.fits").is_file() and (
        folder / "SFD_dust_4096_sgp.fits"
    ).is_file()


def _sfd_query():
    """The dustmaps query object, loaded once."""
    global _SFD_QUERY
    if _SFD_QUERY is None:
        from dustmaps.sfd import SFDQuery

        _SFD_QUERY = SFDQuery(map_dir=str(sfd_directory()))
    return _SFD_QUERY


def ebv_from_sfd(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    query=None,
) -> np.ndarray:
    """SFD E(B-V) times 0.86, at each sky position.

    ``query`` lets a test pass a stand-in instead of the real map.
    The real query returns the raw SFD value. We scale it here.
    Negative numerical noise is lifted to zero.
    """
    coordinates = SkyCoord(
        ra=np.asarray(ra_deg, dtype=float) * u.deg,
        dec=np.asarray(dec_deg, dtype=float) * u.deg,
        frame="icrs",
    )
    if query is None:
        raw = np.asarray(_sfd_query().query(coordinates), dtype=float)
    else:
        raw = np.asarray(query(coordinates), dtype=float)
    return np.maximum(SFD_RECALIBRATION * raw, 0.0)


def fitzpatrick_extinction_mag(ebv: float, band: str) -> float:
    """Magnitudes of dimming in ``band`` for this E(B-V).

    A_V is R_V times E(B-V), with R_V fixed at 3.1. The Fitzpatrick
    1999 curve then gives A_lambda at the band's effective wavelength.
    A larger magnitude is fainter, so the caller adds this number.
    E(B-V) of zero adds nothing.
    """
    if band not in BAND_WAVELENGTH_AA:
        raise KeyError(f"no wavelength is set for band {band!r}.")
    if ebv <= 0.0:
        return 0.0
    from extinction import fitzpatrick99

    a_v = R_V * float(ebv)
    wave = np.array([BAND_WAVELENGTH_AA[band]])
    dimming = fitzpatrick99(wave, a_v, r_v=R_V, unit="aa")
    return float(dimming[0])


def extinction_for_bands(ebv: float, bands: np.ndarray) -> np.ndarray:
    """One dimming per visit, in magnitudes, following ``bands``."""
    by_band = {band: fitzpatrick_extinction_mag(ebv, band) for band in ("g", "r", "i")}
    return np.array([by_band[str(band)] for band in bands], dtype=float)


def dust_map_label() -> str:
    """The dust correction a cache must name before the test set is scored.

    SFD times 0.86 is Schlafly and Finkbeiner 2011. Magnitudes use
    Fitzpatrick 1999 at R_V = 3.1. The text is exact, so a cache from
    the old exponential draw cannot pass as this map.
    """
    return "SFD x 0.86, Fitzpatrick 1999, R_V = 3.1"


@dataclass(frozen=True)
class DustSource:
    """Where E(B-V) came from for this run.

    ``mode`` is ``"sfd"`` when the map was read, or ``"dry-run"`` when
    it was not and the old exponential draw was used instead. A dry run
    still uses the Fitzpatrick 1999 law on whatever E(B-V) it has.
    """

    mode: str
    reason: str = ""

    @property
    def is_dry_run(self) -> bool:
        return self.mode != "sfd"


def open_dust_source() -> DustSource:
    """Use the SFD map when the files are present. Otherwise say why not."""
    if sfd_map_is_available():
        try:
            _sfd_query()
        except (OSError, ValueError, ImportError, RuntimeError) as error:
            return DustSource(mode="dry-run", reason=f"{type(error).__name__}: {error}")
        return DustSource(mode="sfd", reason="")
    return DustSource(
        mode="dry-run",
        reason=(
            "The SFD files SFD_dust_4096_ngp.fits and "
            "SFD_dust_4096_sgp.fits were not in "
            f"{sfd_directory()}. Until those files are there, E(B-V) falls "
            "back to the exponential draw. The Harvard Dataverse files are "
            "doi:10.7910/DVN/EWCNL5, ids 2902687 (north) and 2902695 (south)."
        ),
    )

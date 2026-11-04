"""Instrument-bandpass residuals for AT2017gfo, including CSP for Swope.

Writes two tables:

- ``docs/at2017gfo_bandpass_residuals.csv`` — LSST and instrument residuals
- ``docs/at2017gfo_csp_minus_sdss.csv`` — CSP minus SDSS at the Swope epochs

Swope g/r/i are the CSP bandpasses ``cspg``, ``cspr``, and ``cspi``.
SkyMapper bandpasses are requested from the SVO source server. When that
download fails, those rows stay on the nearest SDSS bandpass.

    uv run python scripts/at2017gfo_bandpass_residuals.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kilonova_hunter.sky import R_V, redshift_at_luminosity_distance
from kilonova_hunter.villar import EXACT_DISTANCE_MPC, VillarEvent, exact_components

PHOTOMETRY = Path("docs/at2017gfo_photometry.csv")
RESIDUALS = Path("docs/at2017gfo_bandpass_residuals.csv")
CSP_MINUS_SDSS = Path("docs/at2017gfo_csp_minus_sdss.csv")
EBV = 0.105

EXACT_BANDS = {
    "DECam": {"g": "desg", "r": "desr", "i": "desi"},
    "PS1": {"g": "ps1::g", "r": "ps1::r", "i": "ps1::i"},
    "Swope": {"g": "cspg", "r": "cspr", "i": "cspi"},
}
NEAREST = {"g": "sdssg", "r": "sdssr", "i": "sdssi"}
LSST = {"g": "lsstg", "r": "lsstr", "i": "lssti"}
SKYMAPPER = {"g": "skymapperg", "r": "skymapperr", "i": "skymapperi"}


def _event() -> VillarEvent:
    blue, purple, red = exact_components()
    return VillarEvent(
        object_id="at2017gfo-bandpass",
        kind="at2017gfo-exact",
        distance_mpc=EXACT_DISTANCE_MPC,
        redshift=redshift_at_luminosity_distance(EXACT_DISTANCE_MPC),
        ebv=0.0,
        ra_deg=float("nan"),
        dec_deg=float("nan"),
        galactic_b_deg=float("nan"),
        merger_mjd=57982.528524,
        viewing_angle_deg=30.0,
        blue=blue,
        purple=purple,
        red=red,
    )


def _extinction(wave_aa: float) -> float:
    from extinction import fitzpatrick99

    dimming = fitzpatrick99(np.array([wave_aa]), R_V * EBV, r_v=R_V, unit="aa")
    return float(dimming[0])


def _magnitudes(times: np.ndarray, sn_band: str, event: VillarEvent) -> np.ndarray:
    from astropy.cosmology import Planck18
    from redback.transient_models.kilonova_models import three_component_kilonova_model

    from kilonova_hunter.redback_fix import corrected_diffusion

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
            bands=np.array([sn_band] * len(times)),
            output_format="magnitude",
            cosmology=Planck18,
        )
    return np.asarray(raw, dtype=float)


def _family(row: pd.Series) -> str:
    instrument = "" if pd.isna(row["instrument"]) else str(row["instrument"])
    telescope = "" if pd.isna(row["telescope"]) else str(row["telescope"])
    if instrument == "DECam":
        return "DECam"
    if instrument == "PS1" or telescope == "PAN-STARRS":
        return "PS1"
    if instrument == "Swope" or telescope == "Swope":
        return "Swope"
    if instrument.lower() == "skymapper" or telescope.lower() == "skymapper":
        return "SkyMapper"
    return "other"


_SKYMAPPER_ERROR: dict[str, str] = {}


def _bandpass(family: str, band: str, failures: list[str]) -> tuple[str, str]:
    """Return (sncosmo name, match kind). ``match`` is exact or nearest."""
    if family in EXACT_BANDS:
        return EXACT_BANDS[family][band], "exact"
    if family == "SkyMapper":
        name = SKYMAPPER[band]
        if name not in _SKYMAPPER_ERROR:
            try:
                import sncosmo

                sncosmo.get_bandpass(name)
                _SKYMAPPER_ERROR[name] = ""
            except Exception as exc:
                _SKYMAPPER_ERROR[name] = f"{name}: {type(exc).__name__}: {exc}"
        if _SKYMAPPER_ERROR[name]:
            failures.append(_SKYMAPPER_ERROR[name])
            return NEAREST[band], "nearest"
        return name, "exact"
    return NEAREST[band], "nearest"


def _evaluate(
    frame: pd.DataFrame,
    rows: pd.Index,
    sn_band: str,
    model_column: str,
    ext_column: str,
    event: VillarEvent,
) -> None:
    import sncosmo

    times = frame.loc[rows, "time_days"].to_numpy(dtype=float)
    order = np.argsort(times, kind="mergesort")
    model = np.empty(len(times), dtype=float)
    model[order] = _magnitudes(times[order], sn_band, event)
    frame.loc[rows, model_column] = model
    bandpass = sncosmo.get_bandpass(sn_band)
    frame.loc[rows, ext_column] = _extinction(float(bandpass.wave_eff))


def main() -> None:
    import matplotlib

    matplotlib.rcParams["text.usetex"] = False
    points = pd.read_csv(PHOTOMETRY, comment="#")
    points = points.loc[points["system"].astype(str).str.upper() == "AB"].copy()
    event = _event()
    failures: list[str] = []
    families = points.apply(_family, axis=1)
    bands = []
    kinds = []
    for family, band in zip(families, points["band"].astype(str), strict=True):
        name, kind = _bandpass(family, band, failures)
        bands.append(name)
        kinds.append(kind)
    points["_family"] = families.to_numpy()
    points["_sn"] = bands
    points["_match"] = kinds
    points["_lsst"] = [LSST[band] for band in points["band"].astype(str)]
    for column in ("lsst_model", "inst_model", "lsst_model_ext", "inst_model_ext"):
        points[column] = np.nan
    for sn_band in sorted(set(points["_lsst"])):
        chosen = points.index[points["_lsst"] == sn_band]
        _evaluate(points, chosen, sn_band, "lsst_model", "lsst_model_ext", event)
    for sn_band in sorted(set(points["_sn"])):
        chosen = points.index[points["_sn"] == sn_band]
        _evaluate(points, chosen, sn_band, "inst_model", "inst_model_ext", event)
    observed = points["magnitude"].to_numpy(dtype=float)
    points["lsst_resid"] = points["lsst_model"] - (observed - points["lsst_model_ext"])
    points["inst_resid"] = points["inst_model"] - (observed - points["inst_model_ext"])
    out = pd.DataFrame(
        {
            "time_days": points["time_days"],
            "band": points["band"],
            "instrument": points["instrument"],
            "telescope": points["telescope"],
            "system": points["system"],
            "obs": points["magnitude"],
            "lsst_model": points["lsst_model"],
            "lsst_resid": points["lsst_resid"],
            "inst_band": points["_sn"],
            "match": points["_match"],
            "inst_model": points["inst_model"],
            "inst_resid": points["inst_resid"],
        }
    )
    RESIDUALS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESIDUALS, index=False)
    swope = points.loc[points["_family"] == "Swope"].copy()
    sdss_name = [NEAREST[band] for band in swope["band"].astype(str)]
    swope["_sdss"] = sdss_name
    for sn_band in sorted(set(sdss_name)):
        chosen = swope.index[swope["_sdss"] == sn_band]
        times = swope.loc[chosen, "time_days"].to_numpy(dtype=float)
        order = np.argsort(times, kind="mergesort")
        model = np.empty(len(times), dtype=float)
        model[order] = _magnitudes(times[order], sn_band, event)
        swope.loc[chosen, "_sdss_model"] = model
    comparison = pd.DataFrame(
        {
            "time_days": swope["time_days"],
            "band": swope["band"],
            "csp_band": swope["_sn"],
            "sdss_band": swope["_sdss"],
            "csp_minus_sdss": swope["inst_model"] - swope["_sdss_model"],
        }
    )
    comparison.to_csv(CSP_MINUS_SDSS, index=False)
    print(f"wrote {RESIDUALS} ({len(out)} rows)")
    print(f"wrote {CSP_MINUS_SDSS}")
    for band in ("g", "r", "i"):
        diff = comparison.loc[comparison["band"] == band, "csp_minus_sdss"]
        print(f"CSP minus SDSS {band}: {diff.min():+.3f} to {diff.max():+.3f} mag")
    if failures:
        print("SkyMapper download failed:")
        for line in sorted(set(failures)):
            print(f"  {line}")
    _print_windows(out)


def _print_windows(frame: pd.DataFrame) -> None:
    windows = (
        (0.4, 0.8),
        (0.8, 1.5),
        (1.5, 3.0),
        (3.0, 6.0),
        (6.0, np.inf),
    )
    for label, column in (("LSST", "lsst_resid"), ("instrument", "inst_resid")):
        print(label)
        for low, high in windows:
            chunk = frame.loc[(frame["time_days"] >= low) & (frame["time_days"] < high)]
            for band in ("g", "r", "i"):
                values = chunk.loc[chunk["band"] == band, column]
                if len(values) == 0:
                    continue
                mean = float(values.mean())
                print(f"  {low:g}-{high:g} {band}: {mean:+.3f} n={len(values)}")


if __name__ == "__main__":
    main()

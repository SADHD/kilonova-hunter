"""ZTF alerts from Fink, used as a proxy for a Rubin night.

The public API does not hand out a raw night of every alert. It hands
out the latest alerts of one Fink class, capped at ``n`` per request.
This module asks for the class ``Unknown``, one calendar day at a time,
and labels the result ``ZTF proxy for Rubin``.

A ZTF night rolls over at local noon at Palomar. Palomar standard time
is UTC-8, so local noon is 20:00 UTC. The night index is

    floor(MJD - 20/24)

which changes at 20:00 UTC. Pacific Daylight Time would put noon at
19:00 UTC. The offset stays fixed at 20:00 so daylight saving never
splits one observing night across two indices.

Simulated Rubin fakes do not use this formula. They use
``floor(MJD - 0.5)``, which changes at noon UTC. That lives in
``judge.rubin_night``.

Tuning and test slices are different nights. Nights are sorted, and
even positions go to tuning, odd positions to test. No night is in both.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from kilonova_hunter.config import Thresholds, resolve_thresholds
from kilonova_hunter.filter import is_kilonova_candidate

# Fink's public class search. Documented at
# https://doc.ztf.fink-broker.org/services/search/classsearch/
FINK_LATESTS_URL = "https://api.ztf.fink-portal.org/api/v1/latests"

# Class the public endpoint will return without a special account.
# "Unknown" is Fink's label for alerts it has not classified further.
FINK_CLASS = "Unknown"

# How many alerts to ask for on one calendar day. The server caps this.
FINK_N_PER_DAY = 1000

# Columns we ask for. fid 1, 2, 3 are ZTF g, r, i.
FINK_COLUMNS = "i:objectId,i:jd,i:fid,i:magpsf,i:sigmapsf,i:ra,i:dec"

SLICE_LABEL = "ZTF proxy for Rubin"

# Local noon at Palomar in Pacific Standard Time, as a fraction of a day.
# 20:00 UTC is 20/24.
ZTF_NOON_UTC_DAYS = 20.0 / 24.0

# ZTF filter id to the band name the kilonova rules use.
FID_TO_BAND = {1: "g", 2: "r", 3: "i"}

# JD and MJD differ by this constant. MJD = JD - 2400000.5.
_JD_MINUS_MJD = 2_400_000.5


def ztf_night(mjd: np.ndarray | float) -> np.ndarray:
    """Night index for a ZTF alert. Changes at 20:00 UTC."""
    values = np.asarray(mjd, dtype=float)
    return np.floor(values - ZTF_NOON_UTC_DAYS)


def jd_to_mjd(jd: np.ndarray | float) -> np.ndarray:
    """Modified Julian Date from a Julian Date."""
    return np.asarray(jd, dtype=float) - _JD_MINUS_MJD


def _post_json(url: str, payload: dict, timeout: float = 120.0) -> list:
    """POST JSON and parse a list. A non-list body is an API error."""
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    parsed = json.loads(raw.decode("utf-8"))
    if isinstance(parsed, dict) and "message" in parsed:
        raise RuntimeError(f"Fink API error: {parsed['message']}")
    if not isinstance(parsed, list):
        raise RuntimeError(f"Fink API returned {type(parsed).__name__}, not a list.")
    return parsed


def fetch_class_between(
    start: date,
    stop: date,
    *,
    class_name: str = FINK_CLASS,
    n_per_day: int = FINK_N_PER_DAY,
) -> pd.DataFrame:
    """Latest ``class_name`` alerts for each calendar day from start to stop.

    ``stop`` is included. Each day is its own request because one request
    over many days returns only the latest ``n`` alerts, which can all
    sit on the last night.
    """
    frames: list[pd.DataFrame] = []
    day = start
    while day <= stop:
        next_day = day + timedelta(days=1)
        payload = {
            "class": class_name,
            "n": str(n_per_day),
            "startdate": day.isoformat(),
            "stopdate": next_day.isoformat(),
            "columns": FINK_COLUMNS,
        }
        rows = _post_json(FINK_LATESTS_URL, payload)
        if rows:
            frame = pd.DataFrame(rows)
            frame["query_date"] = day.isoformat()
            frames.append(frame)
        day = next_day
    if not frames:
        return pd.DataFrame()
    alerts = pd.concat(frames, ignore_index=True)
    alerts = alerts.drop_duplicates(subset=["i:objectId", "i:jd"])
    alerts["mjd"] = jd_to_mjd(alerts["i:jd"].to_numpy(dtype=float))
    alerts["night"] = ztf_night(alerts["mjd"].to_numpy(dtype=float)).astype(int)
    return alerts


def split_nights(alerts: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Even positions to tuning, odd positions to test, after sorting.

    The two lists do not share a night.
    """
    if alerts.empty or "night" not in alerts.columns:
        return [], []
    nights = sorted(int(night) for night in alerts["night"].unique())
    tuning = nights[0::2]
    test = nights[1::2]
    return tuning, test


def slice_summary(alerts: pd.DataFrame, nights: list[int]) -> dict:
    """Night list, alert count, and the count on each night."""
    if not nights:
        return {"nights": [], "n_nights": 0, "n_alerts": 0, "alerts_per_night": {}}
    chosen = alerts.loc[alerts["night"].isin(nights)]
    counts = {
        str(int(night)): int((chosen["night"] == night).sum()) for night in nights
    }
    return {
        "nights": [int(night) for night in nights],
        "n_nights": len(nights),
        "n_alerts": int(len(chosen)),
        "alerts_per_night": counts,
    }


def build_slice_document(
    alerts: pd.DataFrame,
    *,
    start: date,
    stop: date,
    class_name: str = FINK_CLASS,
    n_per_day: int = FINK_N_PER_DAY,
) -> dict:
    """The frozen description of the two slices. This is what gets committed."""
    tuning, test = split_nights(alerts)
    return {
        "label": SLICE_LABEL,
        "query": {
            "endpoint": FINK_LATESTS_URL,
            "method": "POST",
            "parameters": {
                "class": class_name,
                "n": str(n_per_day),
                "columns": FINK_COLUMNS,
                "startdate": "one calendar day at a time, from the date range",
                "stopdate": "the following calendar day",
            },
        },
        "date_range": {"start": start.isoformat(), "stop": stop.isoformat()},
        "night_formula": (
            "floor(MJD - 20/24). The index changes at 20:00 UTC, which is "
            "local noon at Palomar in Pacific Standard Time (UTC-8). "
            "The offset is fixed, so daylight saving does not split a night. "
            "MJD = JD - 2400000.5."
        ),
        "rubin_fake_night_formula": (
            "floor(MJD - 0.5). The index changes at 12:00 UTC. "
            "Simulated fakes use this, not the ZTF formula."
        ),
        "split": (
            "Nights sorted by the night index. Even positions are the tuning "
            "slice. Odd positions are the test slice. No night is in both."
        ),
        "note": (
            "The public latests endpoint returns alerts Fink labeled "
            f"{class_name}, at most n per calendar day, not every ZTF alert "
            "from that night. TNS types are not in these columns. "
            "kilonova_hunter.tns.lookup_classifications is not implemented."
        ),
        "tuning": slice_summary(alerts, tuning),
        "test": slice_summary(alerts, test),
    }


def write_slice_document(document: dict, path: Path) -> None:
    """Write the frozen slice file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def light_curve_from_alerts(alerts: pd.DataFrame) -> pd.DataFrame:
    """One object's ZTF alerts as a light curve the filter can read.

    Time zero is the first alert of this object. Filter ids 1, 2, and 3
    are g, r, and i. Other filters are left out. ``i:sigmapsf`` is the
    magnitude error.
    """
    empty = pd.DataFrame(
        columns=["time_days", "filter", "magnitude", "magnitude_error"]
    )
    if alerts.empty:
        return empty
    frame = alerts.copy()
    if "mjd" not in frame.columns:
        frame["mjd"] = jd_to_mjd(frame["i:jd"].to_numpy(dtype=float))
    kept = frame.loc[frame["i:fid"].astype(int).isin(FID_TO_BAND)]
    if kept.empty:
        return empty
    start = float(frame["mjd"].min())
    return pd.DataFrame(
        {
            "time_days": kept["mjd"].to_numpy(dtype=float) - start,
            "filter": [FID_TO_BAND[int(fid)] for fid in kept["i:fid"].tolist()],
            "magnitude": kept["i:magpsf"].to_numpy(dtype=float),
            "magnitude_error": kept["i:sigmapsf"].to_numpy(dtype=float),
        }
    )


def average_flags_per_night(
    alerts: pd.DataFrame,
    thresholds: Thresholds | None = None,
) -> dict:
    """New candidates per night, with later flags of the same object separate.

    Each night is scored with only the alerts from that night and from
    earlier nights. An alert from a later night is not visible yet.

    A candidate counts on the first night it qualifies. ``n_flagged``
    is the number of those first flags, and ``flags_per_night`` is that
    number divided by the number of nights. The N = 10 per night limit
    uses this new-candidate rate. A later night that still flags the
    same object goes into ``n_repeat_flags`` and does not add to the
    limit.

    This is the plumbing the contamination limit will use. It is not
    applied to the grid until the team says to apply it.
    """
    settings = resolve_thresholds(thresholds)
    empty = {
        "n_nights": 0,
        "n_flagged": 0,
        "n_repeat_flags": 0,
        "flags_per_night": float("nan"),
        "repeat_flags_per_night": float("nan"),
    }
    if alerts.empty:
        return empty
    working = alerts.copy()
    if "night" not in working.columns:
        mjd = working["mjd"] if "mjd" in working.columns else jd_to_mjd(working["i:jd"])
        working["night"] = ztf_night(np.asarray(mjd, dtype=float)).astype(int)
    if "mjd" not in working.columns:
        working["mjd"] = jd_to_mjd(working["i:jd"].to_numpy(dtype=float))
    ordered = sorted(int(night) for night in working["night"].unique())
    already: set[str] = set()
    n_new = 0
    n_repeat = 0
    for night in ordered:
        seen = working.loc[working["night"] <= night]
        flagged = _flagged_object_ids(seen, settings)
        new_ids = [object_id for object_id in flagged if object_id not in already]
        n_new += len(new_ids)
        n_repeat += len(flagged) - len(new_ids)
        already.update(new_ids)
    nights = len(ordered)
    return {
        "n_nights": nights,
        "n_flagged": int(n_new),
        "n_repeat_flags": int(n_repeat),
        "flags_per_night": float(n_new / nights) if nights else float("nan"),
        "repeat_flags_per_night": float(n_repeat / nights) if nights else float("nan"),
    }


def _flagged_object_ids(alerts: pd.DataFrame, settings: Thresholds) -> list[str]:
    """Object ids that are candidates with the alerts available so far."""
    flagged: list[str] = []
    for object_id, group in alerts.groupby("i:objectId", sort=False):
        curve = light_curve_from_alerts(group)
        if curve.empty:
            continue
        if is_kilonova_candidate(curve, settings).is_candidate:
            flagged.append(str(object_id))
    return flagged

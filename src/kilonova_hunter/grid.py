"""Score the tuning grid, and pick a cut when the team asks for one.

The grid is the 72 combinations in ``recall_settings.json``. For each
combination, each population, and each of days 3, 5, and 7, this module
reports:

- recall among judgeable fakes, with a 95 percent Wilson interval
- the catch rate, which is the number caught divided by every fake,
  with one Wilson interval
- how many caught fakes were not judgeable (this should be zero)
- how many caught fakes had the reddening rule skipped
- the same rates inside the distance bins 10-100, 100-200, and 200-300 Mpc

Recall among judgeable fakes describes the filter. The catch rate is
the survey-level number: it counts fakes the survey never saw.

The v1 pick, used when no alert slice is allowed to choose, keeps every
combination within 5 percentage points of the best day-3 recall on
judgeable 2017-like tuning fakes, then picks the strictest. Strictest
means a higher g minus r cut, then a bigger minimum rise, then a larger
minimum gap, then a shorter window. That order lives in the JSON file,
so it can change without a code edit.

The contamination pick is also implemented. It is not applied until the
team logs the frozen ZTF slices. Among combinations that average at
most 10 flags per night on the tuning slice, it takes the highest
day-3 recall, keeps those within 5 points, then uses the same order.
If none average 10 or fewer, it says so and picks the fewest flags
among the combinations within 5 points of the best recall.

Exact AT2017gfo copies are never in these denominators.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from kilonova_hunter.config import THRESHOLDS, Thresholds
from kilonova_hunter.features import g_minus_r_rise
from kilonova_hunter.filter import evaluate_fade_rule, is_kilonova_candidate
from kilonova_hunter.judge import JUDGE_DAYS, is_candidate_judgeable, is_judgeable
from kilonova_hunter.model_grid import (
    REALISTIC_DAY7_LABEL,
    drop_after_two_component_cap,
    uses_two_component_model,
)
from kilonova_hunter.recall import (
    EARLY_GAP_MODEL_LIMIT,
    light_curve_for_the_filter,
    wilson_interval,
)
from kilonova_hunter.tns import TNS_STATUS_LINE
from kilonova_hunter.villar import EXACT_KIND

# Luminosity-distance bins, in Mpc. The last edge is 300 and it is
# inclusive, so a fake at exactly 300 Mpc is in the 200-300 bin and
# nothing past 300 is.
DISTANCE_BINS = (
    ("10-100", 10.0, 100.0, False),
    ("100-200", 100.0, 200.0, False),
    ("200-300", 200.0, 300.0, True),
)

_SETTINGS_PATH = Path(__file__).with_name("recall_settings.json")


def load_settings(path: Path | None = None) -> dict:
    """Read the grid and the pick order. Edit the JSON, not this function."""
    source = _SETTINGS_PATH if path is None else Path(path)
    settings = json.loads(source.read_text(encoding="utf-8"))
    _validate_settings(settings)
    return settings


def _validate_settings(settings: dict) -> None:
    gap = float(settings["recall_gap"])
    if not 0.0 <= gap < 1.0:
        raise ValueError(f"recall_gap must sit between 0 and 1, got {gap}.")
    names = [item["parameter"] for item in settings["strictness"]]
    grid = settings["grid"]
    if set(names) != set(grid):
        raise ValueError("the pick order and the grid must name the same cuts.")
    for combo in grid_combinations(settings):
        if combo["min_color_separation_days"] > combo["reddening_window_days"]:
            raise ValueError(
                "min_color_separation_days cannot be longer than "
                f"reddening_window_days, got {combo}."
            )


def grid_combinations(settings: dict | None = None) -> list[dict[str, float]]:
    """Every cut combination, in a stable order."""
    spec = load_settings() if settings is None else settings
    axes = spec["grid"]
    keys = (
        "redder_than_g_minus_r",
        "min_g_minus_r_rise",
        "reddening_window_days",
        "min_color_separation_days",
    )
    rows = []
    for values in product(*[axes[key] for key in keys]):
        rows.append(
            {key: float(value) for key, value in zip(keys, values, strict=True)}
        )
    return rows


def thresholds_for(combo: dict[str, float]) -> Thresholds:
    """The placeholder cuts, with the four grid numbers replaced."""
    from dataclasses import replace

    return replace(THRESHOLDS, **combo)


def strictness_key(row: dict, settings: dict) -> tuple[float, ...]:
    """Larger means stricter, following the JSON order."""
    parts: list[float] = []
    for item in settings["strictness"]:
        value = float(row[item["parameter"]])
        if item["higher_is_stricter"]:
            parts.append(value)
        else:
            parts.append(-value)
    return tuple(parts)


def _recall_of(row: dict) -> float | None:
    value = row.get("recall_judgeable")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def pick_by_recall(rows: list[dict], settings: dict | None = None) -> dict:
    """Strictest combination within 5 points of the best day-3 recall.

    ``rows`` are day-3 results for the 2017-like tuning set. A row with
    no recall (untestable, or no judgeable fakes) cannot be picked.
    The returned dict is the winning row plus ``best_recall`` and
    ``n_in_band``.
    """
    spec = load_settings() if settings is None else settings
    usable = [
        row for row in rows if _recall_of(row) is not None and not row.get("untestable")
    ]
    if not usable:
        raise RuntimeError("No day-3 recall is available to pick from.")
    best = max(_recall_of(row) for row in usable)
    assert best is not None
    gap = float(spec["recall_gap"])
    band = [row for row in usable if _recall_of(row) >= best - gap]
    winner = max(band, key=lambda row: strictness_key(row, spec))
    chosen = dict(winner)
    chosen["best_recall"] = best
    chosen["n_in_band"] = len(band)
    chosen["pick_mode"] = "recall-only"
    chosen["limit_met"] = None
    return chosen


def pick_with_flag_limit(rows: list[dict], settings: dict | None = None) -> dict:
    """Pick using the flags-per-night limit, then recall, then the order.

    If no combination averages at most ``max_flags_per_night``, the
    winner is the fewest flags among combinations within 5 points of
    the best recall, and ``limit_met`` is false.
    """
    spec = load_settings() if settings is None else settings
    usable = [
        row for row in rows if _recall_of(row) is not None and not row.get("untestable")
    ]
    if not usable:
        raise RuntimeError("No day-3 recall is available to pick from.")
    if any("flags_per_night" not in row for row in usable):
        raise ValueError("each row needs flags_per_night for this pick.")
    limit = float(spec["max_flags_per_night"])
    gap = float(spec["recall_gap"])
    meeting = [row for row in usable if float(row["flags_per_night"]) <= limit]
    if meeting:
        best = max(_recall_of(row) for row in meeting)
        assert best is not None
        band = [row for row in meeting if _recall_of(row) >= best - gap]
        winner = max(band, key=lambda row: strictness_key(row, spec))
        limit_met = True
    else:
        best = max(_recall_of(row) for row in usable)
        assert best is not None
        band = [row for row in usable if _recall_of(row) >= best - gap]
        fewest = min(float(row["flags_per_night"]) for row in band)
        tied = [row for row in band if float(row["flags_per_night"]) <= fewest + 1e-12]
        winner = max(tied, key=lambda row: strictness_key(row, spec))
        limit_met = False
    chosen = dict(winner)
    chosen["best_recall"] = best
    chosen["n_in_band"] = len(band)
    chosen["pick_mode"] = "flags-per-night"
    chosen["limit_met"] = limit_met
    chosen["flag_limit"] = limit
    return chosen


def pick_with_tuning_alerts(
    rows: list[dict],
    alerts_dir: Path | str,
    settings: dict | None = None,
) -> dict:
    """Pick with the flag limit, counting tuning-night Fink alerts only.

    The rate comes from ``ztf_proxy_alerts_tuning.csv``. Rows marked as
    the other night slice are not counted. ``ztf_proxy_alerts.csv`` is
    not opened.
    """
    from kilonova_hunter.alerts import average_flags_per_night, read_tuning_alerts

    alerts = read_tuning_alerts(alerts_dir)
    summary = average_flags_per_night(alerts)
    rated = [dict(row, flags_per_night=summary["flags_per_night"]) for row in rows]
    chosen = pick_with_flag_limit(rated, settings)
    chosen["fink_nights"] = _nights_in(alerts)
    chosen["fink_n_flagged"] = summary["n_flagged"]
    return chosen


def _nights_in(alerts: pd.DataFrame) -> list[int]:
    if alerts.empty or "night" not in alerts.columns:
        return []
    return sorted(int(night) for night in alerts["night"].unique())


def chance_wilson_lower_at_least(
    successes_total: int,
    true_rate: float,
    bound: float = 0.70,
) -> float:
    """Chance the 95 percent Wilson lower bound is at least ``bound``.

    The sample size is ``successes_total`` (the number of judgeable
    fakes). The true recall is assumed equal to ``true_rate``. The
    chance is the probability that a binomial draw of that size would
    land at or above the smallest count whose Wilson lower bound clears
    ``bound``.
    """
    total = int(successes_total)
    if total <= 0:
        return float("nan")
    rate = float(np.clip(true_rate, 0.0, 1.0))
    if wilson_interval(total, total)[0] < bound:
        return 0.0
    low = 0
    high = total
    while low < high:
        mid = (low + high) // 2
        if wilson_interval(mid, total)[0] >= bound:
            high = mid
        else:
            low = mid + 1
    minimum = low
    if rate == 1.0:
        return 1.0
    if rate == 0.0:
        return 1.0 if minimum <= 0 else 0.0
    log_p = math.log(rate)
    log_q = math.log(1.0 - rate)
    logs = [
        math.lgamma(total + 1)
        - math.lgamma(count + 1)
        - math.lgamma(total - count + 1)
        + count * log_p
        + (total - count) * log_q
        for count in range(minimum, total + 1)
    ]
    biggest = max(logs)
    return float(math.exp(biggest) * sum(math.exp(item - biggest) for item in logs))


@dataclass(frozen=True)
class GridRun:
    """What ``kilohunt recall --grid`` prints, and whether sign-off is blocked.

    ``copy_status`` is ``untestable``, ``fail``, or
    ``caught-but-not-judgeable`` when sign-off is blocked. It is
    ``None`` when nothing blocks. A nonzero caught-but-not-judgeable
    count blocks, the same as a sanity FAIL.
    """

    text: str
    sign_off_blocker: bool
    copy_status: str | None
    block_reasons: tuple[str, ...] = ()


def copy_sign_off(copies: dict) -> tuple[bool, str | None]:
    """Whether the exact-copy verdict blocks sign-off, and which status.

    Definition (a) is the official verdict. Definition (b) blocks
    sign-off only when the draw stopped on (b). A (b) count from a
    draw that stopped on (a) is indicative, not a 12(16) result.
    """
    stopping = str(copies.get("stopping_definition", "a"))
    official = copies.get("official") or {}
    candidate = copies.get("candidate") or {}
    if official.get("blocker"):
        return True, str(official.get("status"))
    if stopping == "b" and candidate.get("blocker"):
        return True, str(candidate.get("status"))
    return False, None


def caught_not_judgeable_blocks(rows: list[dict]) -> bool:
    """True when any scored cell caught a fake the judgeable rule rejected.

    Rule 5 requires that count to be 0. A nonzero count blocks sign-off
    the same way a sanity FAIL does.

    Ruling 12(21) keeps this check on every scored cell: all 72 cuts and
    the distance bins. A breach in a cell that is not the chosen cut
    still blocks, and ``kilohunt recall --grid`` exits 2.
    """
    return any(int(row.get("n_caught_not_judgeable", 0)) > 0 for row in rows)


@dataclass
class PreparedFake:
    """Measurements reused across the 72 cut combinations.

    The fade cut and the color time are not on the grid, so they are
    measured once. The reddening rise depends on the window and the
    gap, so it is measured once per pair. ``rise`` is None when the
    reddening rule would be skipped.
    """

    object_id: str
    distance_mpc: float
    judgeable: bool
    model_failed: bool
    fade_passed: bool
    color_value: float | None
    rises: dict[tuple[float, float], float | None]
    # Official definition (a) is ``judgeable``. This is candidate (b).
    # It is not official.
    judgeable_candidate: bool = False
    fade_skipped: bool = False


def _without_exact_copies(events: pd.DataFrame) -> pd.DataFrame:
    """Drop the AT2017gfo copies. They are not part of any recall."""
    if events.empty:
        return events
    kind = events["kind"].astype(str)
    object_id = events["object_id"].astype(str)
    keep = (kind != EXACT_KIND) & ~object_id.str.startswith(f"{EXACT_KIND}-")
    return events.loc[keep].reset_index(drop=True)


def _curves_by_id(curves: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if curves.empty:
        return {}
    return {
        str(object_id): frame.drop(columns=["object_id"])
        for object_id, frame in curves.groupby("object_id", sort=False)
    }


def _color_value(visible: pd.DataFrame) -> float | None:
    """g minus r at the color-rule time, or None when it cannot be read."""
    from kilonova_hunter.features import latest_same_night_color

    return latest_same_night_color(visible).g_minus_r


def _rise_or_skip(visible: pd.DataFrame, window: float, gap: float) -> float | None:
    """The fitted color rise, or None when the reddening rule skips.

    The skip test matches ``evaluate_reddening_rule``: no rise, or a
    missing endpoint, means the rule is not applied.
    """
    settings = thresholds_for(
        {
            "redder_than_g_minus_r": THRESHOLDS.redder_than_g_minus_r,
            "min_g_minus_r_rise": THRESHOLDS.min_g_minus_r_rise,
            "reddening_window_days": window,
            "min_color_separation_days": gap,
        }
    )
    change = g_minus_r_rise(visible, settings)
    if (
        change.rise is None
        or change.earlier_time_days is None
        or change.later_time_days is None
        or change.earlier_g_minus_r is None
        or change.later_g_minus_r is None
    ):
        return None
    return float(change.rise)


def prepare_fakes(
    events: pd.DataFrame,
    curves: pd.DataFrame,
    day: float,
    settings: dict | None = None,
    *,
    new_fade: bool = False,
) -> list[PreparedFake]:
    """Measure each fake once, using only rows at or before ``day``."""
    spec = load_settings() if settings is None else settings
    usable = _without_exact_copies(events)
    by_id = _curves_by_id(curves)
    windows = {
        (
            float(combo["reddening_window_days"]),
            float(combo["min_color_separation_days"]),
        )
        for combo in grid_combinations(spec)
    }
    prepared: list[PreparedFake] = []
    empty = pd.DataFrame(
        columns=["time_days", "filter", "magnitude", "magnitude_error"]
    )
    for row in usable.itertuples(index=False):
        curve = by_id.get(str(row.object_id), empty)
        kind = str(getattr(row, "kind", ""))
        redshift = float(getattr(row, "redshift", 0.0))
        if uses_two_component_model(kind):
            # Day 7 for the realistic set is this same cut: nothing after
            # source-frame day 6 is a detection.
            curve = drop_after_two_component_cap(curve, redshift)
        # 0/1 after a CSV round trip, or a real bool from a fresh frame.
        failed = bool(
            int(pd.to_numeric(pd.Series([row.model_failed])).fillna(0).iloc[0])
        )
        merger = float(row.merger_mjd)
        judgeable = False if failed else is_judgeable(curve, day, merger_mjd=merger)
        candidate = (
            False if failed else is_candidate_judgeable(curve, day, merger_mjd=merger)
        )
        visible = light_curve_for_the_filter(curve, day)
        visible.attrs["merger_mjd"] = float(row.merger_mjd)
        if failed or visible.empty:
            prepared.append(
                PreparedFake(
                    object_id=str(row.object_id),
                    distance_mpc=float(row.distance_mpc),
                    judgeable=judgeable,
                    judgeable_candidate=candidate,
                    model_failed=failed,
                    fade_passed=new_fade and not failed,
                    fade_skipped=new_fade and not failed,
                    color_value=None,
                    rises={pair: None for pair in windows},
                )
            )
            continue
        from dataclasses import replace

        fade_result = evaluate_fade_rule(
            visible, replace(THRESHOLDS, fade_requires_two_after_peak=new_fade)
        )
        fade_passed = fade_result.passed is not False
        color_value = _color_value(visible)
        rises = {pair: _rise_or_skip(visible, pair[0], pair[1]) for pair in windows}
        prepared.append(
            PreparedFake(
                object_id=str(row.object_id),
                distance_mpc=float(row.distance_mpc),
                judgeable=judgeable,
                judgeable_candidate=candidate,
                model_failed=failed,
                fade_passed=fade_passed,
                fade_skipped=fade_result.passed is None,
                color_value=color_value,
                rises=rises,
            )
        )
    return prepared


def caught_by(fake: PreparedFake, combo: dict[str, float]) -> tuple[bool, bool]:
    """Whether the filter flags this fake, and whether reddening was skipped.

    The second value is only meaningful when the fake was flagged.
    Skipping reddening does not by itself reject the object.
    """
    rise = fake.rises[
        (
            float(combo["reddening_window_days"]),
            float(combo["min_color_separation_days"]),
        )
    ]
    skipped = rise is None
    reddening_ok = True if skipped else rise > float(combo["min_g_minus_r_rise"])
    color_ok = fake.color_value is not None and fake.color_value > float(
        combo["redder_than_g_minus_r"]
    )
    caught = fake.fade_passed and color_ok and reddening_ok
    return caught, skipped and caught


def fast_verdict_matches_filter(visible: pd.DataFrame, combo: dict[str, float]) -> bool:
    """True when the fast path and ``is_kilonova_candidate`` agree.

    Used by the tests. The grid uses the fast path so 72 cuts do not
    each refit the same lines.
    """
    settings = thresholds_for(combo)
    direct = is_kilonova_candidate(visible, settings).is_candidate
    if visible.empty:
        return direct is False
    fake = PreparedFake(
        object_id="check",
        distance_mpc=0.0,
        judgeable=False,
        model_failed=False,
        fade_passed=evaluate_fade_rule(visible).passed is True,
        color_value=_color_value(visible),
        rises={
            (
                float(combo["reddening_window_days"]),
                float(combo["min_color_separation_days"]),
            ): _rise_or_skip(
                visible,
                float(combo["reddening_window_days"]),
                float(combo["min_color_separation_days"]),
            )
        },
    )
    caught, _skipped = caught_by(fake, combo)
    return caught is direct


def _rates(caught: int, judge_caught: int, judgeable: int, total: int) -> dict:
    recall = float(judge_caught / judgeable) if judgeable else float("nan")
    recall_low, recall_high = wilson_interval(judge_caught, judgeable)
    catch = float(caught / total) if total else float("nan")
    catch_low, catch_high = wilson_interval(caught, total)
    return {
        "n_total": total,
        "n_judgeable": judgeable,
        "n_caught": caught,
        "n_caught_and_judgeable": judge_caught,
        "n_caught_not_judgeable": caught - judge_caught,
        "judgeable_fraction": float(judgeable / total) if total else float("nan"),
        "recall_judgeable": recall,
        "recall_low": recall_low,
        "recall_high": recall_high,
        "catch_rate": catch,
        "catch_low": catch_low,
        "catch_high": catch_high,
    }


# Measured model g-r minus AT2017gfo. The numbers are read from the
# committed MOSFiT check output, robust night interpolation.
MEASURED_G_MINUS_R_GAP_MAG = 0.4
# +0.3 mag is the same shift with the instrument-to-LSST correction
# taken off. The difference of the two recalls is that uncertainty.
INSTRUMENT_TO_LSST_OFFSET_MAG = 0.3
# Science Lead can narrow this scope by editing these three constants.
# The report is both populations, days 3/5/7, both shifts, and both
# judgeable definitions. It is not used to pick a cut.
COLOR_SENSITIVITY_POPULATIONS = ("2017-like", "realistic")
COLOR_SENSITIVITY_DAYS = (3.0, 5.0, 7.0)
COLOR_SENSITIVITY_SHIFTS_MAG = (
    MEASURED_G_MINUS_R_GAP_MAG,
    INSTRUMENT_TO_LSST_OFFSET_MAG,
)


def g_minus_r_gap_by_day() -> tuple[tuple[float, float], ...]:
    """Model-minus-AT2017gfo g−r at 1.5, 2, and 3 days.

    Read from ``checks/mosfit/gr_gap.out`` (the robust night-interpolation
    block). Rounded half-up to 0.01 mag.
    """
    path = Path(__file__).resolve().parents[2] / "checks" / "mosfit" / "gr_gap.out"
    text = path.read_text(encoding="utf-8")
    started = False
    found: dict[float, float] = {}
    for line in text.splitlines():
        if line.startswith(" t(d)  data g-r"):
            started = True
            continue
        if not started or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 4:
            break
        day = float(parts[0])
        gap = Decimal(parts[-1])
        rounded = gap.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        found[day] = float(rounded)
    return tuple((day, found[day]) for day in (1.5, 2.0, 3.0))


def recall_with_bluer_color(
    prepared: list[PreparedFake],
    combo: dict[str, float],
    offset: float = MEASURED_G_MINUS_R_GAP_MAG,
    *,
    population: str = "2017-like",
    day: float = 3.0,
    status: str = "ready",
) -> dict:
    """Recompute recall after shifting g-r bluer by ``offset``.

    The shift applies to the population that was passed in, both the
    2017-like set and the realistic set. Subtracting ``offset``
    magnitudes from g-r is the sensitivity for the measured
    model-minus-AT2017gfo color gap. The default is +0.4 mag. The
    epoch values are +0.37 at 1.5 d, +0.47 at 2 d, and +0.42 at 3 d,
    so the offset sits in 0.3 to 0.47. A constant shift does not
    change the reddening rise. The tuning grid prints this shift.
    It is not used to pick a cut.
    """
    from dataclasses import replace

    shifted = [
        replace(
            fake,
            color_value=(
                None if fake.color_value is None else fake.color_value - float(offset)
            ),
        )
        for fake in prepared
    ]
    return score_prepared(
        shifted,
        combo,
        population=population,
        day=day,
        status=status,
    )


def color_sensitivity(
    prepared: list[PreparedFake],
    combo: dict[str, float],
    *,
    population: str = "2017-like",
    day: float = 3.0,
    status: str = "ready",
) -> dict:
    """Recall at +0.4 mag and at +0.3 mag, and the difference.

    The difference is the instrument-to-LSST correction uncertainty.
    This is not called by the tuning grid.
    """
    shared = {"population": population, "day": day, "status": status}
    at_0_4 = recall_with_bluer_color(
        prepared, combo, MEASURED_G_MINUS_R_GAP_MAG, **shared
    )
    at_0_3 = recall_with_bluer_color(
        prepared, combo, INSTRUMENT_TO_LSST_OFFSET_MAG, **shared
    )
    return {
        "offset_0_4": at_0_4,
        "offset_0_3": at_0_3,
        "difference": at_0_4["recall_judgeable"] - at_0_3["recall_judgeable"],
    }


def format_color_sensitivity(
    prepared: list[PreparedFake],
    combo: dict[str, float],
    **kwargs,
) -> str:
    """Both g-r shifts, and their difference, for the report."""
    result = color_sensitivity(prepared, combo, **kwargs)

    def _percent(value: float) -> str:
        if value != value:
            return "no recall"
        return f"{100.0 * value:.1f}%"

    return (
        "g-r bluer by +0.4 mag: recall "
        f"{_percent(result['offset_0_4']['recall_judgeable'])}. "
        "g-r bluer by +0.3 mag: recall "
        f"{_percent(result['offset_0_3']['recall_judgeable'])}. "
        "Difference "
        f"{_percent(result['difference'])} "
        "is the instrument to LSST correction uncertainty. "
        "This hook is not used to pick a cut."
    )


def format_color_sensitivity_scope(
    prepared_for_examples: dict,
    combo: dict[str, float],
    *,
    populations: tuple[str, ...] = COLOR_SENSITIVITY_POPULATIONS,
    days: tuple[float, ...] = COLOR_SENSITIVITY_DAYS,
    shifts: tuple[float, ...] = COLOR_SENSITIVITY_SHIFTS_MAG,
) -> str:
    """Recall (a) and (b) for each population, day, and g-r shift.

    ``prepared_for_examples`` is what the tuning grid keeps: a population
    name mapped to ``(prepared_by_day, light_curves)``. A bare
    ``prepared_by_day`` dict is accepted too. Changing the default
    populations, days, or shifts is a constants change. This report is
    not used to pick a cut.
    """

    def _percent(value: float) -> str:
        if value != value:
            return "no recall"
        return f"{100.0 * value:.1f}%"

    lines = [
        "## Color sensitivity",
        "",
        "Placeholder cuts. Each line shifts g-r bluer and reports "
        "recall (a), the official judgeable rule. "
        "This is not used to pick a cut.",
        "",
    ]
    for population in populations:
        stored = prepared_for_examples.get(population)
        if stored is None:
            continue
        prepared_by_day = stored[0] if isinstance(stored, tuple) else stored
        for day in days:
            prepared = prepared_by_day[float(day)]
            for shift in shifts:
                scored = recall_with_bluer_color(
                    prepared,
                    combo,
                    float(shift),
                    population=population,
                    day=float(day),
                )
                lines.append(
                    f"{population} day {float(day):g}, "
                    f"g-r bluer by +{float(shift):.1f} mag: "
                    f"recall (a) {_percent(scored['recall_judgeable'])}."
                )
            if uses_two_component_model(population) and float(day) >= 7.0:
                lines.append(REALISTIC_DAY7_LABEL)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def has_g_minus_r_pair(curve: pd.DataFrame, *, max_gap_days: float = 1.0) -> bool:
    """True when some g point and some r point are at most ``max_gap_days`` apart."""
    if curve is None or curve.empty:
        return False
    g_times = curve.loc[curve["filter"].astype(str) == "g", "time_days"].to_numpy(
        dtype=float
    )
    r_times = curve.loc[curve["filter"].astype(str) == "r", "time_days"].to_numpy(
        dtype=float
    )
    if len(g_times) == 0 or len(r_times) == 0:
        return False
    gaps = np.abs(g_times.reshape(-1, 1) - r_times.reshape(1, -1))
    return bool(np.any(gaps <= float(max_gap_days) + 1e-9))


def same_band_after_r_peak(
    curve: pd.DataFrame, *, separation_days: float = 0.5
) -> dict[str, bool]:
    """Two same-band points at least ``separation_days`` apart after r peak.

    ``any`` is true when some band has such a pair. ``g``, ``r``, and
    ``i`` are that band alone. No r peak means every flag is false.
    """
    flags = {"any": False, "g": False, "r": False, "i": False}
    if curve is None or curve.empty:
        return flags
    from kilonova_hunter.features import peak_time_days

    peak = peak_time_days(curve, "r")
    if peak is None:
        return flags
    after = curve.loc[curve["time_days"].to_numpy(dtype=float) > float(peak)]
    for band in ("g", "r", "i"):
        times = np.sort(
            after.loc[after["filter"].astype(str) == band, "time_days"].to_numpy(
                dtype=float
            )
        )
        ok = len(times) >= 2 and float(times[-1] - times[0]) + 1e-9 >= separation_days
        flags[band] = ok
        flags["any"] = flags["any"] or ok
    return flags


def _curve_up_to_day(curve: pd.DataFrame, day: float, *, redshift: float, kind: str):
    """Rows at or before ``day``. Realistic day 7 stops at source-frame day 6."""
    frame = curve
    if uses_two_component_model(kind) and float(day) >= 7.0:
        frame = drop_after_two_component_cap(frame, redshift)
    if frame is None or frame.empty or "time_days" not in frame.columns:
        return frame
    times = frame["time_days"].to_numpy(dtype=float)
    return frame.loc[times <= float(day) + 1e-9].reset_index(drop=True)


def informational_tuning_counts(
    events: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    population: str,
    day: float,
) -> dict:
    """Shares (ii) and (iii) for one tuning population and one day.

    Informational only. Not a cut and not a recall.
    """
    by_id: dict[str, pd.DataFrame] = {}
    if curves is not None and not curves.empty and "object_id" in curves.columns:
        for object_id, group in curves.groupby("object_id", sort=False):
            by_id[str(object_id)] = group.drop(columns=["object_id"])
    n = int(len(events))
    n_pair = 0
    n_after = 0
    by_band = {"g": 0, "r": 0, "i": 0}
    if n == 0:
        return {
            "population": population,
            "day": day,
            "n": 0,
            "g_minus_r_pair": float("nan"),
            "after_r_peak_any": float("nan"),
            "after_r_peak_g": float("nan"),
            "after_r_peak_r": float("nan"),
            "after_r_peak_i": float("nan"),
        }
    for row in events.itertuples(index=False):
        kind = str(getattr(row, "kind", population))
        redshift = float(getattr(row, "redshift", 0.0))
        curve = by_id.get(str(row.object_id), pd.DataFrame())
        visible = _curve_up_to_day(curve, day, redshift=redshift, kind=kind)
        if has_g_minus_r_pair(visible):
            n_pair += 1
        flags = same_band_after_r_peak(visible)
        if flags["any"]:
            n_after += 1
        for band in ("g", "r", "i"):
            if flags[band]:
                by_band[band] += 1
    return {
        "population": population,
        "day": day,
        "n": n,
        "g_minus_r_pair": n_pair / n,
        "after_r_peak_any": n_after / n,
        "after_r_peak_g": by_band["g"] / n,
        "after_r_peak_r": by_band["r"] / n,
        "after_r_peak_i": by_band["i"] / n,
        "realistic_day7": uses_two_component_model(population) and float(day) >= 7.0,
    }


def format_informational_counts(
    samples: list,
    prepared_for_examples: dict,
    combo: dict[str, float],
) -> str:
    """Tuning-only (a), (b), (ii), and (iii). Not used to pick a cut."""
    lines = [
        "## Informational counts",
        "",
        "These counts are not used to tune or to pick a cut. "
        "Recall (a) uses the placeholder cuts in config.py. "
        "(ii) is the share with any g and r points at most 1.0 day apart. "
        "(iii) is the share with two same-band points at least 0.5 days "
        "apart after the r-band peak. The band lines are that band alone.",
        "",
    ]
    for sample in samples:
        if sample.split != "tuning":
            continue
        population = sample.population
        prepared_by_day, _curves = prepared_for_examples[population]
        for day in JUDGE_DAYS:
            scored = score_prepared(
                prepared_by_day[day],
                combo,
                population=population,
                day=day,
                status=sample.status,
            )
            shares = informational_tuning_counts(
                sample.events,
                sample.light_curves,
                population=population,
                day=day,
            )

            def _percent(value: float) -> str:
                if value != value:
                    return "no number"
                return f"{100.0 * value:.1f}%"

            lines.append(
                f"{population} day {day:g}: "
                f"recall (a) {_percent(scored['recall_judgeable'])}. "
                f"(ii) g-r pair {_percent(shares['g_minus_r_pair'])}. "
                f"(iii) any band {_percent(shares['after_r_peak_any'])} "
                f"(g {_percent(shares['after_r_peak_g'])}, "
                f"r {_percent(shares['after_r_peak_r'])}, "
                f"i {_percent(shares['after_r_peak_i'])})."
            )
            if shares.get("realistic_day7"):
                lines.append(REALISTIC_DAY7_LABEL)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def score_prepared(
    prepared: list[PreparedFake],
    combo: dict[str, float],
    *,
    population: str,
    day: float,
    status: str,
    distance_bin: str = "all",
    low_mpc: float | None = None,
    high_mpc: float | None = None,
    high_inclusive: bool = False,
) -> dict:
    """One grid cell: a cut, a population, a day, and maybe a distance bin."""
    chosen = prepared
    if low_mpc is not None and high_mpc is not None:
        if high_inclusive:
            chosen = [
                fake for fake in prepared if low_mpc <= fake.distance_mpc <= high_mpc
            ]
        else:
            chosen = [
                fake for fake in prepared if low_mpc <= fake.distance_mpc < high_mpc
            ]
    caught = 0
    judgeable = 0
    judgeable_candidate = 0
    judge_caught = 0
    candidate_caught = 0
    skipped = 0
    for fake in chosen:
        is_caught, reddening_skipped = caught_by(fake, combo)
        if fake.judgeable:
            judgeable += 1
        if fake.judgeable_candidate:
            judgeable_candidate += 1
        if is_caught:
            caught += 1
            if fake.judgeable:
                judge_caught += 1
            if fake.judgeable_candidate:
                candidate_caught += 1
            if reddening_skipped:
                skipped += 1
    untestable = day == 3.0 and status == "untestable" and distance_bin == "all"
    # A distance bin can also have too few judgeable fakes to quote a
    # recall. The untestable flag on the headline row is the sample-size
    # rule. A bin with zero judgeable fakes simply has no recall.
    rates = _rates(caught, judge_caught, judgeable, len(chosen))
    if untestable or judgeable == 0:
        rates["recall_judgeable"] = float("nan")
        rates["recall_low"] = float("nan")
        rates["recall_high"] = float("nan")
    row = {
        "population": population,
        "day": day,
        "distance_bin": distance_bin,
        "status": status,
        "untestable": untestable,
        "n_reddening_skipped": skipped,
        "n_fade_skipped": sum(fake.fade_skipped for fake in chosen),
        "n_fade_skipped_judgeable": sum(
            fake.fade_skipped and fake.judgeable for fake in chosen
        ),
        "n_reddening_skipped_judgeable": sum(
            fake.judgeable
            and fake.rises.get(
                (
                    float(combo["reddening_window_days"]),
                    float(combo["min_color_separation_days"]),
                )
            )
            is None
            for fake in chosen
        ),
        "n_judgeable_candidate": judgeable_candidate,
        "n_caught_and_candidate": candidate_caught,
        "recall_candidate": (
            float(candidate_caught / judgeable_candidate)
            if judgeable_candidate
            else float("nan")
        ),
        **combo,
        **rates,
    }
    if untestable:
        row["recall_candidate"] = float("nan")
    if status == "underpowered" and day == 3.0 and distance_bin == "all" and judgeable:
        # Power uses the measured recall. When the recall was blanked
        # above we would not be in this branch, because underpowered
        # means the floor was met, so judgeable is at least the floor.
        measured = judge_caught / judgeable
        row["power_at_70"] = chance_wilson_lower_at_least(judgeable, measured, 0.70)
    else:
        row["power_at_70"] = float("nan")
    return row


def score_grid(
    prepared_by_day: dict[float, list[PreparedFake]],
    *,
    population: str,
    status: str,
    settings: dict | None = None,
) -> list[dict]:
    """Every combination, every day, and the three distance bins."""
    spec = load_settings() if settings is None else settings
    rows: list[dict] = []
    for combo in grid_combinations(spec):
        for day in JUDGE_DAYS:
            prepared = prepared_by_day[day]
            rows.append(
                score_prepared(
                    prepared,
                    combo,
                    population=population,
                    day=day,
                    status=status,
                )
            )
            for name, low, high, high_inclusive in DISTANCE_BINS:
                rows.append(
                    score_prepared(
                        prepared,
                        combo,
                        population=population,
                        day=day,
                        status=status,
                        distance_bin=name,
                        low_mpc=low,
                        high_mpc=high,
                        high_inclusive=high_inclusive,
                    )
                )
    return rows


def examples_caught_but_not_judgeable(
    prepared: list[PreparedFake],
    curves: pd.DataFrame,
    combo: dict[str, float],
    day: float,
    limit: int = 5,
) -> list[dict]:
    """A few fakes the filter caught that were not judgeable, for the log."""
    by_id = _curves_by_id(curves)
    found = []
    for fake in prepared:
        caught, skipped = caught_by(fake, combo)
        if not caught or fake.judgeable:
            continue
        curve = by_id.get(fake.object_id, pd.DataFrame())
        found.append(
            {
                "object_id": fake.object_id,
                "distance_mpc": fake.distance_mpc,
                "reddening_skipped": skipped,
                "n_rows_by_day": int(len(light_curve_for_the_filter(curve, day))),
            }
        )
        if len(found) >= limit:
            break
    return found


def score_exact_copies(
    events: pd.DataFrame,
    curves: pd.DataFrame,
    combo: dict[str, float],
    *,
    day: float = 3.0,
) -> dict:
    """How many exact copies are judgeable, and how many of those are caught.

    The sanity result is tri-state, from ``sanity_copy_verdict``.

    - ``pass``: at least 50 copies are judgeable and at least 95 percent
      of those were caught.
    - ``fail``: at least 50 are judgeable and fewer than 95 percent were
      caught. That blocks sign-off.
    - ``untestable``: fewer than 50 are judgeable, including zero.
      Untestable is its own state. It blocks sign-off and is reported
      as UNTESTABLE. It is not a pass, and it is not printed as NaN.

    Copies are not in the recall. Misses are listed with the start
    time, the sky position, E(B-V), and the rule that failed.
    """
    if events.empty:
        empty_official = sanity_copy_verdict(0, 0)
        empty_candidate = sanity_copy_verdict(0, 0)
        return {
            "n_copies": 0,
            "n_judgeable": 0,
            "n_judgeable_candidate": 0,
            "n_caught_judgeable": 0,
            "n_caught_candidate": 0,
            "fraction_judgeable_caught": empty_official["fraction"],
            "fraction_candidate_caught": empty_candidate["fraction"],
            "blocker": empty_official["blocker"],
            "blocker_candidate": empty_candidate["blocker"],
            "official": empty_official,
            "candidate": empty_candidate,
            "misses": [],
        }
    by_id = _curves_by_id(curves)
    settings = thresholds_for(combo)
    n_judge = 0
    n_candidate = 0
    n_caught = 0
    n_caught_candidate = 0
    misses = []
    for row in events.itertuples(index=False):
        curve = by_id.get(str(row.object_id), pd.DataFrame())
        judgeable = is_judgeable(curve, day, merger_mjd=float(row.merger_mjd))
        if not judgeable:
            continue
        n_judge += 1
        candidate = is_candidate_judgeable(curve, day, merger_mjd=float(row.merger_mjd))
        if candidate:
            n_candidate += 1
        visible = light_curve_for_the_filter(curve, day)
        visible.attrs["merger_mjd"] = float(row.merger_mjd)
        verdict = is_kilonova_candidate(visible, settings)
        if verdict.is_candidate:
            n_caught += 1
            if candidate:
                n_caught_candidate += 1
            continue
        failed_rules = []
        for name in ("fade", "color", "reddening"):
            rule = getattr(verdict, name)
            if rule.passed is False:
                failed_rules.append(f"{name}: {rule.explanation}")
        misses.append(
            {
                "object_id": str(row.object_id),
                "merger_mjd": float(row.merger_mjd),
                "ra_deg": float(row.ra_deg),
                "dec_deg": float(row.dec_deg),
                "ebv": float(row.ebv),
                "failed_rules": failed_rules,
                "candidate": candidate,
            }
        )
    official = sanity_copy_verdict(n_judge, n_caught)
    candidate_verdict = sanity_copy_verdict(n_candidate, n_caught_candidate)
    return {
        "n_copies": int(len(events)),
        "n_judgeable": n_judge,
        "n_judgeable_candidate": n_candidate,
        "n_caught_judgeable": n_caught,
        "n_caught_candidate": n_caught_candidate,
        "fraction_judgeable_caught": official["fraction"],
        "fraction_candidate_caught": candidate_verdict["fraction"],
        "blocker": official["blocker"],
        "blocker_candidate": candidate_verdict["blocker"],
        "official": official,
        "candidate": candidate_verdict,
        "misses": misses,
        "day": day,
        "cuts": combo,
    }


def sanity_copy_verdict(n_judge: int, n_caught: int, *, target: int = 50) -> dict:
    """Tri-state sanity result for one judgeable definition.

    ``untestable`` means fewer than ``target`` judgeable copies,
    including zero. That blocks sign-off. The report says UNTESTABLE.
    It does not say the copies passed, and it does not print a bare NaN.

    ``fail`` means there are enough judgeable copies and fewer than
    95 percent were caught. That is more than 2 misses in 50. It
    blocks sign-off.

    ``pass`` means there are enough judgeable copies and at least
    95 percent were caught. The fraction is an observed catch rate,
    not a 95-percent-confidence claim.
    """
    if n_judge < target:
        return {
            "status": "untestable",
            "phrase": "sanity block UNTESTABLE, v1 not Done",
            "fraction": float("nan"),
            "blocker": True,
            "passed": False,
            "n_judge": n_judge,
            "n_caught": n_caught,
        }
    fraction = float(n_caught / n_judge)
    blocker = fraction < 0.95
    status = "fail" if blocker else "pass"
    return {
        "status": status,
        "phrase": (
            f"observed catch rate {n_caught} of {n_judge} "
            f"({100.0 * fraction:.1f}%). "
            "This is not a 95%-confidence claim. "
            "A fail is fewer than 95% caught, more than 2 misses in 50, "
            "or any miss traced to a bug."
        ),
        "fraction": fraction,
        "blocker": blocker,
        "passed": not blocker,
        "n_judge": n_judge,
        "n_caught": n_caught,
    }


def _rule_word(passed: bool | None) -> str:
    """passed, failed, or skipped. A skip is not a failure."""
    if passed is True:
        return "passed"
    if passed is False:
        return "failed"
    return "skipped"


def exact_copy_rule_breakdown(
    events: pd.DataFrame,
    curves: pd.DataFrame,
    combo: dict[str, float],
    *,
    day: float = 3.0,
) -> dict:
    """Which rule failed on each judgeable exact copy.

    The official judgeable rule is not changed. Reddening that cannot
    run is recorded as skipped, which is different from a rise that was
    measured and failed. The window and the gap are the cuts that were
    used, so a skip can be traced to those two numbers.
    """
    by_id = _curves_by_id(curves)
    settings = thresholds_for(combo)
    rows: list[dict] = []
    n_judge = 0
    n_candidate = 0
    n_skipped = 0
    for row in events.itertuples(index=False):
        curve = by_id.get(str(row.object_id), pd.DataFrame())
        judgeable = is_judgeable(curve, day, merger_mjd=float(row.merger_mjd))
        if not judgeable:
            continue
        n_judge += 1
        candidate = is_candidate_judgeable(curve, day, merger_mjd=float(row.merger_mjd))
        if candidate:
            n_candidate += 1
        visible = light_curve_for_the_filter(curve, day)
        visible.attrs["merger_mjd"] = float(row.merger_mjd)
        verdict = is_kilonova_candidate(visible, settings)
        reddening = _rule_word(verdict.reddening.passed)
        if reddening == "skipped":
            n_skipped += 1
        color_text = _color_next_to_cut(visible)
        rows.append(
            {
                "object_id": str(row.object_id),
                "candidate": "yes" if candidate else "no",
                "fade": _rule_word(verdict.fade.passed),
                "g_minus_r": color_text,
                "reddening": reddening,
                "fade_text": verdict.fade.explanation,
                "g_minus_r_text": color_text,
                "reddening_text": verdict.reddening.explanation,
            }
        )
    finding = ""
    if n_skipped:
        finding = (
            f"Finding: reddening could not run on {n_skipped} of {n_judge} "
            "copies the official judgeable rule accepted. A skip is not a "
            "failed rise. The color points were not there, so the reddening "
            "rule was not applied."
        )
    return {
        "day": day,
        "n_copies": int(len(events)),
        "n_judgeable": n_judge,
        "n_judgeable_candidate": n_candidate,
        "n_judgeable_reddening_skipped": n_skipped,
        "window_days": float(combo["reddening_window_days"]),
        "gap_days": float(combo["min_color_separation_days"]),
        "finding": finding,
        "rows": rows,
    }


def _color_next_to_cut(visible: pd.DataFrame) -> str:
    """g minus r at r peak + 1 day, or why that color is not usable."""
    from kilonova_hunter.features import latest_same_night_color

    color = latest_same_night_color(visible)
    if color.g_minus_r is None:
        return color.explanation
    return (
        f"g-r {color.g_minus_r:.2f} next to cut {THRESHOLDS.redder_than_g_minus_r:.2f}."
    )


def _widest_bracket_gap(visible: pd.DataFrame, when: float) -> float | None:
    """Largest gap the color would have to cross, in days."""
    gaps: list[float] = []
    for band in ("g", "r"):
        times = sorted(
            float(value)
            for value in visible.loc[visible["filter"].astype(str) == band, "time_days"]
        )
        if len(times) < 2 or when < times[0] or when > times[-1]:
            continue
        earlier = times[0]
        for time_days in times:
            if time_days > when:
                gaps.append(time_days - earlier)
                break
            earlier = time_days
    if not gaps:
        return None
    return max(gaps)


def format_exact_copy_breakdown(report: dict) -> str:
    """Plain-language breakdown. This is a finding, not a new score."""
    lines = [
        "# Exact-copy rule breakdown",
        "",
        "Official judgeable definition (a) is unchanged. "
        "Candidate (b) also needs a usable g minus r at r-peak + 1 day "
        "with the 1.0-day gap limit. (b) is not official. "
        "It does not replace (a).",
        "Light curves use the diffusion wrapper (arnett-factor-2) on "
        "redback 1.20.0. The package itself is not edited.",
        "Cuts are the placeholders in config.py, not a cut picked from the grid.",
        f"Day {report['day']:g}. Reddening window {report['window_days']:g} days, "
        f"minimum color gap {report['gap_days']:g} days.",
        f"{report['n_judgeable']} of {report['n_copies']} copies are judgeable "
        f"under (a). {report['n_judgeable_candidate']} of those are judgeable "
        "under (b).",
        "",
    ]
    if report["finding"]:
        lines.append(report["finding"])
        lines.append("")
    if not report["rows"]:
        lines.append("No judgeable copies.")
        return "\n".join(lines) + "\n"
    lines.append("Judgeable copies:")
    lines.append("")
    for row in report["rows"]:
        lines.append(
            f"- {row['object_id']}: definition (b) {row['candidate']}, "
            f"fade {row['fade']}, "
            f"{row['g_minus_r']}, reddening {row['reddening']}."
        )
        lines.append(f"  Fade: {row['fade_text']}")
        lines.append(f"  g-r: {row['g_minus_r_text']}")
        lines.append(f"  Reddening: {row['reddening_text']}")
    return "\n".join(lines) + "\n"


def write_cached_exact_copy_breakdown(
    cache_dir: Path | str,
    destination: Path | str,
    *,
    day: float = 3.0,
) -> str:
    """Write the breakdown from a cache that already exists.

    This reads the saved copies. It does not simulate and it does not
    rerun the grid.
    """
    folder = Path(cache_dir)
    events = pd.read_csv(folder / "events.csv")
    curves = pd.read_csv(folder / "light_curves.csv")
    combo = {
        "redder_than_g_minus_r": THRESHOLDS.redder_than_g_minus_r,
        "min_g_minus_r_rise": THRESHOLDS.min_g_minus_r_rise,
        "reddening_window_days": THRESHOLDS.reddening_window_days,
        "min_color_separation_days": THRESHOLDS.min_color_separation_days,
    }
    text = format_exact_copy_breakdown(
        exact_copy_rule_breakdown(events, curves, combo, day=day)
    )
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def score_with_frozen_cuts(
    events: pd.DataFrame,
    curves: pd.DataFrame,
    combo: dict[str, float],
    *,
    population: str,
    status: str = "ready",
) -> list[dict]:
    """Score one cached split at days 3, 5, and 7. Does not change the cuts."""
    prepared = {day: prepare_fakes(events, curves, day) for day in JUDGE_DAYS}
    return [
        score_prepared(
            prepared[day],
            combo,
            population=population,
            day=day,
            status=status,
        )
        for day in JUDGE_DAYS
    ]


def format_test_score(
    rows_by_population: dict[str, list[dict]],
    *,
    dry_run: bool,
    provisional: bool,
) -> str:
    """Plain-English test-set report. Pass/fail is only the 2017-like day-3 bar."""
    lines = ["# Test-set score", ""]
    if dry_run:
        lines.append("DRY RUN.")
        lines.append("")
    if provisional:
        lines.append("PROVISIONAL. Contamination limit NOT applied.")
        lines.append("")
    lines.append("Passing on ZTF does not mean passing on Rubin.")
    lines.append(TNS_STATUS_LINE)
    lines.append("Judgeable (a) is unchanged. Decision 12(22) drops (b).")
    lines.append(EARLY_GAP_MODEL_LIMIT)
    lines.append("")
    lines.append(
        "Recall among judgeable fakes describes the filter. "
        "The catch rate is caught divided by every fake."
    )
    lines.append("")
    for population, rows in rows_by_population.items():
        lines.append(f"## {population}")
        lines.append("")
        for row in rows:
            day = int(row["day"])
            if population == "2017-like" and day == 3 and not row["untestable"]:
                passed = (
                    row["recall_low"] == row["recall_low"] and row["recall_low"] >= 0.70
                )
                verdict = "PASS" if passed else "FAIL"
                lines.append(f"Day 3: {verdict}. {_combo_sentence(row)}")
            elif row["untestable"]:
                lines.append(f"Day {day}: untestable. No recall number.")
            else:
                lines.append(f"Day {day}: {_combo_sentence(row)}")
            lines.append(
                f"Judgeable fraction: {100.0 * row['judgeable_fraction']:.1f}%. "
                f"Reddening skipped: {row['n_reddening_skipped']}. "
                f"Caught but not judgeable: {row['n_caught_not_judgeable']}."
            )
            if row["population"] == "realistic" and day == 7:
                lines.append(REALISTIC_DAY7_LABEL)
            lines.append("")
    return "\n".join(lines)


def write_grid_csv(rows: list[dict], path: Path) -> None:
    """Write the grid table. Rates that were left blank stay blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame = frame.drop(
        columns=[name for name in frame if "candidate" in name], errors="ignore"
    )
    frame.to_csv(path, index=False)


def summarize_grid(
    rows: list[dict],
    *,
    samples: list,
    dust_mode: str,
    dust_reason: str,
    copies: dict | None,
    contamination_applied: bool,
) -> str:
    """The short markdown summary that sits next to the CSV."""
    lines = [
        "# Tuning grid",
        "",
        "Label for the alert slices, when they exist: ZTF proxy for Rubin.",
        "This file scores simulated fakes. It does not apply a flags-per-night limit.",
        "",
        "Recall among judgeable fakes describes the filter. "
        "The catch rate is caught divided by every fake. "
        "That is the survey-level number.",
        "",
        "Judgeable (a) is unchanged. Decision 12(22) drops (b).",
        EARLY_GAP_MODEL_LIMIT,
        "",
        TNS_STATUS_LINE,
        "",
    ]
    if dust_mode != "sfd":
        lines.append("DRY RUN. The SFD dust map was not used.")
        lines.append(dust_reason)
        lines.append("")
    else:
        lines.append(
            "Dust: SFD E(B-V) times 0.86, then Fitzpatrick 1999 at R_V = 3.1. "
            "Redshift: astropy Planck18. See sky.py."
        )
        lines.append("")
    if not contamination_applied:
        lines.append(
            "Contamination limit NOT applied. "
            "No cuts were picked from this grid, and the test set was not scored."
        )
        lines.append(
            "When Fink flags are counted, the report has two columns: "
            "flags_per_night (new candidates) and repeat_flags_per_night. "
            "A repeat counts only on a night with a new alert for that object. "
            "A night with no new alert counts nothing, not even as a repeat. "
            "The N = 10 limit uses the new-candidate column only."
        )
        lines.append("")
    lines.append("## Sample")
    lines.append("")
    for sample in samples:
        lines.append(
            f"- {sample.population} {sample.split}: {sample.n_simulated} simulated, "
            f"{sample.n_judgeable_day3} judgeable at day 3, "
            f"{sample.n_judgeable_day5} at day 5, "
            f"{sample.n_judgeable_day7} at day 7, status {sample.status}."
        )
        if sample.status == "underpowered":
            lines.append(
                "  Underpowered: the cap was hit below the target "
                "and at or above the floor."
            )
        if sample.status == "untestable":
            lines.append("  Untestable at day 3: no recall number.")
    lines.append("")
    headline = [
        row
        for row in rows
        if row["distance_bin"] == "all"
        and row["population"] == "2017-like"
        and row["day"] == 3.0
    ]
    if headline:
        best = max(
            headline,
            key=lambda row: (
                -1.0
                if row["recall_judgeable"] != row["recall_judgeable"]
                else row["recall_judgeable"]
            ),
        )
        lines.append("## Best day-3 recall on judgeable 2017-like tuning fakes")
        lines.append("")
        lines.append("This is the highest recall in the grid. It is not a picked cut.")
        lines.append("")
        lines.append(_combo_sentence(best))
        offenders = int(best["n_caught_not_judgeable"])
        lines.append("")
        lines.append(
            f"Caught but not judgeable, at that combination and day 3: {offenders}. "
            "Rule 5 requires this count to be 0."
        )
        if offenders:
            lines.append(
                "That count is not zero, so it breaks rule 5 and blocks sign-off. "
                "The filter can flag a "
                "fake the frozen judgeable rule rejects, because the color "
                "rule may interpolate g and r from different nights and the "
                "fade rule may use two points closer together than 0.5 days. "
                "Those fakes are listed in the run log. They stay in the catch "
                "rate, because the catch rate is every fake the filter flagged."
            )
    if copies is not None:
        lines.extend(_copy_lines(copies))
    lines.append("")
    lines.append("## Distance bins")
    lines.append("")
    lines.append(
        "Bins are luminosity distance: 10-100, 100-200, and 200-300 Mpc. "
        "The last bin ends at 300. " + REALISTIC_DAY7_LABEL
    )
    return "\n".join(lines) + "\n"


def _tuning_fink_report(output_dir: Path) -> str:
    """Fink counts from the tuning file, when that file is present.

    Missing file means the count is left out. The full alerts file is
    not read. The count does not pick a cut.
    """
    from kilonova_hunter.alerts import (
        TUNING_ALERTS_FILENAME,
        average_flags_per_night,
        read_tuning_alerts,
    )

    if not (output_dir / TUNING_ALERTS_FILENAME).is_file():
        return ""
    alerts = read_tuning_alerts(output_dir)
    summary = average_flags_per_night(alerts)
    nights = int(summary["n_nights"])
    flagged = int(summary["n_flagged"])
    return (
        "## Fink tuning nights\n\n"
        f"Tuning-night Fink alerts: {nights} nights, {flagged} new candidates. "
        "Nights outside the tuning slice are not counted. "
        "This count is not used to pick a cut.\n"
    )


def _grid_run_header() -> str:
    """Git SHA and redback version, printed at the top of the grid summary."""
    from kilonova_hunter.cli import _git_sha
    from kilonova_hunter.sample import _redback_version

    return f"git {_git_sha()}\nredback {_redback_version()}\n\n"


def run_tuning_grid(
    *,
    output_dir: Path | str = Path("results"),
    cache_dir: Path | str = Path("data") / "recall" / "cache",
    sample_size=None,
    workers: int = 1,
    settings_path: Path | None = None,
    splits: tuple[str, ...] | None = None,
    new_fade: bool = False,
    copy_seed: int = 2026110401,
) -> GridRun:
    """Simulate both populations and the given splits, then score tuning.

    ``splits`` defaults to tuning and test. Pass ``("tuning",)`` to
    leave the test split untouched: it is not simulated, read, or
    written. The test split is never scored here. No cut is picked.
    The returned run says whether the exact-copy verdict blocks
    sign-off. The command exits 2 in that case, and so does a nonzero
    caught-but-not-judgeable count.
    """
    from kilonova_hunter.sample import (
        LIKE_KIND,
        REALISTIC,
        SPLITS,
        SampleSizeSettings,
        simulate_exact_copies,
        simulate_population,
    )
    from kilonova_hunter.sky import open_dust_source

    settings = load_settings(settings_path)
    size = sample_size if sample_size is not None else SampleSizeSettings()
    dust = open_dust_source()
    chosen_splits = SPLITS if splits is None else splits
    samples = []
    rows: list[dict] = []
    prepared_for_examples: dict[str, tuple] = {}
    for population in (LIKE_KIND, REALISTIC):
        for split in chosen_splits:
            sample = simulate_population(
                population,
                split,
                sample_size=size,
                dust=dust,
                cache_dir=cache_dir,
                workers=workers,
            )
            samples.append(sample)
            if split != "tuning":
                continue
            prepared = {
                day: prepare_fakes(
                    sample.events, sample.light_curves, day, settings, new_fade=new_fade
                )
                for day in JUDGE_DAYS
            }
            rows.extend(
                score_grid(
                    prepared,
                    population=population,
                    status=sample.status,
                    settings=settings,
                )
            )
            prepared_for_examples[population] = (prepared, sample.light_curves)
    copies = simulate_exact_copies(
        dust=dust, cache_dir=cache_dir, workers=workers, seed=copy_seed
    )
    placeholder = {
        "redder_than_g_minus_r": THRESHOLDS.redder_than_g_minus_r,
        "min_g_minus_r_rise": THRESHOLDS.min_g_minus_r_rise,
        "reddening_window_days": THRESHOLDS.reddening_window_days,
        "min_color_separation_days": THRESHOLDS.min_color_separation_days,
    }
    copy_score = score_exact_copies(
        copies.events,
        copies.light_curves,
        dict(placeholder, fade_requires_two_after_peak=new_fade),
    )
    # The default draw stops on definition (a). A later caller that
    # stops on (b) must say so, or (b) is only indicative.
    copy_score["stopping_definition"] = "a"
    blocked, copy_status = copy_sign_off(copy_score)
    reasons: list[str] = []
    if blocked:
        reasons.append(
            f"the exact-copy verdict is {(copy_status or 'blocker').upper()}"
        )
    if caught_not_judgeable_blocks(rows):
        reasons.append(
            "caught-but-not-judgeable is not zero in a scored cell, "
            "including one that is not the chosen cut, which breaks rule 5 "
            "and blocks sign-off"
        )
        blocked = True
        if copy_status is None:
            copy_status = "caught-but-not-judgeable"
    destination = Path(output_dir)
    write_grid_csv(rows, destination / "tuning_grid.csv")
    text = summarize_grid(
        rows,
        samples=[item for item in samples if item.split == "tuning"],
        dust_mode=dust.mode,
        dust_reason=dust.reason,
        copies=copy_score,
        contamination_applied=False,
    )
    worst = _worst_not_judgeable(rows, prepared_for_examples)
    if worst:
        text = worst + "\n" + text
    tuning_samples = [item for item in samples if item.split == "tuning"]
    text = (
        text.rstrip()
        + "\n\n"
        + format_informational_counts(
            tuning_samples, prepared_for_examples, placeholder
        )
    )
    if prepared_for_examples:
        text = (
            text.rstrip()
            + "\n\n"
            + format_color_sensitivity_scope(prepared_for_examples, placeholder)
        )
    fink = _tuning_fink_report(destination)
    if fink:
        text = text.rstrip() + "\n\n" + fink
    text = _grid_run_header() + text
    (destination / "tuning_grid.md").write_text(text, encoding="utf-8")
    return GridRun(
        text=text,
        sign_off_blocker=blocked,
        copy_status=copy_status,
        block_reasons=tuple(reasons),
    )


def _worst_not_judgeable(rows: list[dict], prepared_for_examples: dict) -> str:
    """List every caught-but-not-judgeable case at days 3, 5, and 7.

    Rule 5 requires the count to be 0. A nonzero count breaks rule 5
    and blocks sign-off, the same as a sanity FAIL. These fakes are in
    the catch rate and not in
    the judgeable denominator, so that day's recall does not include
    them. The official judgeable rule is not changed.
    """
    headline = [
        row
        for row in rows
        if row["distance_bin"] == "all"
        and float(row["day"]) in JUDGE_DAYS
        and int(row["n_caught_not_judgeable"]) > 0
    ]
    if not headline:
        return ""
    lines = [
        "# Caught but not judgeable",
        "",
        "Every cell below is a day-3, day-5, or day-7 case where the filter "
        "flagged a fake the official judgeable rule did not accept. "
        "Those fakes stay in the catch rate and stay out of the judgeable "
        "denominator, so the recall for that day does not include them. "
        "Rule 5 requires this count to be 0. A nonzero count breaks rule 5 "
        "and blocks sign-off.",
        "",
    ]
    for row in sorted(
        headline, key=lambda item: (item["population"], float(item["day"]))
    ):
        population = row["population"]
        prepared_by_day, curves = prepared_for_examples[population]
        day = float(row["day"])
        prepared = prepared_by_day[day]
        combo = {
            "redder_than_g_minus_r": row["redder_than_g_minus_r"],
            "min_g_minus_r_rise": row["min_g_minus_r_rise"],
            "reddening_window_days": row["reddening_window_days"],
            "min_color_separation_days": row["min_color_separation_days"],
        }
        examples = examples_caught_but_not_judgeable(
            prepared, curves, combo, day=day, limit=int(row["n_caught_not_judgeable"])
        )
        lines.append(
            f"{population} day {day:g}, g-r cut {combo['redder_than_g_minus_r']}, "
            f"rise {combo['min_g_minus_r_rise']}, "
            f"window {combo['reddening_window_days']} days, "
            f"gap {combo['min_color_separation_days']} days: "
            f"{row['n_caught_not_judgeable']} caught and not judgeable."
        )
        if population == "realistic" and day == 7.0:
            lines.append(REALISTIC_DAY7_LABEL)
        for example in examples:
            lines.append(
                f"- {example['object_id']} at {example['distance_mpc']:.1f} Mpc, "
                f"{example['n_rows_by_day']} rows by that day, "
                f"reddening skipped: {example['reddening_skipped']}."
            )
        lines.append("")
    return "\n".join(lines)


def _combo_sentence(row: dict) -> str:
    recall = row["recall_judgeable"]
    if recall != recall:
        recall_text = "no recall (untestable or no judgeable fakes)"
    else:
        recall_text = (
            f"recall {100.0 * recall:.1f}% "
            f"(95% Wilson {100.0 * row['recall_low']:.1f}% to "
            f"{100.0 * row['recall_high']:.1f}%)"
        )
    catch = row["catch_rate"]
    catch_text = (
        f"catch rate {100.0 * catch:.1f}% "
        f"(95% Wilson {100.0 * row['catch_low']:.1f}% "
        f"to {100.0 * row['catch_high']:.1f}%)"
        if catch == catch
        else "no catch rate"
    )
    return (
        f"g-r cut {row['redder_than_g_minus_r']}, "
        f"rise {row['min_g_minus_r_rise']}, "
        f"window {row['reddening_window_days']} days, "
        f"gap {row['min_color_separation_days']} days: "
        f"{recall_text}; {catch_text}; "
        f"judgeable fraction {100.0 * row['judgeable_fraction']:.1f}%; "
        f"reddening skipped on {row['n_reddening_skipped']} flagged fakes."
        f"{_judgeable_clause(row)}"
        f"{_power_clause(row.get('power_at_70', float('nan')))}"
    )


def _judgeable_clause(row: dict) -> str:
    """Both judgeable counts, when the row has the candidate one.

    (a) is the official section 1 definition. (b) also needs a usable
    g minus r at the r-band peak plus 1 day, with the 1.0-day gap limit.
    (b) is not official. It is not adopted.
    """
    if "n_judgeable" not in row:
        return ""
    return f" Judgeable (a, official): {row['n_judgeable']}."


def _power_clause(value: float) -> str:
    """How to print the chance of clearing a 70 percent recall.

    A chance like 3e-80 is a real number, but printing it that way looks
    like a bug. Anything that small is about 0.
    """
    if value != value:
        return ""
    if value < 5e-4:
        shown = "about 0"
    else:
        shown = f"{value:.2f}"
    return f" Chance the Wilson lower bound clears 70% recall: {shown}."


def _sanity_line(
    label: str, verdict: dict, n_copies: int, *, blocks_sign_off: bool = True
) -> str:
    """One sentence for definition (a) or (b). Never prints a bare NaN."""
    if verdict["status"] == "untestable":
        block = "This blocks sign-off. " if blocks_sign_off else ""
        return (
            f"Judgeable ({label}): UNTESTABLE. {verdict['phrase']}. "
            f"{block}"
            f"{verdict['n_judge']} judgeable, {verdict['n_caught']} caught, "
            f"out of {n_copies} copies."
        )
    prefix = "SIGN-OFF BLOCKER. " if verdict["blocker"] and blocks_sign_off else ""
    return f"{prefix}Judgeable ({label}): {verdict['phrase']} Out of {n_copies} copies."


def _candidate_line(verdict: dict, n_copies: int, stopping: str) -> str:
    """Definition (b). Indicative unless the draw stopped on (b)."""
    line = _sanity_line(
        "b, candidate, not adopted",
        verdict,
        n_copies,
        blocks_sign_off=stopping == "b",
    )
    if stopping != "b":
        line += " Indicative, not a 12(16) result."
    return line


def _copy_lines(copies: dict) -> list[str]:
    official = copies.get("official") or sanity_copy_verdict(
        int(copies["n_judgeable"]), int(copies["n_caught_judgeable"])
    )
    lines = [
        "",
        "## Exact AT2017gfo copies",
        "",
        "These copies are not in the recall or the catch rate. "
        "The fractions below use the placeholder cuts in config.py, "
        "not a cut picked from the grid. "
        "Definition (a) is official; (b) was dropped by Decision 12(22). "
        "12(16) is approved: keep drawing until 50 are judgeable, cap 4,000.",
        "",
        _sanity_line("a, official", official, int(copies["n_copies"])),
    ]
    if copies["misses"]:
        lines.append("")
        lines.append("Missed judgeable copies:")
        for miss in copies["misses"]:
            rules = "; ".join(miss["failed_rules"]) or "no rule text"
            lines.append(
                f"- {miss['object_id']}: merger MJD {miss['merger_mjd']:.5f}, "
                f"ra {miss['ra_deg']:.3f}, dec {miss['dec_deg']:.3f}, "
                f"E(B-V) {miss['ebv']:.4f}. {rules}"
            )
    return lines

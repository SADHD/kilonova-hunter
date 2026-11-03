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
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from kilonova_hunter.config import THRESHOLDS, Thresholds
from kilonova_hunter.features import g_minus_r_at_time, g_minus_r_rise, peak_time_days
from kilonova_hunter.filter import evaluate_fade_rule, is_kilonova_candidate
from kilonova_hunter.judge import JUDGE_DAYS, is_judgeable
from kilonova_hunter.recall import light_curve_for_the_filter, wilson_interval
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
    peak = peak_time_days(visible, "r")
    if peak is None:
        return None
    color = g_minus_r_at_time(
        visible,
        peak + THRESHOLDS.color_days_after_peak,
        max_gap_days=THRESHOLDS.max_color_interpolation_gap_days,
    )
    return color.g_minus_r


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
        # 0/1 after a CSV round trip, or a real bool from a fresh frame.
        failed = bool(
            int(pd.to_numeric(pd.Series([row.model_failed])).fillna(0).iloc[0])
        )
        merger = float(row.merger_mjd)
        judgeable = False if failed else is_judgeable(curve, day, merger_mjd=merger)
        visible = light_curve_for_the_filter(curve, day)
        if failed or visible.empty:
            prepared.append(
                PreparedFake(
                    object_id=str(row.object_id),
                    distance_mpc=float(row.distance_mpc),
                    judgeable=judgeable,
                    model_failed=failed,
                    fade_passed=False,
                    color_value=None,
                    rises={pair: None for pair in windows},
                )
            )
            continue
        fade_passed = evaluate_fade_rule(visible).passed is True
        color_value = _color_value(visible)
        rises = {pair: _rise_or_skip(visible, pair[0], pair[1]) for pair in windows}
        prepared.append(
            PreparedFake(
                object_id=str(row.object_id),
                distance_mpc=float(row.distance_mpc),
                judgeable=judgeable,
                model_failed=failed,
                fade_passed=fade_passed,
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
                fake
                for fake in prepared
                if low_mpc <= fake.distance_mpc <= high_mpc
            ]
        else:
            chosen = [
                fake for fake in prepared if low_mpc <= fake.distance_mpc < high_mpc
            ]
    caught = 0
    judgeable = 0
    judge_caught = 0
    skipped = 0
    for fake in chosen:
        is_caught, reddening_skipped = caught_by(fake, combo)
        if fake.judgeable:
            judgeable += 1
        if is_caught:
            caught += 1
            if fake.judgeable:
                judge_caught += 1
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
        **combo,
        **rates,
    }
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

    Copies are not in the recall. Zero judgeable copies is a sign-off
    blocker, and so is catching fewer than 95 percent of the judgeable
    copies. Misses are listed with the start time, the sky position,
    E(B-V), and the rule that failed.
    """
    if events.empty:
        return {
            "n_copies": 0,
            "n_judgeable": 0,
            "n_caught_judgeable": 0,
            "fraction_judgeable_caught": float("nan"),
            "blocker": True,
            "misses": [],
        }
    by_id = _curves_by_id(curves)
    settings = thresholds_for(combo)
    n_judge = 0
    n_caught = 0
    misses = []
    for row in events.itertuples(index=False):
        curve = by_id.get(str(row.object_id), pd.DataFrame())
        judgeable = is_judgeable(curve, day, merger_mjd=float(row.merger_mjd))
        if not judgeable:
            continue
        n_judge += 1
        visible = light_curve_for_the_filter(curve, day)
        verdict = is_kilonova_candidate(visible, settings)
        if verdict.is_candidate:
            n_caught += 1
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
            }
        )
    fraction = float(n_caught / n_judge) if n_judge else float("nan")
    # No judgeable copy is not a pass. A NaN fraction used to hide that.
    blocker = n_judge == 0 or fraction < 0.95
    return {
        "n_copies": int(len(events)),
        "n_judgeable": n_judge,
        "n_caught_judgeable": n_caught,
        "fraction_judgeable_caught": fraction,
        "blocker": blocker,
        "misses": misses,
        "day": day,
        "cuts": combo,
    }


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
            lines.append("")
    return "\n".join(lines)


def write_grid_csv(rows: list[dict], path: Path) -> None:
    """Write the grid table. Rates that were left blank stay blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
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
            f"Caught but not judgeable, at that combination and day 3: {offenders}."
        )
        if offenders:
            lines.append(
                "That count is not zero. The filter can flag a fake the frozen "
                "judgeable rule rejects, because the color rule may interpolate "
                "g and r from different nights and the fade rule may use two "
                "points closer together than 0.5 days. Those fakes are listed "
                "in the run log. They stay in the catch rate, because the catch "
                "rate is every fake the filter flagged."
            )
    if copies is not None:
        lines.extend(_copy_lines(copies))
    lines.append("")
    lines.append("## Distance bins")
    lines.append("")
    lines.append(
        "Bins are luminosity distance: 10-100, 100-200, and 200-300 Mpc. "
        "The last bin ends at 300."
    )
    return "\n".join(lines) + "\n"


def run_tuning_grid(
    *,
    output_dir: Path | str = Path("results"),
    cache_dir: Path | str = Path("data") / "recall" / "cache",
    sample_size=None,
    workers: int = 1,
    settings_path: Path | None = None,
) -> str:
    """Simulate both populations and both splits, then score the tuning grid.

    The test split is simulated and cached so a later score does not
    redraw it. It is not scored here. No cut is picked.
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
    samples = []
    rows: list[dict] = []
    prepared_for_examples: dict[str, tuple] = {}
    for population in (LIKE_KIND, REALISTIC):
        for split in SPLITS:
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
                day: prepare_fakes(sample.events, sample.light_curves, day, settings)
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
    copies = simulate_exact_copies(dust=dust, cache_dir=cache_dir, workers=workers)
    placeholder = {
        "redder_than_g_minus_r": THRESHOLDS.redder_than_g_minus_r,
        "min_g_minus_r_rise": THRESHOLDS.min_g_minus_r_rise,
        "reddening_window_days": THRESHOLDS.reddening_window_days,
        "min_color_separation_days": THRESHOLDS.min_color_separation_days,
    }
    copy_score = score_exact_copies(copies.events, copies.light_curves, placeholder)
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
    (destination / "tuning_grid.md").write_text(text, encoding="utf-8")
    return text


def _worst_not_judgeable(rows: list[dict], prepared_for_examples: dict) -> str:
    """Lead with the investigation when any fake is caught but not judgeable.

    The check covers days 3, 5, and 7. A hit on day 7 is still reported.
    That case is a rule mismatch, not a bug in the day-D cutoff, and the
    official judgeable rule is left as it is.
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
    worst = max(headline, key=lambda row: int(row["n_caught_not_judgeable"]))
    population = worst["population"]
    prepared_by_day, curves = prepared_for_examples[population]
    day = float(worst["day"])
    prepared = prepared_by_day[day]
    combo = {
        "redder_than_g_minus_r": worst["redder_than_g_minus_r"],
        "min_g_minus_r_rise": worst["min_g_minus_r_rise"],
        "reddening_window_days": worst["reddening_window_days"],
        "min_color_separation_days": worst["min_color_separation_days"],
    }
    examples = examples_caught_but_not_judgeable(prepared, curves, combo, day=day)
    lines = [
        "# Caught but not judgeable",
        "",
        f"{worst['population']} day {worst['day']}: "
        f"{worst['n_caught_not_judgeable']} fakes were flagged and were not judgeable.",
        "The filter can do this when g and r fall on different nights "
        "(the color rule still interpolates) or when two points in one band "
        "are closer than 0.5 days (the fade rule can still fit them). "
        "The catch rate still counts them. The recall among judgeable fakes does not.",
        "",
    ]
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
        f"{_power_clause(row.get('power_at_70', float('nan')))}"
    )


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


def _copy_lines(copies: dict) -> list[str]:
    fraction = copies["fraction_judgeable_caught"]
    if fraction != fraction:
        fraction_text = "no judgeable copies"
    else:
        fraction_text = f"{100.0 * fraction:.1f}% of judgeable copies caught"
    blocker = "SIGN-OFF BLOCKER. " if copies["blocker"] else ""
    lines = [
        "",
        "## Exact AT2017gfo copies",
        "",
        "These copies are not in the recall or the catch rate. "
        "The fraction below uses the placeholder cuts in config.py, "
        "not a cut picked from the grid.",
        "",
        f"{blocker}{copies['n_caught_judgeable']} of {copies['n_judgeable']} "
        f"judgeable copies caught ({fraction_text}), "
        f"out of {copies['n_copies']} copies.",
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

"""Keep simulating until enough fakes are judgeable at day 3.

Fakes are drawn in seeded batches. The default batch is 500. After
each batch we count how many are judgeable at day 3, and we do not
look at the filter. We stop after the first batch that reaches the
target. The default target is 400. The hard cap is 20,000 fakes.
Those three numbers, and the untestable floor of 50, are settings,
because the target may move again.

If the cap is hit with fewer than 50 judgeable at day 3, day 3 is
``untestable`` and the recall is left blank. If the cap is hit with
at least 50 but fewer than the target, the run is ``underpowered``:
the recall is reported, along with the chance that a sample of this
size would put the Wilson lower bound at or above 70 percent if the
true recall equals the recall we measured.

Light curves are written under the cache directory. A later grid
reads them instead of simulating again. Raising the target and
running again continues with the next batch and the same seeds. It
does not redraw the batches already on disk.

Tuning and test are separate seeds from one numpy ``SeedSequence``
whose parent entropy is 1100 (the same integer the older tuning set
uses). Realistic and 2017-like are separate streams too.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from kilonova_hunter.judge import JUDGE_DAYS, is_candidate_judgeable, is_judgeable
from kilonova_hunter.recall import (
    MODEL_ERROR_ABORT_FRACTION,
    TUNING_SEED,
    PopulationSettings,
    SurveySettings,
    ejecta_masses_and_speeds,
    sample_ebv,
    sample_masses,
    sample_viewing_angle_deg,
    simulate_event,
)
from kilonova_hunter.redback_fix import DIFFUSION_FIX_ID
from kilonova_hunter.sky import (
    DustSource,
    draw_footprint_positions,
    draw_merger_mjd,
    dust_map_label,
    open_dust_source,
    redshift_at_luminosity_distance,
    sample_luminosity_distances,
)
from kilonova_hunter.villar import (
    EXACT_BATCH_SIZE,
    EXACT_CAP,
    EXACT_COUNT,
    EXACT_DISTANCE_MPC,
    EXACT_KIND,
    EXACT_SEED,
    EXACT_TARGET_JUDGEABLE,
    LIKE_KIND,
    VillarEvent,
    VillarSpread,
    draw_components,
    exact_components,
)

logger = logging.getLogger(__name__)

REALISTIC = "realistic"
POPULATIONS = (LIKE_KIND, REALISTIC)
SPLITS = ("tuning", "test")

# Bump this when a cached light curve would no longer match the code.
# 2: the diffusion wrapper is on. A version-1 cache is pre-fix.
CACHE_VERSION = 2

# Parent entropy shared by the tuning stream and the test stream.
PARENT_SEED = TUNING_SEED


@dataclass(frozen=True)
class SampleSizeSettings:
    """When to stop simulating.

    ``target_judgeable`` is how many day-3 judgeable fakes we want.
    ``cap`` is the most fakes we will simulate. ``untestable_floor``
    is the smallest day-3 count that still gets a recall number.
    ``batch_size`` is how many fakes are drawn between counts.
    """

    target_judgeable: int = 400
    cap: int = 20_000
    untestable_floor: int = 50
    batch_size: int = 500

    def __post_init__(self) -> None:
        if self.target_judgeable < 1:
            raise ValueError("the judgeable target must be at least 1.")
        if self.cap < 1:
            raise ValueError("the cap must be at least 1.")
        if self.batch_size < 1:
            raise ValueError("the batch size must be at least 1.")
        if not 0 <= self.untestable_floor <= self.target_judgeable:
            raise ValueError(
                "the untestable floor must sit between 0 and the target, "
                f"got {self.untestable_floor} with target {self.target_judgeable}."
            )


def sample_status(
    n_simulated: int, n_judgeable: int, settings: SampleSizeSettings
) -> str:
    """``continue``, ``ready``, ``underpowered``, or ``untestable``.

    Call this after a batch, not in the middle of one. A batch that
    crosses the target is kept whole.
    """
    if n_judgeable >= settings.target_judgeable:
        return "ready"
    if n_simulated >= settings.cap:
        if n_judgeable < settings.untestable_floor:
            return "untestable"
        return "underpowered"
    return "continue"


def next_batch_size(n_simulated: int, settings: SampleSizeSettings) -> int:
    """How many fakes the next batch may draw without passing the cap."""
    remaining = settings.cap - n_simulated
    if remaining <= 0:
        return 0
    return min(settings.batch_size, remaining)


def evaluation_survey() -> SurveySettings:
    """Wide-fast-deep visits, long enough to cover day 7.

    Eight days leaves a visit that falls on day 7 inside the survey.
    The official windows are still 3, 5, and 7 days from the merger.
    """
    return SurveySettings(cadence="wfd", survey_length_days=8.0)


@dataclass
class SampleResult:
    """One cached population and split."""

    population: str
    split: str
    status: str
    n_simulated: int
    n_judgeable_day3: int
    n_judgeable_day5: int
    n_judgeable_day7: int
    n_model_failed: int
    dust_mode: str
    dust_reason: str
    events: pd.DataFrame
    light_curves: pd.DataFrame
    cache_dir: Path


def _split_id(split: str) -> int:
    if split == "tuning":
        return 0
    if split == "test":
        return 1
    raise ValueError(f"split must be tuning or test, got {split!r}.")


def _population_id(population: str) -> int:
    if population == REALISTIC:
        return 0
    if population == LIKE_KIND:
        return 1
    raise ValueError(f"unknown population {population!r}.")


def parameter_rng(split: str, population: str, batch_index: int) -> np.random.Generator:
    """The random stream that draws masses, distances, and sky positions.

    Tuning and test differ only by ``split_id``. Both streams are
    children of the same parent entropy, ``PARENT_SEED``.
    """
    entropy = [
        PARENT_SEED,
        _split_id(split),
        _population_id(population),
        int(batch_index),
        1,
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def survey_seed(split: str, population: str, batch_index: int, local_index: int) -> int:
    """Seed for one fake's visit times and noise. Independent of the draw."""
    entropy = [
        PARENT_SEED,
        _split_id(split),
        _population_id(population),
        int(batch_index),
        2,
        int(local_index),
    ]
    state = np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)
    return int(state[0])


def _ebv_for_batch(
    dust: DustSource,
    ra: np.ndarray,
    dec: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """SFD E(B-V), or the exponential stand-in when this is a dry run."""
    if dust.mode == "sfd":
        from kilonova_hunter.sky import ebv_from_sfd

        return ebv_from_sfd(ra, dec)
    settings = PopulationSettings()
    return np.array([sample_ebv(rng, settings) for _ in range(len(ra))], dtype=float)


def draw_realistic_batch(
    split: str,
    batch_index: int,
    count: int,
    start_index: int,
    dust: DustSource,
) -> tuple[list, list[int]]:
    """Draw one batch of realistic mergers. No light curves yet.

    Order inside the random stream, so a seed can be replayed:
    luminosity distances, then sky positions, then for each fake the
    two neutron-star masses, the viewing angle, and the merger time.
    E(B-V) comes from the map, so it does not use the stream. A dry
    run draws the exponential E(B-V) last.
    """
    rng = parameter_rng(split, REALISTIC, batch_index)
    redshifts, distances = sample_luminosity_distances(rng, count)
    ra, dec, galactic_b = draw_footprint_positions(rng, count)
    blanks = PopulationSettings()
    events = []
    for index in range(count):
        mass_1, mass_2 = sample_masses(rng, blanks)
        angle = sample_viewing_angle_deg(rng)
        merger = float(draw_merger_mjd(rng, 1)[0])
        events.append(
            {
                "mass_1": mass_1,
                "mass_2": mass_2,
                "angle": angle,
                "merger_mjd": merger,
                "distance_mpc": float(distances[index]),
                "redshift": float(redshifts[index]),
                "ra_deg": float(ra[index]),
                "dec_deg": float(dec[index]),
                "galactic_b_deg": float(galactic_b[index]),
            }
        )
    ebv = _ebv_for_batch(dust, ra, dec, rng)
    from kilonova_hunter.recall import FakeEvent

    built = []
    seeds = []
    for index, item in enumerate(events):
        number = start_index + index
        built.append(
            FakeEvent(
                object_id=f"realistic-{split}-{number:05d}",
                kind=REALISTIC,
                mass_1=item["mass_1"],
                mass_2=item["mass_2"],
                distance_mpc=item["distance_mpc"],
                redshift=item["redshift"],
                viewing_angle_deg=item["angle"],
                ebv=float(ebv[index]),
                lambda_1=blanks.lambda_1,
                lambda_2=blanks.lambda_2,
                mtov=blanks.mtov,
                zeta=blanks.zeta,
                vej_wind=blanks.vej_wind,
                kappa_dynamical=blanks.kappa_dynamical,
                kappa_wind=blanks.kappa_wind,
                tf_dynamical=blanks.tf_dynamical,
                tf_wind=blanks.tf_wind,
                ra_deg=item["ra_deg"],
                dec_deg=item["dec_deg"],
                galactic_b_deg=item["galactic_b_deg"],
                merger_mjd=item["merger_mjd"],
            )
        )
        seeds.append(survey_seed(split, REALISTIC, batch_index, index))
    return built, seeds


def draw_villar_batch(
    split: str,
    batch_index: int,
    count: int,
    start_index: int,
    dust: DustSource,
    spread: VillarSpread | None = None,
) -> tuple[list[VillarEvent], list[int]]:
    """Draw one batch of 2017-like fakes. Components are independent.

    Order: distances, sky, then for each fake the merger time, the
    viewing angle, and blue, purple, red mass and speed. Opacity stays
    at the table value inside ``draw_components``.
    """
    rng = parameter_rng(split, LIKE_KIND, batch_index)
    settings = VillarSpread() if spread is None else spread
    redshifts, distances = sample_luminosity_distances(rng, count)
    ra, dec, galactic_b = draw_footprint_positions(rng, count)
    partial = []
    for _index in range(count):
        merger = float(draw_merger_mjd(rng, 1)[0])
        angle = sample_viewing_angle_deg(rng)
        blue, purple, red = draw_components(rng, settings)
        partial.append((merger, angle, blue, purple, red))
    ebv = _ebv_for_batch(dust, ra, dec, rng)
    built: list[VillarEvent] = []
    seeds: list[int] = []
    for index, (merger, angle, blue, purple, red) in enumerate(partial):
        number = start_index + index
        built.append(
            VillarEvent(
                object_id=f"2017-like-{split}-{number:05d}",
                kind=LIKE_KIND,
                distance_mpc=float(distances[index]),
                redshift=float(redshifts[index]),
                ebv=float(ebv[index]),
                ra_deg=float(ra[index]),
                dec_deg=float(dec[index]),
                galactic_b_deg=float(galactic_b[index]),
                merger_mjd=merger,
                viewing_angle_deg=angle,
                blue=blue,
                purple=purple,
                red=red,
            )
        )
        seeds.append(survey_seed(split, LIKE_KIND, batch_index, index))
    return built, seeds


def draw_exact_copies(
    dust: DustSource,
    count: int = EXACT_COUNT,
    seed: int = EXACT_SEED,
    *,
    start_index: int = 0,
    batch_index: int = 0,
) -> tuple[list[VillarEvent], list[int]]:
    """``count`` copies of the Table 2 center, at 40 Mpc.

    Start times and sky positions are random. The ejecta are not.
    The seed is ``EXACT_SEED`` plus the batch index, not the tuning or
    test seed, so these copies do not consume either population stream.
    They are not part of a recall denominator. A later batch uses a
    later ``batch_index`` so it does not repeat the first batch.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, batch_index]))
    redshift = redshift_at_luminosity_distance(EXACT_DISTANCE_MPC)
    ra, dec, galactic_b = draw_footprint_positions(rng, count)
    blue, purple, red = exact_components()
    mergers = draw_merger_mjd(rng, count)
    angles = [sample_viewing_angle_deg(rng) for _ in range(count)]
    ebv = _ebv_for_batch(dust, ra, dec, rng)
    events: list[VillarEvent] = []
    seeds: list[int] = []
    for index in range(count):
        number = start_index + index
        events.append(
            VillarEvent(
                object_id=f"{EXACT_KIND}-{number:05d}",
                kind=EXACT_KIND,
                distance_mpc=EXACT_DISTANCE_MPC,
                redshift=redshift,
                ebv=float(ebv[index]),
                ra_deg=float(ra[index]),
                dec_deg=float(dec[index]),
                galactic_b_deg=float(galactic_b[index]),
                merger_mjd=float(mergers[index]),
                viewing_angle_deg=angles[index],
                blue=blue,
                purple=purple,
                red=red,
            )
        )
        state = np.random.SeedSequence([seed, batch_index, index, 3]).generate_state(
            1, dtype=np.uint32
        )
        seeds.append(int(state[0]))
    return events, seeds


def _simulate_one(payload: dict) -> tuple[str, pd.DataFrame, str | None]:
    """Observe one fake. The filter is not called."""
    population = payload["population"]
    if population == "stub":
        from kilonova_hunter.recall import stub_magnitudes

        model = stub_magnitudes
    elif population == REALISTIC:
        from kilonova_hunter.villar import realistic_magnitudes as model
    else:
        from kilonova_hunter.villar import villar_magnitudes as model
    observed, error_reason, _peaks = simulate_event(
        payload["event"],
        payload["survey"],
        np.random.default_rng(payload["seed"]),
        model,
    )
    return payload["event"].object_id, observed, error_reason


def _run_jobs(
    jobs: list[dict], workers: int
) -> list[tuple[str, pd.DataFrame, str | None]]:
    """Simulate a batch, in order. ``workers`` of 1 stays in this process."""
    if workers <= 1 or len(jobs) <= 1:
        return [_simulate_one(job) for job in jobs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_simulate_one, jobs))


def _event_record(event, *, error_reason: str | None, curve: pd.DataFrame) -> dict:
    """One fake as a flat row. Judgeable flags do not use the filter."""
    failed = error_reason is not None
    if isinstance(event, VillarEvent):
        row: dict[str, object] = {
            "object_id": event.object_id,
            "kind": event.kind,
            "distance_mpc": event.distance_mpc,
            "redshift": event.redshift,
            "ebv": event.ebv,
            "ra_deg": event.ra_deg,
            "dec_deg": event.dec_deg,
            "galactic_b_deg": event.galactic_b_deg,
            "merger_mjd": event.merger_mjd,
            "viewing_angle_deg": event.viewing_angle_deg,
            "mej_blue": event.blue.mej,
            "vej_blue": event.blue.vej,
            "kappa_blue": event.blue.kappa,
            "mej_purple": event.purple.mej,
            "vej_purple": event.purple.vej,
            "kappa_purple": event.purple.kappa,
            "mej_red": event.red.mej,
            "vej_red": event.red.vej,
            "kappa_red": event.red.kappa,
        }
    else:
        row = {field.name: getattr(event, field.name) for field in fields(event)}
        row.update(ejecta_masses_and_speeds(event))
    row["model_failed"] = int(failed)
    row["model_error_reason"] = "" if error_reason is None else error_reason
    for day in JUDGE_DAYS:
        flag = (
            False if failed else is_judgeable(curve, day, merger_mjd=event.merger_mjd)
        )
        row[f"judgeable_day{int(day)}"] = int(flag)
        candidate = (
            False
            if failed
            else is_candidate_judgeable(curve, day, merger_mjd=event.merger_mjd)
        )
        row[f"judgeable_candidate_day{int(day)}"] = int(candidate)
    return row


def _curve_frame(object_id: str, curve: pd.DataFrame) -> pd.DataFrame | None:
    if curve is None or curve.empty:
        return None
    frame = curve.copy()
    frame.insert(0, "object_id", object_id)
    return frame


def _abort_if_model_is_broken(n_failed: int, n_total: int) -> None:
    if n_total <= 0:
        return
    fraction = n_failed / n_total
    if fraction > MODEL_ERROR_ABORT_FRACTION:
        raise RuntimeError(
            "The light-curve model failed for "
            f"{n_failed} of {n_total} fakes "
            f"({100.0 * fraction:.0f}%), which is more than "
            f"{100.0 * MODEL_ERROR_ABORT_FRACTION:.0f}%. "
            "That is a broken model, not a survey that missed them."
        )


def _cache_paths(cache_dir: Path) -> tuple[Path, Path, Path]:
    return (
        cache_dir / "manifest.json",
        cache_dir / "events.csv",
        cache_dir / "light_curves.csv",
    )


def _empty_curves() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["object_id", "time_days", "filter", "magnitude", "magnitude_error"]
    )


def _read_cache(cache_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame] | None:
    manifest_path, events_path, curves_path = _cache_paths(cache_dir)
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = pd.read_csv(events_path) if events_path.is_file() else pd.DataFrame()
    curves = pd.read_csv(curves_path) if curves_path.is_file() else _empty_curves()
    return manifest, events, curves


def _write_cache(
    cache_dir: Path,
    manifest: dict,
    events: pd.DataFrame,
    curves: pd.DataFrame,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, events_path, curves_path = _cache_paths(cache_dir)
    events.to_csv(events_path, index=False)
    curves.to_csv(curves_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _survey_dict(survey: SurveySettings) -> dict:
    """The survey numbers a cache must match before it is reused.

    Depths and the error floor are part of the key. A cache built with
    different limits is a different survey, even when the cadence matches.
    """
    assert survey.limits is not None
    return {
        "cadence": survey.cadence,
        "wfd_gap_min_days": survey.wfd_gap_min_days,
        "wfd_gap_max_days": survey.wfd_gap_max_days,
        "survey_length_days": survey.survey_length_days,
        "bands": list(survey.bands),
        "limits": {band: float(survey.limits[band]) for band in survey.bands},
        "error_floor_mag": float(survey.error_floor_mag),
    }


def _redback_version() -> str:
    """Installed redback version, or ``not-installed`` when it is absent."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("redback")
    except PackageNotFoundError:
        return "not-installed"


def _recorded_dust_map(dust: DustSource) -> str:
    """What the manifest should say about the dust map for this run."""
    if dust.mode == "sfd":
        return dust_map_label()
    return "dry-run"


def _same_setup(
    manifest: dict,
    *,
    population: str,
    split: str,
    dust: DustSource,
    survey: SurveySettings,
) -> bool:
    return (
        manifest.get("cache_version") == CACHE_VERSION
        and manifest.get("population") == population
        and manifest.get("split") == split
        and manifest.get("parent_seed") == PARENT_SEED
        and manifest.get("dust_mode") == dust.mode
        and manifest.get("dust_map") == _recorded_dust_map(dust)
        and manifest.get("redback_version") == _redback_version()
        and manifest.get("diffusion_fix") == DIFFUSION_FIX_ID
        and manifest.get("survey") == _survey_dict(survey)
    )


def _result_from_frames(
    *,
    population: str,
    split: str,
    status: str,
    events: pd.DataFrame,
    curves: pd.DataFrame,
    dust: DustSource,
    cache_dir: Path,
) -> SampleResult:
    def _count(day: int) -> int:
        return _judge_count(events, day)

    failed = _flag_sum(events, "model_failed")
    return SampleResult(
        population=population,
        split=split,
        status=status,
        n_simulated=int(len(events)),
        n_judgeable_day3=_count(3),
        n_judgeable_day5=_count(5),
        n_judgeable_day7=_count(7),
        n_model_failed=failed,
        dust_mode=dust.mode,
        dust_reason=dust.reason,
        events=events,
        light_curves=curves,
        cache_dir=cache_dir,
    )


def simulate_population(
    population: str,
    split: str,
    *,
    sample_size: SampleSizeSettings | None = None,
    survey: SurveySettings | None = None,
    dust: DustSource | None = None,
    cache_dir: Path | str = Path("data") / "recall" / "cache",
    workers: int = 1,
) -> SampleResult:
    """Simulate one population and one split, reusing a matching cache.

    The day-3 judgeable count is the only number that decides whether
    another batch is drawn. Day 5 and day 7 are filled in afterwards
    from the same light curves. The filter is not called.
    """
    settings = sample_size if sample_size is not None else SampleSizeSettings()
    sky = survey if survey is not None else evaluation_survey()
    dust_source = dust if dust is not None else open_dust_source()
    folder = Path(cache_dir) / population / split
    loaded = _read_cache(folder)
    events = pd.DataFrame()
    curves = _empty_curves()
    completed = 0
    if loaded is not None and _same_setup(
        loaded[0],
        population=population,
        split=split,
        dust=dust_source,
        survey=sky,
    ):
        manifest, events, curves = loaded
        completed = int(manifest.get("completed_batches", 0))
        status = sample_status(len(events), _judge_count(events, 3), settings)
        if status != "continue":
            manifest["status"] = status
            _write_cache(folder, manifest, events, curves)
            return _result_from_frames(
                population=population,
                split=split,
                status=status,
                events=events,
                curves=curves,
                dust=dust_source,
                cache_dir=folder,
            )
        logger.info(
            "%s %s: cache has %s fakes, continuing from batch %s.",
            population,
            split,
            len(events),
            completed,
        )
    elif loaded is not None:
        logger.info(
            "%s %s: cache settings differ, starting over.",
            population,
            split,
        )

    curve_frames = [curves] if not curves.empty else []
    event_frames = [events] if not events.empty else []
    while True:
        done = 0 if events.empty else len(events)
        # Rebuild the running total from the frames we have appended.
        if event_frames:
            events = pd.concat(event_frames, ignore_index=True)
            done = len(events)
        count = next_batch_size(done, settings)
        if count == 0:
            break
        if population == REALISTIC:
            batch, seeds = draw_realistic_batch(
                split, completed, count, done, dust_source
            )
        elif population == LIKE_KIND:
            batch, seeds = draw_villar_batch(split, completed, count, done, dust_source)
        else:
            raise ValueError(f"unknown population {population!r}.")
        jobs = [
            {"population": population, "event": event, "survey": sky, "seed": seed}
            for event, seed in zip(batch, seeds, strict=True)
        ]
        logger.info(
            "%s %s: simulating batch %s (%s fakes, %s so far).",
            population,
            split,
            completed,
            count,
            done,
        )
        simulated = _run_jobs(jobs, workers)
        rows = []
        new_curves = []
        by_id = {item[0]: item for item in simulated}
        for event in batch:
            object_id, curve, error_reason = by_id[event.object_id]
            rows.append(_event_record(event, error_reason=error_reason, curve=curve))
            framed = _curve_frame(object_id, curve)
            if framed is not None:
                new_curves.append(framed)
        event_frames.append(pd.DataFrame(rows))
        if new_curves:
            curve_frames.extend(new_curves)
        events = pd.concat(event_frames, ignore_index=True)
        curves = (
            pd.concat(curve_frames, ignore_index=True)
            if curve_frames
            else _empty_curves()
        )
        completed += 1
        n_failed = _flag_sum(events, "model_failed")
        _abort_if_model_is_broken(n_failed, len(events))
        n_judge = _judge_count(events, 3)
        status = sample_status(len(events), n_judge, settings)
        manifest = _manifest(
            population=population,
            split=split,
            dust=dust_source,
            survey=sky,
            settings=settings,
            completed=completed,
            events=events,
            status=status,
        )
        _write_cache(folder, manifest, events, curves)
        logger.info(
            "%s %s: %s simulated, %s judgeable at day 3, status %s.",
            population,
            split,
            len(events),
            n_judge,
            status,
        )
        if status != "continue":
            return _result_from_frames(
                population=population,
                split=split,
                status=status,
                events=events,
                curves=curves,
                dust=dust_source,
                cache_dir=folder,
            )
    status = sample_status(len(events), _judge_count(events, 3), settings)
    return _result_from_frames(
        population=population,
        split=split,
        status=status,
        events=events,
        curves=curves,
        dust=dust_source,
        cache_dir=folder,
    )


def simulate_exact_copies(
    *,
    survey: SurveySettings | None = None,
    dust: DustSource | None = None,
    cache_dir: Path | str = Path("data") / "recall" / "cache",
    workers: int = 1,
    count: int | None = None,
    target_judgeable: int = EXACT_TARGET_JUDGEABLE,
    cap: int = EXACT_CAP,
    batch_size: int = EXACT_BATCH_SIZE,
) -> SampleResult:
    """Simulate exact AT2017gfo copies. They are not a recall set.

    The default follows Decision Log 12(16), which is proposed and
    still waiting on Sage. Batches are drawn at 40 Mpc, with random
    cadence dates and random positions, until at least
    ``target_judgeable`` copies are judgeable at day 3 under the
    official definition, or ``cap`` copies have been drawn. Pass
    ``count`` to draw a fixed number instead. That fixed draw is for
    tests. The report uses the default.
    """
    sky = survey if survey is not None else evaluation_survey()
    dust_source = dust if dust is not None else open_dust_source()
    folder = Path(cache_dir) / EXACT_KIND / "copies"
    loaded = _read_cache(folder)
    if loaded is not None and _exact_cache_matches(
        loaded[0],
        dust=dust_source,
        survey=sky,
        count=count,
        target_judgeable=target_judgeable,
        cap=cap,
    ):
        manifest, events, curves = loaded
        return _result_from_frames(
            population=EXACT_KIND,
            split="copies",
            status=str(manifest.get("status", "ready")),
            events=events,
            curves=curves,
            dust=dust_source,
            cache_dir=folder,
        )
    rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    drawn = 0
    batch_index = 0
    goal = count if count is not None else cap
    while drawn < goal:
        if count is None and _flag_sum_rows(rows, "judgeable_day3") >= target_judgeable:
            break
        this_batch = batch_size if count is None else count
        this_batch = min(this_batch, goal - drawn)
        events_list, seeds = draw_exact_copies(
            dust_source,
            count=this_batch,
            start_index=drawn,
            batch_index=batch_index,
        )
        jobs = [
            {"population": LIKE_KIND, "event": event, "survey": sky, "seed": seed}
            for event, seed in zip(events_list, seeds, strict=True)
        ]
        simulated = _run_jobs(jobs, workers)
        by_id = {item[0]: item for item in simulated}
        for event in events_list:
            object_id, curve, error_reason = by_id[event.object_id]
            rows.append(_event_record(event, error_reason=error_reason, curve=curve))
            framed = _curve_frame(object_id, curve)
            if framed is not None:
                frames.append(framed)
        drawn += this_batch
        batch_index += 1
        if count is not None:
            break
    events = pd.DataFrame(rows)
    curves = pd.concat(frames, ignore_index=True) if frames else _empty_curves()
    n_judge = _flag_sum(events, "judgeable_day3")
    if count is None and n_judge < target_judgeable:
        status = "untestable"
    else:
        status = "ready"
    manifest = {
        "cache_version": CACHE_VERSION,
        "population": EXACT_KIND,
        "split": "copies",
        "dust_mode": dust_source.mode,
        "dust_map": _recorded_dust_map(dust_source),
        "dust_reason": dust_source.reason,
        "redback_version": _redback_version(),
        "diffusion_fix": DIFFUSION_FIX_ID,
        "survey": _survey_dict(sky),
        "n_simulated": len(events),
        "n_judgeable_day3": n_judge,
        "seed": EXACT_SEED,
        "target_judgeable": None if count is not None else target_judgeable,
        "cap": None if count is not None else cap,
        "fixed_count": count,
        "status": status,
    }
    _write_cache(folder, manifest, events, curves)
    return _result_from_frames(
        population=EXACT_KIND,
        split="copies",
        status=status,
        events=events,
        curves=curves,
        dust=dust_source,
        cache_dir=folder,
    )


def _flag_sum_rows(rows: list[dict], column: str) -> int:
    return sum(int(row.get(column, 0)) for row in rows)


def _exact_cache_matches(
    manifest: dict,
    *,
    dust: DustSource,
    survey: SurveySettings,
    count: int | None,
    target_judgeable: int,
    cap: int,
) -> bool:
    """True when a saved copy cache is the current code and the same draw."""
    if manifest.get("cache_version") != CACHE_VERSION:
        return False
    if manifest.get("diffusion_fix") != DIFFUSION_FIX_ID:
        return False
    if manifest.get("dust_mode") != dust.mode:
        return False
    if manifest.get("dust_map") != _recorded_dust_map(dust):
        return False
    if manifest.get("redback_version") != _redback_version():
        return False
    if manifest.get("survey") != _survey_dict(survey):
        return False
    if count is not None:
        return int(manifest.get("n_simulated", 0)) == count
    return (
        manifest.get("target_judgeable") == target_judgeable
        and manifest.get("cap") == cap
        and manifest.get("fixed_count") is None
    )


def cache_refusal_reason(manifest: dict) -> str | None:
    """Why a cache must not be scored or reused, or None when it may.

    A cache from before the diffusion wrapper has no ``diffusion_fix``
    marker. That cache is refused. An unfinished run (``continue`` or
    ``running``) is refused too.
    """
    if manifest.get("cache_version") != CACHE_VERSION:
        return (
            f"cache_version is {manifest.get('cache_version')!r}, "
            f"and the current code is {CACHE_VERSION}."
        )
    if manifest.get("diffusion_fix") != DIFFUSION_FIX_ID:
        return (
            "the cache has no diffusion-fix marker "
            f"{DIFFUSION_FIX_ID!r}. A pre-fix cache is refused."
        )
    status = str(manifest.get("status", ""))
    if status in {"continue", "running"}:
        return f"the cache status is {status!r}. Finish the simulation first."
    if status not in {"ready", "underpowered", "untestable"}:
        return f"the cache status is {status!r}, which is not a finished run."
    return None


def _flag_sum(events: pd.DataFrame, column: str) -> int:
    """Sum a yes/no column, including after a CSV round trip.

    The string "False" is not true. ``astype(bool)`` would count
    that word as true because any non-empty string is true in Python.
    """
    if events.empty or column not in events.columns:
        return 0
    raw = events[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    words = raw.astype(str).str.strip().str.lower()
    filled = numeric.where(~numeric.isna(), words.map({"true": 1.0, "false": 0.0}))
    return int(filled.fillna(0).sum())


def _judge_count(events: pd.DataFrame, day: int) -> int:
    if events.empty:
        return 0
    return _flag_sum(events, f"judgeable_day{day}")


def _manifest(
    *,
    population: str,
    split: str,
    dust: DustSource,
    survey: SurveySettings,
    settings: SampleSizeSettings,
    completed: int,
    events: pd.DataFrame,
    status: str,
) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "population": population,
        "split": split,
        "parent_seed": PARENT_SEED,
        "dust_mode": dust.mode,
        "dust_map": _recorded_dust_map(dust),
        "dust_reason": dust.reason,
        "redback_version": _redback_version(),
        "diffusion_fix": DIFFUSION_FIX_ID,
        "survey": _survey_dict(survey),
        "sample_size": asdict(settings),
        "completed_batches": completed,
        "n_simulated": int(len(events)),
        "n_judgeable_day3": _judge_count(events, 3),
        "n_model_failed": _flag_sum(events, "model_failed"),
        "status": status,
    }

"""Synthetic-only held-score harness verification; no real sets, maps or ledger."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from kilonova_hunter import cli, grid, sample, sky
from kilonova_hunter.redback_fix import DIFFUSION_FIX_ID


def synthetic_cache(folder, population):
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "ready",
        "cache_version": sample.CACHE_VERSION,
        "diffusion_fix": DIFFUSION_FIX_ID,
        "dust_mode": "sfd",
        "dust_map": sky.dust_map_label(),
        "n_simulated": 1,
        "parent_seed": sample.PARENT_SEED,
        "population": population,
        "split": "test",
        "sample_size": {"target_judgeable": 400},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(
        [
            {
                "object_id": f"synthetic-{population}",
                "kind": population,
                "redshift": 0.01,
                "model_failed": 0,
                "merger_mjd": 60000.0,
                "distance_mpc": 40.0,
            }
        ]
    ).to_csv(folder / "events.csv", index=False)
    pd.DataFrame(
        columns=[
            "object_id",
            "time_days",
            "filter",
            "magnitude",
            "magnitude_error",
        ]
    ).to_csv(folder / "light_curves.csv", index=False)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    # Generator is a synthetic stub in this fixture; no production build claim.
    monkeypatch.setattr(cli, "_score_generator_sha", cli._git_sha)
    maps = tmp_path / "synthetic_maps"
    maps.mkdir()
    for pole in ("ngp", "sgp"):
        (maps / f"SFD_dust_4096_{pole}.fits").write_bytes(b"synthetic map, NOT FITS")
    monkeypatch.setattr(sky, "sfd_directory", lambda: maps)
    monkeypatch.setattr(cli, "_fink_exposure_note", lambda: ("synthetic", "synthetic"))
    cache = tmp_path / "cache"
    folders = [cache / p / "test" for p in (sample.LIKE_KIND, sample.REALISTIC)]
    folders.append(cache / "at2017gfo-exact" / "copies")
    for folder in folders:
        synthetic_cache(folder, folder.parent.name)
        (folder / "score_provenance.json").write_text(
            json.dumps(cli._score_receipt(folder, cli._score_map_provenance()))
        )
    cuts = tmp_path / "cuts.json"
    cuts.write_text(
        json.dumps(
            {
                "frozen": True,
                **{
                    "redder_than_g_minus_r": 0.0,
                    "min_g_minus_r_rise": 0.3,
                    "reddening_window_days": 2.0,
                    "min_color_separation_days": 1.0,
                },
            }
        )
    )
    args = SimpleNamespace(
        output=tmp_path / "report",
        cuts=cuts,
        cache=cache,
        rescore_decision=None,
        rebuild_score_cache=False,
    )
    monkeypatch.setattr(
        grid,
        "score_exact_copies",
        lambda *a, **k: {
            "n_judgeable": 50,
            "n_caught_judgeable": 50,
            "official": grid.sanity_copy_verdict(50, 50),
        },
    )
    return args, folders, maps


def entry():
    return next(
        iter(json.loads(cli.test_set_scored_path().read_text())["sets"].values())
    )[-1]


@pytest.mark.parametrize(
    "field",
    [
        "generator_sha",
        "diffusion_fix",
        "dust_mode",
        "dust_map_path",
        "dust_map_hashes",
        "test_set_hashes",
    ],
)
def test_missing_attestation_refuses_before_consumption(harness, field):
    args, folders, _ = harness
    receipt = folders[0] / "score_provenance.json"
    payload = json.loads(receipt.read_text())
    del payload[field]
    receipt.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="1"):
        cli._run_score_test(args)
    assert not cli.test_set_scored_path().exists()
    assert not cli.test_set_claim_dir().exists()
    assert not cli.test_set_scored_path().with_suffix(".json.lock").exists()


@pytest.mark.parametrize(
    "target", ["manifest", "events", "curves", "map", "sha", "receipt"]
)
def test_stale_or_modified_cache_refused(harness, target, capsys):
    args, folders, maps = harness
    paths = {
        "manifest": folders[0] / "manifest.json",
        "events": folders[0] / "events.csv",
        "curves": folders[0] / "light_curves.csv",
        "map": maps / "SFD_dust_4096_ngp.fits",
    }
    if target in paths:
        with paths[target].open("a") as stream:
            stream.write("\n")
    else:
        path = folders[0] / "score_provenance.json"
        if target == "receipt":
            path.unlink()
        else:
            payload = json.loads(path.read_text())
            payload["generator_sha"] = "0" * 40
            path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit) as caught:
        cli._run_score_test(args)
    assert caught.value.code == 1
    assert "provenance refused" in capsys.readouterr().err
    assert not cli.test_set_scored_path().exists()


def test_matching_provenance_and_normal_completion(harness):
    args, folders, _ = harness
    cli._run_score_test(args)
    result = entry()
    assert result["status"] == "complete" and result["consumed"] is True
    assert len(result["numbers"]) == 2
    assert len(result["numbers"][sample.LIKE_KIND]) == 3
    assert result["not_computed"] == {}
    receipt = json.loads((folders[0] / "score_provenance.json").read_text())
    assert result["provenance"][sample.LIKE_KIND].items() >= receipt.items()
    assert result["score_sha256"] == cli._score_file_hash(args.output / "test_score.md")


@pytest.mark.parametrize(
    "fault", ["rule5", "sanity", "untestable", "exception", "exit", "report"]
)
def test_durable_failures(harness, monkeypatch, fault):
    args, _, _ = harness
    real_score = grid.score_with_frozen_cuts

    def score(*a, **kw):
        if fault in ("exception", "exit") and kw["population"] == sample.REALISTIC:
            if fault == "exit":
                raise SystemExit(7)
            raise RuntimeError("synthetic failure after first population")
        rows = real_score(*a, **kw)
        if fault == "rule5":
            rows[0]["n_caught_not_judgeable"] = 1
        return rows

    monkeypatch.setattr(grid, "score_with_frozen_cuts", score)
    if fault in ("sanity", "untestable"):
        monkeypatch.setattr(
            grid,
            "score_exact_copies",
            lambda *a, **k: {
                "n_judgeable": 52 if fault == "sanity" else 0,
                "n_caught_judgeable": 33 if fault == "sanity" else 0,
                "official": grid.sanity_copy_verdict(
                    52 if fault == "sanity" else 0,
                    33 if fault == "sanity" else 0,
                ),
            },
        )
    if fault == "report":
        monkeypatch.setattr(grid, "format_test_score", lambda *a, **k: "synthetic")
        original = Path.write_text

        def fail_report(path, *a, **k):
            if path.name == "test_score.md":
                raise OSError("synthetic report-write failure")
            return original(path, *a, **k)

        monkeypatch.setattr(Path, "write_text", fail_report)
    with pytest.raises((SystemExit, RuntimeError, OSError)) as caught:
        cli._run_score_test(args)
    if isinstance(caught.value, SystemExit):
        assert caught.value.code == (7 if fault == "exit" else 2)
    result = entry()
    assert result["status"] == "FAIL" and result["consumed"] is True
    assert result["numbers"][sample.LIKE_KIND]
    assert result["fail_reason"]
    if fault in ("exception", "exit"):
        assert set(result["not_computed"]) == {"realistic", "sanity"}
    if fault == "rule5":
        assert result["numbers"][sample.REALISTIC][0]["n_caught_not_judgeable"] == 1
        assert "rule 5" in result["fail_reason"]
    assert not Path(str(cli.test_set_scored_path()) + ".lock").exists()
    args.output = args.output.parent / "second-report"
    with pytest.raises(SystemExit) as second:
        cli._run_score_test(args)
    assert second.value.code == 1
    assert entry() == result


def test_failure_before_numbers_does_not_invent_them(harness, monkeypatch):
    args, _, _ = harness

    def stop(*a, **k):
        raise SystemExit(9)

    monkeypatch.setattr(grid, "score_with_frozen_cuts", stop)
    with pytest.raises(SystemExit) as caught:
        cli._run_score_test(args)
    assert caught.value.code == 9
    assert entry()["numbers"] == {}
    assert len(entry()["not_computed"]) == 3
    assert entry()["status"] == "FAIL"


def test_rebuild_routes_to_frozen_generators_without_reuse(harness, monkeypatch):
    args, _, _ = harness
    args.cache = args.cache.parent / "fresh-build"
    args.rebuild_score_cache = True
    calls = []
    monkeypatch.setattr(sky, "open_dust_source", lambda: sky.DustSource(mode="sfd"))

    def population(population, split, **kwargs):
        calls.append((population, split, sorted(kwargs)))
        assert not (kwargs["cache_dir"] / population).exists()
        synthetic_cache(kwargs["cache_dir"] / population / split, population)

    def copies(**kwargs):
        calls.append(("copies", kwargs["seed"]))
        synthetic_cache(kwargs["cache_dir"] / "at2017gfo-exact" / "copies", "copies")

    monkeypatch.setattr(sample, "simulate_population", population)
    monkeypatch.setattr(sample, "simulate_exact_copies", copies)
    cli._run_score_test(args)
    assert calls == [
        (sample.LIKE_KIND, "test", ["cache_dir", "dust"]),
        (sample.REALISTIC, "test", ["cache_dir", "dust"]),
        ("copies", 2026110401),
    ]
    assert not cli.test_set_scored_path().exists()
    first = cli._verify_score_receipt(args.cache / sample.LIKE_KIND / "test")
    args.cache = args.cache.parent / "another-fresh-build"
    cli._run_score_test(args)
    assert first == cli._verify_score_receipt(args.cache / sample.LIKE_KIND / "test")
    with pytest.raises(SystemExit):
        cli._run_score_test(args)
    assert len(calls) == 6


@pytest.mark.parametrize("sha,returncode", [("unknown", 0), ("a" * 40, 1)])
def test_generator_attestation_requires_commit_and_clean_source(
    monkeypatch, sha, returncode
):
    monkeypatch.setattr(cli, "_git_sha", lambda: sha)
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: SimpleNamespace(returncode=returncode)
    )
    with pytest.raises(ValueError):
        cli._score_generator_sha()


def test_claim_failure_is_durable_and_records_provenance(harness, monkeypatch):
    args, _, _ = harness

    def fail_claim(*a, **k):
        raise OSError("synthetic claim failure")

    monkeypatch.setattr(cli, "_claim_score_once_marker", fail_claim)
    with pytest.raises(OSError, match="claim failure"):
        cli._run_score_test(args)
    result = entry()
    assert result["status"] == "FAIL" and result["consumed"]
    assert result["numbers"] == {}
    assert len(result["not_computed"]) == 3
    assert result["provenance"][sample.LIKE_KIND]["generator_sha"]
    assert "claim failure" in result["fail_reason"]


def test_missing_sanity_provenance_refuses_before_consumption(harness):
    args, folders, _ = harness
    (folders[-1] / "score_provenance.json").unlink()
    with pytest.raises(SystemExit) as caught:
        cli._run_score_test(args)
    assert caught.value.code == 1
    assert not cli.test_set_scored_path().exists()


def test_stdout_failure_keeps_numbers_and_marks_fail(harness, monkeypatch):
    args, _, _ = harness

    def broken_stdout(*a, **k):
        raise BrokenPipeError("synthetic closed stdout")

    monkeypatch.setattr(cli, "print", broken_stdout, raising=False)
    with pytest.raises(BrokenPipeError):
        cli._run_score_test(args)
    assert entry()["status"] == "FAIL" and entry()["consumed"]
    assert len(entry()["numbers"]) == 2
    assert "closed stdout" in entry()["fail_reason"]

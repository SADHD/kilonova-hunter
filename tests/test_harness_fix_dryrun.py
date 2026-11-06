"""Rerun with uv run pytest -s tests/test_harness_fix_dryrun.py (synthetic only)."""

import json

import pytest

from kilonova_hunter import cli, grid
from test_harness_fix import entry
from test_harness_fix import harness as harness


def test_synthetic_stale_and_hard_failure_demo(harness, monkeypatch, tmp_path):
    args, folders, _ = harness
    receipt = folders[0] / "score_provenance.json"
    valid = receipt.read_text()
    receipt.unlink()
    with pytest.raises(SystemExit) as stale:
        cli._run_score_test(args)
    assert stale.value.code == 1
    assert not cli.test_set_scored_path().exists()
    print("SYNTHETIC stale cache: exit=1; consumed=false; ledger absent")
    receipt.write_text(valid)
    original = grid.score_with_frozen_cuts

    def forced(*a, **k):
        rows = original(*a, **k)
        rows[0]["n_caught_not_judgeable"] = 1
        return rows

    monkeypatch.setattr(grid, "score_with_frozen_cuts", forced)
    monkeypatch.setattr(
        grid, "format_test_score", lambda *a, **k: "Synthetic forced-failure report."
    )
    monkeypatch.setattr(
        grid,
        "score_exact_copies",
        lambda *a, **k: {
            "n_judgeable": 52,
            "n_caught_judgeable": 33,
            "official": grid.sanity_copy_verdict(52, 33),
        },
    )
    with pytest.raises(SystemExit) as failed:
        cli._run_score_test(args)
    result = entry()
    assert failed.value.code == 2
    assert result["status"] == "FAIL" and result["consumed"] is True
    artifact = tmp_path / "synthetic_fail_ledger_entry.json"
    artifact.write_text(json.dumps(result, indent=2) + "\n")
    print("SYNTHETIC forced rule-5 + sanity failure: exit=2")
    print(
        json.dumps(
            {
                "status": result["status"],
                "consumed": result["consumed"],
                "fail_reason": result["fail_reason"],
                "numbers": {
                    population: [
                        {
                            key: row[key]
                            for key in (
                                "day",
                                "n_total",
                                "n_judgeable",
                                "n_caught_not_judgeable",
                            )
                        }
                        for row in rows
                    ]
                    for population, rows in result["numbers"].items()
                },
                "sanity": result["sanity"],
                "not_computed": result["not_computed"],
            },
            indent=2,
        )
    )
    print(f"Full synthetic ledger entry: {artifact}")
    args.output = tmp_path / "reattempt"
    with pytest.raises(SystemExit) as again:
        cli._run_score_test(args)
    assert again.value.code == 1
    assert entry() == result
    print("SYNTHETIC reattempt: exit=1; ledger unchanged; lock cleaned")


def synthetic_process_main():
    """Standalone evidence runner: the CLI's nonzero exit reaches the shell."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="SYNTHETIC ONLY: no held data")
    parser.add_argument(
        "--scenario",
        choices=("stale", "hard-fail", "error", "exit-zero"),
        required=True,
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    options = parser.parse_args()
    root = options.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    with pytest.MonkeyPatch.context() as patch:
        import socket

        def no_network(*a, **k):
            raise AssertionError("Synthetic dry run must not use network")

        patch.setattr(socket.socket, "connect", no_network)
        patch.setattr(socket.socket, "connect_ex", no_network)
        patch.setattr(cli, "test_set_scored_path", lambda: root / "ledger.json")
        args, folders, _ = harness.__wrapped__(root, patch)
        from kilonova_hunter import sample, sky

        def no_generation(*a, **k):
            raise AssertionError("Synthetic runner must not generate or query maps")

        patch.setattr(sample, "simulate_population", no_generation)
        patch.setattr(sample, "simulate_exact_copies", no_generation)
        patch.setattr(sky, "open_dust_source", no_generation)
        print(
            "SYNTHETIC source tree/SHA attestation stub; fake map bytes; "
            "no generators or map queries"
        )
        if options.scenario == "stale":
            (folders[0] / "score_provenance.json").unlink()
        elif options.scenario in ("error", "exit-zero"):

            def forced_error(*a, **k):
                if k["population"] == "realistic":
                    if options.scenario == "exit-zero":
                        raise SystemExit(0)
                    raise RuntimeError("SYNTHETIC evaluator exception")
                return [
                    {
                        "day": day,
                        "n_total": 1,
                        "n_judgeable": 1,
                        "n_caught": 0,
                        "n_caught_not_judgeable": 0,
                    }
                    for day in (3.0, 5.0, 7.0)
                ]

            patch.setattr(grid, "score_with_frozen_cuts", forced_error)
        else:
            original = grid.score_with_frozen_cuts

            def forced(*a, **k):
                rows = original(*a, **k)
                rows[0]["n_caught_not_judgeable"] = 1
                return rows

            patch.setattr(grid, "score_with_frozen_cuts", forced)
            patch.setattr(grid, "format_test_score", lambda *a, **k: "SYNTHETIC report")
            patch.setattr(
                grid,
                "score_exact_copies",
                lambda *a, **k: {
                    "n_judgeable": 52,
                    "n_caught_judgeable": 33,
                    "official": grid.sanity_copy_verdict(52, 33),
                },
            )
        try:
            cli._run_score_test(args)
        except SystemExit as failure:
            if options.scenario != "stale":
                result = entry()
                assert result["status"] == (
                    "FAIL" if options.scenario == "hard-fail" else "ERROR"
                )
                assert result["consumed"] is True
                saved = cli.test_set_scored_path().read_bytes()
                args.output = root / "retry-report"
                with pytest.raises(SystemExit) as retry:
                    cli._run_score_test(args)
                assert retry.value.code == 1
                assert cli.test_set_scored_path().read_bytes() == saved
                print("SYNTHETIC retry: exit=1; refused; ledger unchanged")
            print(f"SYNTHETIC shell exit: {failure.code}")
            raise
        finally:
            if cli.test_set_scored_path().exists():
                print("SYNTHETIC ledger entry:")
                print(json.dumps(entry(), indent=2))
            else:
                print("SYNTHETIC ledger absent; consumed=false")


if __name__ == "__main__":
    synthetic_process_main()

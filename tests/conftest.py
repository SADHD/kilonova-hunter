"""Fixtures shared by the tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_runtime_artifacts(tmp_path, monkeypatch):
    """Every test, including forked probes, uses fresh runtime artifacts."""
    import builtins
    import io
    import json
    import socket

    # Reuse the committed sncosmo mirror before its first bandpass lookup.
    # Setting this only in the MOSFiT test is too late: DataMirror caches its
    # first root directory. No network exception is needed for real models.
    monkeypatch.setenv(
        "SNCOSMO_DATA_DIR",
        str(Path(__file__).resolve().parents[1] / "checks/mosfit/sncosmo_bandpasses"),
    )

    from kilonova_hunter import alerts, cli

    monkeypatch.setattr(cli, "test_set_scored_path", lambda: tmp_path / "ledger.json")
    # Synthetic fixed night lists; never open the committed Fink document.
    frozen = tmp_path / "synthetic_fink_split.json"
    frozen.write_text(
        json.dumps(
            {
                "tuning": {
                    "nights": [
                        60461,
                        60463,
                        60466,
                        60468,
                        60470,
                        60472,
                        60474,
                        60476,
                        60478,
                    ]
                },
                "test": {
                    "nights": [
                        60462,
                        60464,
                        60467,
                        60469,
                        60471,
                        60473,
                        60475,
                        60477,
                        60479,
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(alerts, "FROZEN_SPLIT_PATH", frozen)

    def no_network(*args, **kwargs):
        raise AssertionError("Runtime tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    root = Path(__file__).resolve().parents[1]

    def guarded(opener):
        def open_safe(file, *args, **kwargs):
            if not isinstance(file, int):
                path = Path(file).absolute()
                if path.is_relative_to(root / "data") or (
                    path.is_relative_to(root / "results")
                    and path.name != "test_set_exposure.json"
                ):
                    raise AssertionError(
                        f"Test attempted real runtime artifact: {path}"
                    )
            return opener(file, *args, **kwargs)

        return open_safe

    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    monkeypatch.setattr(io, "open", guarded(io.open))


@pytest.fixture
def scored_test_set_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ledger and claim files for one test, not the committed results/ copies.

    Production reads ``results/test_set_scored.json`` from the repository
    root. There is no environment-variable override. This fixture is the
    only way a test points that path somewhere else.
    """
    ledger = tmp_path / "test_set_scored.json"

    def ledger_path() -> Path:
        return ledger

    monkeypatch.setattr("kilonova_hunter.cli.test_set_scored_path", ledger_path)
    return ledger

"""Fixtures shared by the tests."""

from __future__ import annotations

from pathlib import Path

import pytest


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

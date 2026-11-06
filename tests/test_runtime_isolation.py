"""Offline model support never grants network or held-artifact access."""

import hashlib
import os
import socket
from pathlib import Path

import pytest

from test_harness_fix import harness as harness

ROOT = Path(__file__).resolve().parents[1]
BANDPASSES = ROOT / "checks/mosfit/sncosmo_bandpasses"
# Existing committed sncosmo mirror; source URLs are in its SOURCES.txt.
LSST_SHA256 = {
    "g": "78535bd67e9f001576a30bdc678b7f7a3bc1bdfb1f3a39cb58fd00128980b1d7",
    "r": "be5b9f0a0eecf9cd6032ae7f8a382ab2e97b9b136ee462a8115c0dd64560d78d",
    "i": "f4a79d0e781d3c0e2e43e647b2229be58aa4485362c3d0023e5d408afb8de12c",
}


def assert_network_denied():
    # Numeric address avoids even DNS. Both operations must fail before I/O.
    with socket.socket() as client:
        for operation in (client.connect, client.connect_ex):
            with pytest.raises(AssertionError, match="must not use the network"):
                operation(("192.0.2.1", 443))


def test_model_support_is_pinned_and_offline():
    assert Path(os.environ["SNCOSMO_DATA_DIR"]) == BANDPASSES
    for band, digest in LSST_SHA256.items():
        data = BANDPASSES / f"bandpasses/lsst/total_{band}.dat"
        assert hashlib.sha256(data.read_bytes()).hexdigest() == digest
    assert_network_denied()
    sncosmo = pytest.importorskip("sncosmo")
    for band in LSST_SHA256:
        curve = sncosmo.get_bandpass(f"lsst{band}")
        assert len(curve.wave) > 0
    assert_network_denied()


def test_synthetic_harness_keeps_network_and_artifact_guards(harness):
    assert_network_denied()
    # Invented names: the guard refuses before an underlying file is opened.
    for folder in ("data", "results"):
        with pytest.raises(AssertionError, match="real runtime artifact"):
            (ROOT / folder / "synthetic_guard_probe_never_opened").read_bytes()

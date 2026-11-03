"""The MOSFiT reference copies stay byte-identical to commit 6595248."""

from __future__ import annotations

import hashlib
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / "checks" / "mosfit" / "reference"
SOURCES = REFERENCE / "SOURCES.txt"


def _pins() -> list[tuple[str, str, str]]:
    pins = []
    for line in SOURCES.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, digest, url = line.split()
        pins.append((name, digest, url))
    return pins


def test_reference_files_match_the_upstream_mosfit_hashes() -> None:
    """Each pin is the sha256 of the file at MOSFiT commit 6595248."""
    pins = _pins()
    pinned = {name for name, _digest, _url in pins}
    on_disk = {
        path.name
        for path in REFERENCE.iterdir()
        if path.is_file() and path.name != "SOURCES.txt"
    }
    assert on_disk == pinned
    for name, digest, url in pins:
        assert url == (
            "https://raw.githubusercontent.com/guillochon/MOSFiT/6595248/"
            + url.split("/6595248/", 1)[1]
        )
        found = hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
        assert found == digest, name

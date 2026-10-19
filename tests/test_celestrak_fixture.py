from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from xenia.celestrak import CelestrakFixture, read_celestrak_fixture

FIXTURE = Path(__file__).parent / "fixtures" / "celestrak-one-row.json"


def test_reads_recorded_celestrak_row_without_network(monkeypatch) -> None:
    def fail_network(*args, **kwargs):
        pytest.fail("network access")

    monkeypatch.setattr(urllib.request, "urlopen", fail_network)

    row = read_celestrak_fixture(FIXTURE)

    assert isinstance(row, CelestrakFixture)
    assert row.id == "celestrak"
    assert row.probe_url.endswith("CATNR=25544&FORMAT=json")
    assert row.http_status == 200
    assert row.content_type == "application/json; charset=UTF-8"
    assert row.verdict == "verified"
    assert row.synthetic is False


def test_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "malformed.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        read_celestrak_fixture(path)


def test_rejects_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    path.write_text('{"id": "celestrak"}', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid Celestrak fixture"):
        read_celestrak_fixture(path)


def test_rejects_invalid_provenance(tmp_path: Path) -> None:
    document = FIXTURE.read_text(encoding="utf-8").replace(
        '"synthetic": false', '"synthetic": true'
    )
    path = tmp_path / "synthetic.json"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid Celestrak fixture"):
        read_celestrak_fixture(path)


def test_rejects_missing_fixture() -> None:
    with pytest.raises(ValueError, match="cannot read Celestrak fixture"):
        read_celestrak_fixture(FIXTURE.with_name("missing.json"))

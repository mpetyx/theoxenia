from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "celestrak-one-row.json"


def test_reads_recorded_celestrak_row_without_network(monkeypatch) -> None:
    def fail_network(*args, **kwargs):
        pytest.fail("network access")

    monkeypatch.setattr(urllib.request, "urlopen", fail_network)

    row: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert row["id"] == "celestrak"
    assert row["probe_url"].endswith("CATNR=25544&FORMAT=json")
    assert row["http_status"] == 200
    assert row["content_type"] == "application/json; charset=UTF-8"
    assert row["verdict"] == "verified"
    assert row["synthetic"] is False

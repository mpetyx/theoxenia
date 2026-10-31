from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "noaa-swpc-one-row.json"


def test_reads_recorded_noaa_swpc_row_without_network(monkeypatch) -> None:
    def fail_network(*args, **kwargs):
        pytest.fail("network access")

    monkeypatch.setattr(urllib.request, "urlopen", fail_network)

    row: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert row["id"] == "noaa-swpc"
    assert row["probe_url"].endswith("planetary_k_index_1m.json")
    assert row["http_status"] == 200
    assert row["content_type"] == "application/json"
    assert row["verdict"] == "verified"
    assert row["synthetic"] is False
    assert (
        row["sha256_of_bytes_read"]
        == "1e6ce2d3a25ce0e90092506ad76b5a3101d653a33aff7d8bbe8979926401e81d"
    )
    assert row["bytes_read"] == 27925

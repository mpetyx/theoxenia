"""Offline readers for recorded Celestrak probe evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class CelestrakFixture(BaseModel):
    """Validated metadata for one recorded Celestrak response."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: Literal["celestrak"]
    name: Literal["Celestrak"]
    endpoint: str = Field(pattern=r"^https://celestrak\.org$")
    probe_url: str = Field(
        pattern=r"^https://celestrak\.org/NORAD/elements/gp\.php\?CATNR=\d+&FORMAT=json$"
    )
    method: Literal["GET"]
    http_status: int = Field(ge=200, le=299)
    bytes_read: int = Field(ge=0)
    content_length: int = Field(ge=0)
    content_type: Literal["application/json; charset=UTF-8"]
    content_range: str | None
    date_header: str = Field(min_length=1)
    sha256_of_bytes_read: str
    truncated: bool
    elapsed_s: float = Field(ge=0)
    error: None
    synthetic: Literal[False]
    numbers_from: Literal["live"]
    verdict: Literal["verified"]
    note: str = Field(min_length=1)

    @field_validator("sha256_of_bytes_read")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("sha256_of_bytes_read must be 64 lowercase hexadecimal characters")
        return value


def read_celestrak_fixture(path: Path) -> CelestrakFixture:
    """Read and validate recorded Celestrak evidence without network access."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        return CelestrakFixture.model_validate(document)
    except OSError as exc:
        raise ValueError(f"cannot read Celestrak fixture {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in Celestrak fixture {path}: {exc}") from exc
    except ValidationError as exc:
        raise ValueError(f"invalid Celestrak fixture {path}: {exc}") from exc

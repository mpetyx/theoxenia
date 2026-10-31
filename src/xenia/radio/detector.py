"""turboSETI execution and its tab-separated hit-table adapter."""

from __future__ import annotations

import linecache
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from typing import Any

from xenia.radio.turboseti_compat import COMPATIBILITY_ID, compatible_search

_NUMPY_INT_FORMAT_ERROR = "format argument: %i requires a real number, not numpy.ndarray"


@dataclass(frozen=True, slots=True)
class DopplerHit:
    """One parsed turboSETI top-hit row."""

    top_hit: int
    drift_hz_s: float
    snr: float
    uncorrected_frequency_mhz: float
    corrected_frequency_mhz: float
    frequency_start_mhz: float
    frequency_end_mhz: float

    def as_dict(self) -> dict[str, int | float]:
        """Return the hit as JSON-compatible scalar fields."""
        return {
            "top_hit": self.top_hit,
            "drift_hz_s": self.drift_hz_s,
            "snr": self.snr,
            "uncorrected_frequency_mhz": self.uncorrected_frequency_mhz,
            "corrected_frequency_mhz": self.corrected_frequency_mhz,
            "frequency_start_mhz": self.frequency_start_mhz,
            "frequency_end_mhz": self.frequency_end_mhz,
        }


@dataclass(frozen=True, slots=True)
class DetectorArtifact:
    """Checksum and size of a temporary turboSETI output table."""

    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class DetectorResult:
    """Detector outputs retained in memory for summarization and provenance."""

    hits: tuple[DopplerHit, ...]
    artifacts: tuple[DetectorArtifact, ...]
    turbo_seti_version: str
    blimpy_version: str
    warnings: tuple[str, ...] = ()


def parse_dat_text(text: str) -> tuple[DopplerHit, ...]:
    """Parse turboSETI's documented tab-separated `.dat` output by header name."""
    columns: list[str] | None = None
    hits: list[DopplerHit] = []
    required = {
        "Top_Hit_#",
        "Drift_Rate",
        "SNR",
        "Uncorrected_Frequency",
        "Corrected_Frequency",
        "freq_start",
        "freq_end",
    }

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header = stripped.lstrip("#").split()
            if header and header[0] == "Top_Hit_#":
                missing = required - set(header)
                if missing:
                    raise ValueError(
                        f"turboSETI .dat header is missing columns: {', '.join(sorted(missing))}"
                    )
                columns = header
            continue
        if columns is None:
            continue

        values = stripped.split()
        if len(values) != len(columns):
            raise ValueError(
                f"turboSETI .dat row {line_number} has {len(values)} values; "
                f"expected {len(columns)}"
            )
        row = dict(zip(columns, values, strict=True))
        try:
            numeric = {
                name: float(row[name])
                for name in (
                    "Drift_Rate",
                    "SNR",
                    "Uncorrected_Frequency",
                    "Corrected_Frequency",
                    "freq_start",
                    "freq_end",
                )
            }
            top_hit = int(row["Top_Hit_#"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid turboSETI .dat row {line_number}: {exc}") from exc
        if any(not isfinite(value) for value in numeric.values()):
            raise ValueError(f"turboSETI .dat row {line_number} contains non-finite values")
        hits.append(
            DopplerHit(
                top_hit=top_hit,
                drift_hz_s=numeric["Drift_Rate"],
                snr=numeric["SNR"],
                uncorrected_frequency_mhz=numeric["Uncorrected_Frequency"],
                corrected_frequency_mhz=numeric["Corrected_Frequency"],
                frequency_start_mhz=numeric["freq_start"],
                frequency_end_mhz=numeric["freq_end"],
            )
        )
    if columns is None:
        raise ValueError("turboSETI .dat table has no valid column header")
    return tuple(hits)


def select_known_answer_hit(
    hits: Iterable[DopplerHit],
    *,
    frequency_low_mhz: float,
    frequency_high_mhz: float,
    max_abs_drift_hz_s: float,
    minimum_snr: float,
) -> DopplerHit | None:
    """Choose the strongest hit satisfying the experiment's declared search window."""
    if frequency_low_mhz > frequency_high_mhz:
        raise ValueError("frequency_low_mhz must not exceed frequency_high_mhz")
    if max_abs_drift_hz_s <= 0 or minimum_snr < 0:
        raise ValueError("drift limit must be positive and minimum SNR non-negative")
    eligible = [
        hit
        for hit in hits
        if frequency_low_mhz <= hit.uncorrected_frequency_mhz <= frequency_high_mhz
        and abs(hit.drift_hz_s) <= max_abs_drift_hz_s
        and hit.snr >= minimum_snr
    ]
    return max(eligible, key=lambda hit: (hit.snr, -hit.top_hit), default=None)


def _find_doppler_class() -> Any:
    try:
        from turbo_seti.find_doppler.find_doppler import FindDoppler
    except ImportError as exc:
        raise RuntimeError(
            "Voyager analysis requires turbo-seti; install the optional xenia[radio] extra"
        ) from exc
    return FindDoppler


def _post_search_hit_count_log_error(exc: BaseException) -> bool:
    """True when tophitsearch has returned and only the hit-count log line failed.

    turboSETI 2.3.2 stores ``total_n_hits`` as a length-1 array and then formats
    it with ``%i`` after the ``.dat`` table has been written. NumPy rejects that
    format. The table is complete, including when it contains no hit rows.
    """
    if not isinstance(exc, TypeError) or _NUMPY_INT_FORMAT_ERROR not in str(exc):
        return False
    traceback = exc.__traceback__
    if traceback is None:
        return False
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    code = traceback.tb_frame.f_code
    if code.co_name != "search_coarse_channel":
        return False
    if not code.co_filename.replace("\\", "/").endswith("find_doppler.py"):
        return False
    line = linecache.getline(code.co_filename, traceback.tb_lineno)
    return "total_n_hits" in line


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_turboseti(
    filterbank_path: Path,
    *,
    output_directory: Path,
    max_drift_hz_s: float,
    minimum_snr: float,
    n_coarse_channels: int = 1,
    apply_compatibility: bool = True,
) -> DetectorResult:
    """Run turboSETI on one local, bounded filterbank slice and hash its outputs."""
    if not filterbank_path.is_file():
        raise FileNotFoundError(f"filterbank slice does not exist: {filterbank_path}")
    if max_drift_hz_s <= 0 or minimum_snr < 0 or n_coarse_channels <= 0:
        raise ValueError(
            "drift limit and coarse-channel count must be positive; minimum SNR non-negative"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.glob("*.dat")):
        raise ValueError(
            "detector output directory contains old .dat tables; use a fresh directory"
        )
    finder_class = _find_doppler_class()

    try:
        turbo_version = version("turbo-seti")
        blimpy_version = version("blimpy")
    except PackageNotFoundError as exc:  # pragma: no cover - corrupted installation
        raise RuntimeError("turbo-seti and blimpy package metadata must be installed") from exc

    finder = finder_class(
        filterbank_path.resolve().as_posix(),
        max_drift=max_drift_hz_s,
        min_drift=1e-5,
        snr=minimum_snr,
        out_dir=output_directory.resolve().as_posix(),
        n_coarse_chan=n_coarse_channels,
        blank_dc=True,
        precision=1,
    )
    search_error: TypeError | None = None
    try:
        with compatible_search(enabled=apply_compatibility and turbo_version == "2.3.2"):
            finder.search(n_partitions=1, progress_bar="n")
    except TypeError as exc:
        # A parseable table can still be partial. Only the verified final logging
        # failure of a single-channel search is recoverable; never infer completion
        # from the presence of rows or from the exception text alone.
        if n_coarse_channels != 1 or not _post_search_hit_count_log_error(exc):
            raise
        search_error = exc

    dat_paths = tuple(sorted(output_directory.glob("*.dat")))
    if not dat_paths:
        raise RuntimeError(
            "turboSETI returned without a .dat table; search completion is unverified"
        )
    parsed_hits = tuple(
        hit
        for dat_path in dat_paths
        for hit in parse_dat_text(dat_path.read_text(encoding="utf-8", errors="replace"))
    )
    finished_empty = (
        search_error is not None
        and not parsed_hits
        and bool(dat_paths)
        and _post_search_hit_count_log_error(search_error)
    )
    if search_error is not None and not parsed_hits and not finished_empty:
        raise search_error
    if parsed_hits and search_error is not None:
        warnings: tuple[str, ...] = (
            f"turboSETI {turbo_version} raised its known NumPy-array formatting "
            "TypeError on the post-search hit-count log line after writing parseable .dat "
            "hit rows; those rows were retained.",
        )
    elif finished_empty:
        warnings = (
            f"turboSETI {turbo_version} raised its known NumPy-array formatting "
            "TypeError on the post-search hit-count log line after tophitsearch returned; "
            "the .dat table contained no hit rows, and that empty result was retained.",
        )
    else:
        warnings = ()
    if apply_compatibility and turbo_version == "2.3.2":
        warnings = (*warnings, f"Applied audited compatibility correction: {COMPATIBILITY_ID}.")
    artifacts = tuple(
        DetectorArtifact(
            filename=path.name,
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
        )
        for path in dat_paths
    )
    return DetectorResult(
        hits=parsed_hits,
        artifacts=artifacts,
        turbo_seti_version=turbo_version,
        blimpy_version=blimpy_version,
        warnings=warnings,
    )

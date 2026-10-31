"""Search one archive ABACAD cadence and write its manifest.

The search is the same filter as EXP-004. Labels come from measured scan
membership inside a drift-projected window. This module does not claim a
signal, and it cannot emit UNEXPLAINED.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

import html
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from xenia.live.cadence import DRIFT_MATCH_HZ_S, FREQUENCY_MATCH_HZ
from xenia.live.known_sources import KnownCatalog, load_known_sources
from xenia.live.models import (
    CadenceHit,
    CadencePayload,
    CadenceStatus,
    InputRef,
    RunManifest,
    Verdict,
)
from xenia.live.search import CadenceSearch, FilterEvent, ScanDetection, search_cadence
from xenia.live.slicer import (
    CADENCE_TRANSFER_CAP_BYTES,
    CadenceFile,
    CadenceSlice,
    slice_cadence,
)
from xenia.live.state import sha256_file
from xenia.live.verdict import VerdictDecision, decide_verdict
from xenia.provenance import create_run_manifest, new_run_id, output_ref
from xenia.radio.cadence_synth import ScanTruth

_MAX_ON_ONLY_PLOTS = 24
_MAX_ASSET_BYTES = 2 * 1024 * 1024
_MJD_DAY_SECONDS = 86400.0
_CHECK_NAMES = (
    "not_in_off",
    "not_known_transmitter",
    "drift_sane",
    "session_census_clear",
    "review",
)


@dataclass(frozen=True, slots=True)
class ScanClock:
    """Start time of one scan, used to project a drifting line."""

    scan_index: int
    role: str
    tstart_mjd: float

    @property
    def on_source(self) -> bool:
        return self.role == "A"


@dataclass(frozen=True, slots=True)
class ArchiveHit:
    """One labelled row. Membership counts are measured, not implied."""

    hit_id: str
    frequency_mhz: float
    drift_hz_s: float
    snr: float
    on_count: int
    off_count: int
    passed_filter: bool
    event_id: str | None
    association_note: str
    decision: VerdictDecision


def project_hz(
    frequency_mhz: float, drift_hz_s: float, tstart_mjd: float, epoch_mjd: float
) -> float:
    """Project a scan-start frequency back to the cadence epoch, in Hz."""
    delta_seconds = (tstart_mjd - epoch_mjd) * _MJD_DAY_SECONDS
    return frequency_mhz * 1_000_000.0 - drift_hz_s * delta_seconds


def label_hits(
    search: CadenceSearch,
    clocks: tuple[ScanClock, ...],
    catalog: KnownCatalog,
    *,
    cadence_id: str,
) -> tuple[ArchiveHit, ...]:
    """Cluster detections, then attach any filter-3 event that lands in the window.

    The window is 500 Hz and 0.15 Hz/s after projection to the first scan's
    start. A filter-3 event does not force on_count=3. The verdict follows the
    detections that actually fell in the window.
    """
    if len(clocks) != 6:
        raise ValueError("an ABACAD cadence has six scan clocks")
    by_index = {clock.scan_index: clock for clock in clocks}
    if set(by_index) != set(range(6)):
        raise ValueError("scan clocks must cover indexes 0..5")
    epoch = by_index[0].tstart_mjd
    clusters = _cluster_detections(search.detections, by_index, epoch)
    used_events: set[int] = set()
    drafted: list[tuple[float, float, ArchiveHit]] = []
    for members in clusters:
        seed = max(members, key=lambda item: (item.snr, -item.scan_index))
        seed_clock = by_index[seed.scan_index]
        center = project_hz(seed.frequency_mhz, seed.drift_hz_s, seed_clock.tstart_mjd, epoch)
        event_index = _matching_event(
            search.events, center_hz=center, drift_hz_s=seed.drift_hz_s, clocks=clocks, epoch=epoch
        )
        event = None if event_index is None else search.events[event_index]
        if event_index is not None:
            used_events.add(event_index)
        drafted.append(
            (
                seed.frequency_mhz,
                seed.drift_hz_s,
                _hit_from_detections(
                    members,
                    by_index,
                    catalog,
                    cadence_id=cadence_id,
                    event=event,
                    center_hz=center,
                ),
            )
        )
    for index, event in enumerate(search.events):
        if index in used_events:
            continue
        drafted.append(
            (
                event.frequency_mhz,
                event.drift_hz_s,
                _hit_from_event(event, catalog, cadence_id=cadence_id),
            )
        )
    drafted.sort(key=lambda item: (item[0], item[1]))
    labelled: list[ArchiveHit] = []
    for number, (_, _, hit) in enumerate(drafted, start=1):
        labelled.append(
            ArchiveHit(
                hit_id=f"{cadence_id}-{number:04d}",
                frequency_mhz=hit.frequency_mhz,
                drift_hz_s=hit.drift_hz_s,
                snr=hit.snr,
                on_count=hit.on_count,
                off_count=hit.off_count,
                passed_filter=hit.passed_filter,
                event_id=hit.event_id,
                association_note=hit.association_note,
                decision=hit.decision,
            )
        )
    return tuple(labelled)


def run_archive_cadence(
    repository_root: Path,
    command: list[str],
    *,
    files: tuple[str, ...],
    sources: tuple[str, ...],
    roles: tuple[str, ...],
    window_mhz: tuple[float, float],
    target: str,
    ra_deg: float,
    dec_deg: float,
    experiment_id: str,
    cadence_id: str,
    known_sources_path: Path,
    catalog_md5: tuple[str, ...] | None = None,
    catalog_size_bytes: tuple[int, ...] | None = None,
    max_transfer_bytes: int = CADENCE_TRANSFER_CAP_BYTES,
    allow_plain_http: bool = False,
) -> tuple[RunManifest, Path]:
    """Slice, search, label, and record one real cadence. Temporary files are deleted."""
    if not target.strip():
        raise ValueError("target must be non-empty")
    if not cadence_id.strip():
        raise ValueError("cadence_id must be non-empty")
    root = repository_root.resolve()
    catalog = load_known_sources(known_sources_path)
    started_at = datetime.now(UTC)
    work = root / "data" / "tmp" / f"exp-003-{_slug(cadence_id)}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        slices = slice_cadence(
            _cadence_files(
                files,
                sources,
                roles,
                catalog_md5=catalog_md5,
                catalog_size_bytes=catalog_size_bytes,
            ),
            output_dir=work / "slices",
            frequency_low_mhz=window_mhz[0],
            frequency_high_mhz=window_mhz[1],
            max_transfer_bytes=max_transfer_bytes,
            allow_plain_http=allow_plain_http,
        )
        scans = _scans(slices)
        search = search_cadence(scans, work_dir=work / "search")
        clocks = _clocks(slices)
        # Separate windows share a cadence but must never share public hit URLs.
        hit_prefix = f"{cadence_id}-{window_mhz[0]:.6f}-{window_mhz[1]:.6f}"
        hits = label_hits(search, clocks, catalog, cadence_id=hit_prefix)
        plot_paths, plot_notes = _plot_on_only(hits, slices, work / "plots", target=target)
        finished_at = datetime.now(UTC)
        run_id = new_run_id(started_at, "exp-003-cadence")
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        copied_plots = _copy_plots(plot_paths, run_dir)
        summary = _slice_summary(
            slices,
            cadence_id=cadence_id,
            window_mhz=window_mhz,
            max_transfer_bytes=max_transfer_bytes,
            allow_plain_http=allow_plain_http,
        )
        inputs = _inputs(slices)
        payload_hits = _payload_hits(hits, copied_plots)
        thumb_urls = _write_thumbs(run_dir, payload_hits)
        hits_path = run_dir / "hits.parquet"
        summary_path = run_dir / "slice_summary.json"
        _write_hits(hits_path, hits, finished_at)
        _write_json(summary_path, summary)
        _assert_parquet_counts(hits_path, hits)
        cadence_hits = _cadence_hits(payload_hits, finished_at, thumb_urls, copied_plots)
        payload = _payload(
            cadence_id=cadence_id,
            target=target,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            window_mhz=window_mhz,
            on_source_seconds=_on_source_seconds(slices),
            started_at=started_at,
            hits=cadence_hits,
        )
        outputs = tuple(
            output_ref(path, repository_root=root)
            for path in (
                hits_path,
                summary_path,
                *(run_dir / "thumbs" / f"{hit.hit_id}.svg" for hit in payload_hits),
                *(run_dir / "plots" / f"{hit_id}.png" for hit_id in copied_plots),
            )
        )
        notes = _notes(
            hits,
            search,
            payload_hits,
            plot_notes,
            slices,
            max_transfer_bytes=max_transfer_bytes,
            target=target,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
        )
        manifest, manifest_path = create_run_manifest(
            run_id=run_id,
            command=command,
            parameters=_parameters(
                experiment_id=experiment_id,
                cadence_id=cadence_id,
                target=target,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                window_mhz=window_mhz,
                known_sources_path=known_sources_path,
                max_transfer_bytes=max_transfer_bytes,
                hits=hits,
                slices=slices,
                allow_plain_http=allow_plain_http,
            ),
            inputs=inputs,
            outputs=outputs,
            started_at=started_at,
            finished_at=finished_at,
            outcome=_outcome(hits),
            notes=notes,
            repository_root=root,
            demo=False,
            pillar="I · Sensor",
            experiment_id=experiment_id,
            mission_control=payload,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return manifest, manifest_path


def _cadence_files(
    files: tuple[str, ...],
    sources: tuple[str, ...],
    roles: tuple[str, ...],
    *,
    catalog_md5: tuple[str, ...] | None,
    catalog_size_bytes: tuple[int, ...] | None,
) -> tuple[CadenceFile, ...]:
    if len(files) != 6 or len(sources) != 6 or len(roles) != 6:
        raise ValueError("files, sources, and roles must each contain six entries")
    if catalog_md5 is not None and len(catalog_md5) != 6:
        raise ValueError("catalog_md5 must contain six entries")
    if catalog_size_bytes is not None and len(catalog_size_bytes) != 6:
        raise ValueError("catalog_size_bytes must contain six entries")
    return tuple(
        CadenceFile(
            url=files[index],
            role=roles[index],
            expected_source_name=sources[index],
            catalog_md5=None if catalog_md5 is None else catalog_md5[index],
            catalog_size_bytes=None if catalog_size_bytes is None else catalog_size_bytes[index],
        )
        for index in range(6)
    )


def _scans(slices: tuple[CadenceSlice, ...]) -> tuple[ScanTruth, ...]:
    return tuple(
        ScanTruth(
            scan_index=item.scan_index,
            role=item.role,
            on_source=item.role == "A",
            seed=0,
            mjd=item.info.header.tstart_mjd,
            path=item.path,
            injection_ids=(),
        )
        for item in slices
    )


def _clocks(slices: tuple[CadenceSlice, ...]) -> tuple[ScanClock, ...]:
    return tuple(
        ScanClock(
            scan_index=item.scan_index,
            role=item.role,
            tstart_mjd=item.info.header.tstart_mjd,
        )
        for item in slices
    )


def _cluster_detections(
    detections: tuple[ScanDetection, ...],
    clocks: dict[int, ScanClock],
    epoch: float,
) -> tuple[tuple[ScanDetection, ...], ...]:
    ordered = sorted(detections, key=lambda item: (-item.snr, item.scan_index, item.frequency_mhz))
    used: set[int] = set()
    clusters: list[tuple[ScanDetection, ...]] = []
    for index, seed in enumerate(ordered):
        if index in used:
            continue
        used.add(index)
        center = project_hz(
            seed.frequency_mhz, seed.drift_hz_s, clocks[seed.scan_index].tstart_mjd, epoch
        )
        members = [seed]
        for other_index, other in enumerate(ordered):
            if other_index in used:
                continue
            projected = project_hz(
                other.frequency_mhz,
                other.drift_hz_s,
                clocks[other.scan_index].tstart_mjd,
                epoch,
            )
            drift_error = abs(other.drift_hz_s - seed.drift_hz_s)
            if abs(projected - center) <= FREQUENCY_MATCH_HZ and drift_error <= DRIFT_MATCH_HZ_S:
                used.add(other_index)
                members.append(other)
        clusters.append(tuple(members))
    return tuple(clusters)


def _matching_event(
    events: tuple[FilterEvent, ...],
    *,
    center_hz: float,
    drift_hz_s: float,
    clocks: tuple[ScanClock, ...],
    epoch: float,
) -> int | None:
    for index, event in enumerate(events):
        if abs(event.drift_hz_s - drift_hz_s) > DRIFT_MATCH_HZ_S:
            continue
        for clock in clocks:
            projected = project_hz(event.frequency_mhz, event.drift_hz_s, clock.tstart_mjd, epoch)
            if abs(projected - center_hz) <= FREQUENCY_MATCH_HZ:
                return index
    return None


def _hit_from_detections(
    members: tuple[ScanDetection, ...],
    clocks: dict[int, ScanClock],
    catalog: KnownCatalog,
    *,
    cadence_id: str,
    event: FilterEvent | None,
    center_hz: float,
) -> ArchiveHit:
    best = max(members, key=lambda item: item.snr)
    on_count = len({item.scan_index for item in members if clocks[item.scan_index].on_source})
    off_count = len({item.scan_index for item in members if not clocks[item.scan_index].on_source})
    passed = event is not None
    decision = decide_verdict(
        frequency_mhz=best.frequency_mhz,
        drift_hz_s=best.drift_hz_s,
        on_count=on_count,
        off_count=off_count,
        passed_filter=passed,
        catalog=catalog,
        session_hits=(),
        this_cadence_id=cadence_id,
        synthetic=False,
    )
    note = (
        f"Associated {len(members)} detections inside {FREQUENCY_MATCH_HZ:g} Hz and "
        f"{DRIFT_MATCH_HZ_S:g} Hz/s of the projected seed at {center_hz:.3f} Hz. "
        f"Measured on_count={on_count}, off_count={off_count}, passed_filter={passed}."
    )
    return ArchiveHit(
        hit_id="",
        frequency_mhz=best.frequency_mhz,
        drift_hz_s=best.drift_hz_s,
        snr=best.snr,
        on_count=on_count,
        off_count=off_count,
        passed_filter=passed,
        event_id=None if event is None else event.event_id,
        association_note=note,
        decision=decision,
    )


def _hit_from_event(event: FilterEvent, catalog: KnownCatalog, *, cadence_id: str) -> ArchiveHit:
    decision = decide_verdict(
        frequency_mhz=event.frequency_mhz,
        drift_hz_s=event.drift_hz_s,
        on_count=0,
        off_count=0,
        passed_filter=True,
        catalog=catalog,
        session_hits=(),
        this_cadence_id=cadence_id,
        synthetic=False,
    )
    note = (
        "Filter-3 event had no detection inside the projected "
        f"{FREQUENCY_MATCH_HZ:g} Hz / {DRIFT_MATCH_HZ_S:g} Hz/s window. "
        "on_count and off_count are 0, not the filter's implied 3 and 0."
    )
    return ArchiveHit(
        hit_id="",
        frequency_mhz=event.frequency_mhz,
        drift_hz_s=event.drift_hz_s,
        snr=event.snr,
        on_count=0,
        off_count=0,
        passed_filter=True,
        event_id=event.event_id,
        association_note=note,
        decision=decision,
    )


def _plot_on_only(
    hits: tuple[ArchiveHit, ...],
    slices: tuple[CadenceSlice, ...],
    plot_dir: Path,
    *,
    target: str,
) -> tuple[dict[str, Path], tuple[str, ...]]:
    selected = sorted(
        (hit for hit in hits if hit.decision.verdict is Verdict.ON_ONLY),
        key=lambda hit: (-hit.snr, hit.frequency_mhz),
    )
    notes: list[str] = []
    if len(selected) > _MAX_ON_ONLY_PLOTS:
        notes.append(
            f"{len(selected)} ON_ONLY hits; plotted the {_MAX_ON_ONLY_PLOTS} with the highest SNR."
        )
        selected = selected[:_MAX_ON_ONLY_PLOTS]
    written: dict[str, Path] = {}
    for hit in selected:
        try:
            path = _plot_one(hit, slices, plot_dir, target=target)
        except Exception as exc:
            notes.append(f"Plot for {hit.hit_id} failed: {type(exc).__name__}: {exc}")
            continue
        size = path.stat().st_size
        if size > _MAX_ASSET_BYTES:
            path.unlink()
            notes.append(f"Plot for {hit.hit_id} was {size} bytes, above 2 MiB, and was not kept.")
            continue
        written[hit.hit_id] = path
    return written, tuple(notes)


def _plot_one(
    hit: ArchiveHit, slices: tuple[CadenceSlice, ...], plot_dir: Path, *, target: str
) -> Path:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=False)
    import pandas as pd
    from turbo_seti.find_event.plot_event import plot_candidate_events

    plot_dir.mkdir(parents=True, exist_ok=True)
    before = set(plot_dir.glob("*.png"))
    frame = pd.DataFrame(
        [
            {
                "Source": target,
                "Freq": hit.frequency_mhz,
                "DriftRate": hit.drift_hz_s,
            }
        ]
    )
    plot_candidate_events(
        frame,
        [str(item.path) for item in slices],
        "f3",
        [item.info.header.source_name for item in slices],
        offset=0,
        plot_dir=f"{plot_dir}/",
    )
    created = set(plot_dir.glob("*.png")) - before
    if len(created) != 1:
        raise RuntimeError(f"expected one plot file, found {len(created)}")
    return created.pop()


def _copy_plots(plot_paths: dict[str, Path], run_dir: Path) -> dict[str, Path]:
    destination = run_dir / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for hit_id, source in plot_paths.items():
        target = destination / f"{hit_id}.png"
        shutil.copyfile(source, target)
        copied[hit_id] = target
    return copied


def _payload_hits(
    hits: tuple[ArchiveHit, ...], copied_plots: dict[str, Path]
) -> tuple[ArchiveHit, ...]:
    by_snr = sorted(hits, key=lambda hit: (-hit.snr, hit.frequency_mhz))
    selected: list[ArchiveHit] = []
    seen: set[str] = set()
    for hit in [*by_snr[:3], *(hit for hit in by_snr if hit.hit_id in copied_plots)]:
        if hit.hit_id in seen:
            continue
        seen.add(hit.hit_id)
        selected.append(hit)
    return tuple(selected)


def _write_thumbs(run_dir: Path, hits: tuple[ArchiveHit, ...]) -> dict[str, str]:
    thumb_dir = run_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    urls: dict[str, str] = {}
    for hit in hits:
        relative = f"runs/{run_dir.name}/thumbs/{hit.hit_id}.svg"
        (thumb_dir / f"{hit.hit_id}.svg").write_text(_thumb_svg(hit), encoding="utf-8")
        urls[hit.hit_id] = relative
    return urls


def _thumb_svg(hit: ArchiveHit) -> str:
    freq = f"{hit.frequency_mhz:.6f} MHz"
    drift = f"{hit.drift_hz_s:+.6f} Hz/s"
    snr = f"SNR {hit.snr:.3f}"
    verdict = hit.decision.verdict.value
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="140" role="img"\n'
        '  aria-label="Measured hit frequency, drift, and SNR">\n'
        '  <rect width="320" height="140" fill="#0D1320"/>\n'
        f'  <text x="16" y="36" fill="#E7EEF8" font-size="16">{html.escape(freq)}</text>\n'
        f'  <text x="16" y="64" fill="#E7B15A" font-size="14">{html.escape(drift)}</text>\n'
        f'  <text x="16" y="92" fill="#C5D0DE" font-size="14">{html.escape(snr)}</text>\n'
        f'  <text x="16" y="120" fill="#C5D0DE" font-size="14">{html.escape(verdict)}</text>\n'
        "</svg>\n"
    )


def _cadence_hits(
    hits: tuple[ArchiveHit, ...],
    found_at: datetime,
    thumb_urls: dict[str, str],
    plot_paths: dict[str, Path],
) -> tuple[CadenceHit, ...]:
    rows: list[CadenceHit] = []
    for hit in hits:
        plot_url = None
        if hit.hit_id in plot_paths:
            plot_url = f"runs/{plot_paths[hit.hit_id].parent.parent.name}/plots/{hit.hit_id}.png"
        rows.append(
            CadenceHit(
                hit_id=hit.hit_id,
                found_at=found_at,
                freq_mhz=hit.frequency_mhz,
                drift_hz_s=hit.drift_hz_s,
                snr=hit.snr,
                verdict=hit.decision.verdict,
                explanation=hit.decision.explanation,
                checks_done=hit.decision.checks_done,
                known_source=hit.decision.known_source,
                thumbnail_url=thumb_urls[hit.hit_id],
                plot_url=plot_url,
                review_signoffs=(),
            )
        )
    return tuple(rows)


def _payload(
    *,
    cadence_id: str,
    target: str,
    ra_deg: float,
    dec_deg: float,
    window_mhz: tuple[float, float],
    on_source_seconds: float,
    started_at: datetime,
    hits: tuple[CadenceHit, ...],
) -> CadencePayload:
    return CadencePayload(
        kind="cadence",
        cadence_id=cadence_id,
        target=target,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        status=CadenceStatus.DONE,
        windows_done=1,
        window_mhz=window_mhz,
        on_source_seconds=on_source_seconds,
        started_at=started_at,
        hits=hits,
    )


def _on_source_seconds(slices: tuple[CadenceSlice, ...]) -> float:
    return sum(
        item.info.header.tsamp_seconds * item.info.header.integrations
        for item in slices
        if item.role == "A"
    )


def _inputs(slices: tuple[CadenceSlice, ...]) -> tuple[InputRef, ...]:
    return tuple(
        InputRef(
            uri=item.url,
            sha256=item.info.output_sha256,
            size_bytes=item.info.output_size_bytes,
        )
        for item in slices
    )


def _slice_summary(
    slices: tuple[CadenceSlice, ...],
    *,
    cadence_id: str,
    window_mhz: tuple[float, float],
    max_transfer_bytes: int,
    allow_plain_http: bool,
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for item in slices:
        observed = item.transfer.source_size_bytes
        catalog_size = item.catalog_size_bytes
        size_match = None if catalog_size is None else observed == catalog_size
        files.append(
            {
                "scan_index": item.scan_index,
                "role": item.role,
                "url": item.url,
                "expected_source_name": item.expected_source_name,
                "source_header": item.info.header.as_dict(),
                "slice": item.info.as_dict(),
                "remote_transfer": item.transfer.as_dict(),
                "catalog_md5": item.catalog_md5,
                "catalog_size_bytes": catalog_size,
                "catalog_size_matches_observed_content_length": size_match,
                "coordinates": _coordinate_attrs(item.path),
            }
        )
    return {
        "cadence_id": cadence_id,
        "requested_window_mhz": [window_mhz[0], window_mhz[1]],
        "transfer_cap_bytes_per_file": max_transfer_bytes,
        "allow_plain_http": allow_plain_http,
        "catalog_md5_note": (
            "catalog_md5 is the value returned by the open-data API. "
            "This run did not hash the remote file."
        ),
        "sha256_note": "slice.sha256 is the local frequency slice, not the remote object.",
        "files": files,
    }


def _coordinate_attrs(path: Path) -> dict[str, float | None]:
    import h5py

    found: dict[str, float | None] = {"src_raj": None, "src_dej": None}
    with h5py.File(path, "r") as handle:
        dataset = handle.get("data", None)
        for name in ("src_raj", "src_dej"):
            value = _attr_float(handle.attrs, name)
            if value is None and dataset is not None:
                value = _attr_float(dataset.attrs, name)
            found[name] = value
    return found


def _attr_float(attrs: object, name: str) -> float | None:
    try:
        if name not in attrs:  # type: ignore[operator]
            return None
        raw = attrs[name]  # type: ignore[index]
    except (KeyError, TypeError):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _write_hits(path: Path, hits: tuple[ArchiveHit, ...], found_at: datetime) -> None:
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - radio extra installation failure
        raise RuntimeError("hits.parquet requires polars from the radio extra") from exc

    records: list[dict[str, object]] = []
    for hit in hits:
        checks = {check.name: check for check in hit.decision.checks}
        row: dict[str, object] = {
            "hit_id": hit.hit_id,
            "freq_mhz": hit.frequency_mhz,
            "drift_hz_s": hit.drift_hz_s,
            "snr": hit.snr,
            "verdict": hit.decision.verdict.value,
            "explanation": hit.decision.explanation,
            "checks_done": hit.decision.checks_done,
            "known_source": hit.decision.known_source,
            "found_at": found_at.isoformat(),
            "on_count": hit.on_count,
            "off_count": hit.off_count,
            "passed_filter": hit.passed_filter,
            "event_id": hit.event_id,
            "association_note": hit.association_note,
        }
        for name in _CHECK_NAMES:
            row[name] = checks[name].passed
            row[f"note_{name}"] = checks[name].note
        records.append(row)
    frame = pl.DataFrame(records) if records else pl.DataFrame(schema=_empty_hit_schema())
    frame.write_parquet(path)


def _empty_hit_schema() -> pl.Schema:
    import polars as pl

    schema = pl.Schema(
        {
            "hit_id": pl.String,
            "freq_mhz": pl.Float64,
            "drift_hz_s": pl.Float64,
            "snr": pl.Float64,
            "verdict": pl.String,
            "explanation": pl.String,
            "checks_done": pl.Int64,
            "known_source": pl.String,
            "found_at": pl.String,
            "on_count": pl.Int64,
            "off_count": pl.Int64,
            "passed_filter": pl.Boolean,
            "event_id": pl.String,
            "association_note": pl.String,
        }
    )
    for name in _CHECK_NAMES:
        schema[name] = pl.Boolean
        schema[f"note_{name}"] = pl.String
    return schema


def _assert_parquet_counts(path: Path, hits: tuple[ArchiveHit, ...]) -> None:
    import polars as pl

    frame = pl.read_parquet(path)
    if frame.height != len(hits):
        raise RuntimeError("hits.parquet row count does not match labelled hits")
    counts = _verdict_counts(hits)
    for verdict, count in counts.items():
        observed = int((frame["verdict"] == verdict).sum()) if frame.height else 0
        if observed != count:
            raise RuntimeError(f"verdict count for {verdict} is {observed}, expected {count}")


def _verdict_counts(hits: tuple[ArchiveHit, ...]) -> dict[str, int]:
    counts = {item.value: 0 for item in Verdict}
    for hit in hits:
        counts[hit.decision.verdict.value] += 1
    if counts[Verdict.UNEXPLAINED.value] != 0:
        raise RuntimeError("archive labelling emitted UNEXPLAINED")
    return counts


def _parameters(
    *,
    experiment_id: str,
    cadence_id: str,
    target: str,
    ra_deg: float,
    dec_deg: float,
    window_mhz: tuple[float, float],
    known_sources_path: Path,
    max_transfer_bytes: int,
    hits: tuple[ArchiveHit, ...],
    slices: tuple[CadenceSlice, ...],
    allow_plain_http: bool,
) -> dict[str, object]:
    return {
        "synthetic": False,
        "experiment_id": experiment_id,
        "cadence_id": cadence_id,
        "target": target,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "window_mhz": [window_mhz[0], window_mhz[1]],
        "roles": [item.role for item in slices],
        "filter_threshold": 3,
        "minimum_snr": 10.0,
        "max_drift_hz_s": 4.0,
        "frequency_match_hz": FREQUENCY_MATCH_HZ,
        "drift_match_hz_s": DRIFT_MATCH_HZ_S,
        "transfer_cap_bytes_per_file": max_transfer_bytes,
        "allow_plain_http": allow_plain_http,
        "known_sources_sha256": sha256_file(known_sources_path),
        "bytes_transferred": [item.transfer.bytes_transferred for item in slices],
        "verdict_counts": _verdict_counts(hits),
        "hit_count": len(hits),
    }


def _outcome(hits: tuple[ArchiveHit, ...]) -> str:
    if any(hit.decision.verdict is Verdict.ON_ONLY for hit in hits):
        return "searched_with_survivors"
    return "searched"


def _notes(
    hits: tuple[ArchiveHit, ...],
    search: CadenceSearch,
    payload_hits: tuple[ArchiveHit, ...],
    plot_notes: tuple[str, ...],
    slices: tuple[CadenceSlice, ...],
    *,
    max_transfer_bytes: int,
    target: str,
    ra_deg: float,
    dec_deg: float,
) -> tuple[str, ...]:
    counts = _verdict_counts(hits)
    summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    local = all(
        item.transfer.bytes_transferred == 0 and not item.transfer.ranges for item in slices
    )
    lines = [
        "Archive ABACAD cadence. demo is false. Slice SHA-256 values are local slices, "
        "not remote-file hashes. catalog_md5 is the open-data API field and was not recomputed.",
        f"Transfer cap is {max_transfer_bytes} bytes per file.",
        f"Verdict counts: {summary}.",
        f"hits.parquet rows: {len(hits)}. Page payload hits: {len(payload_hits)} "
        "(top 3 by SNR, plus plotted ON_ONLY).",
        f"Detections: {len(search.detections)}. Filter-3 events retained: {len(search.events)}.",
        f"Pointing {target} uses catalog ra {ra_deg} deg and decl {dec_deg} deg. "
        "Per-file src_raj and src_dej, when the header has them, are in slice_summary.json.",
        "No other cadence from this observing session was supplied.",
    ]
    if local:
        lines.append("Inputs were local files; bytes_transferred is 0.")
    lines.extend(plot_notes)
    lines.extend(search.warnings)
    return tuple(lines)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in slug.split("-") if part) or "cadence"

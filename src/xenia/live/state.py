"""Build the single Mission Control state document from validated run manifests."""

from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from xenia import __version__
from xenia.live.evidence import is_synthetic, selected_cadences, verdict_counts
from xenia.live.models import (
    CadenceHit,
    CadencePayload,
    CadenceStatus,
    DetectionFloor,
    FloorPayload,
    GalleryItem,
    HeroPayload,
    LiveState,
    LogEntry,
    Pointing,
    RunManifest,
    SiteMeta,
    Totals,
    Verdict,
    VoyagerHero,
    WorkerPayload,
    WorkerState,
    WorkerStatus,
)

REPOSITORY_URL = "https://github.com/mpetyx/theoxenia"
DEMO_DATA_CREDIT = "Synthetic M0 fixtures; no telescope observations are included."
OBSERVED_DATA_CREDIT = "Breakthrough Listen open data, Green Bank Telescope."
INJECTION_DATA_CREDIT = "Recovery floor from synthetic signal injections generated with setigen."


class ManifestLoadError(ValueError):
    """Raised when a run manifest is missing or violates its schema."""


def manifest_paths(root: Path) -> tuple[Path, ...]:
    """Return sorted manifest paths below ``root`` without reading raw data files."""
    if not root.exists():
        raise FileNotFoundError(f"manifest directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"manifest path is not a directory: {root}")
    return tuple(sorted(root.rglob("manifest.json")))


def load_manifests(root: Path) -> list[RunManifest]:
    """Load and validate every manifest below ``root``; fail loudly on malformed input."""
    paths = manifest_paths(root)
    if not paths:
        raise ManifestLoadError(f"no manifest.json files found below {root}")

    manifests: list[RunManifest] = []
    seen_run_ids: set[str] = set()
    for path in paths:
        try:
            manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            raise ManifestLoadError(f"invalid manifest {path}: {exc}") from exc
        if manifest.run_id in seen_run_ids:
            raise ManifestLoadError(f"duplicate run_id {manifest.run_id!r} under {root}")
        seen_run_ids.add(manifest.run_id)
        manifests.append(manifest)
    return manifests


def read_git_sha(root: Path | None = None) -> str:
    """Read a short commit ID, using ``unknown`` only outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            cwd=root,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_state(
    manifests: Sequence[RunManifest],
    *,
    generated_at: datetime | None = None,
    git_sha: str | None = None,
    package_version: str = __version__,
    repo_url: str = REPOSITORY_URL,
    mirror_url: str | None = None,
) -> LiveState:
    """Aggregate validated manifests into a page-ready ``LiveState``.

    The result is deterministic for the same manifests, Git SHA, and package version,
    except for ``generated_at``. No page metric is supplied by the renderer.
    """
    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")

    worker_events: list[tuple[RunManifest, WorkerPayload]] = []
    cadence_events = selected_cadences(manifests)
    hero_events: list[tuple[RunManifest, HeroPayload]] = []
    floor_events: list[tuple[RunManifest, FloorPayload]] = []
    for manifest in manifests:
        payload = manifest.mission_control
        if isinstance(payload, WorkerPayload):
            worker_events.append((manifest, payload))
        elif isinstance(payload, HeroPayload):
            hero_events.append((manifest, payload))
        elif isinstance(payload, FloorPayload):
            floor_events.append((manifest, payload))

    latest_worker = max(worker_events, key=lambda item: item[0].started_at, default=None)
    if latest_worker is None:
        worker = WorkerState(
            status=WorkerStatus.PAUSED,
            last_heartbeat=None,
            host="not-started",
        )
    else:
        snapshot = latest_worker[1]
        worker = WorkerState(
            status=snapshot.status,
            last_heartbeat=snapshot.last_heartbeat,
            host=snapshot.host,
            current=snapshot.current,
            next=snapshot.next,
        )

    grouped: dict[str, list[tuple[RunManifest, CadencePayload]]] = defaultdict(list)
    all_hit_records: dict[str, tuple[RunManifest, CadencePayload, CadenceHit]] = {}
    for manifest, cadence in cadence_events:
        grouped[cadence.cadence_id].append((manifest, cadence))
        for hit in cadence.hits:
            previous = all_hit_records.get(hit.hit_id)
            if previous is None or manifest.started_at > previous[0].started_at:
                all_hit_records[hit.hit_id] = (manifest, cadence, hit)

    pointings: list[Pointing] = []
    for cadence_id, records in sorted(grouped.items()):
        latest_manifest, latest_cadence = max(records, key=lambda item: item[0].started_at)
        pointings.append(
            Pointing(
                cadence_id=cadence_id,
                target=latest_cadence.target,
                ra_deg=latest_cadence.ra_deg,
                dec_deg=latest_cadence.dec_deg,
                status=latest_cadence.status,
                windows_done=sum(cadence.windows_done for _, cadence in records),
                hits=sum(sum(verdict_counts(m, c).values()) for m, c in records),
                survivors=sum(verdict_counts(m, c)[Verdict.ON_ONLY] for m, c in records),
                started_at=latest_cadence.started_at,
                manifest_url=_manifest_url(latest_manifest.run_id),
            )
        )

    hits_by_verdict: Counter[Verdict] = Counter()
    for manifest, cadence in cadence_events:
        hits_by_verdict.update(verdict_counts(manifest, cadence))
    completed_pairs = [
        cadence for _, cadence in cadence_events if cadence.status is CadenceStatus.DONE
    ]
    failed_pairs = [
        cadence for _, cadence in cadence_events if cadence.status is CadenceStatus.FAILED
    ]
    completed_cadences = {cadence.cadence_id for cadence in completed_pairs}
    mhz_hours = sum(
        (cadence.window_mhz[1] - cadence.window_mhz[0]) * cadence.on_source_seconds / 3600
        for cadence in completed_pairs
    )
    totals = Totals(
        cadences_searched=len(completed_cadences),
        windows_searched=sum(cadence.windows_done for cadence in completed_pairs),
        mhz_hours_on_target=round(mhz_hours, 6),
        hits=sum(hits_by_verdict.values()),
        known_source=hits_by_verdict[Verdict.KNOWN_SOURCE],
        in_off=hits_by_verdict[Verdict.IN_OFF],
        single_pointing=hits_by_verdict[Verdict.SINGLE_POINTING],
        rfi_census=hits_by_verdict[Verdict.RFI_CENSUS],
        survivors=hits_by_verdict[Verdict.ON_ONLY],
        unexplained=hits_by_verdict[Verdict.UNEXPLAINED],
        targets_distinct=len({cadence.target for _, cadence in cadence_events}),
        failed=len(failed_pairs),
    )

    gallery_candidates: list[GalleryItem] = []
    for manifest, cadence, hit in all_hit_records.values():
        gallery_candidates.append(
            GalleryItem(
                hit_id=hit.hit_id,
                cadence_id=cadence.cadence_id,
                target=cadence.target,
                freq_mhz=hit.freq_mhz,
                drift_hz_s=hit.drift_hz_s,
                snr=hit.snr,
                verdict=hit.verdict,
                badge_text=verdict_badge(hit.verdict, hit.checks_done, hit.known_source),
                explanation=hit.explanation,
                checks_done=hit.checks_done,
                thumbnail_url=hit.thumbnail_url,
                plot_url=hit.plot_url,
                manifest_url=_manifest_url(manifest.run_id),
                found_at=hit.found_at,
            )
        )

    latest_hits = sorted(gallery_candidates, key=lambda item: item.found_at, reverse=True)[:24]
    recent_survivors = [
        item
        for item in gallery_candidates
        if item.verdict is Verdict.ON_ONLY and item.found_at >= timestamp - timedelta(hours=24)
    ]
    retained_by_id = {item.hit_id: item for item in [*latest_hits, *recent_survivors]}
    gallery = tuple(
        sorted(
            retained_by_id.values(),
            key=lambda item: (
                item.verdict is Verdict.ON_ONLY
                and item.found_at >= timestamp - timedelta(hours=24),
                item.found_at,
            ),
            reverse=True,
        )
    )

    latest_hero = max(hero_events, key=lambda item: item[0].started_at, default=None)
    hero: VoyagerHero | None = None
    if latest_hero is not None:
        manifest, payload = latest_hero
        hero = VoyagerHero(
            image_url=payload.image_url,
            freq_mhz=payload.freq_mhz,
            drift_hz_s=payload.drift_hz_s,
            snr=payload.snr,
            run_id=manifest.run_id,
            manifest_url=_manifest_url(manifest.run_id),
        )

    latest_floor = max(floor_events, key=lambda item: item[0].started_at, default=None)
    detection_floor: DetectionFloor | None = None
    if latest_floor is not None:
        manifest, payload = latest_floor
        detection_floor = DetectionFloor(
            run_id=manifest.run_id,
            snr_grid=payload.snr_grid,
            drift_grid=payload.drift_grid,
            trials_per_cell=payload.trials_per_cell,
            recovered_fraction=payload.recovered_fraction,
            summary=floor_summary(payload),
            manifest_url=_manifest_url(manifest.run_id),
            ci_run_url=payload.ci_run_url,
        )

    log_entries = [
        LogEntry(
            ts=manifest.finished_at,
            level="error" if manifest.outcome == "failed" else "info",
            msg=note,
        )
        for manifest in manifests
        for note in manifest.notes
    ]
    log = tuple(sorted(log_entries, key=lambda entry: entry.ts, reverse=True)[:50])
    real_payloads = [
        manifest.mission_control
        for manifest in manifests
        if not manifest.demo
        and (isinstance(manifest.mission_control, FloorPayload) or not is_synthetic(manifest))
        and isinstance(
            manifest.mission_control,
            (CadencePayload, HeroPayload, FloorPayload),
        )
    ]
    has_real_payload = bool(real_payloads)
    is_demo = not has_real_payload
    has_observations = any(
        isinstance(payload, (CadencePayload, HeroPayload)) for payload in real_payloads
    )
    has_injections = any(isinstance(payload, FloorPayload) for payload in real_payloads)
    credits = []
    if has_observations:
        credits.append(OBSERVED_DATA_CREDIT)
    if has_injections:
        credits.append(INJECTION_DATA_CREDIT)
    data_credit = "; ".join(credits) if credits else DEMO_DATA_CREDIT

    mirror = mirror_url is not None
    return LiveState(
        generated_at=timestamp,
        git_sha=git_sha or read_git_sha(),
        package_version=package_version,
        meta=SiteMeta(
            demo=is_demo,
            repo_url=repo_url,
            mirror_url=mirror_url,
            data_credit=data_credit,
            is_mirror=mirror,
            mirror_updated_at=timestamp if mirror else None,
        ),
        worker=worker,
        totals=totals,
        hero=hero,
        detection_floor=detection_floor,
        pointings=tuple(sorted(pointings, key=lambda item: item.started_at, reverse=True)),
        gallery=gallery,
        log=log,
    )


def _manifest_url(run_id: str) -> str:
    """Return a safe, site-root-relative link for one run manifest."""
    return f"runs/{run_id}/manifest.json"


def verdict_badge(verdict: Verdict, checks_done: int, known_source: str | None) -> str:
    """Map a verdict code to the exact short public badge wording."""
    if verdict is Verdict.KNOWN_SOURCE:
        return f"Known transmitter: {known_source}" if known_source else "Known transmitter"
    if verdict is Verdict.IN_OFF:
        return "Also present in OFF scans"
    if verdict is Verdict.SINGLE_POINTING:
        return "Missing from one or more ON scans"
    if verdict is Verdict.RFI_CENSUS:
        return "Same frequency in another star the same day"
    if verdict is Verdict.ON_ONLY:
        return f"Survivor, {checks_done} of 5 checks done"
    return "Unexplained after 5 checks and review"


def floor_summary(payload: FloorPayload) -> str:
    """Compute the lowest fully recovered SNR row within plus/minus 3 Hz/s."""
    eligible_columns = [index for index, drift in enumerate(payload.drift_grid) if abs(drift) <= 3]
    if not eligible_columns:
        return "No drift cells within the configured ±3 Hz/s summary range."

    qualifying_snrs = [
        snr
        for snr, row in zip(payload.snr_grid, payload.recovered_fraction, strict=True)
        if all(row[index] == 1.0 for index in eligible_columns)
    ]
    if not qualifying_snrs:
        center_columns = [index for index, drift in enumerate(payload.drift_grid) if drift == 0]
        nonzero_columns = [index for index in eligible_columns if payload.drift_grid[index] != 0]
        if center_columns and nonzero_columns:
            center_index = center_columns[0]
            center_floor = next(
                (
                    snr
                    for snr, row in zip(payload.snr_grid, payload.recovered_fraction, strict=True)
                    if row[center_index] == 1.0
                ),
                None,
            )
            if center_floor is not None:
                best_nonzero = max(
                    row[index] for row in payload.recovered_fraction for index in nonzero_columns
                )
                nonzero_summary = (
                    "all tested nonzero-drift cells recovered 0%"
                    if best_nonzero == 0.0
                    else f"best nonzero-drift recovery was {best_nonzero:.0%}"
                )
                return (
                    f"At 0 Hz/s, full recovery begins at SNR ≥ {center_floor:g}; "
                    f"{nonzero_summary}. No tested SNR reached full recovery across ±3 Hz/s."
                )
        return "No tested SNR reached full recovery within the configured ±3 Hz/s range."
    threshold = min(qualifying_snrs)
    return f"Recovers SNR ≥ {threshold:g} at drift within ±3 Hz/s."


def manifest_input_refs(paths: Iterable[Path], *, root: Path) -> tuple[dict[str, object], ...]:
    """Build checksummed provenance records for input manifests."""
    from hashlib import sha256

    records: list[dict[str, object]] = []
    for path in sorted(paths):
        digest = sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "uri": path.relative_to(root).as_posix(),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return tuple(records)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a small generated artifact."""
    from hashlib import sha256

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Bounded real-data worker with durable receipts and main-only publication.

Completion is reconstructed from validated cadence manifests, so a lost process
or a fresh machine does not rescan completed windows. No branch is created.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from xenia.live.config import load_live_config
from xenia.live.evidence import is_synthetic, verdict_counts
from xenia.live.models import CadencePayload, CadenceStatus, RunManifest
from xenia.live.queue import QueuePair, read_queue
from xenia.live.state import load_manifests, sha256_file
from xenia.live.worker import PairOutcome, execute_pair
from xenia.provenance import create_run_manifest, input_ref, new_run_id, output_ref

STATUS_PATH = Path("runs/observatory/status.json")
MAX_FAILURES = 3


class Failure(BaseModel):
    attempts: int = Field(ge=1)
    error: str
    last_attempt: datetime


class ObservatoryStatus(BaseModel):
    mode: Literal["scheduled", "hosted"] = "scheduled"
    state: Literal["ready", "needs_attention", "queue_complete"]
    checked_at: datetime
    cycle_id: str
    queue_sha256: str
    queued_windows: int = Field(ge=0)
    completed_windows: int = Field(ge=0)
    pending_windows: int = Field(ge=0)
    blocked_windows: int = Field(ge=0)
    new_runs: list[str] = Field(default_factory=list)
    failures: dict[str, Failure] = Field(default_factory=dict)
    cadence_ids: list[str]
    interval_minutes: int = Field(default=180, ge=1)


def pair_key(cadence_id: str, window: tuple[float, float]) -> str:
    return f"{cadence_id}:{window[0]:.6f}:{window[1]:.6f}"


def validate_cadence(root: Path, manifest: RunManifest) -> None:
    """Verify compact artifacts and independently compare the complete Parquet counts."""
    import polars as pl

    payload = manifest.mission_control
    if not isinstance(payload, CadencePayload) or is_synthetic(manifest):
        raise ValueError("observatory requires a real cadence manifest")
    if payload.status is not CadenceStatus.DONE or len(manifest.inputs) != 6:
        raise ValueError("cadence must be complete and retain six input references")
    for artifact in manifest.outputs:
        path = root / artifact.path
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("artifact escapes repository")
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("artifact missing or exceeds compact-evidence limit")
        if artifact.sha256 and sha256_file(path) != artifact.sha256:
            raise ValueError("artifact checksum mismatch")
    table = pl.read_parquet(root / "runs" / manifest.run_id / "hits.parquet")
    actual = {row["verdict"]: row["len"] for row in table.group_by("verdict").len().to_dicts()}
    expected = {v.value: n for v, n in verdict_counts(manifest, payload).items() if n}
    if actual != expected or table["hit_id"].n_unique() != table.height:
        raise ValueError("complete hit table differs from manifest or has duplicate IDs")


def completed_pairs(root: Path) -> set[str]:
    completed: set[str] = set()
    for manifest in load_manifests(root / "runs"):
        payload = manifest.mission_control
        if (
            isinstance(payload, CadencePayload)
            and payload.status is CadenceStatus.DONE
            and not is_synthetic(manifest)
        ):
            validate_cadence(root, manifest)
            completed.add(pair_key(payload.cadence_id, payload.window_mhz))
    return completed


def read_status(root: Path) -> ObservatoryStatus | None:
    path = root / STATUS_PATH
    return ObservatoryStatus.model_validate_json(path.read_text()) if path.exists() else None


def _write_json(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def run_cycle(
    root: Path,
    *,
    max_pairs: int = 1,
    mode: Literal["scheduled", "hosted"] = "scheduled",
    runner: Callable[[QueuePair], PairOutcome] | None = None,
) -> ObservatoryStatus:
    """Process at most max_pairs real windows, retaining failures for bounded retries."""
    if not 1 <= max_pairs <= 8:
        raise ValueError("max_pairs must be between 1 and 8")
    root = root.resolve()
    lock_path = root / ".preview" / "observatory.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ModuleNotFoundError:
            pass
        return _cycle(root, max_pairs=max_pairs, mode=mode, runner=runner)


def _cycle(
    root: Path,
    *,
    max_pairs: int,
    mode: Literal["scheduled", "hosted"],
    runner: Callable[[QueuePair], PairOutcome] | None,
) -> ObservatoryStatus:
    config_path = root / "configs/live.yaml"
    config = load_live_config(config_path)
    if config.commit_branch != "main":
        raise ValueError("observatory configuration must use main")
    if (config.max_drift_hz_s, config.minimum_snr, config.filter_threshold) != (4.0, 10.0, 3):
        raise ValueError("configured detector differs from the validated cadence search")
    queue_path = root / config.queue_file
    _, pairs = read_queue(queue_path)
    if not pairs or any(p.synthetic or p.band_coverage != "measured" for p in pairs):
        raise ValueError("observatory queue must contain measured, real-data windows only")
    keys = [pair_key(p.cadence_id, p.window_mhz) for p in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("queue contains duplicate windows")
    started = datetime.now(UTC)
    cycle_id = new_run_id(started, "observatory-cycle")
    completed = completed_pairs(root)
    previous = read_status(root)
    failures = {} if previous is None else dict(previous.failures)
    pending = [p for p in pairs if pair_key(p.cadence_id, p.window_mhz) not in completed]
    new_runs: list[str] = []
    attempted = 0
    for pair in pending:
        key = pair_key(pair.cadence_id, pair.window_mhz)
        previous_failure = failures.get(key)
        attempts = 0 if previous_failure is None else previous_failure.attempts
        if attempts >= MAX_FAILURES:
            continue
        if attempted >= max_pairs:
            break
        attempted += 1
        try:
            outcome = (runner or (lambda p: execute_pair(root, p, config=config)))(pair)
            manifest = RunManifest.model_validate_json(
                (root / "runs" / outcome.run_id / "manifest.json").read_text()
            )
            validate_cadence(root, manifest)
            payload = manifest.mission_control
            assert isinstance(payload, CadencePayload)
            if pair_key(payload.cadence_id, payload.window_mhz) != key:
                raise ValueError("worker returned a different window")
            completed.add(key)
            new_runs.append(outcome.run_id)
            failures.pop(key, None)
        except Exception as exc:
            failures[key] = Failure(
                attempts=attempts + 1,
                error=f"{type(exc).__name__}: {exc}"[:300],
                last_attempt=datetime.now(UTC),
            )
    remaining = set(keys) - completed
    failures = {k: v for k, v in failures.items() if k in remaining}
    blocked = sum(f.attempts >= MAX_FAILURES for f in failures.values())
    status = ObservatoryStatus(
        mode=mode,
        state="needs_attention" if failures else "ready" if remaining else "queue_complete",
        checked_at=datetime.now(UTC),
        cycle_id=cycle_id,
        queue_sha256=sha256_file(queue_path),
        queued_windows=len(keys),
        completed_windows=len(set(keys) & completed),
        pending_windows=len(remaining) - blocked,
        blocked_windows=blocked,
        new_runs=new_runs,
        failures=failures,
        cadence_ids=sorted({p.cadence_id for p in pairs}),
    )
    receipt = root / "runs" / cycle_id / "cycle.json"
    _write_json(receipt, status.model_dump_json(indent=2))
    create_run_manifest(
        run_id=cycle_id,
        command=["python", "scripts/observatory.py", "--max-pairs", str(max_pairs)],
        parameters={"mode": mode, "attempted": attempted, "new_runs": new_runs},
        inputs=(
            input_ref(queue_path, repository_root=root),
            input_ref(config_path, repository_root=root),
        ),
        outputs=(output_ref(receipt, repository_root=root),),
        started_at=started,
        finished_at=status.checked_at,
        outcome=status.state,
        notes=[
            "Scheduled archive processing; no new telescope observations or 24-hour acceptance."
        ],
        repository_root=root,
        demo=False,
        experiment_id="OBS-001",
    )
    _write_json(root / STATUS_PATH, status.model_dump_json(indent=2))
    return status


def publish_cycle(root: Path, status: ObservatoryStatus) -> None:
    """Commit only this cycle's evidence, then fast-forward push main without force.

    A concurrent remote update fails the push and leaves the local evidence commit
    for recovery. It never rebases experimental code or overwrites remote changes.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    if git("branch", "--show-current") != "main":
        raise ValueError("publication requires an existing main checkout")
    if git("diff", "--cached", "--name-only"):
        raise ValueError("refusing to publish with unrelated staged changes")
    if git("diff", "--name-only", "--", ".", ":(exclude)runs/observatory/status.json"):
        raise ValueError("refusing to publish with modified tracked files")
    paths = [STATUS_PATH.as_posix(), f"runs/{status.cycle_id}"]
    for run_id in status.new_runs:
        manifest = RunManifest.model_validate_json(
            (root / "runs" / run_id / "manifest.json").read_text()
        )
        validate_cadence(root, manifest)
        paths.extend(a.path for a in manifest.outputs)
    git("add", "--", *paths)
    git("commit", "-m", f"data: observatory cycle {status.cycle_id}")
    git("push", "origin", "main")

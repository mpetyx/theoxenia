"""Reproducible Voyager 1 known-answer run and manifest-backed waterfall plot."""

from __future__ import annotations

import importlib
import json
import platform
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from xenia.live.models import ArtifactRef, HeroPayload, InputRef, RunManifest
from xenia.provenance import create_run_manifest, new_run_id, output_ref
from xenia.radio.detector import DetectorResult, DopplerHit, run_turboseti, select_known_answer_hit
from xenia.radio.hdf5 import Hdf5FilterbankHeader, Hdf5SliceInfo, slice_hdf5_filterbank
from xenia.radio.remote import RangeMetrics, open_counted_http_file

CANONICAL_SOURCE_URL = (
    "http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.h5"
)
SOURCE_MIRROR_REPOSITORY = "RaghavGirgaonkar/SETI-Project-Test"
SOURCE_MIRROR_COMMIT = "9b6ef4ac4fb5f7a9efbcbb12ba202916e9d409eb"
SOURCE_MIRROR_BLOB_SHA1 = "42a41e5c2564afc8e73dc7b23cece3e3123f15d6"
DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    f"{SOURCE_MIRROR_REPOSITORY}/{SOURCE_MIRROR_COMMIT}/"
    "Voyager1.single_coarse.fine_res.h5"
)
SOURCE_SIZE_BYTES = 50_549_227
SOURCE_NAME = "Voyager1"
DEFAULT_FREQUENCY_LOW_MHZ = 8419.24
DEFAULT_FREQUENCY_HIGH_MHZ = 8419.35
KNOWN_ANSWER_LOW_MHZ = 8419.296
KNOWN_ANSWER_HIGH_MHZ = 8419.298
DEFAULT_MAX_DRIFT_HZ_S = 4.0
DEFAULT_MINIMUM_SNR = 25.0
MINIMUM_DRIFT_HZ_S = 1e-5
REMOTE_BLOCK_SIZE_BYTES = 64 * 1024
REMOTE_TRANSFER_LIMIT_BYTES = 24 * 1024 * 1024


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _runtime_versions() -> dict[str, str]:
    """Record the HDF5 runtime stack needed to reproduce this detector input."""
    importlib.import_module("hdf5plugin")
    import h5py
    import numpy

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "h5py": h5py.__version__,
        "libhdf5": h5py.version.hdf5_version,
        "hdf5plugin": version("hdf5plugin"),
    }


def _render_waterfall_png(
    slice_path: Path,
    output_path: Path,
    slice_info: Hdf5SliceInfo,
    *,
    hit: DopplerHit | None,
) -> None:
    """Render an exact-size 16:9 waterfall from the temporary HDF5 frequency slice."""
    import h5py
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with h5py.File(slice_path, "r") as h5:
        dataset = h5["data"]
        if dataset.shape != (
            slice_info.header.integrations,
            slice_info.header.nifs,
            slice_info.channel_count,
        ):
            raise ValueError("temporary HDF5 slice dimensions changed before plotting")
        cube = np.asarray(dataset[:], dtype=np.float32)
    waterfall = np.nanmean(cube, axis=1)
    if not np.isfinite(waterfall).any():
        raise ValueError("Voyager slice contains no finite samples to plot")

    first_center = slice_info.first_center_mhz
    last_center = slice_info.last_center_mhz
    if first_center > last_center:
        waterfall = waterfall[:, ::-1]
    channel_width_mhz = abs(slice_info.header.foff_mhz)
    frequency_min = min(first_center, last_center) - channel_width_mhz / 2
    frequency_max = max(first_center, last_center) + channel_width_mhz / 2
    elapsed_minutes = slice_info.header.integrations * slice_info.header.tsamp_seconds / 60
    finite_values = waterfall[np.isfinite(waterfall)]
    vmin, vmax = np.percentile(finite_values, [2, 99.5]).astype(float)
    if not vmax > vmin:
        midpoint = float(finite_values[0])
        padding = max(abs(midpoint) * 1e-6, 1e-6)
        vmin, vmax = midpoint - padding, midpoint + padding

    background = "#070B14"
    panel = "#0D1320"
    foreground = "#EAF0FA"
    accent = "#4DE2D2"
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=background)
    try:
        grid = figure.add_gridspec(4, 1, height_ratios=(1.0, 1.0, 1.0, 0.65), hspace=0.22)
        waterfall_axis = figure.add_subplot(grid[:3, 0])
        spectrum_axis = figure.add_subplot(grid[3, 0], sharex=waterfall_axis)
        for axis in (waterfall_axis, spectrum_axis):
            axis.set_facecolor(panel)
            axis.tick_params(colors=foreground, labelsize=9)
            for spine in axis.spines.values():
                spine.set_color("#1A2435")
            axis.grid(color="#1A2435", linewidth=0.5, alpha=0.8)

        image = waterfall_axis.imshow(
            waterfall,
            aspect="auto",
            origin="upper",
            extent=(frequency_min, frequency_max, elapsed_minutes, 0),
            interpolation="nearest",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        waterfall_axis.set_title(
            "Voyager 1 · Green Bank Telescope public recording",
            loc="left",
            color=foreground,
            fontsize=14,
            pad=10,
        )
        waterfall_axis.set_ylabel("Elapsed time (minutes)", color=foreground)
        waterfall_axis.tick_params(labelbottom=False)
        colorbar = figure.colorbar(image, ax=waterfall_axis, pad=0.012, fraction=0.025)
        colorbar.ax.tick_params(colors=foreground, labelsize=8)
        colorbar.set_label("Recorded sample value", color=foreground, fontsize=8)

        centers = np.linspace(
            frequency_min + channel_width_mhz / 2,
            frequency_max - channel_width_mhz / 2,
            slice_info.channel_count,
        )
        spectrum = np.nanmean(waterfall, axis=0)
        spectrum_axis.plot(centers, spectrum, color=accent, linewidth=0.65)
        spectrum_axis.set_ylabel("Mean value", color=foreground, fontsize=8)
        spectrum_axis.set_xlabel("Frequency (MHz)", color=foreground)

        if hit is None:
            waterfall_axis.text(
                0.012,
                0.94,
                "No hit passed the predeclared window and thresholds",
                transform=waterfall_axis.transAxes,
                color=foreground,
                fontsize=9,
                va="top",
                bbox={"facecolor": panel, "edgecolor": "#1A2435", "alpha": 0.9},
            )
        else:
            frequency = hit.uncorrected_frequency_mhz
            waterfall_axis.axvline(frequency, color=accent, linewidth=1.1, linestyle="--")
            spectrum_axis.axvline(frequency, color=accent, linewidth=1.0, linestyle="--")
            waterfall_axis.annotate(
                f"Observed carrier {frequency:.6f} MHz · SNR {hit.snr:.2f}",
                xy=(frequency, 0.06),
                xycoords=("data", "axes fraction"),
                xytext=(8, 0),
                textcoords="offset points",
                color=accent,
                fontsize=8,
                va="bottom",
            )

        figure.text(
            0.01,
            0.012,
            "Observed samples · no sidebands are asserted · source and run details in the manifest",
            color="#AAB7C9",
            fontsize=8,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=100, facecolor=background)
    finally:
        plt.close(figure)


def _serialized_hits(hits: Sequence[DopplerHit], *, limit: int = 100) -> dict[str, object]:
    ordered = sorted(hits, key=lambda hit: (-hit.snr, hit.top_hit))
    return {
        "count": len(ordered),
        "truncated": len(ordered) > limit,
        "hits": [hit.as_dict() for hit in ordered[:limit]],
    }


def _detector_artifacts(result: DetectorResult) -> list[dict[str, object]]:
    return [
        {
            "filename": artifact.filename,
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
        }
        for artifact in result.artifacts
    ]


def _manifest_outputs(paths: Sequence[Path], *, repository_root: Path) -> tuple[ArtifactRef, ...]:
    artifacts = []
    for path in paths:
        if path.is_file():
            artifact = output_ref(path, repository_root=repository_root)
            if path.suffix.lower() == ".png":
                artifact = artifact.model_copy(update={"media_type": "image/png"})
            artifacts.append(artifact)
    return tuple(artifacts)


def _record_source_header(parameters: dict[str, object], header: Hdf5FilterbankHeader) -> None:
    parameters["source_header"] = header.as_dict()


def run_voyager(
    repository_root: Path,
    command: Sequence[str],
    *,
    source_url: str = DEFAULT_SOURCE_URL,
    frequency_low_mhz: float = DEFAULT_FREQUENCY_LOW_MHZ,
    frequency_high_mhz: float = DEFAULT_FREQUENCY_HIGH_MHZ,
    max_drift_hz_s: float = DEFAULT_MAX_DRIFT_HZ_S,
    minimum_snr: float = DEFAULT_MINIMUM_SNR,
    experiment_id: str = "EXP-001",
) -> tuple[RunManifest, Path]:
    """Run EXP-001 from bounded remote HDF5 channel reads to a manifest-backed plot."""
    root = repository_root.resolve()
    started_at = datetime.now(UTC)
    run_id = new_run_id(started_at, "exp-001-voyager")
    run_directory = root / "runs" / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    manifest_path = run_directory / "manifest.json"
    slice_summary_path = run_directory / "slice_summary.json"
    hits_path = run_directory / "hits.json"
    plot_path = run_directory / "hero.png"
    failure_path = run_directory / "failure.json"

    inputs = (InputRef(uri=source_url),)
    transfer_metrics: RangeMetrics | None = None
    parameters: dict[str, object] = {
        "source_url": source_url,
        "canonical_source_url": CANONICAL_SOURCE_URL,
        "source_expected_size_bytes": (
            SOURCE_SIZE_BYTES if source_url == DEFAULT_SOURCE_URL else None
        ),
        "mirror_provenance": (
            {
                "repository": SOURCE_MIRROR_REPOSITORY,
                "commit": SOURCE_MIRROR_COMMIT,
                "git_blob_sha1": SOURCE_MIRROR_BLOB_SHA1,
                "note": (
                    "Third-party HTTPS mirror pinned to a commit; cross-validated against "
                    "official blimpy Voyager test values."
                ),
            }
            if source_url == DEFAULT_SOURCE_URL
            else None
        ),
        "requested_window_mhz": [frequency_low_mhz, frequency_high_mhz],
        "known_answer_window_mhz": [KNOWN_ANSWER_LOW_MHZ, KNOWN_ANSWER_HIGH_MHZ],
        "known_answer_frequency_uncertainty_mhz": None,
        "max_drift_hz_s": max_drift_hz_s,
        "minimum_drift_hz_s": MINIMUM_DRIFT_HZ_S,
        "minimum_snr": minimum_snr,
        "detector_options": {
            "input_format": "HDF5 FILTERBANK",
            "n_coarse_chan": 1,
            "n_partitions": 1,
            "progress_bar": "n",
            "blank_dc": True,
            "precision": "single",
        },
        "remote_block_size_bytes": REMOTE_BLOCK_SIZE_BYTES,
        "remote_transfer_limit_bytes": REMOTE_TRANSFER_LIMIT_BYTES,
    }

    try:
        parameters["runtime"] = _runtime_versions()
        if max_drift_hz_s <= 0 or minimum_snr < 0:
            raise ValueError("drift limit must be positive and minimum SNR non-negative")
        if frequency_low_mhz >= frequency_high_mhz:
            raise ValueError("frequency window must have positive width")
        with open_counted_http_file(
            source_url,
            block_size=REMOTE_BLOCK_SIZE_BYTES,
            max_transfer_bytes=REMOTE_TRANSFER_LIMIT_BYTES,
        ) as (remote_file, transfer_metrics):
            inputs = (InputRef(uri=source_url, size_bytes=transfer_metrics.source_size_bytes),)
            parameters["source_size_bytes"] = transfer_metrics.source_size_bytes
            if (
                source_url == DEFAULT_SOURCE_URL
                and transfer_metrics.source_size_bytes != SOURCE_SIZE_BYTES
            ):
                raise ValueError(
                    "pinned Voyager mirror size changed: "
                    f"expected {SOURCE_SIZE_BYTES}, observed {transfer_metrics.source_size_bytes}"
                )

            with tempfile.TemporaryDirectory(
                prefix="xenia-exp001-", ignore_cleanup_errors=True
            ) as temporary_directory:
                temporary_root = Path(temporary_directory)
                slice_path = temporary_root / "voyager-window.h5"
                slice_info = slice_hdf5_filterbank(
                    remote_file,
                    output_path=slice_path,
                    frequency_low_mhz=frequency_low_mhz,
                    frequency_high_mhz=frequency_high_mhz,
                    expected_source_name=SOURCE_NAME,
                    on_header=lambda header: _record_source_header(parameters, header),
                )
                parameters["known_answer_frequency_uncertainty_mhz"] = (
                    abs(slice_info.header.foff_mhz) / 2
                )
                slice_summary = {
                    "source_url": source_url,
                    "canonical_source_url": CANONICAL_SOURCE_URL,
                    "source_name": slice_info.header.source_name,
                    "source_header": slice_info.header.as_dict(),
                    "requested_window_mhz": [frequency_low_mhz, frequency_high_mhz],
                    **slice_info.as_dict(),
                    "remote_transfer": transfer_metrics.as_dict(),
                }
                _write_json(slice_summary_path, slice_summary)
                parameters.update(
                    {
                        "remote_transfer": transfer_metrics.as_dict(),
                        "slice": slice_summary,
                    }
                )
                detector_result = run_turboseti(
                    slice_path,
                    output_directory=temporary_root / "detector",
                    max_drift_hz_s=max_drift_hz_s,
                    minimum_snr=minimum_snr,
                    n_coarse_channels=1,
                )
                hit = select_known_answer_hit(
                    detector_result.hits,
                    frequency_low_mhz=KNOWN_ANSWER_LOW_MHZ,
                    frequency_high_mhz=KNOWN_ANSWER_HIGH_MHZ,
                    max_abs_drift_hz_s=max_drift_hz_s,
                    minimum_snr=minimum_snr,
                )
                _render_waterfall_png(slice_path, plot_path, slice_info, hit=hit)
                detector_summary = {
                    "turbo_seti_version": detector_result.turbo_seti_version,
                    "blimpy_version": detector_result.blimpy_version,
                    "max_drift_hz_s": max_drift_hz_s,
                    "minimum_drift_hz_s": MINIMUM_DRIFT_HZ_S,
                    "minimum_snr": minimum_snr,
                    "n_coarse_chan": 1,
                    "known_answer_window_mhz": [
                        KNOWN_ANSWER_LOW_MHZ,
                        KNOWN_ANSWER_HIGH_MHZ,
                    ],
                    "frequency_channel_width_mhz": abs(slice_info.header.foff_mhz),
                    "dat_outputs": _detector_artifacts(detector_result),
                    "warnings": list(detector_result.warnings),
                    "selected_hit_top_hit": None if hit is None else hit.top_hit,
                }

        _write_json(
            hits_path,
            {
                **_serialized_hits(detector_result.hits),
                "selected_known_answer_hit": None if hit is None else hit.as_dict(),
            },
        )
        parameters.update(
            {
                "remote_transfer": transfer_metrics.as_dict(),
                "slice": slice_summary,
                "detector": detector_summary,
                "hit_count": len(detector_result.hits),
            }
        )
        outputs = _manifest_outputs(
            (slice_summary_path, hits_path, plot_path), repository_root=root
        )
        hero_payload = None
        if hit is not None:
            hero_payload = HeroPayload(
                kind="hero",
                experiment="EXP-001",
                run_id=run_id,
                image_url=plot_path.relative_to(root).as_posix(),
                freq_mhz=hit.uncorrected_frequency_mhz,
                drift_hz_s=hit.drift_hz_s,
                snr=hit.snr,
            )
        finished_at = datetime.now(UTC)
        outcome = "known_answer_detected" if hit is not None else "no_expected_hit"
        notes: tuple[str, ...]
        if hit is not None:
            notes = (
                "A turboSETI hit passed the predeclared Voyager frequency, drift, and SNR checks.",
                "This is a known-transmitter positive control, not an unknown candidate.",
            )
        else:
            notes = (
                "Null known-answer result: no hit passed the predeclared thresholds. "
                "This is not a scientific non-detection conclusion.",
            )
        notes = (*notes, *detector_result.warnings)
        manifest, written_path = create_run_manifest(
            run_id=run_id,
            command=command,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            started_at=started_at,
            finished_at=finished_at,
            outcome=outcome,
            notes=notes,
            repository_root=root,
            demo=False,
            pillar="I · Sensor",
            experiment_id=experiment_id,
            mission_control=hero_payload,
            manifest_path=manifest_path,
        )
        return manifest, written_path
    except Exception as exc:
        if transfer_metrics is not None:
            parameters["remote_transfer"] = transfer_metrics.as_dict()
        _write_json(
            failure_path,
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source_url": source_url,
                "source_header": parameters.get("source_header"),
                "runtime": parameters.get("runtime"),
                "remote_transfer": None if transfer_metrics is None else transfer_metrics.as_dict(),
            },
        )
        finished_at = datetime.now(UTC)
        failure_outputs = _manifest_outputs(
            (slice_summary_path, hits_path, plot_path, failure_path), repository_root=root
        )
        create_run_manifest(
            run_id=run_id,
            command=command,
            parameters=parameters,
            inputs=inputs,
            outputs=failure_outputs,
            started_at=started_at,
            finished_at=finished_at,
            outcome="failed",
            notes=(f"Operational failure: {type(exc).__name__}: {exc}",),
            repository_root=root,
            demo=False,
            pillar="I · Sensor",
            experiment_id=experiment_id,
            mission_control=None,
            manifest_path=manifest_path,
        )
        raise

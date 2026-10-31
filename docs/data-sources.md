# Data sources and verification notes

A cited source or upstream example is not a Theoxenia result. Experiment claims come only from committed run manifests and their recorded code version.

## EXP-001: Voyager 1 known-answer sample

| Item | Source/evidence |
|---|---|
| Primary Breakthrough Listen page | [Open-data page](https://seti.berkeley.edu/listen/data.html) |
| Original tutorial | [Official UCBerkeleySETI Voyager notebook](https://github.com/ucberkeleyseti/breakthrough/blob/master/GBT/voyager/voyager.ipynb) |
| Original tutorial file | [`voyager_f1032192_t300_v2.fil`](https://storage.googleapis.com/gbt_fil/voyager_f1032192_t300_v2.fil), a 528,482,692-byte GBT X-band observation from 2015-12-30. The official notebook reports only **two integrations**; turboSETI 2.3.2 requires at least four, so this file is not used as the M1 detector input. No integrations are padded or fabricated. |
| Canonical 16-integration test file | [`Voyager1.single_coarse.fine_res.h5`](http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.h5), listed by the [official blimpy test downloader](https://github.com/UCBerkeleySETI/blimpy/blob/master/tests/download_data.sh). Its shape is `(16, 1, 1_048_576)`, dtype float32, chunks `(1, 1, 16_384)`, source `Voyager1`, `fch1=8421.386717353016 MHz`, `foff=-2.7939677238464355e-06 MHz`, `tsamp=18.253611008 s`, and `tstart=57650.78209490741 MJD`. |
| Transport actually used | [Pinned HTTPS mirror](https://raw.githubusercontent.com/RaghavGirgaonkar/SETI-Project-Test/9b6ef4ac4fb5f7a9efbcbb12ba202916e9d409eb/Voyager1.single_coarse.fine_res.h5), third-party mirror, commit `9b6ef4ac4fb5f7a9efbcbb12ba202916e9d409eb`, Git blob SHA-1 `42a41e5c2564afc8e73dc7b23cece3e3123f15d6`, length 50,549,227 bytes. The canonical Berkeley URL is HTTP; the project does not bypass TLS validation. The pinned HTTPS mirror is separately cross-checked below and identified as a mirror—not the publisher. |
| Independent upstream comparison | [Official blimpy Voyager regression test](https://github.com/UCBerkeleySETI/blimpy/blob/master/tests/test_voyager_data_load.py) compares `.h5` and `.fil` cutouts over 8419.24–8419.35 MHz, expects shape `(16, 1, 39_370)`, and asserts matching peak-channel indices (18,959 and 18,996). A bounded range-only probe of the pinned mirror read 715 channels around 8419.296–8419.298 MHz (3,003,853 payload bytes); its absolute peak channels map to those published local indices. This task has not independently retrieved the `.fil` payload and rerun the full HDF5-versus-SIGPROC array comparison. The upstream test prints peak amplitudes but does not assert those absolute amplitudes as literals. |
| Detector reference | The [official turboSETI README](https://github.com/UCBerkeleySETI/turbo_seti/blob/master/README.md) publishes an example on this separate 2016 HDF5 observation: 8419.297028 MHz, drift −0.373093 Hz/s, SNR 245.707984; it also lists 8419.319368 MHz, −0.392226 Hz/s, SNR 30.612128. These are upstream references, not Theoxenia measurements. |

The reproducible command is `make voyager` (`xenia live hero --voyager`). It reads whole HDF5 channels in the 8419.24–8419.35 MHz band through HTTPS byte ranges, with 64 KiB blocks and a 24 MiB transfer ceiling. The detector is evaluated only for the predeclared known-answer window 8419.296–8419.298 MHz, max absolute drift 4 Hz/s, and minimum SNR 25. Channel-bin uncertainty is recorded as half the measured channel width. The temporary sliced HDF5 file and turboSETI scratch output live under a temporary directory and are removed after the run; no raw sample or slice is committed.

The successful run `20261012T130904642405Z-exp-001-voyager` (code SHA `706761a`) returned three hit rows and passed the predeclared frequency, drift, and SNR checks. The selected known-answer hit was 8419.297028 MHz, −0.377557 Hz/s, SNR 1244.783936. It transferred 4,627,459 bytes by recorded HTTP ranges from the 50,549,227-byte source and produced a 3,157,056-byte temporary slice; only compact summaries, manifest, and 137,860-byte waterfall PNG are retained. The frequency agrees with the official turboSETI example, while measured drift and SNR differ; this remains a known-transmitter positive control, not an unknown-candidate result. The run manifest records the exact source, parameters, code revision, transfer ranges, versions, outputs, and warning.

An independent reproduction run 20261021T215534947750Z-exp-001-voyager reproduced EXP-001 with outcome known_answer_detected, returning the identical three candidate hits and matching the selected known-answer hit at 8419.297028 MHz, -0.377557 Hz/s, and SNR 1244.783936.

Four failed early attempts are retained as first-class evidence: `20261012T123034402477Z-exp-001-voyager` records blimpy’s two-integration validation failure; `20261012T123734993626Z-exp-001-voyager` records turboSETI’s minimum-four-integration rejection; `20261012T130200938902Z-exp-001-voyager` records a turboSETI FFT-length mismatch because its default coarse-channel heuristic treated the pre-sliced 110 kHz band as 64 coarse channels; and `20261012T130616461127Z-exp-001-voyager` records turboSETI’s final candidate-count logging TypeError after writing parseable hit rows. These are operational failures, not scientific null results. The current implementation uses the validated 16-integration source and explicitly tells turboSETI the sliced input contains one coarse channel; it does not patch blimpy or turboSETI.

## Data-handling rules

- Use HTTPS and byte ranges; do not disable TLS verification or fall back to full-file downloads.
- Pin mutable mirrors to immutable commits and record the canonical publisher endpoint separately.
- Record actual transfer ranges, payload bytes, source size, source metadata, detector/runtime versions, output checksums, command parameters, code SHA, timestamps, and outcome in the run manifest.
- Keep raw telescope data and all temporary frequency slices outside Git and persistent project directories.
- Record operational failures and null known-answer outcomes; neither is silently converted into a positive finding or a scientific non-detection claim.

## Additional references

- [Official blimpy test downloader](https://github.com/UCBerkeleySETI/blimpy/blob/master/tests/download_data.sh)
- [turboSETI 2.3.2 on PyPI](https://pypi.org/project/turbo-seti/2.3.2/)
- [SIGPROC filterbank format guide](http://sigproc.sourceforge.net/sigproc.pdf)
- [Official Python 3.15.0 release](https://www.python.org/downloads/release/python-3150/)

Mutable upstream pages and package metadata were checked on 2026-10-12. Recheck them before future baseline or source changes.

## Other pillar source notes

The exact six-scan public GBT cadence metadata and bounded HDF5 metadata-probe evidence for the later worker slice are tracked in the EXP-030 research notes. Do not infer a universal storage provider from the archive: the Breakthrough Listen archive paper says records may resolve to Berkeley servers, national computing centers, or cloud storage. Each worker run must retain the exact archive query, selected record URLs, and per-object checksums.

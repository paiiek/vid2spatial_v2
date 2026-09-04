# Tracked issues

Known defects and limitations, kept here rather than in the README quick-start
so a caveat cannot be mistaken for setup instructions. Each entry says what is
wrong, what has been done, and what is still open.

Last updated: 2026-09-04.

---

## I1 — Live OSC distance was 2x wrong (FIXED in this repo)

**Was.** `OSCSpatialSender.send_frame` emitted `/vid2spatial/spatial` LAST,
carrying distance in metres. The engine bridge normalises that bundle with its
OWN constant (20 m on the installed bridge, 10 m on the repinned lane bridge),
so the bundle overrode the correct `/vid2spatial/distance` value and every
live-streamed object was placed at half its intended distance. The offline
export path was correct throughout, so offline and live disagreed.

**Now.** `send_frame` no longer emits the bundle. It is available behind
`OSCConfig.legacy_spatial` / `--legacy-spatial` for old bridges.
Measured through the real bridge handlers over real UDP, a 10 m source:

| path | forwarded ADM dist |
|---|---|
| default | 1.0 (correct) |
| `--legacy-spatial` | 0.5 |

**Still open (engine repo).** The installed bridge at
`/home/seung/mmhoa/spatial_engine` still normalises `/vid2spatial/spatial` with
20 m. Unifying it on 10 m is an engine-side item. This repo no longer depends
on that happening.

Re-measured 2026-09-04 through the real bridge and a running
`spatial_engine_core`, streaming a 10 m source: the default path forwards
`/adm/obj/1/aed [-30.0, 10.0, 1.0]` and `--legacy-spatial --distance-max-m 20`
forwards `[-30.0, 10.0, 0.5]`, i.e. the table above still holds exactly.

**The exact engine-side change** (this repo cannot make it; the bridge is owned
by `spatial_engine`). In `bridge/vid2spatial_osc.py`, `_handle_spatial`:

```python
-            dist_norm = max(0.0, min(1.0, 1.0 - dist_m / 20.0))
+            dist_norm = max(0.0, min(1.0, 1.0 - dist_m / 10.0))
```

That single constant is the whole divergence. Until it lands, `--legacy-spatial`
requires `--distance-max-m 20.0` to agree with the bridge, and the preflight
now enforces that agreement against the value actually passed (see I13).

---

## I2 — Occlusion estimation is not implemented

`OcclusionConfig.estimate=True` used to import a `vid2spatial_pkg/occlusion.py`
that has never existed. The failure was swallowed by a broad `except`, so the
run printed one warning and rendered fully un-occluded audio. It now raises at
config construction. The `occlusion.json_path` route (an externally produced
timeline) works and is the supported path.

---

## I3 — No live multi-object path

The offline pipeline renders N sources (`MultiSourceConfig`) and exports one
automation file per ADM object. There is no live multi-object OSC path: the
bridge keys every datagram on the single tracking id `default`, and the ADR's
`/vid2spatial/obj/{N}/azim` family is not implemented engine-side. Streaming N
sources would collapse them onto object 1.

---

## I4 — The headline azimuth metric is circular

`test/run_quant_eval.py` derives ground-truth azimuth from the GT bbox centre
using the same pinhole formula and the same assumed 60 degree FOV as the
prediction. AzMAE is therefore a monotone reparameterisation of bbox-centre
pixel error and cannot detect an FOV error, a principal-point error, or lens
distortion. Handled in a separate work lane; recorded here so the number is not
read as a spatial accuracy result.

---

## I5 — Camera FOV is hardcoded to 60 degrees

`config.py` fixes `fov_deg = 60.0` and never reads it from the file. A 40
degree lens rendered as 60 degrees inflates every azimuth by roughly 1.5x, a
systematic error larger than the whole reported AzMAE. Separate work lane.

---

## I6 — No camera-motion compensation

Azimuth comes from the object's image position, so a camera pan sweeps a
stationary object across the sound field. Separate work lane.

---

## I7 — `bbox_area_log` shipped without an ear check

The default distance mapping changed on user-facing paths at `09289ea` on
geometric evidence alone. The objective half of the check is now done:
`reports/gain_mode_loudness_2026-09-04.md` compares loudness, loudness range
and spectral-centroid range across the 22-clip set.

**Still open, human.** A 2-condition A/B on ~5 clips with 3 listeners. No
objective metric settles whether the new mapping sounds right.

---

## I8 — Depth accuracy is reported for the proxy, not the deployed chain

`test/full_eval/DEPTH_GT_KITTI_RESULTS.md` isolates the bbox-area proxy term
with an oracle `z0`. The composed number, with `z0` estimated by Depth Anything
V2 ViT-S over the same 225 tracks:

| Calibration `z0` | AbsRel | δ1 |
|---|---|---|
| Ground truth (proxy only) | 0.100 | 0.870 |
| Depth model estimate (composed) | 0.247 | 0.484 |

Per-track Spearman is exactly unchanged (0.999 median); pooled Spearman falls
from 0.984 to 0.896 because each track picks up a different scale error.

**Still open.** The checkpoint on this machine is the RELATIVE ViT-S model, so
metres come from a single global affine fitted once against the ground truth,
not from the model. The composed figure is therefore optimistic relative to a
deployment with no ground truth at all. A metric backend (Metric3D v2, or a
Depth-Anything metric checkpoint) would remove the fit. Full detail and the
reproduce steps: `reports/depth_composed_kitti_2026-09-04.md`.

---

## I9 — Unbenchmarked tracker backends

`vid2spatial_pkg/experimental/` holds four tracker backends that are reachable
from `vision.py` but appear in no evaluation table, and cannot go through the
QUANT_EVAL harness. See that package's docstring for why each one is there.

---

## I10 — `pipeline.run()` cannot execute in this environment (librosa vs NumPy 2.4)

`pipeline.py` loads audio with `librosa.load`, which pulls in numba:

```
ImportError: Numba needs NumPy 2.3 or less. Got NumPy 2.4.
```

The installed NumPy is 2.4.4. Every entry point that loads audio through the
pipeline therefore raises before doing any work — the single-source `run()` and
the multi-source path alike.

**Pre-existing**, not introduced by any recent change: the same `librosa.load`
call is at `pipeline.py:728` in `09289ea`. The test suite does not catch it
because the tests exercise the render helpers directly rather than going
through `run()`.

**Fix (CLOSED 2026-09-04)**: `vid2spatial_pkg/audio_io.py` `load_audio` keeps
`librosa.load` as the primary path, unchanged where librosa works, and falls
back to `soundfile.read` plus `scipy.signal.resample_poly` when librosa cannot
be imported *or* raises `ImportError` at call time (numba defers its NumPy check
to first use, so the failure surfaces there). The fallback matches the defaults
this pipeline used: mono mixdown, float32, 1-D, native rate when `sr is None`.
It warns once. It is not bit-identical on the resampling path -- librosa uses
soxr, the fallback a polyphase FIR -- and at `sr=None` no resampling happens at
all. `pipeline.py` now calls `load_audio` at both sites.

Verified by a real end-to-end `run()` on this box, first one ever executed here,
a 60-frame LaSOT clip (`airplane-1`, KCF from the GT first box, `depth=none`)
with a 2 s 220 Hz tone:

```bash
python3 tools/run_e2e_smoke.py --out DIR \
    --lasot /home/seung/mmhoa/vid2spatial_v2/data/lasot/airplane-1 --frames 60
```

That script builds the mp4 and the tone from the clip, tracks with KCF from the
clip's own first ground-truth box, runs with no depth backend, and calls
``SpatialAudioPipeline.run()``.

Outputs: `out.traj.json` 60 frames (38971 B), `out.foa.wav` (96000, 4) @ 48 kHz
(1536104 B), `out.stereo.wav` (384044 B). The fallback warning fires at
`pipeline.py:859`, so the fallback is what carried the run.

## I11 — What `av_confidence` cannot tell you

The A4 gate correlates the audio envelope against the tracked object's motion
energy. Three limits are worth stating, because a 0.0 score is common and
almost never means "wrong pairing":

1. **Any constant series makes the score undefined**, and it is then reported
   as 0.0 with a warning. That covers a steady sound (a running engine), a
   static object, and — most commonly — an object at **constant velocity**,
   whose motion energy is flat. A car crossing the frame at a steady speed is
   the ordinary case. Low confidence means "unverified", never "wrong".

2. **The score is not a p-value.** The chance level subtracted is the 95th
   percentile of the lag-scan maximum over 200 phase-randomised surrogates,
   which is a calibration, not a guarantee about a single clip. Measured
   false-alarm rate against the WARN gate on unrelated pairs: 2.5 percent at
   30 frames, 1.0 percent at 60. Until 2026-09-04 the null was the single-`r`
   form `2/sqrt(n)`, which ignored the maximisation over roughly 31 lags and
   let about 18 percent of unrelated pairings through as verified.

3. **The envelope assumes the audio and the video share a start time.** Windows
   are `sr/fps` samples, so window *i* is frame *i*; audio of a different
   duration is padded or truncated and warns. Offset audio is not detected
   beyond the +/-0.5 s lag scan.

The gate reaches JSON exports in full. The CSV export carries only the scalar
`av_confidence` column, not the warning text; see the schema note in
`vid2spatial_pkg/trajectory_export.py`.

---

## I12 — The exported FOA is ACN/N3D, not the AmbiX (ACN/SN3D) it is labelled

`foa_render.dir_to_foa_acn_sn3d_gains` is named for SN3D and its docstring says
AmbiX (ACN/SN3D), but it computes

    W = 1/sqrt(2),  X = sqrt(3/2)*x,  Y = sqrt(3/2)*y,  Z = sqrt(3/2)*z

which is ACN/**N3D** scaled by `1/sqrt(2)`, not SN3D (SN3D is `W = 1`,
`X = x`). Measured on a demo render of a source near the front
(az 6.7 deg, el 6.8 deg): channel RMS `X/W = 1.72`, and `sqrt(3) = 1.732`.
Channel ORDER is correct ACN (`W, Y, Z, X`).

Consequence: a decoder that takes the file at its word and applies the SN3D
convention renders the source about 4.8 dB over-directional, and the bed sits
3 dB low overall. The repo's own decoders (`foa_to_stereo`, `foa_to_binaural`)
multiply the first order by `sqrt(3)` on the way in, i.e. they also assume SN3D
input, so the internal round trip applies the factor twice.

**Not fixed here.** Correcting the encoder changes every rendered sample and
would break the byte-stability golden (`test/test_geometry_render.py`), which
is the point of that gate. The label is corrected in the web UI and this entry
records the measurement; the normalisation change needs its own lane, with the
golden regenerated deliberately.

---

## I13 — Preflight round-trip listened on a hardcoded 9100 (FIXED in this repo)

**Was.** `tools/attach_engine.py::_check_roundtrip` bound port 9100 no matter
what the bridge's `--target-port` was. That failed in both directions:

- against a bridge forwarding anywhere else, the probe heard nothing and the
  preflight refused to attach with "the bridge is not forwarding. Is it
  running?" — blaming a bridge that was working (reproduced live against a
  bridge on `--target-port 22060`);
- in the documented setup the engine itself holds 9100, so the bind failed and
  the check returned "SKIPPED", meaning it never actually proved the path in
  the one configuration the README tells you to build.

A second defect sat next to it: `_check_constants` read `OSCConfig()` instead
of `--distance-max-m`, so `--legacy-spatial --distance-max-m 20.0` still failed
claiming `sender=10.0 m`. The remedy the error text offered could not be
carried out.

**Now.** `--forward-port` (default 9100) selects the port to listen on, the
failure text names it and says it must equal the bridge's `--target-port`, and
the constants check is given the distance law the stream will actually use.
Verified live: round-trip OK against a bridge on 22060, and
`--legacy-spatial --distance-max-m 20.0` now passes and streams.

---

## I14 — No depth backend produced a NEGATIVE distance (FIXED in this repo)

With `depth.backend="none"` (or any failed load) `initialize_depth_backend`
returns `is_metric=False`, but two fallbacks answered in metres anyway:
`V2SpatialTracker._estimate_depth` set `f.depth_m = 2.0` for every frame, and
`vision.estimate_depth_at_bbox` returned `(2.0, True)`. `compute_3d_position`
then read 2.0 as a relative depth:

    dist_m = 0.5 + (1 - 2.0) * (10.0 - 0.5) = -9.0

Every frame of every source came out at **-9.0 m**. `dist_norm` clamped to 1.0
and `dist_adm` to 0.0, so the whole scene collapsed onto the listener with no
warning anywhere; the automation CSVs shipped `dist_m,-9.000000`.

Both fallbacks now answer in the caller's units (`0.5` relative / `2.0` metric).
The same clip now exports `dist_m,5.250000`.

---

## I15 — Multi-source silently rendered one object N times (FIXED in this repo)

Three defects stacked:

1. `pipeline._compute_trajectory` mapped any unrecognised
   `vision.tracking.method` to `adaptive_k`. `"yw_sam2"` is not in the
   recognised list, so asking for the box-driven tracker silently got a
   detection-driven one that **ignores `init_bbox`**.
2. Every source therefore tracked the same detected object. Two boxes 314.8 px
   apart in a 640x360 frame produced byte-identical automation files differing
   only in `object_id`, and one FOA with two copies of one source summed into
   one direction. Nothing in the output distinguished this from success.
3. `_track_v1_bytetrack`'s centre-distance lock had no bound at all
   (`best_dist = inf`), so with a single detection in frame every `init_bbox`
   locked onto it however far away, printing the distance but never acting on
   it.

Now: the method map is explicit and an unknown name raises; `yw_sam2`/`sam2`
reach the real tracker; the centre-distance lock is gated at the box diagonal
or 10 percent of the frame diagonal, whichever is larger, and raises
`NoTrackForInitBBoxError` past it; and `run_multi_source` refuses when two
sources come out with identical angles. `pipeline._compute_trajectory`'s broad
`except Exception` used to swallow that error and reroute to the legacy path,
whose only symptom was `ValueError: Unknown tracking method: v1_bytetrack` —
naming a method the real tracker supports. It now re-raises.

Verified: the same two boxes now give object 1 at az +6.68 deg and object 2 at
az -21.01 deg, two distinct automation files, one FOA.

---

## I16 — The demo's rendered level is very low

With the distance fallback corrected (I14) a source sits at 5.25 m, and the
default distance-gain law puts the demo's binaural output at peak 0.00065
(about -64 dBFS) for a stem that peaks near 0.02 in the file. It is correct per
the gain law and effectively inaudible without normalising. The web demo offers
no output-gain control and no normalisation, so "listen with headphones" gives
silence on a normal system volume. Changing the law would move every rendered
sample and is blocked by the byte-stability golden, so this is recorded, not
fixed.

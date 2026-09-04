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

## I12 — The FOA default is ACN/N3D at -3 dB, not the AmbiX it was labelled

**Measured.** Synthetic single source, `gain_mode="none"`, steady-state RMS
ratios out of `render_foa_from_trajectory` (the numbers a user's file has):

| direction | X/W | Y/W | Z/W | sqrt(X²+Y²+Z²)/W |
|---|---|---|---|---|
| az 0, el 0   | 1.7321 | 0.0000 | 0.0000 | 1.7321 |
| az 90, el 0  | 0.1173 | 1.7281 | 0.0000 | 1.7321 |
| az 45, el 0  | 1.2305 | 1.2190 | 0.0000 | 1.7321 |
| el 90        | 0.1173 | 0.0000 | 1.7281 | 1.7321 |

`sqrt(3) = 1.7321`, `sqrt(3/2) = 1.2247`. The az 90 and el 90 rows miss their
axis only because the 80 ms angle smoother settles at 86.1 deg; the
direction-independent invariant `sqrt(X²+Y²+Z²)/W` is `sqrt(3)` exactly, which
is what pins the normalisation.

**The formula.** Channel order is ACN `[W, Y, Z, X]` (correct). With
`x = cos(az)cos(el)`, `y = sin(az)cos(el)`, `z = sin(el)`:

```
legacy (default)   W = 1/sqrt(2)   X = sqrt(3/2)*x   Y = sqrt(3/2)*y   Z = sqrt(3/2)*z
sn3d  (opt-in)     W = 1           X = x             Y = y             Z = z
```

So `legacy` is ACN/**N3D** scaled by `1/sqrt(2)` — not SN3D, which is what the
function name `dir_to_foa_acn_sn3d_gains`, its docstring, the package
docstring, the README diagram, the web demo and the UI checkbox all claimed.
A decoder that believes the label renders the source **4.77 dB
over-directional** and 3 dB low overall. The repo's own decoders
(`foa_to_stereo`, `foa_to_binaural`) multiply the first order by `sqrt(3)` on
input, i.e. they also assume SN3D, so the internal round trip applies the
factor twice.

**Now.** Every label states what is produced, and `foa_norm` selects it:
`SpatialConfig.foa_norm`, `--foa-norm {legacy,sn3d}` (`config.add_render_cli_args`),
`encode_mono_to_foa(..., norm=)`, `render_foa_from_trajectory(..., foa_norm=)`,
and a picker in the web demo. The trajectory JSON records what was used under
`render.foa_norm` / `render.foa_norm_detail`, so a WAV is never separated from
its convention again. `sn3d` measures `X/W = 1.0000` on axis and an invariant
of `1.0000` in every direction.

**Why the default cannot flip yet.** `legacy` is not a bug that can simply be
corrected; it is the convention every existing artefact was produced under:

- the byte-stability golden (`test/test_geometry_render.py`,
  `test/make_render_golden.py`) digests the encoder output, and flipping the
  default changes every sample;
- the listening-test stimuli (`test/listening_test_v3/`, `render_listening_test_v3.py`)
  and the MOS results reported in `docs/ismar_final/` were rendered with it,
  so a flip silently invalidates the comparison between old and new stimuli;
- the published azimuth/elevation numbers were computed through decoders that
  assume SN3D input, and the double-`sqrt(3)` cancels part of the error.

Flipping would require, in one deliberate lane: regenerate the render golden,
re-render and re-run the listening test (or state that the stimuli are on the
old convention), re-derive any number that passed through `foa_to_stereo` or
`foa_to_binaural`, and drop the compensating `sqrt(3)` in those decoders so the
round trip stops applying it twice. Until then the honest move is the one taken
here: keep the bytes, fix the words, and offer the correct encoder as an option.

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

## I16 — A correct render can be inaudible (-63.77 dBFS), and nothing adds make-up gain

**Measured** on the demo's own 2 s LaSOT clip against a spatamb stem:

| stage | peak | peak dBFS |
|---|---|---|
| source stem, whole file | 0.153058 | -16.30 |
| the 2 s the video covers | 0.006317 | -43.99 |
| FOA after the distance law | 0.000787 | -62.08 |
| binaural | 0.000648 | -63.77 |

Integrated loudness of the finished binaural is **-inf LUFS**: below the
BS.1770 absolute gate, i.e. formally silent to a loudness meter.

**Two causes, neither a missing fader.**

1. *Content in the trimmed window.* The stem opens near-silent, so the two
   seconds the video covers sit 27.7 dB below the file's own peak. This became
   visible only once the render was trimmed to the video (I15's sibling fix);
   before that the render ran over the whole 62 s stem and picked up the loud
   passage, hiding the problem.
2. *The distance law.* `apply_distance_gain_lpf` maps `d_rel` onto a virtual
   inverse-square range `r = 1..8 m` and raises it to `gain_k`. At the clip's
   `d_rel = 0.79` (5.25 m) that is **-19.72 dB measured on unity**, applied on
   top of source audio nobody normalised. A further -18.1 dB lands between the
   trimmed source and the FOA file.

Every stage is doing exactly what it documents. There is simply no make-up gain
anywhere in the chain, and the reference level is implicit rather than stated.

**Now.** Opt-in output normalisation: `SpatialConfig.peak_dbfs`,
`--peak-dbfs` (`config.add_render_cli_args`), and `peak_dbfs=` on both
`render_foa_from_trajectory` and `render_binaural_from_trajectory`. Off by
default, because make-up gain by default would move every published number and
break the byte-stability golden. The web demo sets `-1.0` (`DEMO_PEAK_DBFS`)
because it exists to be listened to; stimulus generation must leave it off.

A/B on the same clip:

| setting | file | peak dBFS | LUFS-I |
|---|---|---|---|
| default (`peak_dbfs=None`) | binaural | -63.77 | -inf |
| default | FOA | -62.08 | -inf |
| `--peak-dbfs -1` | binaural | -1.00 | -16.44 |
| `--peak-dbfs -1` | FOA | -1.00 | -16.84 |

Normalisation is applied to the FOA bed before stereo and binaural are decoded
from it, so relative levels between the outputs are unchanged.

**Still open.** The real fix is a stated reference level: a documented source
loudness target (say -23 LUFS in) and a `d_ref_m` at which the chain is unity,
so a render is predictable without a normaliser. That is a render-default
change and belongs with the I12 flip, in a lane that regenerates the golden.


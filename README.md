# Vid2Spatial

**Spatial audio authoring from video and freehand sketch.**

Draw one bounding box on any object in any video → automatic tracking + binaural spatialization, no category constraints, no depth sensors, no scene-specific tuning.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## Overview

Vid2Spatial reframes spatial audio authoring as **spatial parameter extraction**: a per-frame stream of azimuth, elevation, and relative distance is extracted automatically from monocular video or freehand 3D sketch, stored as a human-readable editable artifact, and routed to multiple output formats — binaural, First-Order Ambisonics (FOA), and real-time OSC — without re-running extraction.

```
Video / Sketch
      │
      ▼
  Single bbox init ──→ SAM2 + YOLO-World tracker
                              │
                        per-frame (az, el, d_rel)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         Binaural          FOA AmbiX       OSC UDP
       (KEMAR SOFA)      (ACN/SN3D)    (any DAW/engine)
```

### Key properties

- **Category-agnostic** — bbox init only, no semantic labels required
- **Single interaction** — one drawn box → full trajectory
- **Format-agnostic output** — same parameter stream → binaural / FOA / OSC
- **Sketch mode** — freehand 3D sphere input for off-screen or non-literal sources
- **OSC streaming** — verified with Reaper, Max/MSP, TouchDesigner, Unreal Engine, Pure Data

---

## Results

| Metric | Value |
|---|---|
| Tracking (22 LaSOT clips, AzMAE) | **1.36° ± 1.75°** (median 0.70°) |
| vs. ByteTrack baseline | −72% |
| Perceptual study N=20, p | ≤ 0.025 (Friedman) |
| Efficiency study N=12, time reduction | 41–49% |
| Edit operations reduction | 87–90% |

Caveat on the headline AzMAE: ground truth and prediction share the same
pinhole projection and the same assumed 60° FOV, so this number measures
tracking, not camera geometry. See `docs/ISSUES.md` I4.

### Learned end-to-end mono→binaural does not beat a mono floor

The closest published family to an end-to-end alternative, 2.5D Visual Sound,
was run pretrained on FAIR-Play test clips and its output binaural scored
against the recorded ground-truth binaural with the same ITD-inversion harness
used on our own output, so the inversion floor is common-mode.

| Condition (80 FAIR-Play clips) | AzMAE vs recorded GT |
|---|---|
| 2.5D Visual Sound (ILD read-out) | 19.60° |
| 2.5D Visual Sound (ITD read-out) | 18.81° |
| Mono, i.e. no spatialisation at all | 18.82° |

It beats the mono floor on only **25%** of clips, and the correlation between
its ILD and the true ILD is **0.023**. Learned mono→binaural spatialisation, in
this configuration, adds no usable azimuth.

This is a *different dataset and protocol* from the 1.36° tracking number above
and the two must not be compared directly: FAIR-Play is recorded binaural music
in a room, LaSOT is visual tracking. What the result supports is the choice of a
deterministic geometric pipeline over a learned end-to-end one, not a claim that
this system is 14x better. Source: `test/full_eval/E2E_COMPARISON.json`,
produced by `test/run_e2e_comparison.py`; our own binaural ITD-inversion floor
on the same harness is 7.27° (`BINAURAL_AZ_INVERSION.json`).

### Trajectory stabilisation is nearly free

Turning the stabiliser on (80 ms angle smoothing, d_rel attack 0.7 s / release
0.2 s) over the same 22 clips:

| Metric | Off | On | Change |
|---|---|---|---|
| Azimuth jitter | 691 | 17 | **40× lower** |
| Azimuth jerk | 4.69e7 | 6.60e3 | **7000× lower** |
| Elevation jitter | 342 | 8.1 | 42× lower |
| AzMAE | 1.3641° | 1.3669° | +0.0028° |

Four decimal places of accuracy buy two orders of magnitude of smoothness.
Source: `test/full_eval/STABILIZATION_PROXY_ABLATION.json`. The AzMAE column
inherits the circularity caveat above; the jitter and jerk columns do not,
since they are properties of the output trajectory alone.

Known defects and limitations: `docs/ISSUES.md`.

---

## Installation

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install opencv-python numpy scipy h5py

# SAM2 (required for yw_sam2 tracker)
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# YOLO-World (optional, for text-prompt init)
pip install ultralytics

# OSC streaming (optional)
pip install python-osc
```

Download KEMAR SOFA file separately and set `sofa_path` in config.

---

## Quick Start

### Video mode (single bbox)

```python
import json
from vid2spatial_pkg.v2_spatial_tracker import V2SpatialTracker

# Camera/depth options go to the constructor; the clip goes to track().
tracker = V2SpatialTracker(depth_backend="metric3d", fov_deg=60.0)
traj = tracker.track(                 # returns a plain trajectory dict
    "input.mp4",
    init_bbox=(x, y, w, h),           # draw once
    method="adaptive_k",              # or "auto" / "v1_bytetrack"
    yw_det_threshold=0.99,
)
json.dump(traj, open("output/traj.json", "w"))
```

`traj` is a dict — `{"fps": ..., "frames": [{"frame", "az", "el", "dist_m", ...}]}` —
not an object, so write it with `json.dump`. For the full video-to-audio run
(tracking plus render plus export in one call) use
`vid2spatial_pkg.pipeline.SpatialAudioPipeline(...).run()` instead.

### Render binaural

```python
from vid2spatial_pkg.foa_render import direct_binaural_sofa

# direct_binaural_sofa(mono, sr, az_s, el_s, sofa_path, block_ms=10.0)
# az_s/el_s are PER-AUDIO-SAMPLE arrays (radians), not per-frame — build them
# from the trajectory first.  Note the SOFA/AmbiX azimuth sign flip.
from vid2spatial_pkg.foa_render import interpolate_angles_distance

az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(
    traj["frames"], len(audio_mono), 48000, fps=traj["fps"])
binaural = direct_binaural_sofa(audio_mono, 48000, -az_s, el_s, "kemar.sofa")

# Or render straight to a file, which also applies the distance gain/LPF:
from vid2spatial_pkg.foa_render import render_binaural_from_trajectory
render_binaural_from_trajectory("mono.wav", traj, "out.wav", "kemar.sofa",
                                gain_mode="bbox_area_log")
```

### OSC streaming

```python
from vid2spatial_pkg.osc_sender import OSCSpatialSender

sender = OSCSpatialSender(host="127.0.0.1", port=9000)
sender.connect()
sender.stream_trajectory(traj.frames, fps=30.0)
```

### Offline automation export (no bridge needed)

```bash
python -m vid2spatial_pkg.trajectory_export traj.json traj_automation.csv --fps 30 --object-id 1
# or .json; or set OutputConfig.automation_path in the pipeline
```

Writes one row per tracked frame: `frame, t_s, object_id, az_deg, el_deg, dist_m,
dist_norm, gain_lin, az_adm_deg, el_adm_deg, dist_adm, confidence`. The `*_adm`
columns already apply the bridge contract (`az_adm = -az`, `dist_adm = 1 - dist_norm`),
so they map 1:1 onto `/adm/obj/N/aed` via the bridge's `/vid2spatial/distance`
path (10 m), which is now the only distance the sender emits (see `docs/ISSUES.md`
I1 for the legacy `/vid2spatial/spatial` bundle and the `--legacy-spatial` flag).
`automation_path` is
read from an argparse namespace as `--automation-path` if your parser defines it. Note: spatial_engine has no per-object
trajectory loader (its `TimelineJson` only carries scene-snapshot keyframes), so this
is a documented interchange format, not a native engine file.

### Attach to engine in 3 steps

vid2spatial is developed detached from `spatial_engine`; these three commands
are the whole attach procedure. The wire contract is pinned against the engine's
`fix/lane-bridge-handoff` bridge (`build_dispatcher`, `DISTANCE_MAX_M = 10.0`).

```bash
# 1. start the engine-side bridge (in the spatial_engine repo)
python3 bridge/vid2spatial_osc.py --listen-port 9000 --target-port 9100 \
        --config bridge/config.yaml

# 2. preflight: wire contract, boundary constants, bridge mode, reachability, round-trip
python3 tools/attach_engine.py --check-engine --host 127.0.0.1 --port 9000

# 3. stream a trajectory
python3 tools/attach_engine.py traj.json --host 127.0.0.1 --port 9000 \
        --object-id 1 --distance-max-m 10.0 --az-sign right-positive
```

Step 2 refuses to continue rather than streaming into a void. It fails loudly on
a drifted wire contract, on a sender/bridge distance-law mismatch (which would
silently rescale distance), on a non-1-based object id, on an unreachable
bridge, and on the `low_latency` trap below. Step 3 runs the same preflight
first unless you pass `--skip-preflight`.

Everything that could silently disagree across the boundary is an explicit flag
with an engine-matching default:

| Flag | Default | Why it matters |
|---|---|---|
| `--distance-max-m` | `10.0` | Must equal the bridge's `DISTANCE_MAX_M`. A mismatch rescales distance with no error anywhere. |
| `--object-id` | `1` | ADM object numbers are **1-based** (`/adm/obj/1/aed`). |
| `--az-sign` | `right-positive` | vid2spatial azimuth is right-positive, ADM-OSC is left-positive, so the bridge negates. |

Use `--dry-run` to print every packet without sending, and `--limit N` for a
short probe. Verified end to end against the lane bridge: 40 frames in, 40
`/adm/obj/1/aed` out, azimuth `-57.2958` becoming `+57.2958`, object id 1.

**The engine needs a speaker layout, or it renders silence.** Start the engine
with `--layout` pointing at a valid layout (for example
`spatial_engine-proto/configs/lab_8ch.yaml`). Without one it falls back to a
layout that renders digital silence for *everything*, including the
`/noise/{ch}/*` per-speaker verification signal that bypasses objects entirely,
and it says only `[warn] layout load failed: ... using fallback`. Measured on
`build_L1`: no layout gives peak `0.000000` on 0/8 channels; a valid layout
gives `0.404724` on 7/8 for per-speaker noise and `0.242737` on 4/8 for a panned
object. Verify with `make verify-engine-audio`.

Two things that look like causes of that silence but are not: the object id base
(an `/adm/obj/N/aed` message activates the object by itself, so a wrong
`/obj/active` id still makes sound once a layout is loaded) and a missing input
source (`--object-source sine` generates internal tones; no `/noise` or
`/obj/input` routing is required). Note the ids do differ, though: `/obj/active`,
`/obj/gain` and `/obj/input` take **internal 0-based** object ids, while
`/adm/obj/N` is 1-based wire, mapping N to internal N-1.

**The `low_latency` trap.** The bridge polls the global file
`/tmp/.spe_bridge_mode` and, if it holds `low_latency`, forwards **nothing**
while still accepting every packet — overriding both `config.yaml` and
`--mode`, and logging nothing after startup. A stale file left by any WebGUI or
bridge on the machine silences the attach. The preflight checks it; to clear it
by hand, `rm /tmp/.spe_bridge_mode`.

### Bridge contract (plugin boundary with spatial_engine)

`vid2spatial_pkg/bridge_contract.yaml` pins every OSC message vid2spatial emits
(address, args, types, ranges, port) and the expectations of the consuming bridge
`spatial_engine/bridge/vid2spatial_osc.py` (listen 9000 → forward 9100, `az_adm = -az`,
`dist_adm = 1 - dist_norm`, `/adm/obj/1/aed`, 1-based object numbers). The `bridge:`
section is generated from the bridge source; the rest is authored and proven by tests.

```bash
python tools/extract_bridge_contract.py --check     # drift alarm vs the bridge tree (exit 1); SKIP if absent
python tools/extract_bridge_contract.py             # regenerate the bridge: section after an agreed change
CUDA_VISIBLE_DEVICES= python -m pytest test/test_bridge_contract.py -q   # live UDP capture of every sender
```

Run both before touching `osc_sender.py`, the demo's `vid2spatial` format, or
`trajectory_export.py`; a flipped azimuth sign, inverted distance normalisation,
changed port, or renamed address fails.

### Depth heuristic verification

```bash
python tools/verify_depth_heuristic.py            # synthetic pinhole checks (CPU, <1s)
python tools/verify_depth_heuristic.py --gt gt.json   # [{area, depth_m}] → MAE / AbsRel / Spearman
```

The bbox-area depth proxy (`depth_utils.compute_bbox_scale_proxy`, z ∝ 1/√area)
and the `bbox_area` gain mode are checked for exact pinhole recovery, monotonicity,
range, and mutual rank agreement. The repo holds no metric-depth ground truth
(LaSOT is bboxes only; FairPlay/TAU are audio), so absolute accuracy is unverified
until a GT file is supplied.

### Web demo (video + sketch + OSC)

```bash
cd test/demo
python server.py --port 8090
# Open http://localhost:8090
```

---

## Coordinate Convention

- Pipeline: `az = atan2(x, z)` → **right of image = az > 0**
- SOFA/AmbiX standard: left = az > 0 (counterclockwise from front)
- FOA encoding applies `az_SOFA = −az_pipeline`
- Elevation: up = positive (image-y-down corrected in `vision.py`)

---

## Package Structure

```
vid2spatial_pkg/
  v2_spatial_tracker.py   # main tracker (SAM2 + YOLO-World)
  vision.py               # detection, MiDaS depth, ray→angles
  foa_render.py           # FOA encoding, HRTF binaural, distance effects
  pipeline.py             # end-to-end pipeline class
  osc_sender.py           # OSC UDP streaming
  bridge_contract.yaml    # OSC wire contract (emitted + bridge-side, see tools/extract_bridge_contract.py)
  config.py               # dataclass configs
  depth_utils.py          # MiDaS / DA-V2 depth backends
  multi_source.py         # multi-object spatial audio

test/
  test_unit.py            # unit suite (no models)
  test_integration.py     # integration suite
  test_bridge_contract.py # OSC plugin-boundary conformance vs bridge_contract.yaml
  run_e2e_final.py        # full eval on 22 LaSOT clips
  run_quant_eval.py       # AzMAE / ElMAE quantitative eval
  run_baseline_compare.py # stereo pan baseline comparison
  render_listening_test_v3.py  # perceptual study stimuli render
  demo/                   # web demo server (video + sketch + OSC)

demo_video/
  make_demo_v3.py         # build demo video
  capture_ui.py           # playwright UI screen capture

docs/ismar_final/         # evaluation reports and analysis
```

---

## Citation

Paper under review. BibTeX will be added upon acceptance.

---

## License

MIT

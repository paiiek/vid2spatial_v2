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
path (10 m). Caveat: the bridge's `/vid2spatial/spatial` handler (emitted last by
`send_frame`) still normalises with 20 m, so a live bridge currently forwards
`dist_m/20` until it is unified on 10 m (engine-repo item). `automation_path` is
read from an argparse namespace as `--automation-path` if your parser defines it. Note: spatial_engine has no per-object
trajectory loader (its `TimelineJson` only carries scene-snapshot keyframes), so this
is a documented interchange format, not a native engine file.

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

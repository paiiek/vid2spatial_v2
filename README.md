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
from vid2spatial_pkg.v2_spatial_tracker import V2SpatialTracker

tracker = V2SpatialTracker(
    video_path="input.mp4",
    bbox_init=[x, y, w, h],   # draw once
    tracker_type="yw_sam2",
    det_threshold=0.99,
)
traj = tracker.run()           # returns per-frame spatial parameter stream
traj.save("output/traj.json")
```

### Render binaural

```python
from vid2spatial_pkg.foa_render import direct_binaural_sofa

binaural = direct_binaural_sofa(
    audio_mono, sr=48000,
    az_rad=traj.az, el_rad=traj.el,
    sofa_path="kemar.sofa"
)
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
so they map 1:1 onto `/adm/obj/N/aed`. Note: spatial_engine has no per-object
trajectory loader (its `TimelineJson` only carries scene-snapshot keyframes), so this
is a documented interchange format, not a native engine file.

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
  config.py               # dataclass configs
  depth_utils.py          # MiDaS / DA-V2 depth backends
  multi_source.py         # multi-object spatial audio

test/
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

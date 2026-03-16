# Vid2Spatial OSC Interface Specification

## Overview

Vid2Spatial streams spatial audio control parameters to DAW via OSC (Open Sound Control).
This enables real-time spatialization control in professional audio workflows.

**Pipeline:**
```
Video → Object Tracking → Depth Estimation → Trajectory → Normalization → OSC Sender → DAW/Spatial Plugins
```

## OSC Address Table

| Address | Type | Range | Unit | Description |
|---------|------|-------|------|-------------|
| `/vid2spatial/azimuth` | float | [-180, 180] | degrees | Horizontal angle (0=front, +90=left, -90=right) |
| `/vid2spatial/azimuth_norm` | float | [-1, 1] | normalized | Azimuth / 180 |
| `/vid2spatial/elevation` | float | [-90, 90] | degrees | Vertical angle (0=horizon, +90=up, -90=down) |
| `/vid2spatial/elevation_norm` | float | [-1, 1] | normalized | Elevation / 90 |
| `/vid2spatial/distance` | float | [0.5, 10+] | meters | Metric depth from Depth Anything V2 |
| `/vid2spatial/d_rel` | float | [0, 1] | normalized | Relative distance (0=near, 1=far) |
| `/vid2spatial/velocity` | float | [0, ∞) | deg/sec | Angular velocity |
| `/vid2spatial/confidence` | float | [0, 1] | normalized | Tracking confidence |
| `/vid2spatial/timecode` | float | [0, ∞) | seconds | Video timecode |
| `/vid2spatial/frame` | int | [0, ∞) | frame index | Current frame number |

## Parameter Definitions

### Azimuth
- **Unit:** degrees
- **Range:** [-180, 180]
- **Convention:** 0° = front center, +90° = left, -90° = right (listener-centric)
- **Normalized:** `azimuth_norm = azimuth / 180` → [-1, 1]

### Elevation
- **Unit:** degrees
- **Range:** [-90, 90]
- **Convention:** 0° = horizon, +90° = zenith (up), -90° = nadir (down)
- **Normalized:** `elevation_norm = elevation / 90` → [-1, 1]

### Distance
- **Unit:** meters (metric depth from Depth Anything V2)
- **Range:** typically 0.5m to 10m+
- **Note:** Raw metric depth, may have noise from depth estimation

### d_rel (Relative Distance)
- **Unit:** normalized [0, 1]
- **Range:** 0 (near, 0.5m) to 1 (far, 10m)
- **Formula:** `d_rel = clamp((dist_m - 0.5) / 9.5, 0, 1)`
- **Use:** Preferred for perceptual mapping (gain, reverb, LPF)

### Velocity
- **Unit:** degrees per second
- **Computation:** `sqrt(d_az² + d_el²) * fps`
- **Use:** Motion detection, dynamic processing control

### Confidence
- **Range:** [0, 1]
- **Source:** Object detection/tracking confidence
- **Use:** Reliability indicator, can control blend weights

## OSC Message Examples

```
/vid2spatial/azimuth 45.0
/vid2spatial/elevation 10.0
/vid2spatial/distance 2.5
/vid2spatial/d_rel 0.21
/vid2spatial/confidence 0.87
/vid2spatial/timecode 3.5
```

**Bundle format (recommended):**
```
#bundle
  /vid2spatial/azimuth 45.0
  /vid2spatial/elevation 10.0
  /vid2spatial/d_rel 0.21
  /vid2spatial/timecode 3.5
```

## DAW Integration

### Reaper
1. Install ReaControlMIDI or similar OSC receiver
2. Map parameters to track pan/width/distance controls
3. Use `d_rel` for direct 0-1 parameter mapping

### Max/MSP
```max
[udpreceive 9000]
|
[route /vid2spatial/azimuth /vid2spatial/d_rel]
|           |
[spat5.pan~] [gain~]
```

### IRCAM SPAT
- Map `azimuth` to azimuth parameter
- Map `elevation` to elevation parameter
- Map `d_rel` to distance (SPAT expects normalized)

### Ableton Live (via Max for Live)
- Use [mfl.osc] object to receive
- Map to device parameters via M4L

## Synchronization

| Parameter | Value |
|-----------|-------|
| Default FPS | 30 |
| OSC Rate | video FPS (30 Hz typical) |
| Timecode | seconds since video start |
| OSC send latency | ~0.05ms/frame (UDP, non-blocking) |
| E2E pipeline latency | 1.3-4.6x realtime (offline, see below) |

**Timecode formula:**
```
timecode_sec = frame_index / fps
```

**E2E Pipeline Latency** (10s video, 300 frames, RTX 3090):

| Stage | Time | Note |
|-------|------|------|
| DINO tracking | 11.5-39.6s | ~90% of total; varies with motion speed |
| RTS smoothing | <0.01s | Negligible |
| Audio render | ~1.5s | HRTF binaural or stereo pan |
| OSC send | ~0.015s | 300 UDP packets, non-realtime burst |
| **Total** | **13-46s** | **1.3-4.6x realtime** |

Note: OSC streaming is offline (pre-computed trajectory). For real-time preview,
use EMA smoothing instead of RTS and stream as frames are tracked.

## Design Rationale

### Why d_rel instead of raw distance?

1. **Consistent mapping:** Raw distance varies by video (1-3m vs 5-10m). d_rel uses **per-clip min/max normalization** (extracted from the trajectory's own distance range) to always span [0, 1].

2. **Perceptual alignment:** Human distance perception is roughly logarithmic. Using normalized range allows DAW to apply its own perceptual curves.

3. **DAW compatibility:** Most spatial plugins expect 0-1 range for distance parameters.

### Why separate azimuth and azimuth_norm?

- **azimuth (degrees):** Intuitive for visualization, debugging
- **azimuth_norm ([-1,1]):** Direct mapping to stereo pan, plugin parameters

### Coordinate System

```
         +Y (up)
          |
          |
          |_____ +X (right, listener perspective)
         /
        /
       +Z (front)

Azimuth: rotation around Y axis (0° = +Z, +90° = -X)
Elevation: angle from XZ plane (0° = horizon)
```

## Implementation Notes

### FOA Rendering Pipeline

```python
# In foa_render.py
az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, sr)

# Gain: uses dist_s (1/r law)
# LPF: uses d_rel_s (consistent perceptual mapping)
# Reverb: uses d_rel_s (consistent wetness mapping)
```

### Depth Field Priority

When extracting distance from trajectory:
1. `depth_render` (explicit render value from RTS smoothing)
2. `depth_blended` (from depth enhancement)
3. `dist_m_raw` (raw metric depth)
4. `dist_m` (may be smoothed)

### d_rel Computation (depth_utils.py)

```python
d_rel = clamp((dist_m - d_min) / (d_max - d_min), 0, 1)
# d_min, d_max: extracted per-clip from trajectory distance range
# If range < 0.1m, d_rel = 0.5 (constant)
```

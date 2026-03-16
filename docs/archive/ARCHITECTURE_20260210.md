# Vid2Spatial: Technical Architecture Document

**Version**: 1.2
**Date**: 2026-02-10
**Author**: Seungheon Doh

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Overview](#2-system-overview)
3. [Module Descriptions](#3-module-descriptions)
   - 3.1 [HybridTracker](#31-hybridtracker)
   - 3.2 [DINO Adaptive-K Re-detection](#32-dino-adaptive-k-re-detection)
   - 3.3 [Robustness Layer](#33-robustness-layer)
   - 3.4 [Depth Estimation](#34-depth-estimation)
   - 3.5 [3D Projection](#35-3d-projection)
   - 3.6 [RTS Smoother](#36-rts-smoother)
   - 3.7 [Audio Rendering](#37-audio-rendering)
   - 3.8 [OSC Sender](#38-osc-sender)
4. [Data Flow](#4-data-flow)
5. [Performance Analysis](#5-performance-analysis)
6. [Implementation Details](#6-implementation-details)
7. [Configuration Options](#7-configuration-options)
8. [References](#8-references)

---

## 1. Introduction

### 1.1 Problem Statement

Spatial audio authoring for video content requires **stable spatial control trajectories** derived from visual motion. Traditional approaches face several challenges:

1. **Motion Collapse**: State-of-the-art video object segmentation models (e.g., SAM2) experience severe tracking failures at motion frequencies above 0.5Hz
2. **Depth Ambiguity**: Monocular depth estimation is unreliable for fast-moving small objects
3. **Temporal Jitter**: Raw tracking outputs contain high-frequency noise unsuitable for perceptual audio
4. **Control-signal Reliability**: Tracking accuracy alone is insufficient — trajectories must be smooth and stable enough to serve as audio authoring control inputs

### 1.2 Proposed Solution

Vid2Spatial addresses these challenges through a deterministic vision-guided control pipeline:

1. **DINO Adaptive-K Re-detection**: Text-guided detection with adaptive keyframe intervals that prevents trajectory collapse
2. **Confidence-Weighted Depth Blending**: Metric depth + bbox-scale proxy fusion for stable distance control
3. **RTS Smoothing**: Optimal offline trajectory smoothing preserving motion amplitude
4. **FOA/OSC Dual Output**: Integration with authoring workflows via Ambisonics rendering and DAW automation

### 1.3 Target Applications

- Spatial audio post-production for film/video
- Interactive spatial audio in VR/AR
- Research in audio-visual correspondence
- DAW automation via OSC

---

## 2. System Overview

### 2.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT STAGE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Video File (.mp4, .avi)  +  Audio File (.wav)  +  Text Prompt          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DETECTION STAGE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  Grounding DINO                                                          │
│  ├── Text-guided object detection                                        │
│  ├── Returns: bbox (x, y, w, h), confidence                             │
│  └── Model: GroundingDINO-T (Swin-T backbone)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRACKING STAGE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Adaptive-K Tracker                                                      │
│  ├── Keyframe detection at intervals K ∈ [2, 15]                        │
│  ├── K adapts based on:                                                  │
│  │   - Detection confidence                                              │
│  │   - 2D velocity                                                       │
│  │   - BBox scale change                                                 │
│  ├── Linear interpolation between keyframes                              │
│  └── Robustness layer: confidence gating + jump rejection               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     DEPTH ESTIMATION STAGE                               │
├─────────────────────────────────────────────────────────────────────────┤
│  Dual-Layer Depth                                                        │
│  ├── Layer 1: Depth Anything V2 Metric                                  │
│  │   - Indoor: Hypersim (max 20m)                                       │
│  │   - Outdoor: VKITTI (max 80m)                                        │
│  │   - Auto scene detection                                              │
│  ├── Layer 2: BBox-Scale Proxy                                          │
│  │   - z_proxy = z_initial × √(area_initial / area_current)             │
│  │   - Instant response to scale changes                                 │
│  └── Blending: α × metric + (1-α) × proxy                               │
│      where α = f(confidence)                                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      3D PROJECTION STAGE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Camera Model                                                            │
│  ├── Pinhole camera with configurable FOV (default: 60°)                │
│  ├── pixel (cx, cy) → normalized ray (rx, ry, rz)                       │
│  ├── ray → spherical (azimuth, elevation)                               │
│  └── 3D position: (x, y, z) = ray × depth_m                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       SMOOTHING STAGE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  RTS Smoother (Rauch-Tung-Striebel)                                     │
│  ├── Forward pass: Kalman filter predictions                            │
│  ├── Backward pass: Optimal smoothing                                   │
│  ├── State: [position, velocity] per dimension                          │
│  └── Result: 93-97% jerk reduction                                      │
│                                                                          │
│  Alternative: EMA (for real-time)                                       │
│  └── x_smooth = α × x_new + (1-α) × x_prev                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│        FOA RENDERER           │   │        OSC SENDER             │
├───────────────────────────────┤   ├───────────────────────────────┤
│  AmbiX B-format encoding      │   │  Real-time streaming          │
│  ├── W = mono × gain          │   │  ├── /azimuth                 │
│  ├── Y = mono × sin(az)       │   │  ├── /elevation               │
│  ├── Z = mono × sin(el)       │   │  ├── /distance                │
│  └── X = mono × cos(az)cos(el)│   │  ├── /d_rel                   │
│                               │   │  └── /velocity                │
│  Output: 4ch WAV (48kHz)      │   │  Protocol: UDP                │
└───────────────────────────────┘   └───────────────────────────────┘
```

### 2.2 Module Dependencies

```
vid2spatial_pkg/
├── hybrid_tracker.py      ← Entry point
│   ├── Uses: vision.py (detection, geometry)
│   ├── Uses: depth_metric.py (metric depth)
│   └── Uses: depth_utils.py (proxy, blending)
│
├── trajectory_stabilizer.py  ← RTS smoother
│
├── foa_render.py          ← FOA output
│   └── Uses: trajectory data
│
├── osc_sender.py          ← OSC output
│   └── Uses: trajectory data
│
├── depth_metric.py        ← Depth Anything V2
│   └── External: Depth-Anything-V2/
│
├── depth_utils.py         ← Depth enhancement
│   └── Uses: DepthConfig
│
└── vision.py              ← Core utilities
    ├── Detection (GroundingDINO)
    └── Geometry (camera, rays, angles)
```

---

## 3. Module Descriptions

### 3.1 HybridTracker

**File**: `vid2spatial_pkg/hybrid_tracker.py`

**Purpose**: Main entry point for video object tracking with 3D trajectory output.

**Key Classes**:

```python
@dataclass
class HybridTrackingFrame:
    frame_idx: int                    # Frame number
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    center: Tuple[float, float]       # (cx, cy) in pixels
    confidence: float                 # Detection confidence [0, 1]
    depth_m: float                    # Metric depth in meters
    mask: Optional[np.ndarray]        # Segmentation mask (optional)

@dataclass
class HybridTrackingResult:
    frames: List[HybridTrackingFrame]
    video_width: int
    video_height: int
    fps: float
    total_frames: int
    text_prompt: str
    initial_detection_conf: float
    fov_deg: float = 60.0            # Camera FOV for 3D projection
```

**Main Method**:

```python
def track(
    self,
    video_path: str,
    text_prompt: str,
    tracking_method: str = "adaptive_k",  # "sam2", "yolo", "adaptive_k"
    depth_stride: int = 5,
    end_frame: Optional[int] = None,
) -> HybridTrackingResult:
```

**Tracking Methods**:

| Method | Description | Use Case |
|--------|-------------|----------|
| `adaptive_k` | DINO with adaptive re-detection interval | **Recommended** |
| `sam2` | SAM2 video propagation | Slow, smooth motion |
| `yolo` | YOLO + ByteTrack | Real-time, simple scenes |

---

### 3.2 DINO Adaptive-K Re-detection

**Concept**: Instead of continuous tracking (which fails at high motion), perform periodic re-detection with intervals that adapt to motion characteristics.

**Algorithm**:

```python
def compute_adaptive_k(confidence, velocity_2d, bbox_scale_change):
    """
    Compute optimal re-detection interval K.

    K_min = 2   (high motion: detect every 2 frames)
    K_max = 15  (low motion: detect every 15 frames)
    """
    # Base K from confidence
    if confidence < 0.4:
        k_conf = 3  # Low confidence → frequent re-detection
    elif confidence > 0.7:
        k_conf = 12  # High confidence → less frequent
    else:
        k_conf = 8

    # Adjust for velocity
    if velocity_2d > 50:  # Fast motion
        k_vel = 2
    elif velocity_2d < 10:  # Slow motion
        k_vel = 15
    else:
        k_vel = 8

    # Adjust for scale change (approaching/receding)
    if bbox_scale_change > 0.1:
        k_scale = 3
    else:
        k_scale = 10

    return min(k_conf, k_vel, k_scale)
```

**Interpolation Between Keyframes**:

```python
def interpolate_frames(kf1, kf2, num_frames):
    """Linear interpolation between keyframes."""
    interpolated = []
    for i in range(num_frames):
        t = i / num_frames
        frame = HybridTrackingFrame(
            frame_idx=kf1.frame_idx + i,
            bbox=lerp_bbox(kf1.bbox, kf2.bbox, t),
            center=lerp_point(kf1.center, kf2.center, t),
            confidence=lerp(kf1.confidence, kf2.confidence, t),
            depth_m=lerp(kf1.depth_m, kf2.depth_m, t),
        )
        interpolated.append(frame)
    return interpolated
```

**Why This Works**:

1. DINO is robust to appearance changes (text-guided)
2. Short intervals (K=2-3) capture high-frequency motion
3. Long intervals (K=10-15) save computation for slow scenes
4. Interpolation provides smooth trajectories

---

### 3.3 Robustness Layer

**Purpose**: Filter outlier detections that would cause trajectory artifacts.

**Confidence Gating**:

```python
def confidence_gate(detection, threshold=0.35):
    """Reject low-confidence detections."""
    if detection.confidence < threshold:
        return None  # Reject
    return detection
```

**Jump Rejection**:

```python
def jump_rejection(current, previous, max_velocity=150):
    """
    Reject detections with implausible velocity.

    Args:
        max_velocity: Maximum allowed pixels/frame (150 ≈ 4500 px/s at 30fps)
    """
    if previous is None:
        return current

    dx = current.center[0] - previous.center[0]
    dy = current.center[1] - previous.center[1]
    velocity = math.sqrt(dx*dx + dy*dy)

    if velocity > max_velocity:
        return None  # Reject as outlier
    return current
```

**Combined Pipeline**:

```python
def robustness_filter(detections):
    filtered = []
    previous = None

    for det in detections:
        # Stage 1: Confidence gating
        if det.confidence < CONF_THRESHOLD:
            continue

        # Stage 2: Jump rejection
        if previous is not None:
            velocity = compute_velocity(det, previous)
            if velocity > VELOCITY_THRESHOLD:
                continue

        filtered.append(det)
        previous = det

    return filtered
```

---

### 3.4 Depth Estimation (Distance Control Stability Module)

**File**: `vid2spatial_pkg/depth_metric.py`, `vid2spatial_pkg/depth_utils.py`

> Depth 모듈은 파이프라인의 **거리 제어 안정화(distance control stability)** 핵심 컴포넌트입니다. Metric depth와 bbox-scale proxy를 confidence-weighted blending하여, 프레임간 jitter를 60% 감소시키고 오디오 저작에 적합한 안정적 거리 궤적을 생성합니다.

#### 3.4.1 Metric Depth (Layer 1)

**Model**: Depth Anything V2 Metric

```python
class MetricDepthEstimator:
    """
    Returns actual distance in meters (not relative depth).

    Models:
    - Indoor (Hypersim): max_depth=20m
    - Outdoor (VKITTI): max_depth=80m
    """

    def __init__(
        self,
        scene_type: Literal["indoor", "outdoor", "auto"] = "auto",
        model_size: Literal["small", "base", "large"] = "small",
    ):
        # Auto mode loads both models and selects per-frame
        pass

    def infer(self, frame: np.ndarray, bbox: Tuple) -> float:
        """
        Estimate depth at bbox center.

        Returns:
            depth_m: Distance in meters
        """
        # 1. Run full-frame depth estimation
        depth_map = self.model(frame)

        # 2. Extract depth at bbox center
        cx, cy = bbox_center(bbox)
        depth_m = depth_map[int(cy), int(cx)]

        return depth_m
```

**Scene Type Selection**:

| Scene Type | Model | Max Depth | Training Data |
|------------|-------|-----------|---------------|
| indoor | Hypersim | 20m | Synthetic indoor |
| outdoor | VKITTI | 80m | Driving scenes |
| auto | Both | Adaptive | Per-frame selection |

#### 3.4.2 BBox-Scale Proxy (Layer 2)

**Motivation**: Metric depth models are optimized for scene geometry, not fast-moving objects. BBox scale provides instant feedback for distance changes.

```python
def compute_bbox_scale_proxy(bbox_areas, initial_depth_m=2.0):
    """
    Estimate depth from bbox scale changes.

    Physics: For fixed-size objects, depth ∝ 1/√(bbox_area)

    Args:
        bbox_areas: List of (w × h) per frame
        initial_depth_m: Calibration depth at frame 0

    Returns:
        proxy_depths: Estimated depth per frame
    """
    areas = np.array(bbox_areas)
    initial_area = areas[0]

    # z_proxy = z_initial × √(area_initial / area_current)
    proxy_depths = initial_depth_m * np.sqrt(initial_area / areas)

    return proxy_depths
```

**Properties**:
- ✅ Instant response (no model inference)
- ✅ Captures approach/recede motion
- ❌ Assumes fixed object size
- ❌ Affected by partial occlusion

#### 3.4.3 Confidence-Weighted Blending

**Key Insight**: Use confidence as a proxy for tracking reliability. Low confidence often means the object is small, fast, or partially occluded—exactly when proxy depth is more reliable.

```python
def blend_depth_with_proxy(
    metric_depths,
    proxy_depths,
    confidences,
    min_confidence=0.3,
    max_confidence=0.8,
):
    """
    Blend metric depth with proxy based on tracking confidence.

    Low confidence  → α ≈ 0 (use proxy)
    High confidence → α ≈ 1 (use metric)
    """
    blended = []

    for metric, proxy, conf in zip(metric_depths, proxy_depths, confidences):
        # Compute blending weight
        if conf <= min_confidence:
            alpha = 0.0  # 100% proxy
        elif conf >= max_confidence:
            alpha = 1.0  # 100% metric
        else:
            alpha = (conf - min_confidence) / (max_confidence - min_confidence)

        # Blend
        depth = alpha * metric + (1 - alpha) * proxy
        blended.append(depth)

    return blended
```

**Measured Results**:

| Video Type | Metric Jitter | Blended Jitter | Reduction |
|------------|---------------|----------------|-----------|
| Slow (piano) | 0.03cm | 0.01cm | 60% |
| Moderate | 1.5cm | 0.6cm | 60% |
| Fast (soccer) | **18.5cm** | **7.4cm** | **60%** |

#### 3.4.4 Adaptive Depth Stride

**Motivation**: Depth estimation is expensive. For static scenes, we can skip many frames.

```python
def compute_adaptive_depth_stride(recent_depths, recent_bbox_areas, config):
    """
    Compute optimal depth estimation interval.

    Returns stride ∈ [min_stride, max_stride]
    """
    depth_var = np.var(recent_depths[-10:])
    bbox_change = compute_scale_change(recent_bbox_areas[-10:])

    if depth_var < 0.01 and bbox_change < 0.05:
        return config.max_stride  # 30: stable scene
    elif depth_var > 0.5 or bbox_change > 0.2:
        return config.min_stride  # 2: fast changes
    else:
        return config.default_stride  # 5: default
```

**Measured Savings**:

| Motion Type | Baseline Samples | Adaptive Samples | Savings |
|-------------|-----------------|------------------|---------|
| Slow | 150 | 9 | **94%** |
| Moderate | 150 | 20 | **87%** |
| Fast | 150 | 48 | **68%** |

---

### 3.5 3D Projection

**File**: `vid2spatial_pkg/vision.py`

**Camera Model**:

```python
@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fov_deg: float = 60.0  # Horizontal field of view

    @property
    def fx(self):
        """Focal length in pixels."""
        return self.width / (2 * math.tan(math.radians(self.fov_deg / 2)))

    @property
    def fy(self):
        return self.fx  # Assume square pixels

    @property
    def cx(self):
        return self.width / 2

    @property
    def cy(self):
        return self.height / 2
```

**Pixel to Ray**:

```python
def pixel_to_ray(px, py, K):
    """
    Convert pixel coordinates to unit ray in camera frame.

    Camera frame: X=right, Y=down, Z=forward
    """
    x = (px - K.cx) / K.fx
    y = (py - K.cy) / K.fy
    z = 1.0

    # Normalize
    norm = math.sqrt(x*x + y*y + z*z)
    return (x/norm, y/norm, z/norm)
```

**Ray to Spherical**:

```python
def ray_to_angles(ray):
    """
    Convert ray to spherical coordinates.

    Returns:
        azimuth: Horizontal angle, + = right, radians
        elevation: Vertical angle, + = up, radians
    """
    x, y, z = ray

    # Azimuth: angle from forward (Z) in XZ plane
    azimuth = math.atan2(x, z)

    # Elevation: angle from horizontal
    elevation = -math.asin(y)  # Negative because Y points down

    return azimuth, elevation
```

**Complete 3D Position**:

```python
def compute_3d_position(px, py, depth_m, K):
    """
    Compute 3D position from pixel + depth.
    """
    ray = pixel_to_ray(px, py, K)

    x = ray[0] * depth_m
    y = ray[1] * depth_m
    z = ray[2] * depth_m

    az, el = ray_to_angles(ray)

    return {
        "x": x, "y": y, "z": z,
        "az": az, "el": el,
        "dist_m": depth_m,
    }
```

---

### 3.6 RTS Smoother

**File**: `vid2spatial_pkg/trajectory_stabilizer.py`

**Concept**: Rauch-Tung-Striebel (RTS) smoother is the optimal linear smoother for offline data. It runs a Kalman filter forward, then a backward pass to incorporate future information.

**State Model**:

```
State: x = [position, velocity]^T
Process: x_{k+1} = A × x_k + w    (constant velocity model)
Observation: z_k = H × x_k + v   (we observe position)

A = [1, dt]    H = [1, 0]
    [0,  1]
```

**Algorithm**:

```python
def rts_smooth_trajectory(frames):
    """
    Apply RTS smoothing to trajectory.

    1. Forward pass (Kalman filter)
    2. Backward pass (RTS smoother)
    """
    # Extract positions
    positions = np.array([[f["az"], f["el"], f["dist_m"]] for f in frames])

    # For each dimension
    smoothed = np.zeros_like(positions)
    for dim in range(3):
        smoothed[:, dim] = rts_smooth_1d(positions[:, dim])

    # Reconstruct frames
    result = []
    for i, f in enumerate(frames):
        result.append({
            **f,
            "az": smoothed[i, 0],
            "el": smoothed[i, 1],
            "dist_m": smoothed[i, 2],
        })

    return result


def rts_smooth_1d(observations):
    """RTS smoother for 1D signal."""
    n = len(observations)

    # State: [position, velocity]
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    x_smooth = np.zeros((n, 2))

    # Process noise
    Q = np.array([[0.01, 0], [0, 0.1]])
    # Measurement noise
    R = np.array([[0.1]])

    # Forward pass (Kalman filter)
    x_filt[0] = [observations[0], 0]
    P_filt[0] = np.eye(2)

    for k in range(1, n):
        # Predict
        A = np.array([[1, 1], [0, 1]])
        x_pred[k] = A @ x_filt[k-1]
        P_pred[k] = A @ P_filt[k-1] @ A.T + Q

        # Update
        H = np.array([[1, 0]])
        y = observations[k] - H @ x_pred[k]
        S = H @ P_pred[k] @ H.T + R
        K = P_pred[k] @ H.T @ np.linalg.inv(S)

        x_filt[k] = x_pred[k] + K @ y
        P_filt[k] = (np.eye(2) - K @ H) @ P_pred[k]

    # Backward pass (RTS smoother)
    x_smooth[n-1] = x_filt[n-1]

    for k in range(n-2, -1, -1):
        A = np.array([[1, 1], [0, 1]])
        C = P_filt[k] @ A.T @ np.linalg.inv(P_pred[k+1])
        x_smooth[k] = x_filt[k] + C @ (x_smooth[k+1] - x_pred[k+1])

    return x_smooth[:, 0]  # Return position component
```

**Results**:

| Metric | Before RTS | After RTS | Reduction |
|--------|-----------|-----------|-----------|
| Jerk (az) | 0.089 | **0.0026** | **97%** |
| Jerk (el) | 0.076 | **0.0018** | **98%** |
| Jerk (dist) | 0.12 | **0.0089** | **93%** |

---

### 3.7 Audio Rendering

**File**: `vid2spatial_pkg/foa_render.py`

Vid2Spatial provides **two independent rendering paths**, both driven by the same trajectory:

```
trajectory_3d.json
       │
       ├──▶ FOA Renderer ──▶ foa.wav (4ch AmbiX, for Ambisonics playback)
       │
       └──▶ HRTF Binaural ──▶ proposed.wav (2ch, for headphone listening)
```

#### 3.7.1 Coordinate Convention (CRITICAL)

The pipeline and spatial audio standards use **opposite azimuth conventions**:

| Convention | Right of center | Left of center |
|-----------|-----------------|----------------|
| Pipeline (`atan2(x, z)`) | az > 0 | az < 0 |
| AmbiX / SOFA standard | az < 0 | az > 0 |

**Solution**: Negate azimuth before FOA encoding or HRTF lookup:
```python
az_ambiX = -az_pipeline
```

#### 3.7.2 FOA Encoding (AmbiX ACN/SN3D)

| Channel | Name | Encoding |
|---------|------|----------|
| W | Omnidirectional | `mono × gain` |
| Y | Left-Right | `mono × gain × sin(az)` |
| Z | Up-Down | `mono × gain × sin(el)` |
| X | Front-Back | `mono × gain × cos(az) × cos(el)` |

#### 3.7.3 HRTF Binaural Rendering (Overlap-Add)

Direct HRIR convolution from KEMAR SOFA (64,800 measurements) for full-bandwidth spatial cues (ILD, ITD, pinna filtering). Uses **overlap-add with Hann window** for artifact-free HRIR transitions:

- **Block size**: 50ms (2,400 samples at 48kHz), hop = 50% (1,200 samples)
- **Window**: Hann → 50% overlap guarantees perfect reconstruction
- **HRIR lookup**: nearest-neighbor via Cartesian dot-product per block

Each block: convolve extended input `[start - hrir_len + 1, end)` with block's HRIR, window with Hann, accumulate via overlap-add. Adjacent blocks with different HRIRs blend smoothly through the overlapping Hann windows.

#### 3.7.4 Shared Processing Chain

Both FOA and binaural paths share identical post-processing:
- **Distance gain + LPF**: `apply_distance_gain_lpf(d_rel_s)` — d_rel-driven attenuation and low-pass
- **Reverb**: `schroeder_ir(sr, rt60=0.4)` — Schroeder reverberator
- **d_rel normalization**: Per-clip min/max (not global [0.5m, 10m])

```python
# d_rel: per-clip normalization
d_min, d_max = dist.min(), dist.max()
d_range = d_max - d_min
d_rel = 0.5 if d_range < 0.1 else clip((dist - d_min) / d_range, 0, 1)
```

#### 3.7.5 Baseline Condition (Stereo Pan)

For listening test comparison:
- **Stereo pan**: `pan = sin(az)` (constant-power pan law)
- Same distance gain/LPF and reverb as proposed
- **Only difference** between proposed and baseline: HRTF binaural vs. stereo pan

---

### 3.8 OSC Sender

**File**: `vid2spatial_pkg/osc_sender.py`

**OSC Protocol**:

```python
class OSCSpatialSender:
    """
    Stream spatial parameters via OSC for DAW automation.
    """

    def __init__(self, host="127.0.0.1", port=9000):
        self.client = udp_client.SimpleUDPClient(host, port)

    def send_frame(self, frame):
        """Send single frame data."""
        self.client.send_message("/vid2spatial/azimuth",
                                 math.degrees(frame["az"]))
        self.client.send_message("/vid2spatial/elevation",
                                 math.degrees(frame["el"]))
        self.client.send_message("/vid2spatial/distance",
                                 frame["dist_m"])
        self.client.send_message("/vid2spatial/d_rel",
                                 frame.get("d_rel", 0.5))
        self.client.send_message("/vid2spatial/timecode",
                                 frame["frame"] / self.fps)

    def stream_trajectory(self, trajectory, fps=30, realtime=True):
        """Stream entire trajectory."""
        self.fps = fps
        frame_duration = 1.0 / fps

        for frame in trajectory["frames"]:
            self.send_frame(frame)

            if realtime:
                time.sleep(frame_duration)
```

**DAW Integration Example (Reaper)**:

```lua
-- Reaper OSC pattern for azimuth
OSC_PATTERN = "/vid2spatial/azimuth"
TRACK_FX_PARAM = 1  -- Panner azimuth parameter

function on_osc_message(address, value)
    if address == OSC_PATTERN then
        -- Map -180..180 to 0..1
        local normalized = (value + 180) / 360
        reaper.TrackFX_SetParam(track, fx_index, TRACK_FX_PARAM, normalized)
    end
end
```

---

## 4. Data Flow

### 4.1 Trajectory Data Format

```python
trajectory = {
    "intrinsics": {
        "width": 1920,
        "height": 1080,
        "fov_deg": 60.0,
    },
    "frames": [
        {
            "frame": 0,
            "az": 0.1234,           # Azimuth in radians
            "el": -0.0567,          # Elevation in radians
            "dist_m": 2.5,          # Metric depth in meters
            "d_rel": 0.35,          # Relative distance [0, 1]
            "depth_blended": 2.5,   # Blended depth
            "depth_proxy": 2.4,     # BBox-scale proxy depth
            "confidence": 0.82,     # Detection confidence
            "x": 0.123,             # 3D position X
            "y": -0.056,            # 3D position Y
            "z": 2.49,              # 3D position Z
        },
        # ... more frames
    ]
}
```

### 4.2 Processing Pipeline Timing

**Measured E2E latency** (10s video, 300 frames @ 30fps, NVIDIA RTX 3090):

| Stage | Slow Motion | Fast Motion | Note |
|-------|-------------|-------------|------|
| Tracker init | 4.2s | 4.2s | One-time model load (DINO+SAM2+DepthAnything) |
| **DINO tracking** | **11.5s** | **39.6s** | Bottleneck (~90% of total) |
| RTS smoothing | 0.006s | 0.008s | Negligible |
| **Binaural render** | **1.3s** | **1.4s** | HRTF OLA convolution (10s audio) |
| Baseline render | 0.4s | 0.4s | Stereo pan + reverb |
| OSC send | 0.009s | 0.015s | 300 UDP packets |
| **Total** | **13.3s** | **46.3s** | |
| **Realtime ratio** | **1.3x** | **4.6x** | |

Tracking time varies with motion speed (adaptive K):
- Slow motion (motorcycle-17): avg K=14.4, 18 keyframes → 11.5s
- Fast motion (dog-14): avg K=2.2, 137 keyframes → 39.6s

Post-tracking rendering is ~1.5s regardless of content — essentially real-time.

---

## 5. Performance Analysis

### 5.1 Tracking Accuracy

**Synthetic Test (0.6Hz Sinusoidal Motion)**:

| Method | Amplitude Recovery | MAE | Phase Error |
|--------|-------------------|-----|-------------|
| SAM2 | 3.4% | 142.9px | N/A (collapsed) |
| YOLO+ByteTrack | 78% | 45.2px | 12° |
| **DINO Adaptive-K** | **100%** | **16.1px** | **<5°** |

**Real Video Comparison (13 videos)**:

| Category | Win Rate | Best Cases |
|----------|----------|------------|
| Fast motion | 85% | Sports, dance |
| Moderate motion | 60% | Walking, instruments |
| Slow motion | 40% | Static scenes |

### 5.2 Depth Quality

**Metric vs Proxy Comparison**:

| Motion Type | Metric Jitter | Proxy Jitter | Blended Jitter |
|-------------|---------------|--------------|----------------|
| Slow | 0.03cm | 0.00cm | 0.01cm |
| Moderate | 1.5cm | 0.00cm | 0.6cm |
| Fast | **18.5cm** | 0.00cm | **7.4cm** |

### 5.3 Computational Cost

**GPU**: NVIDIA RTX 3090

| Component | VRAM | Time per Frame |
|-----------|------|----------------|
| Grounding DINO | 2.1GB | 45ms |
| SAM2 (optional) | 1.8GB | 35ms |
| Depth Anything V2 | 1.2GB | 30ms |
| YOLO (optional) | 0.5GB | 8ms |
| **Total** | **~5GB** | ~80ms (keyframe) |

---

## 6. Implementation Details

### 6.1 Environment Setup

```bash
# Python version
Python 3.10+

# Core dependencies
torch>=2.0
torchvision
numpy
opencv-python
scipy
soundfile
python-osc

# Model dependencies
transformers
timm
segment-anything-2
```

### 6.2 Model Weights

| Model | Source | Size |
|-------|--------|------|
| Grounding DINO | HuggingFace | 340MB |
| Depth Anything V2 Metric (Small) | HuggingFace | 99MB |
| SAM2 Hiera Small | Meta | 185MB |

### 6.3 GPU Memory Management

```python
# Clear cache between heavy operations
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use mixed precision where possible
with torch.cuda.amp.autocast():
    output = model(input)
```

---

## 7. Configuration Options

### 7.1 Tracker Configuration

```python
tracker = HybridTracker(
    device="cuda",              # "cuda" or "cpu"
    box_threshold=0.15,         # DINO detection threshold
    fov_deg=60.0,              # Camera field of view
)

result = tracker.track(
    video_path="input.mp4",
    text_prompt="person",
    tracking_method="adaptive_k",  # or "sam2", "yolo"
    depth_stride=5,             # Depth estimation interval
    end_frame=300,              # Max frames to process
)
```

### 7.2 Depth Configuration

```python
from vid2spatial_pkg.depth_utils import DepthConfig, process_trajectory_depth

config = DepthConfig(
    # BBox proxy blending
    use_bbox_proxy=True,
    proxy_blend_by_confidence=True,

    # Adaptive stride
    use_adaptive_stride=True,
    min_stride=2,
    max_stride=30,
    default_stride=5,

    # d_rel output
    output_d_rel=True,
    d_rel_min=0.5,   # Maps to d_rel=0
    d_rel_max=10.0,  # Maps to d_rel=1
)

enhanced_frames = process_trajectory_depth(trajectory["frames"], config)
```

### 7.3 RTS Smoother Configuration

```python
from vid2spatial_pkg.trajectory_stabilizer import rts_smooth_trajectory

smoothed = rts_smooth_trajectory(
    frames,
    process_noise=0.01,      # Lower = smoother
    measurement_noise=0.1,   # Higher = trusts model more
)
```

---

## 8. References

### 8.1 Models

1. **Grounding DINO**: Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection", 2023
2. **Depth Anything V2**: Yang et al., "Depth Anything V2", 2024
3. **SAM2**: Kirillov et al., "Segment Anything 2", 2024

### 8.2 Algorithms

1. **RTS Smoother**: Rauch, H. E., Tung, F., & Striebel, C. T. (1965). "Maximum likelihood estimates of linear dynamic systems"
2. **Kalman Filter**: Kalman, R. E. (1960). "A new approach to linear filtering and prediction problems"

### 8.3 Spatial Audio

1. **Ambisonics**: Gerzon, M. A. (1973). "Periphony: With-height sound reproduction"
2. **FOA Encoding**: Daniel, J. (2000). "Représentation de champs acoustiques"

---

## Appendix A: Output Format Specifications

### A.1 FOA WAV Format

```
Format: WAV, PCM, 32-bit float
Sample Rate: 48000 Hz
Channels: 4 (AmbiX ACN SN3D)
  Channel 0: W (omnidirectional)
  Channel 1: Y (left-right)
  Channel 2: Z (up-down)
  Channel 3: X (front-back)
```

### A.2 Trajectory JSON Format

```json
{
  "intrinsics": {
    "width": 1920,
    "height": 1080,
    "fov_deg": 60.0
  },
  "frames": [
    {
      "frame": 0,
      "az": 0.0,
      "el": 0.0,
      "dist_m": 2.0,
      "d_rel": 0.35,
      "x": 0.0,
      "y": 0.0,
      "z": 2.0,
      "confidence": 0.85
    }
  ]
}
```

---

**Document Version History**:
- v1.0 (2026-02-05): Initial release

---

*This document is part of the Vid2Spatial project. For questions, contact: seungheon.doh@kaist.ac.kr*

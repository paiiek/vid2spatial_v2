"""
Depth Utilities for Vid2Spatial

Provides:
1. bbox-scale proxy depth with confidence blending
2. d_rel (relative distance) output
3. adaptive depth stride based on motion/variance

Key insight: For spatial audio, relative distance changes matter more than
absolute metric accuracy. Fast-moving objects benefit from bbox-scale proxy.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DepthConfig:
    """Configuration for depth processing."""
    # Blending strategy: "metric_default" (conservative) or "proxy_blend" (aggressive)
    # metric_default: Use metric depth, only blend proxy when stable AND fast motion
    # proxy_blend: Original confidence-based blending
    blend_strategy: str = "metric_default"

    # Blending parameters (for proxy_blend strategy)
    use_bbox_proxy: bool = True
    proxy_blend_by_confidence: bool = True  # blend based on tracking confidence
    use_proxy_variance_gating: bool = True  # gate proxy by its stability

    # Proxy variance gating parameters
    proxy_var_window: int = 5  # Window size for variance computation
    proxy_var_threshold: float = 0.1  # Variance threshold for stability
    proxy_diff_threshold: float = 0.3  # Max metric-proxy diff (meters) for stability

    # Fast motion threshold for metric_default strategy
    fast_motion_threshold: float = 0.1  # bbox area change rate threshold

    # Adaptive stride parameters
    use_adaptive_stride: bool = True
    min_stride: int = 2
    max_stride: int = 30
    default_stride: int = 5

    # Variance thresholds for adaptive stride
    depth_var_low: float = 0.01  # Below this -> increase stride
    depth_var_high: float = 0.5  # Above this -> decrease stride
    bbox_change_low: float = 0.05
    bbox_change_high: float = 0.2

    # Output options (d_rel always computed from metric depth)
    d_rel_min: float = 0.5  # Min distance for d_rel normalization
    d_rel_max: float = 10.0  # Max distance for d_rel normalization


def compute_bbox_scale_proxy(
    bbox_areas: List[float],
    initial_depth_m: float = 2.0,
) -> List[float]:
    """
    Compute depth proxy from bbox scale changes.

    Assumes: depth ∝ 1/sqrt(bbox_area)
    Calibrated using initial frame's metric depth.

    Args:
        bbox_areas: List of bbox areas (w*h) per frame
        initial_depth_m: Metric depth at frame 0 for calibration

    Returns:
        List of proxy depth values in meters
    """
    if not bbox_areas:
        return []

    areas = np.array(bbox_areas, dtype=np.float32)
    areas = np.maximum(areas, 1.0)  # Avoid division by zero

    # Calibrate: z_proxy = z_initial * sqrt(area_initial / area_current)
    initial_area = areas[0]
    proxy_depths = initial_depth_m * np.sqrt(initial_area / areas)

    return proxy_depths.tolist()


def compute_proxy_stability(
    proxy_depths: List[float],
    metric_depths: List[float],
    window: int = 5,
    var_threshold: float = 0.1,
    diff_threshold: float = 1.0,
) -> List[float]:
    """
    Compute proxy stability score based on:
    1. Local variance (proxy 자체 안정성)
    2. Metric-proxy difference (proxy 신뢰성)

    High variance OR large diff -> low stability -> use metric depth
    Low variance AND small diff -> high stability -> can use proxy

    Args:
        proxy_depths: Proxy depth values
        metric_depths: Metric depth values for comparison
        window: Window size for variance computation
        var_threshold: Variance threshold (higher = more tolerant)
        diff_threshold: Max acceptable metric-proxy difference in meters

    Returns:
        Stability scores (0-1) per frame
    """
    n = len(proxy_depths)
    if n == 0:
        return []

    proxy_arr = np.array(proxy_depths)
    metric_arr = np.array(metric_depths) if metric_depths else np.ones(n)
    stability = []

    for i in range(n):
        # Get window around current frame
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        window_vals = proxy_arr[start:end]

        if len(window_vals) < 2:
            stability.append(1.0)  # Not enough data, assume stable
            continue

        # 1. Local variance stability
        local_var = np.var(window_vals)
        var_stab = float(np.exp(-local_var / var_threshold))

        # 2. Metric-proxy difference stability
        # Large difference means bbox-scale calibration is off
        diff = abs(proxy_arr[i] - metric_arr[i])
        diff_stab = float(np.exp(-diff / diff_threshold))

        # Combined: both must be stable (product is stricter)
        stab = var_stab * diff_stab
        stability.append(stab)

    return stability


def blend_depth_with_proxy(
    metric_depths: List[float],
    proxy_depths: List[float],
    confidences: List[float],
    proxy_stability: Optional[List[float]] = None,
    min_confidence: float = 0.3,
    max_confidence: float = 0.8,
) -> List[float]:
    """
    Blend metric depth with bbox-scale proxy based on confidence AND proxy stability.

    Blending logic:
    - confidence: detection reliability (객체 찾기 신뢰도)
    - proxy_stability: bbox depth reliability (깊이 추정 신뢰도)

    Final alpha = confidence_alpha * stability
    - Low confidence + stable proxy -> use proxy
    - Low confidence + unstable proxy -> use metric (safer)
    - High confidence -> use metric

    Args:
        metric_depths: Metric depth values from Depth Anything V2
        proxy_depths: Proxy depth from bbox scale
        confidences: Tracking confidence per frame
        proxy_stability: Stability scores for proxy (0-1), None = assume stable
        min_confidence: Below this, consider using proxy
        max_confidence: Above this, use 100% metric

    Returns:
        Blended depth values
    """
    n = len(metric_depths)
    blended = []

    for i in range(n):
        conf = confidences[i] if i < len(confidences) else 0.5
        stab = proxy_stability[i] if proxy_stability and i < len(proxy_stability) else 1.0

        # Compute base alpha (metric weight) based on confidence
        if conf <= min_confidence:
            base_alpha = 0.0  # Would use 100% proxy
        elif conf >= max_confidence:
            base_alpha = 1.0  # 100% metric
        else:
            # Linear interpolation
            base_alpha = (conf - min_confidence) / (max_confidence - min_confidence)

        # Apply proxy stability gating:
        # If proxy is unstable (stab low), increase alpha toward metric
        # alpha = base_alpha + (1 - base_alpha) * (1 - stab)
        # Simplified: alpha = 1 - (1 - base_alpha) * stab
        alpha = 1.0 - (1.0 - base_alpha) * stab

        metric = metric_depths[i] if i < len(metric_depths) else 2.0
        proxy = proxy_depths[i] if i < len(proxy_depths) else 2.0

        # Blend
        blended_depth = alpha * metric + (1 - alpha) * proxy
        blended.append(blended_depth)

    return blended


def compute_d_rel(
    depths_m: List[float],
    d_min: float = 0.5,
    d_max: float = 10.0,
) -> List[float]:
    """
    Convert metric depth to relative distance (0-1).

    0 = close (d_min), 1 = far (d_max)
    Useful for direct control of gain/reverb in spatial audio.

    Args:
        depths_m: Depth values in meters
        d_min: Minimum distance (maps to 0)
        d_max: Maximum distance (maps to 1)

    Returns:
        Relative distance values (0-1)
    """
    d_rel = []
    for d in depths_m:
        # Clamp and normalize
        clamped = max(d_min, min(d_max, d))
        normalized = (clamped - d_min) / (d_max - d_min)
        d_rel.append(normalized)

    return d_rel


def compute_adaptive_depth_stride(
    recent_depths: List[float],
    recent_bbox_areas: List[float],
    config: DepthConfig = None,
) -> int:
    """
    Compute optimal depth stride based on recent motion/variance.

    Logic:
    - Low variance + low bbox change -> high stride (save computation)
    - High variance or fast motion -> low stride (need accuracy)

    Args:
        recent_depths: Last N depth values
        recent_bbox_areas: Last N bbox areas
        config: DepthConfig with thresholds

    Returns:
        Recommended depth stride
    """
    if config is None:
        config = DepthConfig()

    if len(recent_depths) < 5:
        return config.default_stride

    # Compute depth variance
    depth_var = np.var(recent_depths[-10:]) if len(recent_depths) >= 10 else np.var(recent_depths)

    # Compute bbox scale change rate
    if len(recent_bbox_areas) >= 2:
        areas = np.array(recent_bbox_areas[-10:])
        scale_changes = np.abs(np.diff(areas) / (areas[:-1] + 1e-8))
        bbox_change_rate = np.mean(scale_changes)
    else:
        bbox_change_rate = 0.0

    # Decision logic
    if depth_var < config.depth_var_low and bbox_change_rate < config.bbox_change_low:
        # Very stable -> use max stride
        return config.max_stride
    elif depth_var > config.depth_var_high or bbox_change_rate > config.bbox_change_high:
        # Fast changing -> use min stride
        return config.min_stride
    else:
        # Moderate -> use default
        return config.default_stride


def process_trajectory_depth(
    frames: List[Dict],
    config: DepthConfig = None,
) -> List[Dict]:
    """
    Post-process trajectory with enhanced depth handling.

    Adds:
    - depth_blended: Blended metric+proxy depth (with variance gating)
    - d_rel: Relative distance (0-1) from METRIC depth (always computed)
    - depth_proxy: Pure bbox-scale proxy (for debugging)
    - proxy_stability: Stability score of proxy (for debugging)

    Args:
        frames: Trajectory frames with dist_m, confidence, bbox info
        config: DepthConfig

    Returns:
        Enhanced trajectory frames
    """
    if config is None:
        config = DepthConfig()

    if not frames:
        return frames

    # Extract data
    metric_depths = [f.get("dist_m", 2.0) for f in frames]
    confidences = [f.get("confidence", 0.5) for f in frames]

    # Compute bbox areas
    bbox_areas = []
    for f in frames:
        w = f.get("w", f.get("bbox_w", 100))
        h = f.get("h", f.get("bbox_h", 100))
        bbox_areas.append(w * h)

    # 1. Compute bbox-scale proxy
    initial_depth = metric_depths[0] if metric_depths[0] > 0 else 2.0
    proxy_depths = compute_bbox_scale_proxy(bbox_areas, initial_depth)

    # 2. Compute proxy stability (variance + metric-proxy diff gating)
    proxy_stability = compute_proxy_stability(
        proxy_depths,
        metric_depths,
        window=config.proxy_var_window,
        var_threshold=config.proxy_var_threshold,
        diff_threshold=config.proxy_diff_threshold,
    )

    # 3. Compute bbox motion rate (for fast motion detection)
    areas = np.array(bbox_areas)
    if len(areas) > 1:
        area_changes = np.abs(np.diff(areas) / (areas[:-1] + 1e-8))
        # Pad to match length
        motion_rate = np.concatenate([[0], area_changes])
    else:
        motion_rate = np.zeros(len(frames))

    # 4. Blend depths based on strategy
    if config.blend_strategy == "metric_default":
        # Conservative: Use metric depth by default
        # Only blend proxy when: (1) proxy is stable AND (2) fast motion detected
        blended_depths = []
        for i in range(len(metric_depths)):
            is_fast = motion_rate[i] > config.fast_motion_threshold
            is_stable = proxy_stability[i] > 0.5

            if is_fast and is_stable:
                # Fast motion + stable proxy: blend with proxy
                alpha = 0.7  # Still mostly metric
                blended = alpha * metric_depths[i] + (1 - alpha) * proxy_depths[i]
            else:
                # Default: use metric
                blended = metric_depths[i]
            blended_depths.append(blended)
    else:
        # Original proxy_blend strategy
        if config.use_bbox_proxy and config.proxy_blend_by_confidence:
            blended_depths = blend_depth_with_proxy(
                metric_depths, proxy_depths, confidences, proxy_stability
            )
        else:
            blended_depths = metric_depths

    # 4. Compute d_rel from METRIC depth (always, for consistent OSC output)
    # This ensures d_rel reflects actual distance, not proxy artifacts
    d_rel_values = compute_d_rel(metric_depths, config.d_rel_min, config.d_rel_max)

    # 5. Enhance frames
    enhanced_frames = []
    for i, f in enumerate(frames):
        enhanced = f.copy()
        enhanced["depth_proxy"] = proxy_depths[i] if i < len(proxy_depths) else 2.0
        enhanced["depth_blended"] = blended_depths[i] if i < len(blended_depths) else 2.0
        enhanced["d_rel"] = d_rel_values[i] if i < len(d_rel_values) else 0.5
        if proxy_stability:
            enhanced["proxy_stability"] = proxy_stability[i] if i < len(proxy_stability) else 1.0
        enhanced_frames.append(enhanced)

    return enhanced_frames


class AdaptiveDepthEstimator:
    """
    Wrapper for adaptive depth estimation during tracking.

    Maintains state to compute adaptive stride on-the-fly.
    """

    def __init__(self, config: DepthConfig = None):
        self.config = config or DepthConfig()
        self.recent_depths: List[float] = []
        self.recent_bbox_areas: List[float] = []
        self.frame_count = 0
        self.last_depth_frame = -1
        self._current_stride = self.config.default_stride

    def should_estimate_depth(self, frame_idx: int) -> bool:
        """Check if we should estimate depth for this frame."""
        if not self.config.use_adaptive_stride:
            # Fixed stride
            return (frame_idx - self.last_depth_frame) >= self.config.default_stride

        # Adaptive stride
        frames_since_last = frame_idx - self.last_depth_frame
        return frames_since_last >= self._current_stride

    def update(self, depth_m: float, bbox_area: float, frame_idx: int):
        """Update state after depth estimation."""
        self.recent_depths.append(depth_m)
        self.recent_bbox_areas.append(bbox_area)
        self.last_depth_frame = frame_idx
        self.frame_count += 1

        # Keep only recent history
        max_history = 30
        if len(self.recent_depths) > max_history:
            self.recent_depths = self.recent_depths[-max_history:]
            self.recent_bbox_areas = self.recent_bbox_areas[-max_history:]

        # Recompute adaptive stride
        if self.config.use_adaptive_stride:
            self._current_stride = compute_adaptive_depth_stride(
                self.recent_depths, self.recent_bbox_areas, self.config
            )

    @property
    def current_stride(self) -> int:
        return self._current_stride

    def get_stats(self) -> Dict:
        """Get statistics for debugging."""
        if not self.recent_depths:
            return {"stride": self._current_stride, "depth_var": 0, "bbox_change": 0}

        depth_var = float(np.var(self.recent_depths))

        if len(self.recent_bbox_areas) >= 2:
            areas = np.array(self.recent_bbox_areas)
            changes = np.abs(np.diff(areas) / (areas[:-1] + 1e-8))
            bbox_change = float(np.mean(changes))
        else:
            bbox_change = 0.0

        return {
            "stride": self._current_stride,
            "depth_var": depth_var,
            "bbox_change": bbox_change,
            "num_samples": len(self.recent_depths),
        }

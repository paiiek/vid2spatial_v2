"""
Video Processing Utilities for Vid2Spatial.

Provides:
1. Scene cut detection (histogram diff, frame diff)
2. Camera zoom detection (optical flow scale, bbox rate)
3. Frame quality assessment

These utilities improve tracking robustness for real-world videos.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class SceneCutConfig:
    """Configuration for scene cut detection."""
    # Histogram-based detection
    hist_threshold: float = 0.5  # Correlation threshold (below = cut)

    # Frame difference-based detection
    frame_diff_threshold: float = 0.3  # Normalized diff threshold (above = cut)

    # Combined detection (use both methods)
    use_histogram: bool = True
    use_frame_diff: bool = True
    require_both: bool = False  # True = AND, False = OR


@dataclass
class ZoomConfig:
    """Configuration for zoom detection."""
    # BBox area change rate threshold
    bbox_area_rate_threshold: float = 0.3  # 30% change = zoom

    # Global motion threshold (optical flow)
    global_motion_threshold: float = 0.15  # Normalized motion

    # Window for temporal smoothing
    window_size: int = 5


def compute_histogram(frame: np.ndarray, bins: int = 64) -> np.ndarray:
    """Compute normalized grayscale histogram."""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-10)  # Normalize
    return hist


def histogram_correlation(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute correlation between two histograms (1.0 = identical)."""
    return float(cv2.compareHist(
        hist1.astype(np.float32),
        hist2.astype(np.float32),
        cv2.HISTCMP_CORREL
    ))


def frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute normalized frame difference (0.0 = identical, 1.0 = completely different)."""
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray1, gray2 = frame1, frame2

    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)

    # Normalize by max possible difference
    normalized_diff = diff.mean() / 255.0
    return float(normalized_diff)


def detect_scene_cut(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    config: SceneCutConfig = None,
    prev_hist: Optional[np.ndarray] = None,
) -> Tuple[bool, float, Optional[np.ndarray]]:
    """
    Detect if a scene cut occurred between two frames.

    Args:
        prev_frame: Previous frame (BGR or grayscale)
        curr_frame: Current frame (BGR or grayscale)
        config: Detection configuration
        prev_hist: Pre-computed histogram for prev_frame (optimization)

    Returns:
        (is_scene_cut, confidence, curr_hist)
        - is_scene_cut: True if scene cut detected
        - confidence: 0-1 confidence score (higher = more likely cut)
        - curr_hist: Current frame histogram (for reuse)
    """
    if config is None:
        config = SceneCutConfig()

    hist_cut = False
    frame_diff_cut = False
    confidence = 0.0
    curr_hist = None

    # Histogram-based detection
    if config.use_histogram:
        if prev_hist is None:
            prev_hist = compute_histogram(prev_frame)
        curr_hist = compute_histogram(curr_frame)

        correlation = histogram_correlation(prev_hist, curr_hist)
        hist_cut = correlation < config.hist_threshold

        # Convert correlation to confidence (1 - correlation)
        hist_confidence = 1.0 - max(0, correlation)
        confidence = max(confidence, hist_confidence)

    # Frame difference-based detection
    if config.use_frame_diff:
        diff = frame_difference(prev_frame, curr_frame)
        frame_diff_cut = diff > config.frame_diff_threshold

        # Normalize to confidence
        diff_confidence = min(1.0, diff / config.frame_diff_threshold) if config.frame_diff_threshold > 0 else 0
        confidence = max(confidence, diff_confidence)

    # Combine results
    if config.require_both:
        is_cut = hist_cut and frame_diff_cut
    else:
        is_cut = hist_cut or frame_diff_cut

    return is_cut, confidence, curr_hist


class SceneCutDetector:
    """Stateful scene cut detector for video processing."""

    def __init__(self, config: SceneCutConfig = None):
        self.config = config or SceneCutConfig()
        self.prev_frame = None
        self.prev_hist = None
        self.frame_idx = 0
        self.cuts = []  # List of (frame_idx, confidence)

    def update(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Process a new frame and detect scene cut.

        Returns:
            (is_scene_cut, confidence)
        """
        if self.prev_frame is None:
            # First frame
            self.prev_frame = frame.copy()
            self.prev_hist = compute_histogram(frame)
            self.frame_idx = 0
            return False, 0.0

        is_cut, confidence, curr_hist = detect_scene_cut(
            self.prev_frame, frame, self.config, self.prev_hist
        )

        if is_cut:
            self.cuts.append((self.frame_idx, confidence))

        # Update state
        self.prev_frame = frame.copy()
        self.prev_hist = curr_hist
        self.frame_idx += 1

        return is_cut, confidence

    def reset(self):
        """Reset detector state."""
        self.prev_frame = None
        self.prev_hist = None
        self.frame_idx = 0
        self.cuts = []


def detect_zoom(
    bbox_areas: List[float],
    config: ZoomConfig = None,
) -> Tuple[bool, float]:
    """
    Detect camera zoom from bbox area changes.

    Args:
        bbox_areas: List of recent bbox areas (newest last)
        config: Detection configuration

    Returns:
        (is_zoom, zoom_rate)
        - is_zoom: True if zoom detected
        - zoom_rate: Rate of area change (positive = zoom in, negative = zoom out)
    """
    if config is None:
        config = ZoomConfig()

    if len(bbox_areas) < 2:
        return False, 0.0

    # Use recent window
    window = bbox_areas[-config.window_size:] if len(bbox_areas) >= config.window_size else bbox_areas

    if len(window) < 2:
        return False, 0.0

    # Compute area change rate
    areas = np.array(window)
    changes = np.diff(areas) / (areas[:-1] + 1e-10)

    # Average change rate
    avg_rate = float(np.mean(changes))

    # Check if consistent direction (all positive or all negative)
    is_consistent = np.all(changes > 0) or np.all(changes < 0)

    # Detect zoom if rate exceeds threshold and direction is consistent
    is_zoom = abs(avg_rate) > config.bbox_area_rate_threshold and is_consistent

    return is_zoom, avg_rate


class ZoomDetector:
    """Stateful zoom detector for video processing."""

    def __init__(self, config: ZoomConfig = None):
        self.config = config or ZoomConfig()
        self.bbox_areas = []
        self.frame_idx = 0
        self.zooms = []  # List of (frame_idx, zoom_rate)

    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """
        Process a new bbox and detect zoom.

        Args:
            bbox: (x1, y1, x2, y2) or (x, y, w, h) format

        Returns:
            (is_zoom, zoom_rate)
        """
        # Compute area
        if len(bbox) == 4:
            # Detect format
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                # (x1, y1, x2, y2) format
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            else:
                # (x, y, w, h) format
                area = bbox[2] * bbox[3]
        else:
            area = 0

        self.bbox_areas.append(float(area))

        # Keep only recent history
        max_history = self.config.window_size * 2
        if len(self.bbox_areas) > max_history:
            self.bbox_areas = self.bbox_areas[-max_history:]

        is_zoom, zoom_rate = detect_zoom(self.bbox_areas, self.config)

        if is_zoom:
            self.zooms.append((self.frame_idx, zoom_rate))

        self.frame_idx += 1
        return is_zoom, zoom_rate

    def reset(self):
        """Reset detector state."""
        self.bbox_areas = []
        self.frame_idx = 0
        self.zooms = []


def assess_frame_quality(
    frame: np.ndarray,
    blur_threshold: float = 100.0,
    brightness_range: Tuple[float, float] = (30, 220),
) -> Tuple[bool, dict]:
    """
    Assess frame quality for reliable tracking.

    Args:
        frame: Input frame (BGR)
        blur_threshold: Laplacian variance threshold (below = blurry)
        brightness_range: (min, max) acceptable brightness

    Returns:
        (is_good_quality, metrics)
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Blur detection (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_sharp = laplacian_var >= blur_threshold

    # Brightness check
    mean_brightness = gray.mean()
    is_brightness_ok = brightness_range[0] <= mean_brightness <= brightness_range[1]

    # Contrast check
    contrast = gray.std()
    is_contrast_ok = contrast > 20  # Minimum contrast

    metrics = {
        "laplacian_var": float(laplacian_var),
        "mean_brightness": float(mean_brightness),
        "contrast": float(contrast),
        "is_sharp": is_sharp,
        "is_brightness_ok": is_brightness_ok,
        "is_contrast_ok": is_contrast_ok,
    }

    is_good = is_sharp and is_brightness_ok and is_contrast_ok
    return is_good, metrics


__all__ = [
    'SceneCutConfig',
    'ZoomConfig',
    'detect_scene_cut',
    'detect_zoom',
    'SceneCutDetector',
    'ZoomDetector',
    'assess_frame_quality',
    'compute_histogram',
    'histogram_correlation',
    'frame_difference',
]

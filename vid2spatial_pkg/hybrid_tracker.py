"""
Hybrid Video Object Tracker.

Combines the strengths of multiple trackers:
1. Grounding-DINO + SAM2: Initial object selection with text prompt (high quality mask)
2. YOLO/ByteTrack: Fast frame-to-frame tracking
3. SAM2: Periodic mask refinement

This approach provides:
- Open-vocabulary object selection (text prompt)
- Fast processing speed
- Lower memory usage (no full video in memory)
- Robust tracking with periodic refinement

Usage:
    tracker = HybridTracker()
    result = tracker.track("video.mp4", "the red ball")
"""

import os
import sys
import logging
import numpy as np
import cv2
import torch
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Import video utilities for robustness
try:
    from .video_utils import (
        SceneCutDetector, SceneCutConfig,
        ZoomDetector, ZoomConfig,
        assess_frame_quality,
    )
    VIDEO_UTILS_AVAILABLE = True
except ImportError:
    VIDEO_UTILS_AVAILABLE = False
    logger.warning("video_utils not available, scene cut/zoom detection disabled")


@dataclass
class HybridTrackingFrame:
    """Single frame tracking result."""
    frame_idx: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]  # (cx, cy)
    confidence: float
    depth_m: float = 0.0  # Metric depth in meters
    mask: Optional[np.ndarray] = None  # Only on refinement frames


@dataclass
class HybridTrackingResult:
    """Complete tracking result."""
    frames: List[HybridTrackingFrame]
    video_width: int
    video_height: int
    fps: float
    total_frames: int
    text_prompt: str
    initial_detection_conf: float
    fov_deg: float = 60.0  # Camera FOV for 3D projection

    def get_trajectory(self) -> List[Dict]:
        """Convert to vid2spatial trajectory format with 3D info."""
        from .vision import CameraIntrinsics, pixel_to_ray, ray_to_angles

        K = CameraIntrinsics(width=self.video_width, height=self.video_height, fov_deg=self.fov_deg)

        trajectory = []
        for f in self.frames:
            cx, cy = f.center
            dist_m = f.depth_m if f.depth_m > 0 else 2.0  # Fallback to 2m

            # Compute 3D position
            ray = pixel_to_ray(cx, cy, K)
            az, el = ray_to_angles(ray)

            x = ray[0] * dist_m
            y = ray[1] * dist_m
            z = ray[2] * dist_m

            trajectory.append({
                "frame": f.frame_idx,
                "cx": cx,
                "cy": cy,
                "w": f.bbox[2],
                "h": f.bbox[3],
                "x": f.bbox[0],
                "y": f.bbox[1],
                "confidence": f.confidence,
                "dist_m": dist_m,
                "az": float(az),
                "el": float(el),
                "pos_x": float(x),
                "pos_y": float(y),
                "pos_z": float(z),
            })

        return trajectory

    def get_trajectory_3d(
        self,
        smooth: bool = True,
        smooth_alpha: float = 0.3,
        stabilizer: Optional[str] = None,
        enhance_depth: bool = True,
    ) -> Dict:
        """
        Get full 3D trajectory in vid2spatial format.

        Args:
            smooth: Apply exponential smoothing to reduce jitter (default)
            smooth_alpha: EMA smoothing factor (0 = no smoothing, 1 = no memory)
            stabilizer: Optional advanced stabilizer for extreme noise:
                        - None: Use EMA smoothing (default, recommended)
                        - "kalman": Kalman filter (for very noisy/fast objects)
                        - "one_euro": One Euro filter (adaptive smoothing)
            enhance_depth: Apply depth enhancement (bbox proxy blending + d_rel)
        """
        raw_traj = self.get_trajectory()

        # Apply advanced stabilizer if requested
        if stabilizer is not None and len(raw_traj) > 1:
            try:
                from .trajectory_stabilizer import stabilize_trajectory_3d

                # Convert to frames format (include w, h, confidence for depth enhancement)
                frames_for_stabilizer = [
                    {
                        "frame": t["frame"],
                        "az": t["az"],
                        "el": t["el"],
                        "dist_m": t["dist_m"],
                        "x": t["pos_x"],
                        "y": t["pos_y"],
                        "z": t["pos_z"],
                        "w": t.get("w", 100),
                        "h": t.get("h", 100),
                        "confidence": t.get("confidence", 0.5),
                    }
                    for t in raw_traj
                ]

                stabilized = stabilize_trajectory_3d(
                    {"frames": frames_for_stabilizer},
                    method=stabilizer,
                )

                frames_data = stabilized["frames"]

                # Apply depth enhancement (bbox proxy blending with variance gating + d_rel)
                if enhance_depth and len(frames_data) > 0:
                    try:
                        from .depth_utils import process_trajectory_depth, DepthConfig

                        config = DepthConfig(
                            blend_strategy="metric_default",
                            use_bbox_proxy=True,
                            proxy_blend_by_confidence=True,
                            use_proxy_variance_gating=True,
                        )
                        frames_data = process_trajectory_depth(frames_data, config)
                    except ImportError:
                        pass

                return {
                    "intrinsics": {
                        "width": self.video_width,
                        "height": self.video_height,
                        "fov_deg": self.fov_deg,
                    },
                    "frames": frames_data,
                    "stabilizer": stabilizer,
                }
            except ImportError:
                print("[warning] trajectory_stabilizer not available, using EMA")

        # Default: EMA smoothing
        if smooth and len(raw_traj) > 1:
            # Apply exponential moving average smoothing
            smoothed = []
            prev = raw_traj[0]
            smoothed.append(prev)

            for t in raw_traj[1:]:
                smoothed_t = {
                    "frame": t["frame"],
                    "az": prev["az"] * (1 - smooth_alpha) + t["az"] * smooth_alpha,
                    "el": prev["el"] * (1 - smooth_alpha) + t["el"] * smooth_alpha,
                    "dist_m": prev["dist_m"] * (1 - smooth_alpha) + t["dist_m"] * smooth_alpha,
                    "pos_x": prev["pos_x"] * (1 - smooth_alpha) + t["pos_x"] * smooth_alpha,
                    "pos_y": prev["pos_y"] * (1 - smooth_alpha) + t["pos_y"] * smooth_alpha,
                    "pos_z": prev["pos_z"] * (1 - smooth_alpha) + t["pos_z"] * smooth_alpha,
                    # Preserve w, h, confidence for depth enhancement
                    "w": t.get("w", 100),
                    "h": t.get("h", 100),
                    "confidence": t.get("confidence", 0.5),
                }
                smoothed.append(smoothed_t)
                prev = smoothed_t

            frames_data = [
                {
                    "frame": t["frame"],
                    "az": t["az"],
                    "el": t["el"],
                    "dist_m": t["dist_m"],
                    "x": t["pos_x"],
                    "y": t["pos_y"],
                    "z": t["pos_z"],
                    "w": t.get("w", 100),
                    "h": t.get("h", 100),
                    "confidence": t.get("confidence", 0.5),
                }
                for t in smoothed
            ]
        else:
            frames_data = [
                {
                    "frame": t["frame"],
                    "az": t["az"],
                    "el": t["el"],
                    "dist_m": t["dist_m"],
                    "x": t["pos_x"],
                    "y": t["pos_y"],
                    "z": t["pos_z"],
                    "w": t.get("w", 100),
                    "h": t.get("h", 100),
                    "confidence": t.get("confidence", 0.5),
                }
                for t in raw_traj
            ]

        # Apply depth enhancement (bbox proxy blending with variance gating + d_rel)
        if enhance_depth and len(frames_data) > 0:
            try:
                from .depth_utils import process_trajectory_depth, DepthConfig

                config = DepthConfig(
                    blend_strategy="metric_default",
                    use_bbox_proxy=True,
                    proxy_blend_by_confidence=True,
                    use_proxy_variance_gating=True,
                )
                frames_data = process_trajectory_depth(frames_data, config)
            except ImportError:
                pass  # depth_utils not available, skip enhancement

        return {
            "intrinsics": {
                "width": self.video_width,
                "height": self.video_height,
                "fov_deg": self.fov_deg,
            },
            "frames": frames_data,
        }


class HybridTracker:
    """
    Hybrid tracker combining Grounding-DINO + SAM2 + YOLO.

    Strategy:
    1. Use Grounding-DINO for text-to-bbox detection
    2. Use SAM2 Image Predictor for initial mask
    3. Use YOLO+ByteTrack for fast frame-to-frame tracking
    4. Periodically refine with SAM2 (every N frames)
    """

    def __init__(
        self,
        device: str = "cuda",
        box_threshold: float = 0.25,  # Lower for small objects
        text_threshold: float = 0.2,
        refine_interval: int = 30,  # SAM2 refinement every N frames
        sam2_model: str = "facebook/sam2-hiera-small",  # Smaller for speed
        scene_type: str = "auto",  # For metric depth estimation
        fov_deg: float = 60.0,  # Camera FOV for 3D projection
        redetect_interval: int = 0,  # Re-detection interval (0 = disabled, K = every K frames)
        trajectory_source: str = "propagation",  # "propagation" or "detection"
    ):
        """
        Initialize Hybrid Tracker.

        Args:
            device: "cuda" or "cpu"
            box_threshold: Grounding-DINO confidence threshold
            text_threshold: Grounding-DINO text threshold
            refine_interval: How often to refine mask with SAM2
            sam2_model: SAM2 model for refinement
            scene_type: Scene type for depth estimation ('indoor', 'outdoor', 'auto')
            fov_deg: Camera FOV for 3D projection
            redetect_interval: Re-run detection every K frames (0 = disabled).
                               Recommended: K=5 for fast-moving objects.
                               When enabled, trajectory is derived from detection centers
                               rather than propagation, improving high-frequency tracking.
            trajectory_source: Source of trajectory centroids:
                               - "propagation": Use SAM2/YOLO propagation (default, good for slow motion)
                               - "detection": Use Grounding-DINO detection centers (better for fast motion)
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.refine_interval = refine_interval
        self.scene_type = scene_type
        self.fov_deg = fov_deg
        self.redetect_interval = redetect_interval
        self.trajectory_source = trajectory_source

        # Models
        self.grounding_dino = None
        self.sam2_image_predictor = None
        self.yolo_model = None
        self.depth_estimator = None

        # Robustness detectors (scene cut, zoom)
        self.scene_cut_detector = None
        self.zoom_detector = None
        self._init_robustness_detectors()

        # Load models
        self._load_grounding_dino()
        self._load_sam2(sam2_model)
        self._load_yolo()
        self._load_depth_estimator()

    def _init_robustness_detectors(self):
        """Initialize scene cut and zoom detectors for pipeline robustness."""
        if VIDEO_UTILS_AVAILABLE:
            self.scene_cut_detector = SceneCutDetector(SceneCutConfig(
                hist_threshold=0.5,
                frame_diff_threshold=0.3,
                use_histogram=True,
                use_frame_diff=True,
                require_both=False,  # Either method triggers detection
            ))
            self.zoom_detector = ZoomDetector(ZoomConfig(
                bbox_area_rate_threshold=0.3,
                window_size=5,
            ))
            logger.info("[hybrid] Scene cut and zoom detectors initialized")
        else:
            self.scene_cut_detector = None
            self.zoom_detector = None

    def _load_grounding_dino(self):
        """Load Grounding-DINO with improved error handling."""
        try:
            from groundingdino.util.inference import load_model, predict, load_image

            weights_dir = os.path.expanduser("~/.cache/groundingdino")
            config_path = os.path.join(weights_dir, "GroundingDINO_SwinT_OGC.py")
            weights_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")

            if not os.path.exists(weights_path):
                logger.error(
                    f"[hybrid] Grounding-DINO weights not found at {weights_path}\n"
                    "  Please download from: https://github.com/IDEA-Research/GroundingDINO\n"
                    "  Or run: python -m groundingdino.util.get_pretrained_weights"
                )
                return

            self.grounding_dino = load_model(config_path, weights_path, device=self.device)
            self._gd_predict = predict
            self._gd_load_image = load_image
            print("[hybrid] Loaded Grounding-DINO")

        except torch.cuda.OutOfMemoryError:
            logger.warning("[hybrid] CUDA OOM loading Grounding-DINO, trying CPU...")
            self._try_load_grounding_dino_cpu()
        except ImportError as e:
            logger.error(f"[hybrid] Grounding-DINO not installed: {e}\n"
                        "  Install with: pip install groundingdino")
        except Exception as e:
            logger.error(f"[hybrid] Failed to load Grounding-DINO: {e}")

    def _try_load_grounding_dino_cpu(self):
        """Fallback: load Grounding-DINO on CPU."""
        try:
            from groundingdino.util.inference import load_model, predict, load_image
            weights_dir = os.path.expanduser("~/.cache/groundingdino")
            config_path = os.path.join(weights_dir, "GroundingDINO_SwinT_OGC.py")
            weights_path = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
            self.grounding_dino = load_model(config_path, weights_path, device="cpu")
            self._gd_predict = predict
            self._gd_load_image = load_image
            self.device = "cpu"  # Update device
            print("[hybrid] Loaded Grounding-DINO on CPU (fallback)")
        except Exception as e:
            logger.error(f"[hybrid] CPU fallback also failed: {e}")

    def _load_sam2(self, model_name: str):
        """Load SAM2 Image Predictor with improved error handling."""
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.sam2_image_predictor = SAM2ImagePredictor.from_pretrained(model_name)
            print(f"[hybrid] Loaded SAM2 ({model_name})")

        except torch.cuda.OutOfMemoryError:
            logger.warning("[hybrid] CUDA OOM loading SAM2, trying smaller model...")
            self._try_load_sam2_fallback()
        except ImportError as e:
            logger.error(f"[hybrid] SAM2 not installed: {e}\n"
                        "  Install with: pip install sam2")
        except Exception as e:
            logger.error(f"[hybrid] Failed to load SAM2: {e}")

    def _try_load_sam2_fallback(self):
        """Fallback: try loading smaller SAM2 model."""
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            # Try tiny model as fallback
            self.sam2_image_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")
            print("[hybrid] Loaded SAM2 (tiny, fallback)")
        except Exception as e:
            logger.error(f"[hybrid] SAM2 fallback also failed: {e}")

    def _load_yolo(self):
        """Load YOLO for tracking with improved error handling."""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolo11n.pt")
            print("[hybrid] Loaded YOLO")
        except ImportError as e:
            logger.error(f"[hybrid] Ultralytics not installed: {e}\n"
                        "  Install with: pip install ultralytics")
        except Exception as e:
            logger.error(f"[hybrid] Failed to load YOLO: {e}")

    def _load_depth_estimator(self):
        """Load Metric Depth Estimator with improved error handling."""
        try:
            from .depth_metric import MetricDepthEstimator
            self.depth_estimator = MetricDepthEstimator(
                scene_type=self.scene_type,
                model_size="small",
                device=self.device,
            )
            print(f"[hybrid] Loaded MetricDepthEstimator (scene_type={self.scene_type})")

        except torch.cuda.OutOfMemoryError:
            logger.warning("[hybrid] CUDA OOM loading depth estimator, trying CPU...")
            self._try_load_depth_estimator_cpu()
        except ImportError as e:
            logger.error(f"[hybrid] Depth Anything V2 not available: {e}")
            self.depth_estimator = None
        except Exception as e:
            logger.error(f"[hybrid] Failed to load MetricDepthEstimator: {e}")
            self.depth_estimator = None

    def _try_load_depth_estimator_cpu(self):
        """Fallback: load depth estimator on CPU."""
        try:
            from .depth_metric import MetricDepthEstimator
            self.depth_estimator = MetricDepthEstimator(
                scene_type=self.scene_type,
                model_size="small",
                device="cpu",
            )
            print("[hybrid] Loaded MetricDepthEstimator on CPU (fallback)")
        except Exception as e:
            logger.error(f"[hybrid] Depth estimator CPU fallback failed: {e}")
            self.depth_estimator = None

    def estimate_depth(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
        bbox: Tuple[int, int, int, int],
    ) -> float:
        """
        Estimate metric depth at object center.

        Args:
            frame: BGR image
            cx, cy: Object center
            bbox: Bounding box (x, y, w, h)

        Returns:
            Depth in meters
        """
        if self.depth_estimator is None:
            return 2.0  # Fallback

        try:
            x, y, w, h = bbox
            depth_m = self.depth_estimator.infer_at_point(
                frame, cx, cy,
                bbox_size=(w, h),
                method="median"
            )
            return float(np.clip(depth_m, 0.3, 50.0))
        except Exception as e:
            print(f"[hybrid] Depth estimation failed: {e}")
            return 2.0

    def detect_with_text(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> Optional[Tuple[Tuple[int, int, int, int], float, str]]:
        """
        Detect object with text prompt using Grounding-DINO.

        Returns:
            (bbox, confidence, label) or None
        """
        if self.grounding_dino is None:
            return None

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, image)
            temp_path = f.name

        try:
            image_source, image_tensor = self._gd_load_image(temp_path)

            if not text_prompt.endswith("."):
                text_prompt = text_prompt + "."

            boxes, logits, phrases = self._gd_predict(
                model=self.grounding_dino,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

        finally:
            os.unlink(temp_path)

        if len(boxes) == 0:
            return None

        H, W = image.shape[:2]

        # Convert from cxcywh (normalized) to xyxy (pixel)
        # boxes format from Grounding-DINO: (cx, cy, w, h) normalized [0, 1]
        from groundingdino.util.box_ops import box_cxcywh_to_xyxy
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)  # Convert to (x1, y1, x2, y2) normalized
        boxes_pixel = boxes_xyxy.cpu().numpy() * np.array([W, H, W, H])

        # Get best detection
        best_idx = logits.argmax()
        x1, y1, x2, y2 = map(int, boxes_pixel[best_idx])
        conf = float(logits[best_idx])
        label = phrases[best_idx]

        return ((x1, y1, x2 - x1, y2 - y1), conf, label)

    def check_scene_cut(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Check if a scene cut occurred.

        Returns:
            (is_scene_cut, confidence)
        """
        if self.scene_cut_detector is None:
            return False, 0.0
        return self.scene_cut_detector.update(frame)

    def check_zoom(self, bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
        """
        Check if camera zoom is occurring.

        Args:
            bbox: (x, y, w, h) bounding box

        Returns:
            (is_zoom, zoom_rate) - positive rate = zoom in, negative = zoom out
        """
        if self.zoom_detector is None:
            return False, 0.0
        return self.zoom_detector.update(bbox)

    def reset_tracking_state(self):
        """Reset tracking state after scene cut or re-initialization."""
        if self.scene_cut_detector:
            self.scene_cut_detector.reset()
        if self.zoom_detector:
            self.zoom_detector.reset()
        logger.info("[hybrid] Tracking state reset")

    def get_sam2_mask(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """Get SAM2 mask from bbox."""
        if self.sam2_image_predictor is None:
            return None

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with torch.inference_mode():
                self.sam2_image_predictor.set_image(image_rgb)

                x, y, w, h = bbox
                box = np.array([[x, y, x + w, y + h]])

                masks, scores, _ = self.sam2_image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,
                    multimask_output=True,
                )

            best_idx = scores.argmax()
            return masks[best_idx].astype(np.uint8)

        except Exception as e:
            print(f"[hybrid] SAM2 mask failed: {e}")
            return None

    def track(
        self,
        video_path: str,
        text_prompt: str,
        sample_stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        depth_stride: int = 5,  # Estimate depth every N frames for speed
        tracking_method: str = "sam2",  # "sam2", "yolo", "auto", "redetect", or "hybrid"
        redetect_interval: Optional[int] = None,  # Override instance setting
        ema_alpha: float = 0.3,  # EMA smoothing for hybrid mode
        use_kalman: bool = False,  # Use Kalman filter instead of EMA
        estimate_depth: bool = True,  # Deprecated, kept for backward compatibility
        detect_scene_cuts: bool = True,  # Enable scene cut detection
        detect_zoom: bool = True,  # Enable zoom detection for bbox proxy protection
    ) -> HybridTrackingResult:
        """
        Track object in video using hybrid approach with metric depth.

        Depth estimation is always enabled for proper 3D spatial audio rendering.

        Args:
            video_path: Path to video
            text_prompt: Object description
            sample_stride: Process every Nth frame
            start_frame: Starting frame
            end_frame: Ending frame
            depth_stride: Estimate depth every N frames (for speed)
            tracking_method: Tracking strategy:
                - "sam2": SAM2 propagation (accurate for slow motion, default)
                - "yolo": YOLO+ByteTrack (fast, less accurate)
                - "auto": YOLO with SAM2 fallback
                - "redetect": K-frame re-detection with interpolation (best for fast motion)
                - "adaptive_k": Adaptive K-frame with confidence-based re-detection (recommended)
                - "hybrid": DINO K-frame + SAM2 propagation + EMA/Kalman (experimental)
            redetect_interval: Override instance redetect_interval for this call
            ema_alpha: EMA smoothing factor for hybrid mode (0-1, lower = smoother)
            use_kalman: Use Kalman filter instead of EMA for hybrid mode
            estimate_depth: Deprecated, depth is always estimated
            detect_scene_cuts: Enable scene cut detection and tracker reset
            detect_zoom: Enable zoom detection to protect bbox proxy

        Returns:
            HybridTrackingResult with 3D trajectory
        """
        # Depth estimation is always enabled
        if not estimate_depth:
            print("[hybrid] Warning: estimate_depth=False is deprecated, depth will be estimated anyway")
        # Use instance setting if not overridden
        redetect_k = redetect_interval if redetect_interval is not None else self.redetect_interval
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame is None:
            end_frame = total_frames

        # Step 1: Get first frame and detect with text
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read first frame")

        print(f"[hybrid] Detecting: '{text_prompt}'")
        detection = self.detect_with_text(first_frame, text_prompt)

        if detection is None:
            # Fallback: try YOLO with class name parsing
            detection = self._fallback_yolo_detect(first_frame, text_prompt)
            if detection is None:
                raise RuntimeError(f"No object found matching: '{text_prompt}'")

        init_bbox, init_conf, init_label = detection
        print(f"[hybrid] Found: {init_label} (conf={init_conf:.2f})")

        # Step 2: Get initial SAM2 mask
        init_mask = self.get_sam2_mask(first_frame, init_bbox)

        # Step 3: Track based on selected method
        # Use redetect mode if specified or if redetect_interval is set
        if tracking_method == "adaptive_k":
            # Adaptive K mode: motion-based triggered re-detection
            print(f"[hybrid] Tracking with ADAPTIVE K mode...")
            frames = self._track_with_adaptive_k(
                video_path, text_prompt, init_bbox,
                sample_stride, start_frame, end_frame,
                k_min=2, k_max=15,
                velocity_threshold_fast=30.0,
                velocity_threshold_slow=5.0,
            )
        elif tracking_method == "hybrid":
            # Hybrid mode: DINO K-frame + SAM2 propagation + EMA/Kalman
            k = redetect_k if redetect_k > 0 else 5
            stab = "Kalman" if use_kalman else f"EMA(α={ema_alpha})"
            print(f"[hybrid] Tracking with HYBRID mode (DINO K={k} + SAM2 + {stab})...")
            frames = self._track_with_hybrid_sam2_dino(
                video_path, text_prompt, init_bbox, init_mask, k,
                sample_stride, start_frame, end_frame,
                ema_alpha=ema_alpha, use_kalman=use_kalman
            )
        elif tracking_method == "redetect" or (redetect_k > 0 and tracking_method != "sam2" and tracking_method != "yolo"):
            # K-frame re-detection mode (best for fast motion)
            k = redetect_k if redetect_k > 0 else 5  # Default K=5
            print(f"[hybrid] Tracking with K-frame re-detection (K={k})...")
            frames = self._track_with_redetection(
                video_path, text_prompt, init_bbox, k, sample_stride, start_frame, end_frame
            )
        elif tracking_method == "sam2" and self.sam2_image_predictor is not None:
            # SAM2 tracking (accurate, slower)
            print(f"[hybrid] Tracking with SAM2 (accurate mode)...")
            frames = self._track_with_sam2_propagation(
                video_path, init_bbox, init_mask, sample_stride, start_frame, end_frame
            )
        elif tracking_method == "yolo":
            # YOLO tracking (fast, less accurate for small objects)
            print(f"[hybrid] Tracking with YOLO (fast mode)...")
            frames = self._track_with_yolo(
                video_path, init_bbox, init_label, sample_stride, start_frame, end_frame
            )
        else:
            # Auto mode: Try YOLO first, fallback to SAM2 if quality is low
            print(f"[hybrid] Tracking with YOLO...")
            frames = self._track_with_yolo(
                video_path, init_bbox, init_label, sample_stride, start_frame, end_frame
            )

            # Check tracking quality - if too many low confidence, try SAM2
            low_conf_count = sum(1 for f in frames if f.confidence < 0.4)
            low_conf_pct = low_conf_count / max(len(frames), 1) * 100

            if low_conf_pct > 50 and self.sam2_image_predictor is not None:
                print(f"[hybrid] YOLO tracking quality low ({low_conf_pct:.0f}% low confidence)")
                print(f"[hybrid] Falling back to SAM2 propagation...")
                sam2_frames = self._track_with_sam2_propagation(
                    video_path, init_bbox, init_mask, sample_stride, start_frame, end_frame
                )
                if sam2_frames:
                    frames = sam2_frames

        # Step 4: Estimate metric depth (always enabled for proper 3D audio)
        if self.depth_estimator is not None:
            print(f"[hybrid] Estimating metric depth (stride={depth_stride})...")
            frames = self._estimate_depths(video_path, frames, depth_stride)
        else:
            print("[hybrid] Warning: depth_estimator not loaded, using default depth=2.0m")

        cap.release()

        return HybridTrackingResult(
            frames=frames,
            video_width=W,
            video_height=H,
            fps=fps,
            total_frames=total_frames,
            text_prompt=text_prompt,
            initial_detection_conf=init_conf,
            fov_deg=self.fov_deg,
        )

    def _estimate_depths(
        self,
        video_path: str,
        frames: List[HybridTrackingFrame],
        depth_stride: int,
    ) -> List[HybridTrackingFrame]:
        """Estimate metric depth for tracking frames."""
        if not frames:
            return frames

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return frames

        # Create frame index lookup
        frame_by_idx = {f.frame_idx: f for f in frames}

        # Estimate depth at intervals
        depth_values = {}
        for i, f in enumerate(frames):
            if i % depth_stride == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f.frame_idx)
                ret, frame = cap.read()
                if ret:
                    depth_m = self.estimate_depth(frame, f.center[0], f.center[1], f.bbox)
                    depth_values[f.frame_idx] = depth_m

        cap.release()

        # Interpolate depths for non-estimated frames
        if depth_values:
            sorted_indices = sorted(depth_values.keys())

            for f in frames:
                if f.frame_idx in depth_values:
                    f.depth_m = depth_values[f.frame_idx]
                else:
                    # Linear interpolation
                    prev_idx = max([idx for idx in sorted_indices if idx < f.frame_idx], default=None)
                    next_idx = min([idx for idx in sorted_indices if idx > f.frame_idx], default=None)

                    if prev_idx is not None and next_idx is not None:
                        # Interpolate
                        t = (f.frame_idx - prev_idx) / (next_idx - prev_idx)
                        f.depth_m = depth_values[prev_idx] * (1 - t) + depth_values[next_idx] * t
                    elif prev_idx is not None:
                        f.depth_m = depth_values[prev_idx]
                    elif next_idx is not None:
                        f.depth_m = depth_values[next_idx]
                    else:
                        f.depth_m = 2.0  # Fallback

        return frames

    def _fallback_yolo_detect(
        self,
        image: np.ndarray,
        text_prompt: str,
    ) -> Optional[Tuple[Tuple[int, int, int, int], float, str]]:
        """Fallback detection using YOLO."""
        if self.yolo_model is None:
            return None

        # Parse class from prompt
        cls_name = self._parse_class(text_prompt)

        results = self.yolo_model(image, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                detected = self.yolo_model.names[cls_id].lower()

                if cls_name in detected or detected in cls_name:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    return ((x1, y1, x2 - x1, y2 - y1), conf, detected)

        return None

    def _parse_class(self, text_prompt: str) -> str:
        """Parse class name from text prompt."""
        common = ["person", "dog", "cat", "car", "ball", "bicycle", "bird"]
        text_lower = text_prompt.lower()
        for c in common:
            if c in text_lower:
                return c
        return "person"

    def _track_with_yolo(
        self,
        video_path: str,
        init_bbox: Tuple[int, int, int, int],
        init_label: str,
        sample_stride: int,
        start_frame: int,
        end_frame: int,
    ) -> List[HybridTrackingFrame]:
        """Track using YOLO+ByteTrack with IoU fallback."""

        # Determine YOLO class to track
        cls_to_track = self._parse_class(init_label)

        cap = cv2.VideoCapture(video_path)
        frames = []
        fidx = 0

        # Initialize tracker state
        current_bbox = init_bbox
        track_id = None
        consecutive_misses = 0
        max_misses_for_iou = 3  # After N misses, use any IoU match

        while fidx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if fidx < start_frame:
                fidx += 1
                continue

            if (fidx - start_frame) % sample_stride != 0:
                fidx += 1
                continue

            # Run YOLO tracking
            results = self.yolo_model.track(frame, persist=True, verbose=False)

            best_match = None
            best_iou = 0
            best_any_iou = 0
            best_any_match = None  # Best IoU match regardless of class

            for r in results:
                if r.boxes is None or r.boxes.id is None:
                    continue

                for i, box in enumerate(r.boxes):
                    cls_id = int(box.cls[0])
                    detected_cls = self.yolo_model.names[cls_id].lower()

                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    new_bbox = (x1, y1, x2 - x1, y2 - y1)
                    conf = float(box.conf[0])
                    tid = int(box.id[0]) if box.id is not None else None

                    # Calculate IoU
                    iou = self._calculate_iou(current_bbox, new_bbox)

                    # Track best IoU match regardless of class (for fallback)
                    if iou > best_any_iou:
                        best_any_iou = iou
                        best_any_match = (new_bbox, conf * 0.7, tid)  # Lower conf for non-class match

                    # Check class match
                    if cls_to_track not in detected_cls and detected_cls not in cls_to_track:
                        continue

                    # If we have a track_id, prefer same track
                    if track_id is not None and tid == track_id:
                        best_match = (new_bbox, conf, tid)
                        break

                    # Otherwise, find best IoU match within class
                    if iou > best_iou:
                        best_iou = iou
                        best_match = (new_bbox, conf, tid)

            # Use class match if found
            if best_match is not None:
                current_bbox, conf, track_id = best_match
                consecutive_misses = 0
            # Fallback to IoU match if too many consecutive misses
            elif best_any_match is not None and (consecutive_misses >= max_misses_for_iou or best_any_iou > 0.3):
                current_bbox, conf, track_id = best_any_match
                consecutive_misses = 0
            else:
                # Keep last known position
                conf = 0.3  # Low confidence for static position
                consecutive_misses += 1

            x, y, w, h = current_bbox
            cx, cy = x + w / 2, y + h / 2

            frames.append(HybridTrackingFrame(
                frame_idx=fidx,
                bbox=current_bbox,
                center=(cx, cy),
                confidence=conf,
                mask=None,
            ))

            fidx += 1

        cap.release()
        return frames

    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """Calculate IoU between two boxes (x,y,w,h format)."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to x1,y1,x2,y2
        ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
        bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

        # Intersection
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _track_with_hybrid_sam2_dino(
        self,
        video_path: str,
        text_prompt: str,
        init_bbox: Tuple[int, int, int, int],
        init_mask: Optional[np.ndarray],
        redetect_interval: int,
        sample_stride: int,
        start_frame: int,
        end_frame: int,
        ema_alpha: float = 0.3,
        use_kalman: bool = False,
    ) -> List[HybridTrackingFrame]:
        """
        Hybrid tracking: SAM2 propagation + DINO K-frame correction + EMA stabilization.

        Architecture:
        - Trajectory authority: DINO detection at K-frames
        - Propagation: SAM2 mask propagation between keyframes
        - Stabilization: EMA (default) or Kalman filter

        This combines:
        1. SAM2's smooth propagation between frames
        2. DINO's accurate re-detection to prevent drift
        3. EMA/Kalman smoothing for stable trajectory

        Args:
            video_path: Path to video
            text_prompt: Object description for DINO
            init_bbox: Initial bounding box
            init_mask: Initial SAM2 mask
            redetect_interval: Run DINO detection every K frames
            sample_stride: Process every Nth frame
            start_frame: Starting frame
            end_frame: Ending frame
            ema_alpha: EMA smoothing factor (0-1, lower = smoother)
            use_kalman: Use Kalman filter instead of EMA
        """
        if self.sam2_image_predictor is None:
            print("[hybrid] SAM2 not available, falling back to redetection mode")
            return self._track_with_redetection(
                video_path, text_prompt, init_bbox, redetect_interval,
                sample_stride, start_frame, end_frame
            )

        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        current_bbox = init_bbox
        prev_mask = init_mask

        # EMA state
        ema_cx, ema_cy = None, None

        # Kalman filter state (if enabled)
        if use_kalman:
            # State: [cx, cy, vx, vy]
            kalman_state = np.array([
                init_bbox[0] + init_bbox[2] / 2,
                init_bbox[1] + init_bbox[3] / 2,
                0.0, 0.0
            ], dtype=np.float64)
            kalman_P = np.eye(4) * 100  # Covariance
            kalman_Q = np.diag([1, 1, 10, 10])  # Process noise
            kalman_R = np.diag([5, 5])  # Measurement noise

        fidx = 0
        frame_count = 0
        while fidx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if fidx < start_frame:
                fidx += 1
                continue

            if (fidx - start_frame) % sample_stride != 0:
                fidx += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            is_keyframe = (frame_count % redetect_interval == 0)

            # --- Get raw position ---
            raw_cx, raw_cy = None, None
            conf = 0.5

            if is_keyframe:
                # DINO detection at keyframe
                detection = self.detect_with_text(frame, text_prompt)
                if detection:
                    bbox, det_conf, label = detection
                    x, y, w, h = bbox
                    raw_cx, raw_cy = x + w / 2, y + h / 2
                    current_bbox = bbox
                    conf = det_conf

                    # Also update SAM2 mask for better propagation
                    try:
                        with torch.inference_mode():
                            self.sam2_image_predictor.set_image(frame_rgb)
                            box = np.array([[x, y, x + w, y + h]])
                            masks, scores, _ = self.sam2_image_predictor.predict(
                                point_coords=None, point_labels=None,
                                box=box, multimask_output=True
                            )
                            best_idx = scores.argmax()
                            prev_mask = masks[best_idx].astype(np.uint8)
                    except Exception:
                        pass
            else:
                # SAM2 propagation between keyframes
                try:
                    with torch.inference_mode():
                        self.sam2_image_predictor.set_image(frame_rgb)

                        # Use previous bbox as prompt
                        x, y, w, h = current_bbox
                        box = np.array([[x, y, x + w, y + h]])

                        masks, scores, _ = self.sam2_image_predictor.predict(
                            point_coords=None, point_labels=None,
                            box=box, multimask_output=True
                        )

                        best_idx = scores.argmax()
                        mask = masks[best_idx].astype(np.uint8)
                        conf = float(scores[best_idx])

                        # Get centroid from mask
                        ys, xs = np.where(mask > 0)
                        if len(xs) > 0 and len(ys) > 0:
                            x1, x2 = xs.min(), xs.max()
                            y1, y2 = ys.min(), ys.max()
                            raw_cx = (x1 + x2) / 2
                            raw_cy = (y1 + y2) / 2
                            current_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                            prev_mask = mask
                except Exception:
                    pass

            # Fallback if no position obtained
            if raw_cx is None:
                x, y, w, h = current_bbox
                raw_cx, raw_cy = x + w / 2, y + h / 2
                conf = 0.3

            # --- Apply stabilization ---
            if use_kalman:
                # Kalman prediction
                F = np.array([
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float64)
                kalman_state = F @ kalman_state
                kalman_P = F @ kalman_P @ F.T + kalman_Q

                # Kalman update
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
                z = np.array([raw_cx, raw_cy])
                y_innov = z - H @ kalman_state
                S = H @ kalman_P @ H.T + kalman_R
                K = kalman_P @ H.T @ np.linalg.inv(S)
                kalman_state = kalman_state + K @ y_innov
                kalman_P = (np.eye(4) - K @ H) @ kalman_P

                final_cx, final_cy = kalman_state[0], kalman_state[1]
            else:
                # EMA smoothing
                if ema_cx is None:
                    ema_cx, ema_cy = raw_cx, raw_cy
                else:
                    ema_cx = ema_alpha * raw_cx + (1 - ema_alpha) * ema_cx
                    ema_cy = ema_alpha * raw_cy + (1 - ema_alpha) * ema_cy

                final_cx, final_cy = ema_cx, ema_cy

            frames.append(HybridTrackingFrame(
                frame_idx=fidx,
                bbox=current_bbox,
                center=(final_cx, final_cy),
                confidence=conf,
                mask=prev_mask if is_keyframe else None,
            ))

            fidx += 1
            frame_count += 1

        cap.release()
        return frames

    def _track_with_redetection(
        self,
        video_path: str,
        text_prompt: str,
        init_bbox: Tuple[int, int, int, int],
        redetect_interval: int,
        sample_stride: int,
        start_frame: int,
        end_frame: int,
        # Robustness parameters
        low_conf_threshold: float = 0.35,
        max_velocity_px: float = 150.0,
    ) -> List[HybridTrackingFrame]:
        """
        Track using K-frame re-detection with linear interpolation between detections.

        Two-pass approach:
        1. Pass 1: Run detection every K frames, collect keyframes
        2. Pass 2: Linear interpolation between keyframes for smooth trajectory

        Robustness features:
        - Confidence gating: low conf detection → use previous position
        - Jump reject: velocity > max_velocity_px → outlier, use previous

        Best for fast-moving objects where propagation loses amplitude.
        """
        cap = cv2.VideoCapture(video_path)

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Pass 1: Collect detection keyframes
        keyframes = []  # List of (frame_idx, cx, cy, bbox, conf)
        frame_indices_to_process = []

        # Robustness stats
        jump_reject_count = 0
        low_conf_reject_count = 0
        last_valid_cx, last_valid_cy = None, None

        fidx = 0
        while fidx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if fidx < start_frame:
                fidx += 1
                continue

            if (fidx - start_frame) % sample_stride != 0:
                fidx += 1
                continue

            frame_indices_to_process.append(fidx)

            # Detect every K frames
            frames_since_start = len(frame_indices_to_process) - 1
            should_redetect = (frames_since_start % redetect_interval == 0)

            if should_redetect:
                detection = self.detect_with_text(frame, text_prompt)

                if detection:
                    bbox, conf, label = detection
                    x, y, w, h = bbox
                    cx, cy = x + w / 2, y + h / 2

                    # === ROBUSTNESS: Confidence gating ===
                    if conf < low_conf_threshold and keyframes:
                        low_conf_reject_count += 1
                        last_kf = keyframes[-1]
                        keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.2))
                        fidx += 1
                        continue

                    # === ROBUSTNESS: Jump reject ===
                    if last_valid_cx is not None:
                        frames_gap = redetect_interval
                        jump_distance = np.sqrt((cx - last_valid_cx)**2 + (cy - last_valid_cy)**2)
                        velocity_per_frame = jump_distance / max(frames_gap, 1)

                        if velocity_per_frame > max_velocity_px:
                            jump_reject_count += 1
                            last_kf = keyframes[-1]
                            keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.25))
                            fidx += 1
                            continue

                    # Valid detection
                    keyframes.append((fidx, cx, cy, bbox, conf))
                    last_valid_cx, last_valid_cy = cx, cy

                elif keyframes:
                    # Detection failed, use last keyframe position with low confidence
                    last_kf = keyframes[-1]
                    keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.3))
                else:
                    # First frame detection failed, use init_bbox
                    x, y, w, h = init_bbox
                    cx, cy = x + w / 2, y + h / 2
                    keyframes.append((fidx, cx, cy, init_bbox, 0.3))
                    last_valid_cx, last_valid_cy = cx, cy

            fidx += 1

        cap.release()

        # Log robustness stats
        if jump_reject_count > 0 or low_conf_reject_count > 0:
            print(f"[redetect] Robustness: {jump_reject_count} jump rejects, {low_conf_reject_count} low-conf rejects")

        # Pass 2: Linear interpolation between keyframes
        if not keyframes:
            return []

        frames = []

        # Build keyframe lookup
        kf_dict = {kf[0]: kf for kf in keyframes}
        kf_indices = sorted(kf_dict.keys())

        for fidx in frame_indices_to_process:
            if fidx in kf_dict:
                # This is a keyframe - use detection result directly
                _, cx, cy, bbox, conf = kf_dict[fidx]
                frames.append(HybridTrackingFrame(
                    frame_idx=fidx,
                    bbox=bbox,
                    center=(cx, cy),
                    confidence=conf,
                    mask=None,
                ))
            else:
                # Interpolate between surrounding keyframes
                prev_kf_idx = max([k for k in kf_indices if k < fidx], default=None)
                next_kf_idx = min([k for k in kf_indices if k > fidx], default=None)

                if prev_kf_idx is not None and next_kf_idx is not None:
                    # Linear interpolation
                    prev_kf = kf_dict[prev_kf_idx]
                    next_kf = kf_dict[next_kf_idx]

                    t = (fidx - prev_kf_idx) / (next_kf_idx - prev_kf_idx)

                    cx = prev_kf[1] * (1 - t) + next_kf[1] * t
                    cy = prev_kf[2] * (1 - t) + next_kf[2] * t

                    # Interpolate bbox
                    px, py, pw, ph = prev_kf[3]
                    nx, ny, nw, nh = next_kf[3]
                    bx = int(px * (1 - t) + nx * t)
                    by = int(py * (1 - t) + ny * t)
                    bw = int(pw * (1 - t) + nw * t)
                    bh = int(ph * (1 - t) + nh * t)

                    # Interpolated confidence (slightly lower than keyframes)
                    conf = min(prev_kf[4], next_kf[4]) * 0.9

                    frames.append(HybridTrackingFrame(
                        frame_idx=fidx,
                        bbox=(bx, by, bw, bh),
                        center=(cx, cy),
                        confidence=conf,
                        mask=None,
                    ))
                elif prev_kf_idx is not None:
                    # After last keyframe - hold position
                    prev_kf = kf_dict[prev_kf_idx]
                    frames.append(HybridTrackingFrame(
                        frame_idx=fidx,
                        bbox=prev_kf[3],
                        center=(prev_kf[1], prev_kf[2]),
                        confidence=prev_kf[4] * 0.8,
                        mask=None,
                    ))
                elif next_kf_idx is not None:
                    # Before first keyframe - hold position
                    next_kf = kf_dict[next_kf_idx]
                    frames.append(HybridTrackingFrame(
                        frame_idx=fidx,
                        bbox=next_kf[3],
                        center=(next_kf[1], next_kf[2]),
                        confidence=next_kf[4] * 0.8,
                        mask=None,
                    ))

        return frames

    def _track_with_adaptive_k(
        self,
        video_path: str,
        text_prompt: str,
        init_bbox: Tuple[int, int, int, int],
        sample_stride: int,
        start_frame: int,
        end_frame: int,
        k_min: int = 2,
        k_max: int = 15,
        velocity_threshold_fast: float = 30.0,  # pixels/frame
        velocity_threshold_slow: float = 5.0,   # pixels/frame
        # Robustness parameters
        low_conf_threshold: float = 0.35,  # Below this: force immediate re-detect or reject
        max_velocity_px: float = 150.0,    # Max allowed velocity (pixels/frame) - jump reject
    ) -> List[HybridTrackingFrame]:
        """
        Track using adaptive K-frame re-detection based on motion velocity.

        Adaptive strategy:
        - Fast motion (velocity > threshold_fast): K = k_min (frequent detection)
        - Slow motion (velocity < threshold_slow): K = k_max (save compute)
        - Medium motion: linear interpolation between k_min and k_max

        Robustness features:
        - Confidence-aware gating: low conf → force re-detect or reject observation
        - Jump reject: velocity > max_velocity_px → outlier, use previous position

        Two-pass approach with RTS smoothing option.
        """
        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Pass 1: Collect keyframes with adaptive K + robustness
        keyframes = []  # List of (frame_idx, cx, cy, bbox, conf, k_used)
        frame_indices_to_process = []

        # State for adaptive K
        last_cx, last_cy = None, None
        frames_since_detection = 0
        current_k = 5  # Start with default K
        velocity_history = []  # Rolling window for velocity estimation

        # Stats for robustness
        jump_reject_count = 0
        low_conf_reject_count = 0

        fidx = 0
        while fidx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if fidx < start_frame:
                fidx += 1
                continue

            if (fidx - start_frame) % sample_stride != 0:
                fidx += 1
                continue

            frame_indices_to_process.append(fidx)
            frames_since_detection += 1

            # Decide whether to re-detect
            should_redetect = (frames_since_detection >= current_k) or (len(keyframes) == 0)

            if should_redetect:
                detection = self.detect_with_text(frame, text_prompt)

                if detection:
                    bbox, conf, label = detection
                    x, y, w, h = bbox
                    cx, cy = x + w / 2, y + h / 2

                    # === ROBUSTNESS: Confidence gating ===
                    if conf < low_conf_threshold and keyframes:
                        # Low confidence detection - reject and force re-detect next frame
                        low_conf_reject_count += 1
                        current_k = 1  # Force immediate re-detection
                        # Use previous position with low confidence marker
                        last_kf = keyframes[-1]
                        keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.2, current_k))
                        frames_since_detection = 0
                        fidx += 1
                        continue

                    # === ROBUSTNESS: Jump reject ===
                    if last_cx is not None:
                        jump_distance = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                        velocity_per_frame = jump_distance / max(frames_since_detection, 1)

                        if velocity_per_frame > max_velocity_px:
                            # Outlier detected - reject this observation
                            jump_reject_count += 1
                            current_k = 1  # Force immediate re-detection
                            # Use previous position
                            last_kf = keyframes[-1]
                            keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.25, current_k))
                            frames_since_detection = 0
                            fidx += 1
                            continue

                        # Valid detection - update velocity history
                        velocity_history.append(velocity_per_frame)

                        # Keep rolling window of last 5 velocities
                        if len(velocity_history) > 5:
                            velocity_history.pop(0)

                        # Compute adaptive K based on average velocity
                        avg_velocity = np.mean(velocity_history)

                        if avg_velocity > velocity_threshold_fast:
                            current_k = k_min
                        elif avg_velocity < velocity_threshold_slow:
                            current_k = k_max
                        else:
                            # Linear interpolation
                            t = (avg_velocity - velocity_threshold_slow) / (velocity_threshold_fast - velocity_threshold_slow)
                            current_k = int(k_max - t * (k_max - k_min))
                            current_k = max(k_min, min(k_max, current_k))

                    keyframes.append((fidx, cx, cy, bbox, conf, current_k))
                    last_cx, last_cy = cx, cy
                    frames_since_detection = 0

                elif keyframes:
                    # Detection failed, use last keyframe
                    last_kf = keyframes[-1]
                    keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.3, current_k))
                    frames_since_detection = 0
                else:
                    # First frame detection failed
                    x, y, w, h = init_bbox
                    cx, cy = x + w / 2, y + h / 2
                    keyframes.append((fidx, cx, cy, init_bbox, 0.3, current_k))
                    last_cx, last_cy = cx, cy
                    frames_since_detection = 0

            fidx += 1

        cap.release()

        if not keyframes:
            return []

        # Log robustness stats
        if jump_reject_count > 0 or low_conf_reject_count > 0:
            print(f"[adaptive_k] Robustness: {jump_reject_count} jump rejects, {low_conf_reject_count} low-conf rejects")

        # Pass 2: Linear interpolation between keyframes
        frames = []
        kf_dict = {kf[0]: kf for kf in keyframes}
        kf_indices = sorted(kf_dict.keys())

        for fidx in frame_indices_to_process:
            if fidx in kf_dict:
                kf = kf_dict[fidx]
                frames.append(HybridTrackingFrame(
                    frame_idx=fidx,
                    bbox=kf[3],
                    center=(kf[1], kf[2]),
                    confidence=kf[4],
                    mask=None,
                ))
            else:
                prev_kf_idx = max([k for k in kf_indices if k < fidx], default=None)
                next_kf_idx = min([k for k in kf_indices if k > fidx], default=None)

                if prev_kf_idx is not None and next_kf_idx is not None:
                    prev_kf = kf_dict[prev_kf_idx]
                    next_kf = kf_dict[next_kf_idx]

                    t = (fidx - prev_kf_idx) / (next_kf_idx - prev_kf_idx)

                    cx = prev_kf[1] * (1 - t) + next_kf[1] * t
                    cy = prev_kf[2] * (1 - t) + next_kf[2] * t

                    px, py, pw, ph = prev_kf[3]
                    nx, ny, nw, nh = next_kf[3]
                    bx = int(px * (1 - t) + nx * t)
                    by = int(py * (1 - t) + ny * t)
                    bw = int(pw * (1 - t) + nw * t)
                    bh = int(ph * (1 - t) + nh * t)

                    conf = min(prev_kf[4], next_kf[4]) * 0.9

                    frames.append(HybridTrackingFrame(
                        frame_idx=fidx,
                        bbox=(bx, by, bw, bh),
                        center=(cx, cy),
                        confidence=conf,
                        mask=None,
                    ))
                elif prev_kf_idx is not None:
                    prev_kf = kf_dict[prev_kf_idx]
                    frames.append(HybridTrackingFrame(
                        frame_idx=fidx,
                        bbox=prev_kf[3],
                        center=(prev_kf[1], prev_kf[2]),
                        confidence=prev_kf[4] * 0.8,
                        mask=None,
                    ))
                elif next_kf_idx is not None:
                    next_kf = kf_dict[next_kf_idx]
                    frames.append(HybridTrackingFrame(
                        frame_idx=fidx,
                        bbox=next_kf[3],
                        center=(next_kf[1], next_kf[2]),
                        confidence=next_kf[4] * 0.8,
                        mask=None,
                    ))

        # Collect stats about adaptive K usage
        k_values = [kf[5] for kf in keyframes]
        if k_values:
            avg_k = np.mean(k_values)
            print(f"[adaptive_k] Avg K={avg_k:.1f}, range=[{min(k_values)}, {max(k_values)}], keyframes={len(keyframes)}")

        return frames

    def _track_with_sam2_propagation(
        self,
        video_path: str,
        init_bbox: Tuple[int, int, int, int],
        init_mask: Optional[np.ndarray],
        sample_stride: int,
        start_frame: int,
        end_frame: int,
    ) -> List[HybridTrackingFrame]:
        """
        Track using SAM2 mask propagation.

        This is slower but more accurate for objects YOLO can't detect.
        """
        if self.sam2_image_predictor is None:
            return []

        cap = cv2.VideoCapture(video_path)
        frames = []

        # Get video info
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize with first frame
        current_bbox = init_bbox
        prev_mask = init_mask

        fidx = 0
        while fidx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if fidx < start_frame:
                fidx += 1
                continue

            if (fidx - start_frame) % sample_stride != 0:
                fidx += 1
                continue

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                with torch.inference_mode():
                    self.sam2_image_predictor.set_image(frame_rgb)

                    # Use previous bbox as prompt
                    x, y, w, h = current_bbox
                    box = np.array([[x, y, x + w, y + h]])

                    masks, scores, _ = self.sam2_image_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box,
                        multimask_output=True,
                    )

                    # Get best mask
                    best_idx = scores.argmax()
                    mask = masks[best_idx].astype(np.uint8)
                    conf = float(scores[best_idx])

                    # Get bbox from mask
                    ys, xs = np.where(mask > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        x1, x2 = xs.min(), xs.max()
                        y1, y2 = ys.min(), ys.max()
                        current_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                    else:
                        # Keep previous bbox
                        x, y, w, h = current_bbox
                        cx, cy = x + w / 2, y + h / 2
                        conf = 0.3

                    frames.append(HybridTrackingFrame(
                        frame_idx=fidx,
                        bbox=current_bbox,
                        center=(cx, cy),
                        confidence=conf,
                        mask=mask,
                    ))

                    prev_mask = mask

            except Exception as e:
                # Fallback to previous position
                x, y, w, h = current_bbox
                cx, cy = x + w / 2, y + h / 2

                frames.append(HybridTrackingFrame(
                    frame_idx=fidx,
                    bbox=current_bbox,
                    center=(cx, cy),
                    confidence=0.3,
                    mask=None,
                ))

            fidx += 1

        cap.release()
        return frames


def create_hybrid_tracker(device: str = "cuda", **kwargs) -> HybridTracker:
    """Factory function."""
    return HybridTracker(device=device, **kwargs)


__all__ = [
    "HybridTracker",
    "HybridTrackingResult",
    "HybridTrackingFrame",
    "create_hybrid_tracker",
]

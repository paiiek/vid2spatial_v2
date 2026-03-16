"""
Depth Anything V2 Metric Depth Estimation

상대적 depth가 아닌 **실제 미터 단위** depth 추정

Models:
- Indoor (Hypersim): max_depth=20m, 실내 장면에 최적화
- Outdoor (VKITTI): max_depth=80m, 실외 장면에 최적화

Usage:
    depth_model = MetricDepthEstimator(scene_type="indoor")
    depth_meters = depth_model.infer(frame)  # Returns depth in meters
"""

import sys
import os
import numpy as np
import cv2
from typing import Optional, Tuple, Literal
from pathlib import Path


class MetricDepthEstimator:
    """
    Depth Anything V2 Metric Depth model.

    Returns actual distance in meters, not relative depth.
    """

    # Checkpoint URLs
    CHECKPOINT_URLS = {
        "indoor": {
            "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth",
            "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth",
            "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth",
        },
        "outdoor": {
            "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Small/resolve/main/depth_anything_v2_metric_vkitti_vits.pth",
            "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Base/resolve/main/depth_anything_v2_metric_vkitti_vitb.pth",
            "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth",
        }
    }

    MODEL_CONFIGS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    def __init__(
        self,
        scene_type: Literal["indoor", "outdoor", "auto"] = "auto",
        model_size: Literal["small", "base", "large"] = "small",
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize Metric Depth Estimator.

        Args:
            scene_type: "indoor" (max 20m), "outdoor" (max 80m), or "auto" (detect)
            model_size: "small", "base", or "large"
            device: "cuda" or "cpu"
            checkpoint_dir: Directory to store checkpoints
        """
        self.scene_type = scene_type
        self.model_size = model_size
        self.device = device
        self.checkpoint_dir = checkpoint_dir or "/home/seung/Depth-Anything-V2/metric_depth/checkpoints"

        # Map size to encoder name
        self.encoder = {"small": "vits", "base": "vitb", "large": "vitl"}[model_size]

        # Max depth based on scene type
        self.max_depth = 20 if scene_type == "indoor" else 80

        # Add metric_depth to path
        metric_depth_path = "/home/seung/Depth-Anything-V2/metric_depth"
        if os.path.exists(metric_depth_path) and metric_depth_path not in sys.path:
            sys.path.insert(0, metric_depth_path)

        self.model = None
        self.indoor_model = None
        self.outdoor_model = None

        self._load_model()

    def _load_model(self):
        """Load metric depth model."""
        try:
            import torch
            from depth_anything_v2.dpt import DepthAnythingV2

            if self.scene_type == "auto":
                # Load both models for auto-detection
                print("[depth_metric] Loading both indoor and outdoor models for auto-detection...")
                self._load_single_model("indoor")
                self.indoor_model = self.model
                self._load_single_model("outdoor")
                self.outdoor_model = self.model
                self.model = self.indoor_model  # Default to indoor
            else:
                self._load_single_model(self.scene_type)

        except Exception as e:
            print(f"[depth_metric] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def _load_single_model(self, scene_type: str):
        """Load a single model for specific scene type."""
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2

        max_depth = 20 if scene_type == "indoor" else 80
        dataset = "hypersim" if scene_type == "indoor" else "vkitti"

        # Build model config with max_depth
        config = {**self.MODEL_CONFIGS[self.encoder], 'max_depth': max_depth}

        # Create model
        self.model = DepthAnythingV2(**config)

        # Load checkpoint
        checkpoint_name = f"depth_anything_v2_metric_{dataset}_{self.encoder}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        if not os.path.exists(checkpoint_path):
            print(f"[depth_metric] Downloading {scene_type} checkpoint...")
            self._download_checkpoint(scene_type, checkpoint_path)

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[depth_metric] Loaded {scene_type} model ({self.encoder}, max_depth={max_depth}m)")

    def _download_checkpoint(self, scene_type: str, checkpoint_path: str):
        """Download checkpoint from HuggingFace."""
        import urllib.request

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        url = self.CHECKPOINT_URLS[scene_type][self.encoder]

        print(f"[depth_metric] Downloading from {url}")
        urllib.request.urlretrieve(url, checkpoint_path)
        print(f"[depth_metric] Saved to {checkpoint_path}")

    def _detect_scene_type(self, image: np.ndarray) -> str:
        """
        Auto-detect if scene is indoor or outdoor.

        Based on:
        - Sky detection (top region brightness + low saturation)
        - Green vegetation ratio
        - Depth distribution from relative model
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Sky detection in top 20%
        top_h = image.shape[0] // 5
        top_region_s = s[:top_h, :]
        top_region_v = v[:top_h, :]

        # Sky: high brightness, low saturation
        sky_mask = (top_region_v > 150) & (top_region_s < 80)
        sky_ratio = np.mean(sky_mask)

        # Green vegetation
        green_mask = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 50)
        green_ratio = np.mean(green_mask)

        # Decision
        outdoor_score = sky_ratio * 0.6 + green_ratio * 0.4

        if outdoor_score > 0.15:
            return "outdoor"
        else:
            return "indoor"

    def infer(self, image: np.ndarray, return_relative: bool = False) -> np.ndarray:
        """
        Estimate metric depth from image.

        Args:
            image: BGR image (H, W, 3)
            return_relative: If True, also return relative depth [0, 1]

        Returns:
            depth: Depth map in meters (H, W)
        """
        if self.model is None:
            print("[depth_metric] Model not loaded, returning fallback")
            h, w = image.shape[:2]
            return np.ones((h, w), dtype=np.float32) * 2.0

        try:
            import torch

            # Auto scene detection
            if self.scene_type == "auto":
                detected = self._detect_scene_type(image)
                if detected == "indoor" and self.indoor_model is not None:
                    model = self.indoor_model
                    max_depth = 20
                else:
                    model = self.outdoor_model if self.outdoor_model else self.indoor_model
                    max_depth = 80
            elif self.scene_type == "outdoor" and self.outdoor_model is not None:
                # Use outdoor model if available (from auto-init)
                model = self.outdoor_model
                max_depth = 80
            elif self.scene_type == "indoor" and self.indoor_model is not None:
                # Use indoor model if available (from auto-init)
                model = self.indoor_model
                max_depth = 20
            else:
                model = self.model
                max_depth = self.max_depth

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Inference - returns depth in meters directly
            with torch.no_grad():
                depth_meters = model.infer_image(image_rgb)

            # Clip to valid range
            depth_meters = np.clip(depth_meters, 0.1, max_depth)

            if return_relative:
                depth_relative = depth_meters / max_depth
                return depth_meters.astype(np.float32), depth_relative.astype(np.float32)

            return depth_meters.astype(np.float32)

        except Exception as e:
            print(f"[depth_metric] Inference failed: {e}")
            h, w = image.shape[:2]
            return np.ones((h, w), dtype=np.float32) * 2.0

    def infer_at_point(
        self,
        image: np.ndarray,
        cx: float,
        cy: float,
        bbox_size: Optional[Tuple[float, float]] = None,
        method: str = "median"
    ) -> float:
        """
        Get metric depth at specific point/region.

        Args:
            image: BGR image
            cx, cy: Center point coordinates
            bbox_size: Optional (width, height) for region averaging
            method: "median", "mean", or "min"

        Returns:
            depth_m: Depth in meters at the point
        """
        depth_map = self.infer(image)

        h, w = depth_map.shape
        cx_int = int(np.clip(cx, 0, w - 1))
        cy_int = int(np.clip(cy, 0, h - 1))

        if bbox_size is None:
            # Single point
            return float(depth_map[cy_int, cx_int])

        # Region average
        bw, bh = bbox_size
        x0 = int(max(0, cx - bw * 0.25))
        x1 = int(min(w, cx + bw * 0.25))
        y0 = int(max(0, cy - bh * 0.25))
        y1 = int(min(h, cy + bh * 0.25))

        region = depth_map[y0:y1, x0:x1]

        if region.size == 0:
            return float(depth_map[cy_int, cx_int])

        if method == "median":
            return float(np.median(region))
        elif method == "mean":
            return float(np.mean(region))
        elif method == "min":
            return float(np.min(region))
        else:
            return float(np.median(region))


def create_metric_depth_backend(
    scene_type: str = "auto",
    model_size: str = "small",
    device: str = "cuda",
) -> callable:
    """
    Create metric depth estimation function for vision pipeline.

    Args:
        scene_type: "indoor", "outdoor", or "auto"
        model_size: "small", "base", or "large"
        device: "cuda" or "cpu"

    Returns:
        depth_fn: Function that takes image and returns depth in meters
    """
    estimator = MetricDepthEstimator(
        scene_type=scene_type,
        model_size=model_size,
        device=device,
    )

    def depth_fn(image: np.ndarray) -> np.ndarray:
        return estimator.infer(image)

    return depth_fn


__all__ = [
    "MetricDepthEstimator",
    "create_metric_depth_backend",
]

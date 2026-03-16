"""
Metric3D v2 Depth Estimation Backend

Single-model metric depth (no indoor/outdoor split needed).
Uses canonical camera space + focal length de-canonicalization.

Models (via torch.hub):
- metric3d_vit_small  (~150MB, fast)
- metric3d_vit_large  (~1.65GB, accurate)
- metric3d_vit_giant2 (~5.51GB, best)

Reference: Hu et al., "Metric3D v2: A Versatile Monocular Geometric Foundation Model"
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class Metric3DEstimator:
    """
    Metric3D v2 depth estimator via torch.hub.

    Returns actual distance in meters without requiring scene-type selection.
    """

    MODEL_MAP = {
        "small": "metric3d_vit_small",
        "large": "metric3d_vit_large",
        "giant": "metric3d_vit_giant2",
    }

    # ViT models expect (616, 1064) input
    INPUT_SIZE = (616, 1064)

    # ImageNet normalization (0-255 scale)
    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        focal_length_px: Optional[float] = None,
    ):
        """
        Args:
            model_size: "small", "large", or "giant"
            device: "cuda" or "cpu"
            focal_length_px: Known focal length in pixels. If None, heuristic used.
        """
        self.model_size = model_size
        self.device = device
        self.focal_length_px = focal_length_px
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load model via torch.hub."""
        import torch

        model_name = self.MODEL_MAP.get(self.model_size, "metric3d_vit_small")
        print(f"[metric3d] Loading {model_name} via torch.hub...")

        try:
            self.model = torch.hub.load(
                "yvanyin/metric3d",
                model_name,
                pretrain=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[metric3d] Loaded {model_name} on {self.device}")
        except Exception as e:
            print(f"[metric3d] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def _get_focal_length(self, h: int, w: int) -> float:
        """Get focal length in pixels. Use provided or heuristic."""
        if self.focal_length_px is not None:
            return self.focal_length_px
        # Heuristic: assume ~60° horizontal FOV
        return max(h, w)

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Estimate metric depth from BGR image.

        Args:
            image_bgr: (H, W, 3) BGR uint8

        Returns:
            depth_meters: (H, W) float32, depth in meters
        """
        if self.model is None:
            h, w = image_bgr.shape[:2]
            return np.ones((h, w), dtype=np.float32) * 2.0

        import torch

        orig_h, orig_w = image_bgr.shape[:2]
        target_h, target_w = self.INPUT_SIZE

        # BGR → RGB
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Keep-aspect-ratio resize
        scale = min(target_h / orig_h, target_w / orig_w)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Scale focal length by the same ratio
        focal = self._get_focal_length(orig_h, orig_w) * scale

        # Pad to target size with ImageNet mean
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        canvas = np.full((target_h, target_w, 3), 0, dtype=np.float32)
        canvas[:] = self.MEAN  # fill with ImageNet mean
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = rgb_resized

        # Normalize
        canvas = (canvas - self.MEAN) / self.STD

        # HWC → CHW → batch
        tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).unsqueeze(0).float()
        tensor = tensor.to(self.device)

        # Inference
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.model.inference(
                {"input": tensor}
            )

        # Extract depth (canonical space)
        depth_canonical = pred_depth.squeeze().cpu().numpy()

        # Unpad
        depth_unpadded = depth_canonical[
            pad_top:pad_top + new_h,
            pad_left:pad_left + new_w,
        ]

        # Resize to original resolution
        depth_orig = cv2.resize(
            depth_unpadded,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # De-canonicalize: metric_depth = canonical_depth * (focal / 1000.0)
        depth_meters = depth_orig * (focal / 1000.0)

        # Clamp to reasonable range
        depth_meters = np.clip(depth_meters, 0.1, 300.0)

        return depth_meters.astype(np.float32)

    def infer_at_point(
        self,
        image_bgr: np.ndarray,
        cx: float,
        cy: float,
        bbox_size: Optional[Tuple[float, float]] = None,
        method: str = "median",
    ) -> float:
        """Get metric depth at specific point/region."""
        depth_map = self.infer(image_bgr)

        h, w = depth_map.shape
        cx_int = int(np.clip(cx, 0, w - 1))
        cy_int = int(np.clip(cy, 0, h - 1))

        if bbox_size is None:
            return float(depth_map[cy_int, cx_int])

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
        return float(np.median(region))


__all__ = ["Metric3DEstimator"]

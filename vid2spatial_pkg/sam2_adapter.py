"""SAM2/SAM adapter.

This module tries to integrate Segment Anything 2 (SAM2) if available, else
falls back to SAM (v1). If neither is available, raise a clear error.

API
- build_sam2_predictor(...) -> callable (frame_bgr, bbox_xywh) -> mask_uint8
  The callable matches vision.compute_trajectory_3d(sam2_mask_fn=...).
"""
from typing import Tuple, Callable, Optional
import numpy as np


def _build_sam1_predictor(ckpt: Optional[str] = None):
    try:
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    except Exception as e:
        return None
    # pick a default model type if not specified
    model_type = "vit_b"
    try:
        sam = sam_model_registry[model_type](checkpoint=ckpt) if ckpt else sam_model_registry[model_type](checkpoint=None)
        predictor = SamPredictor(sam)
        return predictor
    except Exception:
        return None


def build_sam2_predictor(checkpoint_path: Optional[str] = None,
                         config_path: Optional[str] = None,
                         model_id: Optional[str] = None) -> Callable[[np.ndarray, Tuple[float, float, float, float]], np.ndarray]:
    """Return a function that predicts a binary mask given (frame_bgr, bbox_xywh).

    Tries SAM2 first if importable; else SAM (v1). The returned function is deterministic
    given the same inputs.
    """
    predictor = None

    # Try SAM2 image predictor via HuggingFace model id
    try:
        if model_id is not None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
            predictor = SAM2ImagePredictor.from_pretrained(model_id)
    except Exception:
        predictor = None

    # Fallback to SAM1
    if predictor is None:
        predictor = _build_sam1_predictor(checkpoint_path)

    if predictor is None:
        raise RuntimeError("Neither SAM2 nor SAM (segment-anything) is available. Install one and retry.")

    def predict_mask(frame_bgr: np.ndarray, bbox_xywh: Tuple[float, float, float, float]) -> np.ndarray:
        # Convert to RGB for SAM
        import cv2
        x, y, w, h = bbox_xywh
        x, y, w, h = int(x), int(y), int(w), int(h)
        H, W = frame_bgr.shape[:2]
        x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)
        box = np.array([x, y, x + w, y + h])
        # SAM2ImagePredictor and SAM (v1) SamPredictor share a similar predict signature
        masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=box[None, :], multimask_output=False)
        mask = masks[0].astype(np.uint8)
        return mask

    return predict_mask


__all__ = ["build_sam2_predictor"]

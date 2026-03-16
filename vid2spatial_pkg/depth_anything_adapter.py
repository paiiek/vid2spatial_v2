"""Depth backend adapter.

Tries Depth-Anything v2 if importable, else falls back to MiDaS via torch.hub.
Exposes build_depth_predictor(device) -> callable(frame_bgr)->depth[H,W] float32 in [0,1].
"""
from typing import Optional, Callable
import numpy as np


def _build_midas(device: str = "cpu"):
    import torch
    midas = None
    last_err = None
    for name in ("DPT_Small", "MiDaS_small"):
        try:
            midas = torch.hub.load("intel-isl/MiDaS", name, trust_repo=True)
            break
        except Exception as e:
            last_err = e
    if midas is None:
        raise last_err
    midas.to(device)
    midas.eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = getattr(transforms, 'small_transform', transforms.default_transform)
    def predict(frame_bgr):
        import cv2, torch
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = transform(frame_rgb).to(device)
        with torch.no_grad():
            pred = midas(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=frame_rgb.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
            d = pred.cpu().numpy().astype(np.float32)
        d -= d.min();
        if d.max() > 1e-8:
            d /= d.max()
        return d
    return predict


def _build_depth_anything_v2(device: str = "cpu", model_size: str = "small"):
    """
    Build Depth Anything V2 predictor.

    Args:
        device: 'cpu' or 'cuda'
        model_size: 'small', 'base', or 'large'

    Returns:
        Depth prediction function
    """
    import sys
    _DA_ROOT = "/home/seung/Depth-Anything-V2"
    if _DA_ROOT not in sys.path:
        sys.path.insert(0, _DA_ROOT)
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        raise ImportError(
            "Depth Anything V2 not installed. Install with:\n"
            "  pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git"
        )

    import torch
    import cv2

    # Model configs
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Map size to encoder
    encoder_map = {
        'small': 'vits',
        'base': 'vitb',
        'large': 'vitl',
        'giant': 'vitg',
    }

    encoder = encoder_map.get(model_size, 'vits')

    # Initialize model
    model = DepthAnythingV2(**model_configs[encoder])

    # Try to load pretrained weights
    model_urls = {
        'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
        'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
        'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
    }

    if encoder in model_urls:
        import os
        local_ckpt = f"/home/seung/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth"
        try:
            if os.path.exists(local_ckpt):
                state_dict = torch.load(local_ckpt, map_location='cpu', weights_only=True)
                print(f'[info] Loaded Depth Anything V2 {model_size} from local {local_ckpt}')
            else:
                state_dict = torch.hub.load_state_dict_from_url(
                    model_urls[encoder],
                    map_location='cpu',
                    progress=True
                )
                print(f'[info] Loaded Depth Anything V2 {model_size} pretrained weights')
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f'[warn] Failed to load pretrained weights: {e}')
            print('[warn] Using uninitialized Depth Anything V2 model')

    model.to(device)
    model.eval()

    def predict(frame_bgr: np.ndarray) -> np.ndarray:
        """
        Predict depth from BGR image.

        Args:
            frame_bgr: Input image in BGR format (H, W, 3) uint8

        Returns:
            Depth map normalized to [0, 1], float32 (H, W)
        """
        import torch

        # infer_image expects BGR uint8 numpy array directly
        with torch.no_grad():
            depth_np = model.infer_image(frame_bgr).astype(np.float32)

            # Normalize to [0, 1]
            depth_np -= depth_np.min()
            if depth_np.max() > 1e-8:
                depth_np /= depth_np.max()

        return depth_np

    return predict


def build_depth_predictor(
    device: Optional[str] = None,
    backend: str = "auto",
    model_size: str = "small"
) -> Callable:
    """
    Build depth predictor with automatic backend selection.

    Args:
        device: 'cpu', 'cuda', or None (auto-detect)
        backend: 'auto', 'depth_anything_v2', or 'midas'
        model_size: For Depth Anything V2: 'small', 'base', 'large', 'giant'

    Returns:
        Depth prediction function: frame_bgr (H,W,3) -> depth (H,W) float32 [0,1]
    """
    try:
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    # Try Depth Anything V2 if requested or auto
    if backend in ("auto", "depth_anything_v2"):
        try:
            print(f'[info] Attempting to load Depth Anything V2 ({model_size}) on {device}...')
            predictor = _build_depth_anything_v2(device, model_size)
            print('[info] Successfully loaded Depth Anything V2')
            return predictor
        except ImportError as e:
            if backend == "depth_anything_v2":
                # User explicitly requested it, raise error
                raise e
            else:
                # Auto mode: fall back to MiDaS
                print(f'[info] Depth Anything V2 not available, falling back to MiDaS')
        except Exception as e:
            print(f'[warn] Depth Anything V2 initialization failed: {e}')
            if backend == "depth_anything_v2":
                raise e
            else:
                print('[info] Falling back to MiDaS')

    # Fall back to or explicitly use MiDaS
    print(f'[info] Loading MiDaS on {device}...')
    return _build_midas(device)


__all__ = ["build_depth_predictor"]

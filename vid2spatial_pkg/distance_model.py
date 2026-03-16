"""
Learned distance-to-parameter mapping for spatial audio rendering.

Replaces hardcoded gain/LPF/wet curves with data-driven curves fitted
to the text2hoa training data (17,252 labeled samples).

Supports two modes:
- "poly": 3rd-degree polynomial (default, fastest, numpy only)
- "mlp":  2-layer MLP with numpy forward pass (slightly more precise)

Usage:
    model = DistanceParamModel.load("weights/distance_params_v1.npz")
    gain, lpf_hz, wet = model.predict(d_rel_array)
"""
import os
import numpy as np
from numpy.polynomial import polynomial as P
from typing import Tuple, Optional


# Default weights path (relative to package root)
_DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "weights", "distance_params_v1.npz"
)


class DistanceParamModel:
    """Learned mapping from d_rel -> (gain_lin, lpf_hz, wet).

    Both poly and mlp modes are pure numpy — no torch at inference.
    """

    def __init__(self, mode: str = "poly", **kwargs):
        self.mode = mode
        if mode == "poly":
            self.gain_coeffs = np.asarray(kwargs["gain_coeffs"], dtype=np.float64)
            self.wet_coeffs = np.asarray(kwargs["wet_coeffs"], dtype=np.float64)
            self.lpf_log_coeffs = np.asarray(kwargs["lpf_log_coeffs"], dtype=np.float64)
        elif mode == "mlp":
            self.W1 = np.asarray(kwargs["mlp_W1"], dtype=np.float64)  # [1, H]
            self.b1 = np.asarray(kwargs["mlp_b1"], dtype=np.float64)  # [H]
            self.W2 = np.asarray(kwargs["mlp_W2"], dtype=np.float64)  # [H, 3]
            self.b2 = np.asarray(kwargs["mlp_b2"], dtype=np.float64)  # [3]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def load(cls, path: Optional[str] = None) -> "DistanceParamModel":
        """Load from .npz file. Auto-detects mode from stored keys."""
        path = path or _DEFAULT_WEIGHTS
        d = np.load(path, allow_pickle=True)
        mode = str(d.get("mode", "poly"))
        if mode == "mlp" and "mlp_W1" in d:
            return cls(mode="mlp",
                       mlp_W1=d["mlp_W1"], mlp_b1=d["mlp_b1"],
                       mlp_W2=d["mlp_W2"], mlp_b2=d["mlp_b2"])
        else:
            return cls(mode="poly",
                       gain_coeffs=d["gain_coeffs"],
                       wet_coeffs=d["wet_coeffs"],
                       lpf_log_coeffs=d["lpf_log_coeffs"])

    def predict(self, d_rel: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict (gain_lin, lpf_hz, wet) from d_rel array.

        Args:
            d_rel: normalized distance [0, 1], any shape

        Returns:
            gain_lin: linear gain (clipped to [0.3, 1.1])
            lpf_hz:   LPF cutoff in Hz (clipped to [200, 20000])
            wet:      reverb wet mix (clipped to [0.0, 1.0])
        """
        d = np.clip(np.asarray(d_rel, dtype=np.float64), 0.0, 1.0)

        if self.mode == "poly":
            gain = P.polyval(d, self.gain_coeffs)
            wet = P.polyval(d, self.wet_coeffs)
            lpf = np.exp(P.polyval(d, self.lpf_log_coeffs))
        else:  # mlp
            # Forward: h = relu(d @ W1 + b1); out = h @ W2 + b2
            x = d.reshape(-1, 1)  # [N, 1]
            h = np.maximum(0, x @ self.W1 + self.b1)  # [N, H]
            out = h @ self.W2 + self.b2  # [N, 3] = [gain, log_lpf, wet]
            gain = out[:, 0].reshape(d.shape)
            lpf = np.exp(out[:, 1].reshape(d.shape))
            wet = out[:, 2].reshape(d.shape)

        gain = np.clip(gain, 0.3, 1.1).astype(np.float32)
        lpf = np.clip(lpf, 200.0, 20000.0).astype(np.float32)
        wet = np.clip(wet, 0.0, 1.0).astype(np.float32)
        return gain, lpf, wet

    def predict_gain(self, d_rel: np.ndarray) -> np.ndarray:
        """Predict gain_lin only."""
        d = np.clip(np.asarray(d_rel, dtype=np.float64), 0.0, 1.0)
        if self.mode == "poly":
            return np.clip(P.polyval(d, self.gain_coeffs), 0.3, 1.1).astype(np.float32)
        else:
            g, _, _ = self.predict(d)
            return g

    def predict_lpf(self, d_rel: np.ndarray) -> np.ndarray:
        """Predict LPF cutoff (Hz) only."""
        d = np.clip(np.asarray(d_rel, dtype=np.float64), 0.0, 1.0)
        if self.mode == "poly":
            return np.clip(np.exp(P.polyval(d, self.lpf_log_coeffs)),
                           200.0, 20000.0).astype(np.float32)
        else:
            _, l, _ = self.predict(d)
            return l

    def predict_wet(self, d_rel: np.ndarray) -> np.ndarray:
        """Predict wet only."""
        d = np.clip(np.asarray(d_rel, dtype=np.float64), 0.0, 1.0)
        if self.mode == "poly":
            return np.clip(P.polyval(d, self.wet_coeffs), 0.0, 1.0).astype(np.float32)
        else:
            _, _, w = self.predict(d)
            return w


# Module-level singleton (lazy-loaded)
_cached_models = {}  # path -> DistanceParamModel


def get_distance_model(path: Optional[str] = None,
                       mode: Optional[str] = None) -> Optional["DistanceParamModel"]:
    """Get cached model instance. Returns None if weights file not found."""
    key = path or _DEFAULT_WEIGHTS
    if mode:
        key = f"{key}:{mode}"
    if key in _cached_models:
        return _cached_models[key]
    try:
        model = DistanceParamModel.load(path)
        _cached_models[key] = model
        return model
    except (FileNotFoundError, KeyError):
        return None

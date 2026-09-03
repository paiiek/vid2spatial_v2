"""Audio-visual correlation gate.

Nothing in this pipeline ever checked that the object being tracked is the
object making the sound. The mono audio is assumed, wholesale, to belong to
whatever box the user drew, so a user can spatialise a car engine onto a
pedestrian and get a confident, entirely wrong trajectory with no warning.

This module does not attempt audio-visual segmentation. It ships the cheap,
honest version: correlate the audio envelope against the tracked object's
visual motion energy, frame by frame, and report the result as an
``av_confidence`` scalar with a warning when it is near zero. It answers
"is there any relationship at all between this sound and this object's
motion", not "which pixels are sounding".

What it cannot do
-----------------
A steady sound from a moving object (a drone at constant throttle, a running
engine) has a flat envelope and will score near zero even though the pairing
is correct. Low confidence therefore means "unverified", never "wrong".
High confidence is the informative direction: it is hard to get by accident.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "audio_envelope",
    "visual_motion_energy",
    "av_confidence",
    "AV_CONFIDENCE_WARN",
]

# Below this, the pairing is unverified and the caller is warned.
AV_CONFIDENCE_WARN = 0.15


def _standardise(x: np.ndarray) -> Optional[np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if not np.isfinite(s) or s < 1e-12:
        return None
    return (x - x.mean()) / s


def audio_envelope(audio: np.ndarray, sr: int, fps: float, n_frames: int) -> np.ndarray:
    """Per-video-frame RMS envelope of a mono signal.

    The audio is chopped into ``n_frames`` windows on the video clock, so the
    result is directly comparable to a per-frame visual series.
    """
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    if n_frames <= 0:
        return np.zeros(0)
    edges = np.linspace(0, len(audio), n_frames + 1).astype(int)
    env = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = audio[edges[i]:edges[i + 1]]
        env[i] = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    return env


def visual_motion_energy(frames: Sequence[Dict], img_w: Optional[float] = None,
                         img_h: Optional[float] = None) -> np.ndarray:
    """Per-frame motion energy of the tracked object.

    Uses the bbox centre displacement in pixels when the tracker recorded a
    bbox, and falls back to angular displacement (az/el) otherwise, so it works
    on both raw tracker output and exported trajectories. Length matches
    ``frames``; the first entry repeats the second so the series is not shifted.
    """
    n = len(frames)
    if n < 2:
        return np.zeros(max(n, 0))

    def _centre(fr: Dict):
        b = fr.get("bbox")
        if b is not None and len(b) == 4:
            x, y, w, h = (float(v) for v in b)
            return x + w / 2.0, y + h / 2.0
        return None

    have_bbox = all(_centre(fr) is not None for fr in frames)
    if have_bbox:
        pts = np.array([_centre(fr) for fr in frames], dtype=np.float64)
        if img_w and img_h:  # scale-free, so clips of different sizes compare
            pts /= np.array([float(img_w), float(img_h)])
    else:
        pts = np.array([[float(fr.get("az", 0.0)), float(fr.get("el", 0.0))]
                        for fr in frames], dtype=np.float64)

    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([d[:1], d])


def _lagged_pearson(a: np.ndarray, b: np.ndarray, max_lag: int):
    """Best Pearson r of ``a`` against ``b`` over integer lags in +/-max_lag.

    Returns (r_at_zero_lag, best_r, best_lag). A positive lag means the audio
    series had to be delayed to line up with the visual one.
    """
    za, zb = _standardise(a), _standardise(b)
    if za is None or zb is None:
        return float("nan"), float("nan"), 0
    n = len(za)
    max_lag = int(max(0, min(max_lag, n // 4)))
    r0 = float(np.dot(za, zb) / n)
    best_r, best_lag = r0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            continue
        if lag > 0:
            x, y = za[lag:], zb[:n - lag]
        else:
            x, y = za[:n + lag], zb[-lag:]
        zx, zy = _standardise(x), _standardise(y)
        if zx is None or zy is None:
            continue
        r = float(np.dot(zx, zy) / len(zx))
        if r > best_r:
            best_r, best_lag = r, lag
    return r0, best_r, best_lag


def av_confidence(audio: np.ndarray, sr: int, frames: Sequence[Dict], fps: float,
                  img_w: Optional[float] = None, img_h: Optional[float] = None,
                  max_lag_s: float = 0.5) -> Dict:
    """Score how well the audio envelope tracks the object's visual motion.

    Returns a dict suitable for embedding in a trajectory JSON:
      av_confidence  null-corrected best lagged Pearson r, in [0, 1]
      r_null         the ~2-sigma chance level subtracted, 2/sqrt(n)
      pearson        zero-lag Pearson r, in [-1, 1]
      lag_max_r      best r over the lag search
      lag_frames     the lag that achieved it
      n_frames       number of frames scored
      warning        set when the pairing is unverified, else None
    """
    n = len(frames)
    vis = visual_motion_energy(frames, img_w, img_h)
    env = audio_envelope(audio, sr, fps, n)
    out: Dict = {"n_frames": int(n), "fps": float(fps)}

    if n < 4:
        out.update(av_confidence=0.0, pearson=float("nan"), lag_max_r=float("nan"),
                   lag_frames=0,
                   warning="too few frames to correlate audio with visual motion")
        return out

    r0, best_r, best_lag = _lagged_pearson(env, vis, int(round(max_lag_s * fps)))
    if not np.isfinite(best_r):
        out.update(av_confidence=0.0, pearson=r0, lag_max_r=best_r, lag_frames=0,
                   r_null=float("nan"),
                   warning="audio or visual motion is constant; correlation undefined "
                           "(a steady sound or a static object cannot be verified)")
        return out

    # Null correction. Two unrelated series of length n already produce
    # |r| ~ 1/sqrt(n) by chance, and taking the MAXIMUM over the lag scan
    # inflates that further, so an uncorrected score reports confident
    # agreement for pure noise. Subtract a ~2-sigma null band and rescale, so
    # an unrelated pairing lands at 0 rather than at the noise floor.
    r_null = 2.0 / np.sqrt(max(n, 2))
    conf = float(np.clip((best_r - r_null) / max(1.0 - r_null, 1e-6), 0.0, 1.0))
    out.update(av_confidence=conf, pearson=float(r0), lag_max_r=float(best_r),
               lag_frames=int(best_lag), r_null=float(r_null))
    if conf < AV_CONFIDENCE_WARN:
        out["warning"] = (
            f"av_confidence {conf:.3f} < {AV_CONFIDENCE_WARN}: the audio envelope does "
            "not follow this object's motion. The pairing is UNVERIFIED -- it may be "
            "wrong, or the sound may simply be steady. Check that the tracked object "
            "is the sounding one.")
    else:
        out["warning"] = None
    return out

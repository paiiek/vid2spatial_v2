"""Audio loading with a librosa-free fallback.

Why
---
``librosa.load`` is the reference loader for this pipeline, but librosa imports
numba, and numba refuses NumPy 2.4 (``Numba needs NumPy 2.3 or less``). On a box
with a newer NumPy the import fails and every entry point that loads audio dies
before doing any work -- see ``docs/ISSUES.md`` I10.

``load_audio`` keeps librosa as the primary path, unchanged, and falls back to
``soundfile.read`` plus ``scipy.signal.resample_poly`` when librosa cannot be
imported. The fallback matches librosa's defaults used here: mono mixdown by
channel mean, float32, shape ``(n,)``, native sample rate when ``sr is None``.

It is NOT bit-identical: librosa resamples with soxr by default, the fallback
uses a polyphase FIR. The difference is a resampler-quality difference on the
resampling path only; at ``sr=None`` (no resampling) both return the same
samples.
"""
from __future__ import annotations

import warnings
from math import gcd
from typing import Optional, Tuple

import numpy as np

_WARNED = False


def _librosa():
    """Import librosa, or None if it is unusable in this environment.

    numba raises ImportError on a too-new NumPy; other environments have managed
    to raise other things from the same import, so this is deliberately broad.
    """
    try:
        import librosa  # noqa: PLC0415
    except Exception:  # pragma: no cover - depends on the installed stack
        return None
    return librosa


def load_audio(path, sr: Optional[int] = None,
               mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load ``path`` as float32 audio, resampled to ``sr`` (native if None).

    Returns ``(samples, sample_rate)``. With ``mono=True`` the samples are 1-D.
    """
    global _WARNED
    lib = _librosa()
    if lib is not None:
        try:
            return lib.load(str(path), sr=sr, mono=mono)
        except ImportError:
            # librosa imports fine but defers numba to first use, so the NumPy
            # incompatibility can surface here rather than at import.
            pass

    import soundfile as sf  # noqa: PLC0415

    if not _WARNED:
        warnings.warn(
            "librosa is unavailable (typically numba vs the installed NumPy); "
            "falling back to soundfile + scipy.signal.resample_poly for audio "
            "loading. Resampling quality differs from librosa's soxr path. "
            "See docs/ISSUES.md I10.",
            RuntimeWarning, stacklevel=2,
        )
        _WARNED = True

    y, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if mono:
        y = y.mean(axis=1)
    else:
        y = y.T  # librosa returns (channels, samples)

    if sr is not None and int(sr) != int(native_sr):
        from scipy.signal import resample_poly  # noqa: PLC0415
        g = gcd(int(sr), int(native_sr))
        up, down = int(sr) // g, int(native_sr) // g
        axis = -1
        y = resample_poly(y, up, down, axis=axis)
        native_sr = int(sr)

    return np.ascontiguousarray(y, dtype=np.float32), int(native_sr)

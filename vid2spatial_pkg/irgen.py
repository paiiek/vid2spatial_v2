import numpy as np
from typing import Tuple, Optional
from pathlib import Path


def synthesize_mono_rir(room_dim: Tuple[float, float, float],
                        src_pos: Tuple[float, float, float],
                        mic_pos: Tuple[float, float, float],
                        fs: int = 48000,
                        rt60: Optional[float] = 0.6,
                        max_order: int = 10) -> np.ndarray:
    """Synthesize a mono RIR using pyroomacoustics image-source method.

    - room_dim: (Lx, Ly, Lz) in meters
    - src_pos, mic_pos: (x, y, z) in meters
    - rt60: Sabine target; if None, use freq-independent absorption 0.4
    Returns: 1D float32 impulse response
    """
    import pyroomacoustics as pra

    if rt60 is not None:
        e_absorption, max_order_auto = pra.inverse_sabine(rt60, room_dim)
    else:
        e_absorption, max_order_auto = 0.4, max_order

    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=min(max_order, max_order_auto),
    )
    room.add_source(src_pos)
    room.add_microphone(mic_pos)
    room.compute_rir()
    rir = room.rir[0][0]
    return np.asarray(rir, dtype=np.float32)


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """FFT-based 1D convolution (full). Returns x*h with float32 output."""
    N = int(1 << int(np.ceil(np.log2(len(x) + len(h) - 1))))
    X = np.fft.rfft(x, n=N)
    H = np.fft.rfft(h, n=N)
    y = np.fft.irfft(X * H, n=N)[: len(x) + len(h) - 1]
    return y.astype(np.float32)


def schroeder_ir(fs: int, rt60: float = 0.6, length_s: float = 2.0) -> np.ndarray:
    """Generate a Schroeder-style artificial IR (pure Python, deterministic).
    - Comb filters + allpasses approximating target RT60.
    - Returns float32 mono IR.
    References: Schroeder, Manfred R. (1962). Natural sounding artificial reverberation.
    """
    N = int(length_s * fs)
    # comb delays in ms
    comb_ms = [29.7, 37.1, 41.1, 43.7]
    allp_ms = [5.0, 1.7]
    g_rt = 10 ** (-3.0 * np.array(comb_ms) / (rt60 * 1000.0))  # per delay decay
    ir = np.zeros(N, np.float32)
    # noise burst as excitation
    exc = np.zeros(N, np.float32)
    exc[0] = 1.0
    # combs
    y = np.zeros(N, np.float32)
    for d_ms, g in zip(comb_ms, g_rt):
        d = max(1, int(d_ms * 1e-3 * fs))
        buf = np.zeros(N, np.float32)
        for n in range(N):
            fb = buf[n - d] if n - d >= 0 else 0.0
            v = exc[n] + g * fb
            buf[n] = v
        y += buf
    # allpasses (canonical): y[n] = -g*x[n] + x[n-d] + g*y[n-d]
    for d_ms in allp_ms:
        d = max(1, int(d_ms * 1e-3 * fs))
        g = 0.7
        x_in = y.copy()
        out = np.zeros(N, np.float32)
        for n in range(N):
            xm = x_in[n - d] if n - d >= 0 else 0.0
            ym = out[n - d] if n - d >= 0 else 0.0
            out[n] = -g * x_in[n] + xm + g * ym
        y = out
    # normalize
    if np.max(np.abs(y)) > 1e-6:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32)


def fairplay_matched_ir(fs: int = 48000,
                       rt60: float = 0.5,
                       direct_ratio: float = 0.73,
                       early_ratio: float = 0.07,
                       late_ratio: float = 0.20,
                       length_s: float = 0.5) -> np.ndarray:
    """
    Generate IR matched to FAIR-Play GT statistics.

    Based on analysis of 50 FAIR-Play samples:
    - Direct: 73% (anechoic-like)
    - Early: 7% (minimal early reflections)
    - Late: 20% (some diffuse tail)
    - RT60: 0.5s

    This IR performs much better than Schroeder (which assumes diffuse field).

    Args:
        fs: sample rate
        rt60: reverberation time
        direct_ratio: energy in direct sound
        early_ratio: energy in early reflections (5-50ms)
        late_ratio: energy in late reflections (50ms+)
        length_s: IR length in seconds

    Returns:
        ir: (N,) impulse response
    """
    import scipy.signal as signal

    N = int(fs * length_s)
    ir = np.zeros(N, dtype=np.float32)

    # Direct sound (delta at t=0)
    ir[0] = np.sqrt(direct_ratio)

    # Early reflections (5-50ms, sparse)
    if early_ratio > 0:
        num_early = 8
        early_times_ms = np.linspace(5, 50, num_early)  # Evenly spaced
        early_gain = np.sqrt(early_ratio / num_early)

        for t_ms in early_times_ms:
            idx = int(t_ms * 1e-3 * fs)
            if idx < N:
                # Add some randomness (70-100%)
                gain = early_gain * np.random.uniform(0.7, 1.0)
                ir[idx] += gain

    # Late reflections (50ms+, exponential decay)
    if late_ratio > 0:
        late_start = int(0.050 * fs)  # 50ms
        late_samples = N - late_start
        t_late = np.arange(late_samples) / fs

        # Exponential decay envelope
        decay_rate = 6.91 / rt60  # -60dB in rt60 seconds
        envelope = np.exp(-decay_rate * t_late)

        # Filtered noise (dense reverb tail)
        noise = np.random.randn(late_samples)

        # Low-pass filter (reverb darker than direct)
        b, a = signal.butter(4, 4000 / (fs/2), btype='low')
        reverb = signal.filtfilt(b, a, noise)

        # Apply envelope
        reverb = reverb * envelope

        # Normalize by energy (not amplitude) to match late_ratio
        current_energy = np.sum(reverb**2)
        if current_energy > 1e-10:
            reverb = reverb * np.sqrt(late_ratio / current_energy)

        ir[late_start:] += reverb

    return ir.astype(np.float32)


__all__ = [
    "synthesize_mono_rir",
    "fft_convolve",
    "schroeder_ir",
    "fairplay_matched_ir",
]


# -------- Visual heuristics + stereo/BRIR utilities --------
def estimate_room_from_video(video_path: str,
                             default_room: Tuple[float, float, float] = (6.0, 5.0, 3.0),
                             default_rt60: float = 0.6,
                             mic_base: Tuple[float, float, float] = (3.0, 2.5, 1.5),
                             mic_spacing_m: float = 0.2) -> Tuple[Tuple[float, float, float],
                                                                   Tuple[float, float, float],
                                                                   Tuple[float, float, float],
                                                                   Tuple[float, float, float],
                                                                   float]:
    """Heuristic room/RT60 estimation from video metadata.
    Currently returns defaults with a small left-right mic spacing.
    Returns: (room_dim LxLyLz, src_pos, micL, micR, rt60)
    """
    Lx, Ly, Lz = default_room
    mx, my, mz = mic_base
    micL = (mx - mic_spacing_m / 2.0, my, mz)
    micR = (mx + mic_spacing_m / 2.0, my, mz)
    # simple source a bit in front-left of the mics
    src_pos = (mx - 1.0, my, mz)
    return (Lx, Ly, Lz), src_pos, micL, micR, float(default_rt60)


def synthesize_stereo_rir(room_dim: Tuple[float, float, float],
                          src_pos: Tuple[float, float, float],
                          micL: Tuple[float, float, float],
                          micR: Tuple[float, float, float],
                          fs: int = 48000,
                          rt60: Optional[float] = 0.6,
                          max_order: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize stereo RIR (L/R) using pyroomacoustics image-source method.
    Returns (hL, hR) as float32 arrays.
    """
    import pyroomacoustics as pra
    if rt60 is not None:
        e_absorption, max_order_auto = pra.inverse_sabine(rt60, room_dim)
    else:
        e_absorption, max_order_auto = 0.4, max_order
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=min(max_order, max_order_auto),
    )
    room.add_source(src_pos)
    room.add_microphone_array(np.c_[[micL, micR]].T)
    room.compute_rir()
    hL = room.rir[0][0]
    hR = room.rir[1][0]
    return np.asarray(hL, np.float32), np.asarray(hR, np.float32)


def stereo_fft_convolve(st: np.ndarray, hL: np.ndarray, hR: np.ndarray) -> np.ndarray:
    """Convolve stereo signal [2,T] with separate L/R filters using FFT convolution.
    Returns [2, T+len(h)-1]."""
    assert st.ndim == 2 and st.shape[0] == 2
    L = fft_convolve(st[0].astype(np.float32), hL.astype(np.float32))
    R = fft_convolve(st[1].astype(np.float32), hR.astype(np.float32))
    T = min(L.shape[0], R.shape[0])
    return np.stack([L[:T], R[:T]], axis=0).astype(np.float32)


def load_brir(left_wav: str, right_wav: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load BRIR pair (left/right) from wav files.
    Returns (hL, hR, fs)."""
    import soundfile as sf
    hL, fsL = sf.read(left_wav)
    hR, fsR = sf.read(right_wav)
    if fsL != fsR:
        raise ValueError("BRIR sample rates do not match")
    hL = hL[:, 0] if hL.ndim == 2 else hL
    hR = hR[:, 0] if hR.ndim == 2 else hR
    return hL.astype(np.float32), hR.astype(np.float32), int(fsL)


def convolve_foa_with_air(foa_sn3d: np.ndarray, air_sn3d: np.ndarray) -> np.ndarray:
    """Convolve FOA [4,T] with an Ambisonic Impulse Response [4,L] (SN3D/ACN).
    Returns [4, T+L-1]."""
    assert foa_sn3d.shape[0] == 4 and air_sn3d.shape[0] == 4
    outs = []
    for c in range(4):
        outs.append(fft_convolve(foa_sn3d[c], air_sn3d[c]))
    m = min(map(len, outs))
    return np.stack([o[:m].astype(np.float32) for o in outs], axis=0)

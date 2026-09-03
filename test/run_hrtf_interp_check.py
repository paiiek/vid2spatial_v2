#!/usr/bin/env python3
"""
run_hrtf_interp_check.py — does HRIR interpolation remove the 5 deg staircase?
=============================================================================
Gap item A8 verification.

Before 2026-09-04 both binaural paths picked the single nearest measured HRIR
(`argmax` of the dot product).  A KEMAR SOFA grid is typically 5 deg in azimuth,
so a smoothly moving source snapped between HRIRs: zipper artefacts, and an
effective angular resolution floor of ~5 deg -- coarser than the trajectory
feeding it.

This sweeps a broadband source smoothly through an azimuth arc, renders it with
`direct_binaural_sofa` in both modes, and recovers the azimuth back out of the
rendered stereo with GCC-PHAT + Woodworth ITD inversion -- the same inversion
`test/run_binaural_az_inversion.py` uses.  The measurement is the *staircase*:
how much of the recovered azimuth trajectory's motion happens in discrete jumps.

  staircase_ratio = (number of analysis windows whose azimuth step is below
                     1/4 of the grid spacing) / (all windows)

Nearest-neighbour holds the azimuth flat inside each 5 deg cell and then jumps,
so its steps are bimodal.  Interpolation should spread the motion evenly.

Usage:
  python test/run_hrtf_interp_check.py
  python test/run_hrtf_interp_check.py --sofa /path/to/kemar.sofa --arc 90
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vid2spatial_pkg.foa_render import direct_binaural_sofa  # noqa: E402

# A sparse (5 deg) grid shows the effect; a dense 1 deg grid will not, by
# construction.  Both are searched so the check runs wherever a SOFA exists.
SOFA_CANDIDATES = [
    "/home/seung/mmhoa/spatial_engine-proto/.localdeps/libmysofa-v1.3.2/share/"
    "MIT_KEMAR_normal_pinna.sofa",
    "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa",
    "/home/seung/mmhoa/roomestim/roomestim_web/data/hrtf/kemar.sofa",
]

R_HEAD = 0.0875
C_SND = 343.0


def find_sofa(explicit: str = "") -> str:
    if explicit:
        return explicit
    for p in SOFA_CANDIDATES:
        if Path(p).exists():
            return p
    return ""


def grid_spacing_deg(sofa_path: str) -> float:
    """Median azimuth spacing of the horizontal ring of the SOFA grid."""
    import h5py
    with h5py.File(sofa_path, "r") as f:
        sp = f["SourcePosition"][:]
    ring = np.sort(sp[np.abs(sp[:, 1]) < 1e-6][:, 0])
    if ring.size < 3:
        return 0.0
    return float(np.median(np.diff(ring)))


def gcc_phat(a: np.ndarray, b: np.ndarray, fs: int, max_tau: float) -> float:
    n = 1
    while n < len(a) + len(b):
        n <<= 1
    A = np.fft.rfft(a, n=n)
    B = np.fft.rfft(b, n=n)
    R = A * np.conj(B)
    mag = np.abs(R)
    mag[mag < 1e-12] = 1e-12
    cc = np.fft.irfft(R / mag, n=n)
    max_shift = int(min(n // 2, fs * max_tau))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    k = int(np.argmax(np.abs(cc)))
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = np.abs(cc[k - 1]), np.abs(cc[k]), np.abs(cc[k + 1])
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    return ((k - max_shift) + delta) / fs


def itd_to_az_deg(itd: float) -> float:
    """Invert Woodworth ITD = (r/c)(theta + sin theta)."""
    target = itd * C_SND / R_HEAD
    if target <= -(math.pi / 2 + 1):
        return -90.0
    if target >= (math.pi / 2 + 1):
        return 90.0
    lo, hi = -math.pi / 2, math.pi / 2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mid + math.sin(mid) < target:
            lo = mid
        else:
            hi = mid
    return math.degrees(0.5 * (lo + hi))


def invert_azimuth(stereo: np.ndarray, sr: int, win_s: float = 0.050,
                   hop_s: float = 0.010):
    L, R = stereo[0], stereo[1]
    max_tau = (R_HEAD / C_SND) * (math.pi / 2 + 1) * 1.2
    win_n, hop_n = int(win_s * sr), int(hop_s * sr)
    times, az = [], []
    for start in range(0, len(L) - win_n, hop_n):
        a, b = L[start:start + win_n], R[start:start + win_n]
        if np.sqrt(np.mean(a ** 2)) < 1e-6 and np.sqrt(np.mean(b ** 2)) < 1e-6:
            continue
        times.append((start + win_n / 2) / sr)
        az.append(itd_to_az_deg(gcc_phat(a, b, sr, max_tau)))
    return np.asarray(times), np.asarray(az)


def staircase_metrics(az: np.ndarray, spacing_deg: float) -> dict:
    """How much of the trajectory's motion happens in discrete jumps."""
    steps = np.abs(np.diff(az))
    if steps.size == 0:
        return {}
    flat_thresh = max(spacing_deg / 4.0, 1e-6)
    return {
        "n_windows": int(az.size),
        "step_mean_deg": float(np.mean(steps)),
        "step_max_deg": float(np.max(steps)),
        "step_std_deg": float(np.std(steps)),
        # a staircase is mostly-flat with occasional big jumps: high ratio here
        "frac_steps_below_quarter_grid": float(np.mean(steps < flat_thresh)),
        # second difference: a smooth ramp has near-zero curvature
        "mean_abs_second_diff_deg": float(np.mean(np.abs(np.diff(az, 2))))
        if az.size > 2 else 0.0,
    }


def run(sofa_path: str, *, arc_deg: float = 90.0, dur_s: float = 4.0,
        sr: int = 48000, block_ms: float = 10.0, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    n = int(dur_s * sr)
    mono = rng.standard_normal(n).astype(np.float32) * 0.2
    az_deg = np.linspace(0.0, arc_deg, n).astype(np.float32)
    az_s = np.radians(az_deg)
    el_s = np.zeros(n, dtype=np.float32)

    spacing = grid_spacing_deg(sofa_path)
    out = {"sofa": sofa_path, "grid_spacing_deg": spacing,
           "arc_deg": arc_deg, "duration_s": dur_s, "modes": {}}
    for mode in ("nearest", "barycentric"):
        stereo = direct_binaural_sofa(mono, sr, az_s, el_s, sofa_path,
                                      block_ms=block_ms, hrir_interp=mode)
        _, az_est = invert_azimuth(stereo, sr)
        m = staircase_metrics(az_est, spacing if spacing > 0 else 5.0)
        m["recovered_range_deg"] = float(az_est.max() - az_est.min())
        out["modes"][mode] = m
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sofa", default="")
    ap.add_argument("--arc", type=float, default=90.0)
    ap.add_argument("--dur", type=float, default=4.0)
    ap.add_argument("--json", default=str(REPO_ROOT / "reports/hrtf_interp_2026-09-04.json"))
    args = ap.parse_args()

    sofa = find_sofa(args.sofa)
    if not sofa:
        print("[skip] no SOFA file found; pass --sofa")
        return 1
    res = run(sofa, arc_deg=args.arc, dur_s=args.dur)
    print(f"SOFA {res['sofa']}  grid spacing {res['grid_spacing_deg']:.1f} deg")
    for mode, m in res["modes"].items():
        print(f"  {mode:12s} flat-step frac={m['frac_steps_below_quarter_grid']:.3f}  "
              f"max step={m['step_max_deg']:6.3f} deg  "
              f"|2nd diff|={m['mean_abs_second_diff_deg']:6.3f} deg  "
              f"range={m['recovered_range_deg']:6.2f} deg")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2) + "\n")
        print(f"[ok] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

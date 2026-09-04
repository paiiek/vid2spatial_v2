#!/usr/bin/env python3
"""
run_binaural_az_inversion.py  (E2E prerequisite: azimuth-from-binaural harness)
===============================================================================
To compare an end-to-end spatial-audio model (which outputs audio, not a
trajectory) against Vid2Spatial on AzMAE, we must recover an azimuth trajectory
from a stereo/binaural signal. This builds and VALIDATES that inversion on our
own 22 HRTF (KEMAR) binaural renders, where the GT azimuth trajectory (the one
that drove the render, traj.json `az`) is known. If the inversion recovers our
own renders' azimuth well, the harness is trustworthy enough to apply to an
end-to-end model's output (and to recorded GT binaural) for a fair AzMAE.

Method: per analysis window, GCC-PHAT cross-correlation between L and R gives the
inter-aural time difference (ITD, sub-sample via parabolic interp). Woodworth's
ray model maps ITD->azimuth:  ITD = (r/c)(theta + sin theta),  r=0.0875 m head
radius, c=343 m/s; inverted numerically. Sign is calibrated once per clip against
GT (the L/R lead convention of the renderer is fixed but we resolve it robustly).

Output: test/full_eval/BINAURAL_AZ_INVERSION.json
Usage:  python test/run_binaural_az_inversion.py
"""
import json, math
from pathlib import Path
import numpy as np
import soundfile as sf

TEST_DIR = Path(__file__).parent
E2E_ROOT = TEST_DIR / "e2e_final"
R_HEAD = 0.0875
C_SND = 343.0
WIN = 0.100      # 100 ms analysis window
HOP = 0.040      # 40 ms hop

CLIPS = ["car-5", "car-10", "car-13", "dog-1", "dog-3", "dog-9", "dog-14",
         "horse-1", "horse-3", "horse-11", "motorcycle-1", "motorcycle-3",
         "motorcycle-6", "skateboard-8", "skateboard-11", "skateboard-17",
         "train-16", "train-17", "drone-2", "drone-6", "drone-13", "guitar-9"]


def gcc_phat(a, b, fs, max_tau):
    n = 1
    while n < len(a) + len(b):
        n <<= 1
    A = np.fft.rfft(a, n=n); B = np.fft.rfft(b, n=n)
    R = A * np.conj(B)
    mag = np.abs(R)
    mag[mag < 1e-12] = 1e-12
    cc = np.fft.irfft(R / mag, n=n)
    max_shift = int(min(n // 2, fs * max_tau))
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    k = int(np.argmax(np.abs(cc)))
    # parabolic interpolation for sub-sample peak
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = np.abs(cc[k - 1]), np.abs(cc[k]), np.abs(cc[k + 1])
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        delta = 0.0
    lag = (k - max_shift) + delta
    return lag / fs        # seconds (positive => a lags b)


def itd_to_az(itd):
    """Invert Woodworth ITD = (r/c)(theta + sin theta), theta in rad; deg out."""
    target = itd * C_SND / R_HEAD          # = theta + sin theta
    lo, hi = -math.pi / 2, math.pi / 2
    if target <= -(math.pi / 2 + 1): return -90.0
    if target >= (math.pi / 2 + 1): return 90.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mid + math.sin(mid) < target: lo = mid
        else: hi = mid
    return math.degrees(0.5 * (lo + hi))


def invert_clip(clip):
    binp = E2E_ROOT / clip / "binaural.wav"
    trp = E2E_ROOT / clip / "traj.json"
    if not binp.exists() or not trp.exists():
        return None
    x, fs = sf.read(str(binp))
    if x.ndim != 2:
        return None
    L, Rr = x[:, 0], x[:, 1]
    tr = json.load(open(trp))
    frames = tr["frames"]
    fps = float(tr.get("fps", 25.0))
    az_gt_frames = np.degrees([f["az"] for f in frames])
    dur = len(L) / fs

    max_tau = (R_HEAD / C_SND) * (math.pi / 2 + 1) * 1.2
    win_n = int(WIN * fs); hop_n = int(HOP * fs)
    centers, az_est = [], []
    for start in range(0, len(L) - win_n, hop_n):
        a = L[start:start + win_n]; b = Rr[start:start + win_n]
        if np.sqrt(np.mean(a ** 2)) < 1e-4 and np.sqrt(np.mean(b ** 2)) < 1e-4:
            continue
        itd = gcc_phat(a, b, fs, max_tau)
        centers.append((start + win_n / 2) / fs)
        az_est.append(itd_to_az(itd))
    if len(centers) < 5:
        return None
    centers = np.array(centers); az_est = np.array(az_est)

    # GT azimuth at window centers (map time -> frame index)
    fidx = np.clip((centers * fps).astype(int), 0, len(az_gt_frames) - 1)
    az_gt = az_gt_frames[fidx]

    # resolve renderer L/R sign convention robustly (pick sign minimizing MAE)
    mae_pos = np.mean(np.abs(az_est - az_gt))
    mae_neg = np.mean(np.abs(-az_est - az_gt))
    sign = 1.0 if mae_pos <= mae_neg else -1.0
    az_est *= sign
    mae = float(np.mean(np.abs(az_est - az_gt)))
    corr = float(np.corrcoef(az_est, az_gt)[0, 1]) if np.std(az_est) > 1e-6 else 0.0
    return dict(clip=clip, n_win=len(centers), sign=sign,
                inv_azmae=mae, corr=corr,
                gt_range=float(az_gt.max() - az_gt.min()),
                est_range=float(az_est.max() - az_est.min()))


def main():
    rows = []
    for c in CLIPS:
        r = invert_clip(c)
        if r:
            rows.append(r)
            print(f"{c:14s} inv-AzMAE={r['inv_azmae']:6.2f}deg  corr={r['corr']:+.2f}  "
                  f"range est/gt={r['est_range']:.1f}/{r['gt_range']:.1f}  sign={r['sign']:+.0f}")
    if not rows:
        print("no clips"); return
    mae = np.array([r["inv_azmae"] for r in rows])
    corr = np.array([r["corr"] for r in rows])
    summ = dict(n=len(rows), mean_inv_azmae=round(float(mae.mean()), 3),
                median_inv_azmae=round(float(np.median(mae)), 3),
                mean_corr=round(float(np.nanmean(corr)), 3))
    (TEST_DIR / "full_eval" / "BINAURAL_AZ_INVERSION.json").write_text(
        json.dumps(dict(summary=summ, per_clip=rows), indent=2))
    print("\n" + "=" * 70)
    print("AZIMUTH-FROM-BINAURAL INVERSION — validation on our own KEMAR renders")
    print("=" * 70)
    print(f"  clips: {summ['n']}")
    print(f"  inversion AzMAE vs known GT trajectory: mean={summ['mean_inv_azmae']:.2f}  "
          f"median={summ['median_inv_azmae']:.2f} deg")
    print(f"  mean trajectory correlation: {summ['mean_corr']:+.2f}")
    print("  (this is the floor of the measurement harness; an e2e model's AzMAE "
          "would be its own error + this harness error)")


if __name__ == "__main__":
    main()

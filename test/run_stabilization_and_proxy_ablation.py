#!/usr/bin/env python3
"""
run_stabilization_and_proxy_ablation.py
=======================================
FRESH ablations on the CURRENT pipeline (SAM2 + arctan unwrap + 80ms angle
moving-average + asymmetric d_rel IIR). Does NOT reuse any old RTS/Kalman/
Adaptive-K numbers from docs (those are a different experiment).

Produces:
  A1. Stabilization ON/OFF ablation  -> jerk/jitter reduction + AzMAE preserved
  A3. Distance-proxy objective sanity check -> bbox-area cue vs MiDaS metric depth
      (NOTE: correlating d_rel vs MiDaS is circular -- d_rel IS normalized MiDaS,
       corr=1.0 by construction. The meaningful independent check is bbox-area.)

Stabilization params are taken verbatim from the paper:
  - angle moving-average window: 80 ms  (smooth_limit_angles)
  - d_rel asymmetric one-pole IIR: attack tau=0.7 s, release tau=0.2 s
  - arctan-domain unwrap on azimuth

Usage:
    cd /home/seung/mmhoa/vid2spatial_v2
    python test/run_stabilization_and_proxy_ablation.py
"""
import json, math, sys
from pathlib import Path
import numpy as np

TEST_DIR = Path(__file__).parent
V2_ROOT  = TEST_DIR.parent
sys.path.insert(0, str(V2_ROOT))

from scipy.ndimage import uniform_filter1d
from vid2spatial_pkg.foa_render import smooth_limit_angles, _asymmetric_smooth_d_rel

TRAJ_ROOT = TEST_DIR / "e2e_final"
# GT search paths (LaSOT bbox)
LASOT_CANDIDATES = [
    Path("/home/seung/mmhoa/vid2spatial/data/lasot"),
    V2_ROOT / "data" / "lasot",
]

# paper-stated stabilization params
ANGLE_SMOOTH_MS = 80.0
DREL_ATTACK_S   = 0.7
DREL_RELEASE_S  = 0.2

EVAL_CLIPS = {
    "car-5": ("vehicle", True), "car-10": ("vehicle", True), "car-13": ("vehicle", True),
    "dog-1": ("animal", True), "dog-3": ("animal", True), "dog-9": ("animal", True),
    "dog-14": ("animal", True), "horse-1": ("animal", True), "horse-3": ("animal", True),
    "horse-11": ("animal", True), "motorcycle-1": ("vehicle", True),
    "motorcycle-3": ("vehicle", True), "motorcycle-6": ("vehicle", True),
    "skateboard-8": ("sports", True), "skateboard-11": ("sports", True),
    "skateboard-17": ("sports", True), "train-16": ("vehicle", True),
    "train-17": ("vehicle", True), "drone-2": ("drone", False),
    "drone-6": ("drone", False), "drone-13": ("drone", False),
    "guitar-9": ("instrument", False),
}


def find_gt(clip):
    for root in LASOT_CANDIDATES:
        for p in (root / clip.split("-")[0] / clip / "groundtruth.txt",
                  root / clip / "groundtruth.txt"):
            if p.exists():
                return p
    return None


def bbox_to_az_gt(bboxes, W, fov_deg):
    f = W / (2.0 * math.tan(math.radians(fov_deg / 2.0)))
    cx = bboxes[:, 0] + bboxes[:, 2] / 2.0
    return np.degrees(np.arctan2(cx - W / 2.0, f))


SR = 48000  # render sample rate (the domain where the stabilizer acts)


def jitter_jerk_audiorate(sig_deg, sr):
    """sig_deg: control signal at audio rate (deg). Returns (jitter, jerk) as
    RMS angular acceleration (deg/s^2) and RMS angular jerk (deg/s^3).

    This is the domain the renderer drives the HRTF/pan with, and where the
    80ms filter removes the stair-step ('zipper') artifacts that piecewise-linear
    frame->sample interpolation introduces (25 velocity discontinuities/sec).
    Standard motion-smoothness measures; lower = smoother audible motion.
    """
    x = np.asarray(sig_deg, dtype=np.float64)
    if len(x) < 4:
        return float("nan"), float("nan")
    accel = np.diff(x, n=2) * (sr ** 2)   # deg/s^2
    jerk  = np.diff(x, n=3) * (sr ** 3)   # deg/s^3
    return float(np.sqrt(np.mean(accel ** 2))), float(np.sqrt(np.mean(jerk ** 2)))


def interp_to_audiorate(az_deg, el_deg, d_rel, fps, sr):
    """Frame -> audio-rate, arctan-unwrap on az (always; geometric necessity).
    Returns the OFF control streams (interp only, no smoothing)."""
    n = len(az_deg)
    spf = sr / fps
    T = int(round((n - 1) * spf)) + 1
    s = np.arange(T)
    fidx = np.arange(n) * spf
    az_rad = np.unwrap(np.radians(az_deg))
    az_s = np.degrees(np.interp(s, fidx, az_rad))
    el_s = np.degrees(np.interp(s, fidx, np.radians(el_deg)))
    drel_s = np.interp(s, fidx, d_rel)
    return az_s, el_s, drel_s, fidx, T


def stabilize_audiorate(az_off, el_off, drel_off, sr):
    """Apply the CURRENT pipeline stabilization to the audio-rate streams:
    80ms moving average on az/el (= smooth_limit_angles), asymmetric one-pole
    IIR on d_rel (= _asymmetric_smooth_d_rel).

    Implemented in float64 with uniform_filter1d(mode='nearest'). This is
    interior-identical to the shipping np.convolve(...,'same') 80ms box filter,
    but avoids two measurement artifacts irrelevant to the smoothing behaviour:
      (1) float32 quantization amplified by the sr^2/sr^3 derivative scaling,
      (2) the convolve zero-pad boundary transient (an ~80ms fade at clip start).
    """
    win = max(1, int(sr * (ANGLE_SMOOTH_MS / 1000.0)))
    az_sm = uniform_filter1d(az_off.astype(np.float64), size=win, mode='nearest')
    el_sm = uniform_filter1d(el_off.astype(np.float64), size=win, mode='nearest')
    drel_sm = np.asarray(
        _asymmetric_smooth_d_rel(drel_off.astype(np.float32), sr,
                                 DREL_ATTACK_S, DREL_RELEASE_S), np.float64)
    return az_sm, el_sm, drel_sm


def main():
    rows = []
    proxy_rows = []
    pooled_bbox, pooled_midas = [], []

    for clip, (cat, coco) in EVAL_CLIPS.items():
        tp = TRAJ_ROOT / clip / "traj.json"
        if not tp.exists():
            continue
        traj = json.load(open(tp))
        fr = traj["frames"]
        fps = float(traj.get("fps", 25.0))
        W = int(fr[0]["width"]); H = int(fr[0]["height"]); fov = float(fr[0].get("fov_deg", 60.0))

        az = np.degrees([f["az"] for f in fr])
        el = np.degrees([f["el"] for f in fr])
        d_rel = np.array([f["d_rel"] for f in fr])
        dist_m = np.array([f["dist_m"] for f in fr])
        area = np.array([f["w"] * f["h"] for f in fr], dtype=np.float64)

        # ── A1: ON vs OFF jitter/jerk at AUDIO RATE (where the filter acts) ──
        az_off_s, el_off_s, drel_off_s, fidx_s, T = interp_to_audiorate(az, el, d_rel, fps, SR)
        az_on_s, el_on_s, drel_on_s = stabilize_audiorate(az_off_s, el_off_s, drel_off_s, SR)
        j_off_az, k_off_az = jitter_jerk_audiorate(az_off_s, SR)
        j_on_az,  k_on_az  = jitter_jerk_audiorate(az_on_s, SR)
        j_off_el, k_off_el = jitter_jerk_audiorate(el_off_s, SR)
        j_on_el,  k_on_el  = jitter_jerk_audiorate(el_on_s, SR)

        # AzMAE OFF vs ON (vs GT) -- accuracy preservation (sample ON stream at frame centers)
        samp = np.clip(np.round(fidx_s).astype(int), 0, T - 1)
        az_on_frame = az_on_s[samp]
        az_mae_off = az_mae_on = float("nan")
        gtp = find_gt(clip)
        if gtp is not None:
            bb = np.loadtxt(str(gtp), delimiter=",")
            fidx = np.array([f["frame"] for f in fr], dtype=int)
            valid = fidx < len(bb)
            gtb = bb[fidx[valid]]
            present = (gtb[:, 2] > 0) & (gtb[:, 3] > 0)
            if present.sum() >= 5:
                az_gt = bbox_to_az_gt(gtb[present], W, fov)
                az_mae_off = float(np.mean(np.abs(az[valid][present] - az_gt)))
                az_mae_on  = float(np.mean(np.abs(az_on_frame[valid][present] - az_gt)))

        rows.append(dict(clip=clip, cat=cat, coco=coco,
                         j_off_az=j_off_az, j_on_az=j_on_az,
                         k_off_az=k_off_az, k_on_az=k_on_az,
                         j_off_el=j_off_el, j_on_el=j_on_el,
                         k_off_el=k_off_el, k_on_el=k_on_el,
                         az_mae_off=az_mae_off, az_mae_on=az_mae_on))

        # ── A3: bbox-area cue vs MiDaS metric depth ──
        # bbox-area relative distance: larger area = nearer. proxy_far = normalized(1/area-cue)
        # Use the same monotonic relation the renderer's bbox_area mode uses:
        #   area_rel large -> near -> d_rel small ; so d_rel_bbox = 1 - area_norm
        a_min, a_max = area.min(), area.max()
        if (a_max - a_min) > 1e-9:
            d_rel_bbox = 1.0 - (area - a_min) / (a_max - a_min)
        else:
            d_rel_bbox = np.full_like(area, 0.5)
        # MiDaS-based relative distance (what the system actually uses)
        dm_min, dm_max = dist_m.min(), dist_m.max()
        if (dm_max - dm_min) > 1e-9:
            d_rel_midas = (dist_m - dm_min) / (dm_max - dm_min)
        else:
            d_rel_midas = np.full_like(dist_m, 0.5)
        # Pearson between two INDEPENDENT distance cues
        if np.std(d_rel_bbox) > 1e-9 and np.std(d_rel_midas) > 1e-9:
            r = float(np.corrcoef(d_rel_bbox, d_rel_midas)[0, 1])
        else:
            r = float("nan")
        proxy_rows.append(dict(clip=clip, cat=cat, coco=coco, r=r,
                               area_var=float(np.var(area))))
        pooled_bbox.append(d_rel_bbox); pooled_midas.append(d_rel_midas)

    # ── aggregate + print ──
    def agg(key):
        v = [r[key] for r in rows if not math.isnan(r[key])]
        return np.mean(v), np.std(v)

    print("\n" + "=" * 90)
    print("A1. STABILIZATION ON/OFF ABLATION  (current pipeline: unwrap+80ms MA+asym IIR)")
    print("    jitter = RMS angular accel (deg/s^2); jerk = RMS angular jerk (deg/s^3); audio-rate")
    print("=" * 90)
    print(f"{'clip':14s} {'jit_az_off':>11s} {'jit_az_on':>11s} {'jrk_az_off':>12s} {'jrk_az_on':>12s} {'AzMAE_off':>9s} {'AzMAE_on':>9s}")
    for r in rows:
        print(f"{r['clip']:14s} {r['j_off_az']:11.1f} {r['j_on_az']:11.1f} "
              f"{r['k_off_az']:12.0f} {r['k_on_az']:12.0f} "
              f"{r['az_mae_off']:9.2f} {r['az_mae_on']:9.2f}")

    jaz_off = np.mean([r['j_off_az'] for r in rows]); jaz_on = np.mean([r['j_on_az'] for r in rows])
    kaz_off = np.mean([r['k_off_az'] for r in rows]); kaz_on = np.mean([r['k_on_az'] for r in rows])
    jel_off = np.mean([r['j_off_el'] for r in rows]); jel_on = np.mean([r['j_on_el'] for r in rows])
    kel_off = np.mean([r['k_off_el'] for r in rows]); kel_on = np.mean([r['k_on_el'] for r in rows])
    mae_off = agg('az_mae_off'); mae_on = agg('az_mae_on')
    print("-" * 78)
    print(f"MEAN  az jitter: OFF={jaz_off:.4f}  ON={jaz_on:.4f}  "
          f"reduction={100*(1-jaz_on/jaz_off):.1f}%")
    print(f"MEAN  az jerk  : OFF={kaz_off:.4f}  ON={kaz_on:.4f}  "
          f"reduction={100*(1-kaz_on/kaz_off):.1f}%")
    print(f"MEAN  el jitter: OFF={jel_off:.4f}  ON={jel_on:.4f}  "
          f"reduction={100*(1-jel_on/jel_off):.1f}%")
    print(f"MEAN  el jerk  : OFF={kel_off:.4f}  ON={kel_on:.4f}  "
          f"reduction={100*(1-kel_on/kel_off):.1f}%")
    print(f"MEAN  AzMAE    : OFF={mae_off[0]:.3f}+/-{mae_off[1]:.3f}  "
          f"ON={mae_on[0]:.3f}+/-{mae_on[1]:.3f}  delta={mae_on[0]-mae_off[0]:+.3f} deg")

    print("\n" + "=" * 78)
    print("A3. DISTANCE-PROXY SANITY CHECK  (bbox-area cue vs MiDaS metric depth)")
    print("   [literal 'd_rel vs MiDaS' is circular: corr=1.0 by construction]")
    print("=" * 78)
    print(f"{'clip':14s} {'coco':>5s} {'Pearson r (bbox vs MiDaS)':>26s}")
    for r in proxy_rows:
        print(f"{r['clip']:14s} {str(r['coco']):>5s} {r['r']:26.3f}")
    rs = [r['r'] for r in proxy_rows if not math.isnan(r['r'])]
    pb = np.concatenate(pooled_bbox); pm = np.concatenate(pooled_midas)
    pooled_r = float(np.corrcoef(pb, pm)[0, 1])
    print("-" * 78)
    print(f"Per-clip mean Pearson r = {np.mean(rs):.3f} +/- {np.std(rs):.3f}  (n={len(rs)})")
    print(f"Median per-clip r       = {np.median(rs):.3f}")
    print(f"Pooled r (all frames)   = {pooled_r:.3f}  (N={len(pb)} frames)")
    frac_pos = np.mean([1 for x in rs if x > 0.5]) if rs else 0
    print(f"Clips with r>0.5        = {sum(1 for x in rs if x>0.5)}/{len(rs)}")

    # save JSON for downstream writeup
    out = dict(
        stabilization=dict(
            params=dict(angle_smooth_ms=ANGLE_SMOOTH_MS, drel_attack_s=DREL_ATTACK_S,
                        drel_release_s=DREL_RELEASE_S),
            mean=dict(jit_az_off=jaz_off, jit_az_on=jaz_on, jrk_az_off=kaz_off, jrk_az_on=kaz_on,
                      jit_el_off=jel_off, jit_el_on=jel_on, jrk_el_off=kel_off, jrk_el_on=kel_on,
                      azmae_off=mae_off[0], azmae_on=mae_on[0]),
            per_clip=rows),
        distance_proxy=dict(per_clip_mean_r=float(np.mean(rs)), median_r=float(np.median(rs)),
                            pooled_r=pooled_r, per_clip=proxy_rows),
    )
    op = TEST_DIR / "full_eval" / "STABILIZATION_PROXY_ABLATION.json"
    op.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[json] -> {op}")


if __name__ == "__main__":
    main()

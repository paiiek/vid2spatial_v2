#!/usr/bin/env python3
"""
run_full_eval.py
================
22개 LaSOT 클립 × 트래커/depth 조합 정량 평가.

평가 지표 (GT bbox 기반):
  - az_mae_deg    : GT center pixel → GT azimuth vs 추정 azimuth (MAE, °)
  - center_err_px : 추정 az/el → 재투영 픽셀 vs GT center (MAE, px)
  - az_range_deg  : 추정 azimuth 범위 (넓을수록 공간 표현력 높음)
  - az_smooth     : azimuth 2차 미분 절대값 평균 (낮을수록 smooth)
  - coverage_pct  : 추적 유지율 (%)
  - conf_mean     : 평균 detection confidence
  - dist_std_m    : distance 변동폭 (depth 정보량)
  - wall_sec      : 처리 시간

조건:
  Tracker × Depth = 5조건
    BT_s1  × midas
    AK_s1  × midas
    AK_s3  × midas
    BT_s1  × depth_anything    (depth 변수만 다름)
    AK_s1  × depth_anything

클립: 8카테고리 × 2-4개 = 22개 (첫 500프레임 사용)

Usage:
    python test/run_full_eval.py [--clips all | car-5 dog-1 ...]
                                 [--conds all | BT_s1 AK_s1 ...]
                                 [--max_frames 500]
                                 [--workers 1]
                                 [--skip_existing]
                                 [--report_only]
"""

import argparse, json, math, os, sys, time, traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

TEST_DIR = Path(__file__).parent
V2_ROOT  = TEST_DIR.parent
LASOT    = Path("/home/seung/mmhoa/vid2spatial/data/lasot")
SOFA_PATH = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"
OUT_ROOT  = TEST_DIR / "full_eval"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(V2_ROOT))

# ── 평가 클립 목록 ─────────────────────────────────────────────────────────────
EVAL_CLIPS = {
    # car: 고속 이동, 넓은 시야
    "car-5":   dict(cls="car",  speed="fast"),
    "car-10":  dict(cls="car",  speed="fast"),
    "car-13":  dict(cls="car",  speed="fast"),
    # dog: 빠르고 방향 전환 많음
    "dog-1":   dict(cls="dog",  speed="fast"),
    "dog-3":   dict(cls="dog",  speed="fast"),
    "dog-9":   dict(cls="dog",  speed="fast"),
    "dog-14":  dict(cls="dog",  speed="fast"),
    # drone: 넓은 이동 범위
    "drone-2":  dict(cls="drone", speed="medium"),
    "drone-6":  dict(cls="drone", speed="medium"),
    "drone-13": dict(cls="drone", speed="medium"),
    # guitar: 느린 이동 (정적 기준)
    "guitar-9": dict(cls="guitar", speed="slow"),
    # horse: 중속
    "horse-1":  dict(cls="horse", speed="medium"),
    "horse-3":  dict(cls="horse", speed="medium"),
    "horse-11": dict(cls="horse", speed="medium"),
    # motorcycle: 중-고속
    "motorcycle-1": dict(cls="motorcycle", speed="fast"),
    "motorcycle-3": dict(cls="motorcycle", speed="fast"),
    "motorcycle-6": dict(cls="motorcycle", speed="fast"),
    # skateboard: 빠름 + 작은 물체
    "skateboard-8":  dict(cls="skateboard", speed="fast"),
    "skateboard-11": dict(cls="skateboard", speed="fast"),
    "skateboard-17": dict(cls="skateboard", speed="fast"),
    # train: 고속 직선
    "train-16": dict(cls="train", speed="fast"),
    "train-17": dict(cls="train", speed="fast"),
}

# ── 평가 조건 ─────────────────────────────────────────────────────────────────
EVAL_CONDS = {
    "BT_s1_midas":    dict(method="v1_bytetrack", stride=1, depth="midas"),
    "BT_s3_midas":    dict(method="v1_bytetrack", stride=3, depth="midas"),
    "AK_s1_midas":    dict(method="adaptive_k",   stride=1, depth="midas"),
    "AK_s3_midas":    dict(method="adaptive_k",   stride=3, depth="midas"),
    "BT_s1_da":       dict(method="v1_bytetrack", stride=1, depth="depth_anything"),
    "AK_s3_da":       dict(method="adaptive_k",   stride=3, depth="depth_anything"),
    # metric depth backends (returns metres, is_metric=True)
    "BT_s1_metric3d": dict(method="v1_bytetrack", stride=1, depth="metric3d"),
    "BT_s1_da_metric":dict(method="v1_bytetrack", stride=1, depth="metric"),
}


# ── GT 로드 ────────────────────────────────────────────────────────────────────

def load_gt(clip: str, max_frames: int = 0):
    gt_path = LASOT / clip / "groundtruth.txt"
    occ_path = LASOT / clip / "full_occlusion.txt"
    ov_path  = LASOT / clip / "out_of_view.txt"

    gt = []
    for line in gt_path.read_text().strip().split("\n"):
        parts = [float(x) for x in line.strip().replace("\t", ",").split(",")]
        if len(parts) >= 4:
            cx = parts[0] + parts[2] / 2
            cy = parts[1] + parts[3] / 2
            gt.append((cx, cy, parts[2], parts[3]))

    occ = []
    if occ_path.exists():
        occ = [int(x.strip()) for x in occ_path.read_text().strip().split("\n") if x.strip().isdigit()]
    ov = []
    if ov_path.exists():
        ov  = [int(x.strip()) for x in ov_path.read_text().strip().split("\n") if x.strip().isdigit()]

    if max_frames > 0:
        gt  = gt[:max_frames]
        occ = occ[:max_frames]
        ov  = ov[:max_frames]

    return gt, occ, ov


# ── 파이프라인 실행 ────────────────────────────────────────────────────────────

def run_pipeline(clip: str, clip_info: dict, cond: str, cond_info: dict,
                 max_frames: int, skip_existing: bool) -> dict:
    """V2SpatialTracker를 직접 호출해 traj.json 생성 (audio 불필요)."""
    out_dir = OUT_ROOT / f"{clip}__{cond}"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "traj.json"

    if skip_existing and traj_path.exists():
        return {"clip": clip, "cond": cond, "status": "skipped", "traj_path": str(traj_path)}

    img_dir = LASOT / clip / "img"
    if not img_dir.exists():
        return {"clip": clip, "cond": cond, "status": "error",
                "error": f"img dir not found: {img_dir}"}

    # max_frames 제한: 이미지 파일을 임시 디렉토리에 심볼릭 링크로 제한
    # 더 간단하게: track() 실행 후 traj에서 frame < max_frames만 보관
    video_path = str(img_dir / "%08d.jpg")

    t0 = time.time()
    try:
        from vid2spatial_pkg.v2_spatial_tracker import V2SpatialTracker
        from vid2spatial_pkg.temporal_smoother import smooth_trajectory_batch
        import cv2

        tracker = V2SpatialTracker(
            depth_fn=None,
            depth_backend=cond_info["depth"],
            fov_deg=60.0,
        )

        # max_frames 제한을 위해 임시 디렉토리에 처음 N개 이미지만 symlink
        import tempfile, shutil
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"feval_{clip}_{cond}_"))
        try:
            imgs_all = sorted(img_dir.glob("*.jpg"))
            if max_frames > 0:
                imgs_all = imgs_all[:max_frames]
            for i, img_p in enumerate(imgs_all, start=1):
                dst = tmp_dir / f"{i:08d}.jpg"
                dst.symlink_to(img_p.resolve())
            video_path_limited = str(tmp_dir / "%08d.jpg")

            traj_raw = tracker.track(
                video_path=video_path_limited,
                cls_name=clip_info["cls"],
                sample_stride=cond_info["stride"],
                method=cond_info["method"],
                enhance_depth=False,   # 평가에서는 raw depth 사용
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Kalman smoothing
        cap = cv2.VideoCapture(str(img_dir / "%08d.jpg"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps <= 0:
            fps = 25.0

        traj_raw["frames"] = smooth_trajectory_batch(
            traj_raw["frames"], fps=fps,
            process_noise=0.01, measurement_noise=0.1,
        )
        traj_raw["fps"] = fps

        # d_rel 재계산 (bbox_area mode)
        frames = traj_raw["frames"]
        if frames:
            W = frames[0].get("width", 1280)
            H = frames[0].get("height", 720)
            areas = [f.get("w", 80) * f.get("h", 80) for f in frames]
            a_min, a_max = min(areas), max(areas)
            frame_area = W * H
            for f, a in zip(frames, areas):
                a_rel = a / frame_area
                if (a_max - a_min) > 1e-3:
                    f["d_rel"] = float(1.0 - np.clip((a - a_min) / (a_max - a_min), 0, 1))
                else:
                    f["d_rel"] = 0.5

        traj_path.write_text(json.dumps(traj_raw, indent=2))
        wall = time.time() - t0
        return {"clip": clip, "cond": cond, "status": "ok",
                "traj_path": str(traj_path), "wall_sec": wall}
    except Exception as e:
        return {"clip": clip, "cond": cond, "status": "error",
                "error": str(e), "tb": traceback.format_exc()}


# ── 지표 계산 ──────────────────────────────────────────────────────────────────

def compute_metrics(clip: str, cond: str, max_frames: int = 0) -> dict:
    traj_path = OUT_ROOT / f"{clip}__{cond}" / "traj.json"
    if not traj_path.exists():
        return {}

    traj = json.loads(traj_path.read_text())
    frames = traj.get("frames", [])
    if not frames:
        return {}

    if max_frames > 0:
        frames = [f for f in frames if f["frame"] < max_frames]

    W = int(traj.get("intrinsics", {}).get("width",  frames[0].get("width",  1280)))
    H = int(traj.get("intrinsics", {}).get("height", frames[0].get("height",  720)))
    fov = float(traj.get("intrinsics", {}).get("fov_deg", 60.0))
    f_len = (W / 2) / math.tan(math.radians(fov / 2))

    gt, occ, ov = load_gt(clip, max_frames=max_frames if max_frames > 0 else 0)

    az_errs, px_errs, azs, dists, confs, d_rels = [], [], [], [], [], []
    valid_frames = 0

    for fr in frames:
        fi = fr["frame"]
        az_est = fr["az"]
        el_est = fr["el"]
        azs.append(az_est)
        dists.append(fr.get("dist_m", 1.0))
        confs.append(fr.get("confidence", 0.5))
        d_rels.append(fr.get("d_rel", 0.5))
        valid_frames += 1

        if fi < len(gt):
            if (occ and fi < len(occ) and occ[fi]) or (ov and fi < len(ov) and ov[fi]):
                continue  # skip occluded/OOV frames for GT metrics
            cx_gt, cy_gt, _, _ = gt[fi]
            az_gt = math.atan2(cx_gt - W / 2, f_len)
            az_errs.append(abs(math.degrees(az_gt - az_est)))

            # back-project estimated az/el to pixel
            try:
                cx_est = W / 2 + f_len * math.tan(az_est)
                cy_est = H / 2 - f_len * math.tan(el_est) * math.sqrt(1 + math.tan(az_est) ** 2)
                px_errs.append(math.hypot(cx_est - cx_gt, cy_est - cy_gt))
            except (ValueError, OverflowError):
                pass

    az_arr = np.array(azs, dtype=np.float64)
    az_unwrapped = np.unwrap(az_arr)
    az_range = float(np.degrees(az_unwrapped.max() - az_unwrapped.min()))
    az_d2 = np.abs(np.diff(az_unwrapped, n=2))
    az_smooth = float(az_d2.mean()) if len(az_d2) > 0 else float("nan")
    dist_std = float(np.std(dists))
    d_jitter = float(np.mean(np.abs(np.diff(d_rels)))) if len(d_rels) > 1 else float("nan")

    return {
        "az_mae_deg":    float(np.mean(az_errs))  if az_errs  else float("nan"),
        "center_err_px": float(np.mean(px_errs))  if px_errs  else float("nan"),
        "az_range_deg":  az_range,
        "az_smooth":     az_smooth,
        "coverage_pct":  valid_frames / max(len(frames), 1) * 100.0,
        "conf_mean":     float(np.mean(confs))  if confs  else float("nan"),
        "dist_std_m":    dist_std,
        "d_rel_jitter":  d_jitter,
        "n_frames":      valid_frames,
        "n_gt_compared": len(az_errs),
    }


# ── 결과 출력 ──────────────────────────────────────────────────────────────────

def print_report(all_metrics: dict, conds: list):
    """all_metrics[clip][cond] = dict"""
    clips = sorted(all_metrics.keys())

    # ── per-clip table ──────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("FULL EVAL — per-clip az_mae / center_err (°/px)")
    print("=" * 110)
    header = f"{'Clip':<22}"
    for c in conds:
        short = c.replace("_midas", "").replace("_da", "(da)")
        header += f" {short:>16}"
    print(header)
    print("-" * 110)
    for clip in clips:
        row = f"{clip:<22}"
        for c in conds:
            m = all_metrics[clip].get(c, {})
            if m:
                row += f" {m['az_mae_deg']:5.2f}/{m['center_err_px']:6.0f}"
            else:
                row += f"     N/A       "
        print(row)

    # ── aggregate ──────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("AGGREGATE (mean over all clips)")
    print("=" * 110)
    metric_keys = [
        ("az_mae_deg",    "Az MAE (°)",      "{:.3f}"),
        ("center_err_px", "Px Err (px)",     "{:.1f}"),
        ("az_range_deg",  "Az Range (°)",    "{:.1f}"),
        ("az_smooth",     "Az Smooth",       "{:.5f}"),
        ("coverage_pct",  "Coverage (%)",    "{:.1f}"),
        ("conf_mean",     "Conf Mean",       "{:.3f}"),
        ("dist_std_m",    "Dist Std (m)",    "{:.3f}"),
        ("d_rel_jitter",  "dRel Jitter",     "{:.5f}"),
    ]
    cond_short = {c: c.replace("_midas","").replace("_da","(DA)") for c in conds}
    hdr = f"  {'Metric':<22}"
    for c in conds:
        hdr += f" {cond_short[c]:>14}"
    print(hdr)
    print("-" * 110)
    for key, label, fmt in metric_keys:
        row = f"  {label:<22}"
        for c in conds:
            vals = [all_metrics[clip][c][key]
                    for clip in clips
                    if c in all_metrics[clip] and key in all_metrics[clip][c]
                    and not math.isnan(all_metrics[clip][c][key])]
            if vals:
                row += f" {fmt.format(np.mean(vals)):>14}"
            else:
                row += f" {'N/A':>14}"
        print(row)

    # ── category breakdown ─────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("Az MAE by category (mean °)")
    print("=" * 110)
    cats = {}
    for clip in clips:
        cat = clip.split("-")[0]
        cats.setdefault(cat, []).append(clip)

    hdr = f"  {'Category':<16}"
    for c in conds:
        hdr += f" {cond_short[c]:>14}"
    print(hdr)
    print("-" * 110)
    for cat in sorted(cats.keys()):
        row = f"  {cat:<16}"
        for c in conds:
            vals = [all_metrics[clip][c]["az_mae_deg"]
                    for clip in cats[cat]
                    if c in all_metrics[clip] and "az_mae_deg" in all_metrics[clip][c]
                    and not math.isnan(all_metrics[clip][c].get("az_mae_deg", float("nan")))]
            row += f" {'{:.2f}'.format(np.mean(vals)) if vals else 'N/A':>14}"
        print(row)
    print("=" * 110)

    # ── composite score ────────────────────────────────────────────────────
    print("\nComposite Score  = (1 - az_mae_norm) * 0.5 + (1 - px_err_norm) * 0.3 + az_range_norm * 0.2")
    print("(higher = better)")
    print("-" * 60)

    def safe_mean(key, cond):
        vals = [all_metrics[clip][cond][key]
                for clip in clips
                if cond in all_metrics[clip] and key in all_metrics[clip][cond]
                and not math.isnan(all_metrics[clip][cond].get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    mae_vals   = {c: safe_mean("az_mae_deg",    c) for c in conds}
    px_vals    = {c: safe_mean("center_err_px", c) for c in conds}
    range_vals = {c: safe_mean("az_range_deg",  c) for c in conds}

    mae_min, mae_max   = min(v for v in mae_vals.values()   if not math.isnan(v)), \
                         max(v for v in mae_vals.values()   if not math.isnan(v))
    px_min,  px_max    = min(v for v in px_vals.values()    if not math.isnan(v)), \
                         max(v for v in px_vals.values()    if not math.isnan(v))
    rng_min, rng_max   = min(v for v in range_vals.values() if not math.isnan(v)), \
                         max(v for v in range_vals.values() if not math.isnan(v))

    scores = {}
    for c in conds:
        if math.isnan(mae_vals[c]): continue
        mae_n  = (mae_vals[c]   - mae_min)  / max(mae_max  - mae_min,  1e-6)
        px_n   = (px_vals[c]    - px_min)   / max(px_max   - px_min,   1e-6)
        rng_n  = (range_vals[c] - rng_min)  / max(rng_max  - rng_min,  1e-6)
        scores[c] = (1 - mae_n) * 0.5 + (1 - px_n) * 0.3 + rng_n * 0.2

    for c in sorted(conds, key=lambda x: -scores.get(x, 0)):
        s = scores.get(c, float("nan"))
        print(f"  {cond_short[c]:<16}  score={s:.3f}  "
              f"az_mae={mae_vals[c]:.2f}°  px_err={px_vals[c]:.0f}px  "
              f"az_range={range_vals[c]:.1f}°")
    print()


def save_results(all_metrics: dict, path: Path):
    path.write_text(json.dumps(all_metrics, indent=2))
    print(f"Results saved → {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=["all"])
    ap.add_argument("--conds", nargs="+", default=["all"])
    ap.add_argument("--max_frames", type=int, default=500)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--save_json", type=str, default="test/full_eval/results.json")
    args = ap.parse_args()

    skip = args.skip_existing and not args.no_skip

    clips = list(EVAL_CLIPS.keys()) if args.clips == ["all"] else args.clips
    conds = list(EVAL_CONDS.keys()) if args.conds == ["all"] else args.conds

    # 이미 결과 있으면 로드
    results_path = Path(args.save_json)
    all_metrics = {}
    if results_path.exists():
        all_metrics = json.loads(results_path.read_text())
        print(f"Loaded existing results from {results_path}")

    if not args.report_only:
        total = len(clips) * len(conds)
        done = 0
        for clip in clips:
            clip_info = EVAL_CLIPS.get(clip, dict(cls=clip.split("-")[0], speed="medium"))
            all_metrics.setdefault(clip, {})
            for cond in conds:
                cond_info = EVAL_CONDS[cond]
                traj_path = OUT_ROOT / f"{clip}__{cond}" / "traj.json"
                done += 1

                if skip and traj_path.exists():
                    print(f"[{done}/{total}] {clip}/{cond}  [skip - traj exists]")
                else:
                    print(f"[{done}/{total}] {clip}/{cond}  ...", flush=True)
                    t0 = time.time()
                    res = run_pipeline(clip, clip_info, cond, cond_info,
                                       args.max_frames, skip_existing=False)
                    if res["status"] == "error":
                        print(f"  ERROR: {res.get('error', '')[:100]}")
                        continue
                    print(f"  done in {time.time()-t0:.1f}s")

                # compute metrics
                m = compute_metrics(clip, cond, max_frames=args.max_frames)
                if m:
                    all_metrics[clip][cond] = m
                    print(f"  az_mae={m['az_mae_deg']:.2f}°  px={m['center_err_px']:.0f}px  "
                          f"az_range={m['az_range_deg']:.1f}°  conf={m['conf_mean']:.3f}")

        save_results(all_metrics, results_path)

    # compute metrics for any clips that have traj but no metrics yet
    for clip in clips:
        all_metrics.setdefault(clip, {})
        for cond in conds:
            if cond not in all_metrics[clip]:
                traj_path = OUT_ROOT / f"{clip}__{cond}" / "traj.json"
                if traj_path.exists():
                    m = compute_metrics(clip, cond, max_frames=args.max_frames)
                    if m:
                        all_metrics[clip][cond] = m

    print_report(all_metrics, conds)

    # save final
    save_results(all_metrics, results_path)


if __name__ == "__main__":
    main()

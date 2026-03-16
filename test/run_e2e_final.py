#!/usr/bin/env python3
"""
run_e2e_final.py
================
22개 LaSOT 클립 E2E 최종 렌더:
  1. yw_sam2 트래커 (yw_det_threshold=0.99) 로 traj.json 생성
  2. Jupiter 악기 오디오 → HRTF binaural 렌더
  3. render_sample_overlays 스타일 오버레이 + binaural 오디오 mux

Output: test/e2e_final/{clip}/{clip}_e2e.mp4

Usage:
    cd /home/seung/mmhoa/vid2spatial_v2
    python test/run_e2e_final.py [--clips all | car-5 dog-1 ...]
                                  [--max_frames 300]
                                  [--skip_existing]
                                  [--audio_only]
                                  [--overlay_only]
"""

import argparse, json, math, os, sys, time, traceback, tempfile
from pathlib import Path

import cv2
import numpy as np

TEST_DIR  = Path(__file__).parent
V2_ROOT   = TEST_DIR.parent
LASOT     = Path("/home/seung/mmhoa/vid2spatial/data/lasot")
SOFA_PATH = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"
FFMPEG    = "/home/seung/miniforge3/bin/ffmpeg"
TRAJ_SRC  = TEST_DIR / "yw_sam2_eval"   # 기존 traj 재사용 (skip_existing)
OUT_ROOT  = TEST_DIR / "e2e_final"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Jupiter 악기 음원 (62.7s, 48kHz stereo → foa_render가 자동으로 mono로 변환)
AUDIO_SRC = Path("/home/seung/mmhoa/pro/Jupiter_pro_stereo_ext.wav")

sys.path.insert(0, str(V2_ROOT))

# ── 22개 클립 정의 ────────────────────────────────────────────────────────────
EVAL_CLIPS = {
    "car-5":         {"cls": "car",        "text": "car"},
    "car-10":        {"cls": "car",        "text": "car"},
    "car-13":        {"cls": "car",        "text": "car"},
    "dog-1":         {"cls": "dog",        "text": "dog"},
    "dog-3":         {"cls": "dog",        "text": "dog"},
    "dog-9":         {"cls": "dog",        "text": "dog"},
    "dog-14":        {"cls": "dog",        "text": "dog"},
    "horse-1":       {"cls": "horse",      "text": "horse"},
    "horse-3":       {"cls": "horse",      "text": "horse"},
    "horse-11":      {"cls": "horse",      "text": "horse"},
    "motorcycle-1":  {"cls": "motorcycle", "text": "motorcycle"},
    "motorcycle-3":  {"cls": "motorcycle", "text": "motorcycle"},
    "motorcycle-6":  {"cls": "motorcycle", "text": "motorcycle"},
    "skateboard-8":  {"cls": "skateboard", "text": "skateboard"},
    "skateboard-11": {"cls": "skateboard", "text": "skateboard"},
    "skateboard-17": {"cls": "skateboard", "text": "skateboard"},
    "train-16":      {"cls": "train",      "text": "train"},
    "train-17":      {"cls": "train",      "text": "train"},
    "drone-2":       {"cls": "drone",      "text": "flying drone"},
    "drone-6":       {"cls": "drone",      "text": "flying drone"},
    "drone-13":      {"cls": "drone",      "text": "flying drone"},
    "guitar-9":      {"cls": "guitar",     "text": "guitar"},
}

# 카테고리별 오디오 시작 오프셋 (초) — Jupiter 62.7s에서 다른 구간 사용
CAT_AUDIO_OFFSET = {
    "car":        0.0,
    "dog":        8.0,
    "horse":      16.0,
    "motorcycle": 24.0,
    "skateboard": 32.0,
    "train":      40.0,
    "drone":      48.0,
    "guitar":     56.0,
}

# 카테고리 색상 (BGR)
CAT_COLORS = {
    "car":        (0,   165, 255),
    "dog":        (0,   255, 128),
    "horse":      (64,  128, 255),
    "motorcycle": (255, 128,   0),
    "skateboard": (255,   0, 128),
    "train":      (128, 255,   0),
    "drone":      (200, 200,  50),
    "guitar":     (255,  50, 200),
}

CLIP_META = {
    "car-5":         {"cat": "car",        "motion": "slow pan left"},
    "car-10":        {"cat": "car",        "motion": "wide arc crossing"},
    "car-13":        {"cat": "car",        "motion": "fast lateral crossing"},
    "dog-1":         {"cat": "dog",        "motion": "running, large az swing"},
    "dog-3":         {"cat": "dog",        "motion": "walking toward camera"},
    "dog-9":         {"cat": "dog",        "motion": "running, erratic"},
    "dog-14":        {"cat": "dog",        "motion": "running, occlusion"},
    "horse-1":       {"cat": "horse",      "motion": "galloping, moderate arc"},
    "horse-3":       {"cat": "horse",      "motion": "galloping wide"},
    "horse-11":      {"cat": "horse",      "motion": "trotting, steady"},
    "motorcycle-1":  {"cat": "motorcycle", "motion": "road curve"},
    "motorcycle-3":  {"cat": "motorcycle", "motion": "slow approach"},
    "motorcycle-6":  {"cat": "motorcycle", "motion": "fast wide arc"},
    "skateboard-8":  {"cat": "skateboard", "motion": "trick, wide swing"},
    "skateboard-11": {"cat": "skateboard", "motion": "long run left-right"},
    "skateboard-17": {"cat": "skateboard", "motion": "stationary trick"},
    "train-16":      {"cat": "train",      "motion": "slow linear"},
    "train-17":      {"cat": "train",      "motion": "departing"},
    "drone-2":       {"cat": "drone",      "motion": "hovering, slight drift"},
    "drone-6":       {"cat": "drone",      "motion": "overhead static"},
    "drone-13":      {"cat": "drone",      "motion": "wide aerial arc"},
    "guitar-9":      {"cat": "guitar",     "motion": "held/strummed, close-up"},
}


# ── 1. GT 로드 ────────────────────────────────────────────────────────────────
def load_gt(clip: str, max_frames: int = 0):
    gt_path  = LASOT / clip / "groundtruth.txt"
    occ_path = LASOT / clip / "full_occlusion.txt"
    ov_path  = LASOT / clip / "out_of_view.txt"
    gt = []
    for line in gt_path.read_text().strip().split("\n"):
        parts = [float(x) for x in line.strip().replace("\t", ",").split(",")]
        if len(parts) >= 4:
            gt.append((parts[0] + parts[2]/2, parts[1] + parts[3]/2, parts[2], parts[3]))
    occ = [int(x.strip()) for x in occ_path.read_text().strip().split("\n")
           if x.strip().isdigit()] if occ_path.exists() else []
    ov  = [int(x.strip()) for x in ov_path.read_text().strip().split("\n")
           if x.strip().isdigit()] if ov_path.exists() else []
    if max_frames > 0:
        gt, occ, ov = gt[:max_frames], occ[:max_frames], ov[:max_frames]
    return gt, occ, ov


def get_init_bbox(clip: str):
    gt_path = LASOT / clip / "groundtruth.txt"
    first = gt_path.read_text().strip().split("\n")[0]
    p = [float(x) for x in first.strip().replace("\t", ",").split(",")]
    return (int(p[0]), int(p[1]), int(p[2]), int(p[3]))


# ── 2. 트래킹 (yw_sam2) ────────────────────────────────────────────────────────
def run_tracking(clip: str, clip_info: dict, max_frames: int,
                 out_dir: Path, skip_existing: bool) -> dict:
    """yw_sam2 트래커 실행 → traj.json 저장. 기존 traj_src 에서 복사 가능하면 복사."""
    traj_path = out_dir / "traj.json"

    # 기존 yw_sam2_eval 결과 재사용
    src_traj = TRAJ_SRC / clip / "traj.json"
    if skip_existing and traj_path.exists():
        print(f"  [track] skipped (existing)")
        return {"status": "skipped", "traj_path": str(traj_path)}
    if src_traj.exists() and skip_existing:
        import shutil
        shutil.copy2(src_traj, traj_path)
        print(f"  [track] copied from yw_sam2_eval")
        return {"status": "copied", "traj_path": str(traj_path)}

    img_dir = LASOT / clip / "img"
    if not img_dir.exists():
        return {"status": "error", "error": f"img dir not found: {img_dir}"}

    init_bbox = get_init_bbox(clip)
    t0 = time.time()

    try:
        from vid2spatial_pkg.v2_spatial_tracker import V2SpatialTracker
        from vid2spatial_pkg.temporal_smoother import smooth_trajectory_batch

        tracker = V2SpatialTracker(depth_fn=None, depth_backend="midas", fov_deg=60.0)

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"e2e_{clip}_"))
        try:
            imgs_all = sorted(img_dir.glob("*.jpg"))
            if max_frames > 0:
                imgs_all = imgs_all[:max_frames]
            for i, img_p in enumerate(imgs_all, start=1):
                (tmp_dir / f"{i:08d}.jpg").symlink_to(img_p.resolve())

            traj_raw = tracker.track(
                video_path=str(tmp_dir / "%08d.jpg"),
                cls_name=clip_info["cls"],
                text_query=clip_info["text"],
                init_bbox=init_bbox,
                sample_stride=1,
                method="yw_sam2",
                enhance_depth=False,
            )
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        fps = float(traj_raw.get("fps", 25.0))
        traj_raw["frames"] = smooth_trajectory_batch(
            traj_raw["frames"], fps=fps,
            process_noise=0.01, measurement_noise=0.1,
        )
        traj_path.write_text(json.dumps(traj_raw, indent=2))
        elapsed = time.time() - t0
        print(f"  [track] done in {elapsed:.1f}s → {traj_path}")
        return {"status": "ok", "traj_path": str(traj_path), "wall_sec": elapsed}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


# ── 3. 오디오 렌더 ────────────────────────────────────────────────────────────
def render_audio(clip: str, cat: str, traj_path: Path, out_dir: Path) -> Path:
    """traj.json + Jupiter 악기 음원 → binaural WAV."""
    import soundfile as sf

    binaural_path = out_dir / "binaural.wav"

    # 이미 존재하면 재사용
    if binaural_path.exists():
        print(f"  [audio] reusing existing binaural.wav")
        return binaural_path

    with open(traj_path) as f:
        trajectory = json.load(f)

    frames = trajectory.get("frames", [])
    fps    = float(trajectory.get("fps", 25.0))
    n_sec  = len(frames) / fps
    sr_audio = 48000

    # Jupiter 음원에서 카테고리별 오프셋으로 n_sec 추출 → 임시 mono wav
    offset_sec = CAT_AUDIO_OFFSET.get(cat, 0.0)
    src_audio, sr_src = sf.read(str(AUDIO_SRC), dtype="float32")
    if src_audio.ndim == 2:
        src_audio = src_audio.mean(axis=1)

    # 리샘플링 (source가 다른 sr일 경우)
    if sr_src != sr_audio:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_audio, sr_src)
        src_audio = resample_poly(src_audio, sr_audio // g, sr_src // g)

    start_smp = int(offset_sec * sr_audio)
    n_smp     = int(n_sec * sr_audio) + sr_audio  # 1s 여유
    chunk = src_audio[start_smp : start_smp + n_smp]

    # 부족하면 반복
    while len(chunk) < n_smp:
        chunk = np.concatenate([chunk, src_audio])
    chunk = chunk[:n_smp]

    # 임시 mono wav 저장
    tmp_wav = out_dir / "_tmp_mono.wav"
    sf.write(str(tmp_wav), chunk, sr_audio)

    # binaural 렌더
    try:
        sys.path.insert(0, str(V2_ROOT))
        from vid2spatial_pkg.foa_render import render_binaural_from_trajectory

        result = render_binaural_from_trajectory(
            audio_path=str(tmp_wav),
            trajectory=trajectory,
            output_path=str(binaural_path),
            sofa_path=SOFA_PATH,
            smooth_ms=50.0,
            dist_gain_k=1.5,
            dist_lpf_min_hz=1200.0,
            dist_lpf_max_hz=8000.0,
            apply_reverb=False,
            block_ms=10.0,
            gain_mode="bbox_area",
        )
        print(f"  [audio] binaural rendered → {binaural_path}")
    except Exception as e:
        traceback.print_exc()
        print(f"  [audio] ERROR: {e}")
        binaural_path = None
    finally:
        if tmp_wav.exists():
            tmp_wav.unlink()

    return binaural_path


# ── 4. 오버레이 렌더 ──────────────────────────────────────────────────────────
def render_overlay(clip: str, traj_path: Path, out_dir: Path, binaural_path: Path) -> Path:
    """traj.json → 오버레이 MP4 + binaural 오디오 mux."""

    final_mp4 = out_dir / f"{clip}_e2e.mp4"

    with open(traj_path) as f:
        traj = json.load(f)

    frames_list = traj.get("frames", [])
    intr = traj.get("intrinsics", {})
    fps  = float(traj.get("fps", 25.0))
    W_orig = intr.get("width", frames_list[0].get("width", 1280) if frames_list else 1280)
    H_orig = intr.get("height", frames_list[0].get("height", 720) if frames_list else 720)
    fov_deg = intr.get("fov_deg", 60.0)
    f_px = (W_orig / 2) / math.tan(math.radians(fov_deg / 2))

    # GT 로드
    meta = CLIP_META.get(clip, {"cat": "car", "motion": ""})
    cat  = meta["cat"]
    color = CAT_COLORS.get(cat, (200, 200, 200))
    gt, occ, ov = load_gt(clip, max_frames=len(frames_list))

    # GT az 계산
    gt_az = []
    for (gcx, gcy, gw, gh) in gt:
        dx = gcx - W_orig / 2
        gt_az.append(math.atan2(dx, f_px))

    # 추정 az 시퀀스
    est_az = [fd["az"] for fd in frames_list]

    # az MAE (visible frames only)
    valid = [(g, e) for i, (g, e) in enumerate(zip(gt_az[:len(est_az)], est_az))
             if i < len(occ) and occ[i] == 0 and i < len(ov) and ov[i] == 0]
    az_mae = float(np.mean([abs(math.degrees(g - e)) for g, e in valid])) if valid else 0.0

    # by-frame lookup
    by_frame = {}
    for fd in frames_list:
        az, el = fd["az"], fd["el"]
        xn = math.tan(az)
        yn = math.tan(el) * math.sqrt(1 + xn * xn)
        fd["_cx"] = W_orig / 2 + f_px * xn
        fd["_cy"] = H_orig / 2 + f_px * yn
        by_frame[fd["frame"]] = fd

    # img dir
    img_dir = LASOT / clip / "img"
    imgs = sorted(img_dir.glob("*.jpg"))[:len(frames_list)]

    # output size
    OUT_W, OUT_H = 960, 540
    GRAPH_H = 90
    FULL_H  = OUT_H + GRAPH_H
    sx = OUT_W / W_orig
    sy = OUT_H / H_orig

    raw_mp4 = out_dir / "_overlay_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_mp4), fourcc, fps, (OUT_W, FULL_H))

    trail = []
    n_frames = len(imgs)

    for fidx, img_p in enumerate(imgs):
        frame = cv2.imread(str(img_p))
        if frame is None:
            continue
        frame = cv2.resize(frame, (OUT_W, OUT_H))

        fd = by_frame.get(fidx)

        # ── GT bbox (초록, 점선 느낌으로 얇게) ──────────────────────────────
        if fidx < len(gt):
            gcx, gcy, gw, gh = gt[fidx]
            gx1 = int((gcx - gw/2) * sx); gy1 = int((gcy - gh/2) * sy)
            gx2 = int((gcx + gw/2) * sx); gy2 = int((gcy + gh/2) * sy)
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (50, 200, 50), 1, cv2.LINE_AA)
            # GT 중심
            cv2.circle(frame, (int(gcx * sx), int(gcy * sy)), 3, (50, 220, 50), -1, cv2.LINE_AA)

        if fd:
            cx  = float(fd.get("cx", fd["_cx"])) * sx
            cy  = float(fd.get("cy", fd["_cy"])) * sy
            bw  = float(fd.get("w", 60)) * sx
            bh  = float(fd.get("h", 40)) * sy
            az_d = math.degrees(fd["az"])
            el_d = math.degrees(fd["el"])
            dist = fd.get("dist_m", 1.5)
            conf = fd.get("confidence", 0.5)
            d_rel = fd.get("d_rel", 0.5)
            bk   = fd.get("depth_backend", "?")

            # trail
            trail.append((cx, cy))
            if len(trail) > 60:
                trail.pop(0)
            for j in range(1, len(trail)):
                a = j / len(trail)
                tc = (int(color[0]*a*0.6), int(color[1]*a*0.6), int(color[2]*a*0.8))
                cv2.line(frame,
                         (int(trail[j-1][0]), int(trail[j-1][1])),
                         (int(trail[j][0]),   int(trail[j][1])),
                         tc, max(1, int(2*a)), cv2.LINE_AA)

            # 추정 bbox
            x1, y1 = int(cx - bw/2), int(cy - bh/2)
            cv2.rectangle(frame, (x1, y1), (int(cx+bw/2), int(cy+bh/2)), color, 2, cv2.LINE_AA)
            cv2.circle(frame, (int(cx), int(cy)), 5, color, -1, cv2.LINE_AA)

            # ── HUD (왼쪽) ───────────────────────────────────────────────────
            hud_lines = [
                (f"Az  {az_d:+.1f} deg", (180, 220, 255)),
                (f"El  {el_d:+.1f} deg", (180, 220, 255)),
                (f"Dst {dist:.2f} m",   (255, 200, 150)),
                (f"d_r {d_rel:.2f}",    (180, 180, 255)),
                (f"Conf {conf:.2f}",    (200, 255, 200)),
                (f"Bk  {bk}",           (150, 150, 150)),
            ]
            for k, (txt, hcol) in enumerate(hud_lines):
                cv2.putText(frame, txt, (8, 22 + k * 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, hcol, 1, cv2.LINE_AA)

            # ── Spatial gauge (오른쪽 위) ─────────────────────────────────
            gx, gy, gr = OUT_W - 60, 55, 38
            cv2.circle(frame, (gx, gy), gr, (50, 50, 50), -1, cv2.LINE_AA)
            cv2.circle(frame, (gx, gy), gr, color, 1, cv2.LINE_AA)
            # concentric circles (거리)
            for ri in [0.3, 0.6, 0.9]:
                cv2.circle(frame, (gx, gy), int(gr * ri), (80, 80, 80), 1, cv2.LINE_AA)
            # N/S/E/W ticks
            for ang_t in [0, 90, 180, 270]:
                a_r = math.radians(ang_t - 90)
                ex = int(gx + (gr - 3) * math.cos(a_r))
                ey = int(gy + (gr - 3) * math.sin(a_r))
                cv2.circle(frame, (ex, ey), 2, (120, 120, 120), -1, cv2.LINE_AA)
            # az needle
            a_r = math.radians(az_d - 90)
            ex = int(gx + (gr - 5) * math.cos(a_r))
            ey = int(gy + (gr - 5) * math.sin(a_r))
            cv2.arrowedLine(frame, (gx, gy), (ex, ey), (0, 255, 255), 2,
                            cv2.LINE_AA, tipLength=0.25)
            # dist dot
            d_norm = max(0.05, min(0.95, d_rel))
            dx_n = int(gx + (gr - 5) * d_norm * math.cos(a_r))
            dy_n = int(gy + (gr - 5) * d_norm * math.sin(a_r))
            cv2.circle(frame, (dx_n, dy_n), 4, (255, 200, 50), -1, cv2.LINE_AA)
            cv2.putText(frame, f"{az_d:+.0f}d", (gx - 18, gy + gr + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # ── 상단 클립 정보 ─────────────────────────────────────────────────
        az_mae_str = f"AzMAE {az_mae:.2f}deg"
        cv2.putText(frame, f"{clip} | {meta.get('motion','')} | {az_mae_str}",
                    (8, OUT_H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
        # frame counter
        cv2.putText(frame, f"{fidx+1}/{n_frames}", (OUT_W - 80, OUT_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        # progress bar
        pb_w = int(OUT_W * (fidx + 1) / max(n_frames, 1))
        cv2.rectangle(frame, (0, OUT_H - 3), (pb_w, OUT_H - 1), color, -1)

        # ── 하단 az 그래프 패널 ───────────────────────────────────────────
        graph = np.full((GRAPH_H, OUT_W, 3), 18, dtype=np.uint8)
        G_AZ_RANGE = 60.0
        def gx_f(i): return int(10 + (OUT_W - 20) * i / max(n_frames - 1, 1))
        def gy_f(az_val): return int(GRAPH_H / 2 - (GRAPH_H / 2 - 8) * az_val / G_AZ_RANGE)

        # zero line
        cv2.line(graph, (10, GRAPH_H // 2), (OUT_W - 10, GRAPH_H // 2), (40, 40, 40), 1)
        # ±30° lines
        for ref_az in [-30.0, 30.0]:
            ry = gy_f(ref_az)
            cv2.line(graph, (10, ry), (OUT_W - 10, ry), (35, 35, 35), 1)

        # GT az (초록)
        for i in range(1, fidx + 1):
            if i < len(gt_az) and i - 1 < len(gt_az):
                cv2.line(graph,
                         (gx_f(i-1), gy_f(math.degrees(gt_az[i-1]))),
                         (gx_f(i),   gy_f(math.degrees(gt_az[i]))),
                         (50, 200, 50), 1, cv2.LINE_AA)

        # 추정 az (카테고리 색)
        for i in range(1, fidx + 1):
            if i < len(est_az) and i - 1 < len(est_az):
                cv2.line(graph,
                         (gx_f(i-1), gy_f(math.degrees(est_az[i-1]))),
                         (gx_f(i),   gy_f(math.degrees(est_az[i]))),
                         color, 1, cv2.LINE_AA)

        # 현재 위치 마커
        if fidx < len(est_az):
            cv2.circle(graph, (gx_f(fidx), gy_f(math.degrees(est_az[fidx]))), 3,
                       (0, 255, 255), -1, cv2.LINE_AA)

        cv2.putText(graph, "Az(deg)", (12, GRAPH_H - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1)
        cv2.putText(graph, "GT", (OUT_W - 45, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (50, 200, 50), 1)
        cv2.putText(graph, "Est", (OUT_W - 45, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)

        combined = np.vstack([frame, graph])
        writer.write(combined)

    writer.release()

    # ffmpeg mux (video + binaural)
    if binaural_path and binaural_path.exists():
        cmd = (f"{FFMPEG} -y -i {raw_mp4} -i {binaural_path} "
               f"-c:v libx264 -crf 18 -preset fast "
               f"-c:a aac -b:a 256k -shortest "
               f"{final_mp4} -loglevel warning")
        ret = os.system(cmd)
        if ret != 0:
            print(f"  [mux] ffmpeg returned {ret}")
    else:
        cmd = (f"{FFMPEG} -y -i {raw_mp4} "
               f"-c:v libx264 -crf 18 -preset fast "
               f"{final_mp4} -loglevel warning")
        os.system(cmd)

    if raw_mp4.exists():
        raw_mp4.unlink()

    print(f"  [overlay] → {final_mp4}")
    return final_mp4


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="*", default=["all"])
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Reuse existing traj.json (default: True)")
    parser.add_argument("--no_skip", action="store_true",
                        help="Force re-run even if traj.json exists")
    parser.add_argument("--audio_only", action="store_true")
    parser.add_argument("--overlay_only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.no_skip:
        args.skip_existing = False

    clips = list(EVAL_CLIPS.keys()) if "all" in args.clips else args.clips
    clips = [c for c in clips if c in EVAL_CLIPS]

    print(f"E2E Final Render: {len(clips)} clips, max_frames={args.max_frames}")
    print(f"  skip_existing={args.skip_existing}, audio_src={AUDIO_SRC}")

    results = []
    t_global = time.time()

    for clip in clips:
        clip_info = EVAL_CLIPS[clip]
        meta = CLIP_META.get(clip, {"cat": "car", "motion": ""})
        cat  = meta["cat"]
        out_dir = OUT_ROOT / clip
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Clip: {clip}  [{cat}]  {meta.get('motion','')}")

        t0 = time.time()

        # 1. Tracking
        if not args.audio_only and not args.overlay_only:
            track_r = run_tracking(clip, clip_info, args.max_frames, out_dir, args.skip_existing)
            if track_r["status"] == "error":
                print(f"  [SKIP] tracking error: {track_r.get('error','')}")
                results.append({"clip": clip, "status": "track_error"})
                continue
        elif args.audio_only or args.overlay_only:
            # 기존 traj 복사
            src = TRAJ_SRC / clip / "traj.json"
            dst = out_dir / "traj.json"
            if not dst.exists() and src.exists():
                import shutil
                shutil.copy2(src, dst)

        traj_path = out_dir / "traj.json"
        if not traj_path.exists():
            print(f"  [SKIP] no traj.json")
            results.append({"clip": clip, "status": "no_traj"})
            continue

        # 2. Audio
        if not args.overlay_only:
            binaural_path = render_audio(clip, cat, traj_path, out_dir)
        else:
            binaural_path = out_dir / "binaural.wav"
            if not binaural_path.exists():
                binaural_path = render_audio(clip, cat, traj_path, out_dir)

        # 3. Overlay
        final = render_overlay(clip, traj_path, out_dir, binaural_path)

        elapsed = time.time() - t0
        results.append({"clip": clip, "status": "ok",
                        "output": str(final), "elapsed": elapsed})
        print(f"  [done] {elapsed:.1f}s")

    total = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"Total: {total:.1f}s  ({len([r for r in results if r['status']=='ok'])}/{len(clips)} ok)")
    print(f"Output dir: {OUT_ROOT}")
    for r in results:
        st = r['status']
        if st == 'ok':
            print(f"  ✓ {r['clip']} ({r.get('elapsed',0):.1f}s) → {Path(r['output']).name}")
        else:
            print(f"  ✗ {r['clip']} [{st}]")


if __name__ == "__main__":
    main()

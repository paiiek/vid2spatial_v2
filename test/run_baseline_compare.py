"""
run_baseline_compare.py
Stereo pan baseline vs HRTF binaural 비교 렌더.

조건:
  baseline_pan      – stereo pan only (sin law, no distance gain, no HRTF)
  baseline_pan_lpf  – stereo pan + distance gain/LPF (no HRTF)
  hrtf_binaural     – full HRTF binaural (A_tuned: bbox_area, k=1.5, min=0.1)

공통: 동일한 traj.json(BT_s1_midas, final setting), 동일한 오디오 소스

Usage:
    python test/run_baseline_compare.py [--clips car-5 dog-1 motorcycle-3]
"""
import argparse, json, math, os, shutil, sys, subprocess
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from vid2spatial_pkg.foa_render import (
    render_binaural_from_trajectory,
    interpolate_angles_distance,
    apply_distance_gain_lpf,
    render_stereo_pan_baseline,
    render_stereo_pan_reverb_baseline,
)

# ── 경로 ──────────────────────────────────────────────────────────────────────
LASOT_ROOT = "/home/seung/mmhoa/vid2spatial/data/lasot"
TRAJ_ROOT  = "test/full_eval"
SOFA_PATH  = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"
OUT_ROOT   = "test/baseline_compare"
FFMPEG     = "/home/seung/miniforge3/bin/ffmpeg"
TRAJ_COND  = "BT_s1_midas"   # full_eval 디렉토리 내 traj 조건명

# 전체 22 클립 (az_range 내림차순)
CLIPS = {
    "car-13":        {"audio": "/tmp/v2s_m3d_trajectory/car-13_m3d_v2.wav"},
    "car-10":        {"audio": "/tmp/v2s_m3d_trajectory/car-10_m3d_v2.wav"},
    "motorcycle-6":  {"audio": "/tmp/v2s_m3d_trajectory/motorcycle-6_m3d_v2.wav"},
    "dog-9":         {"audio": "/tmp/v2s_m3d_trajectory/dog-9_m3d_v2.wav"},
    "drone-13":      {"audio": "/tmp/v2s_m3d_trajectory/drone-13_m3d_v2.wav"},
    "motorcycle-1":  {"audio": "/tmp/v2s_m3d_trajectory/motorcycle-1_m3d_v2.wav"},
    "dog-1":         {"audio": "/tmp/v2s_m3d_trajectory/dog-1_m3d_v2.wav"},
    "skateboard-11": {"audio": "/tmp/v2s_m3d_trajectory/skateboard-11_m3d_v2.wav"},
    "skateboard-8":  {"audio": "/tmp/v2s_m3d_trajectory/skateboard-8_m3d_v2.wav"},
    "dog-14":        {"audio": "/tmp/v2s_m3d_trajectory/dog-14_m3d_v2.wav"},
    "horse-3":       {"audio": "/tmp/v2s_m3d_trajectory/horse-3_m3d_v2.wav"},
    "horse-11":      {"audio": "/tmp/v2s_m3d_trajectory/horse-11_m3d_v2.wav"},
    "drone-2":       {"audio": "/tmp/v2s_m3d_trajectory/drone-2_m3d_v2.wav"},
    "guitar-9":      {"audio": "/tmp/v2s_m3d_trajectory/guitar-9_m3d_v2.wav"},
    "train-16":      {"audio": "/tmp/v2s_m3d_trajectory/train-16_m3d_v2.wav"},
    "skateboard-17": {"audio": "/tmp/v2s_m3d_trajectory/skateboard-17_m3d_v2.wav"},
    "car-5":         {"audio": "/tmp/v2s_m3d_trajectory/car-5_m3d_v2.wav"},
    "train-17":      {"audio": "/tmp/v2s_m3d_trajectory/train-17_m3d_v2.wav"},
    "dog-3":         {"audio": "/tmp/v2s_m3d_trajectory/dog-3_m3d_v2.wav"},
    "motorcycle-3":  {"audio": "/tmp/v2s_m3d_trajectory/motorcycle-3_m3d_v2.wav"},
    "horse-1":       {"audio": "/tmp/v2s_m3d_trajectory/horse-1_m3d_v2.wav"},
    "drone-6":       {"audio": "/tmp/v2s_m3d_trajectory/drone-6_m3d_v2.wav"},
}

# A_tuned 공통 렌더 파라미터
HRTF_PARAMS = dict(
    gain_mode="bbox_area", metric_alpha=0.5,
    gain_k=1.5, gain_min=0.1, lpf_min=1200, lpf_max=8000,
    confidence_fade=False,
)

CONDITIONS = {
    "baseline_pan": dict(
        label="Stereo Pan\n(sin-law, no gain, no HRTF)",
        color=(100, 100, 255),
    ),
    "baseline_pan_lpf": dict(
        label="Stereo Pan + Gain/LPF\n(no HRTF)",
        color=(100, 200, 255),
    ),
    "hrtf_binaural": dict(
        label="HRTF Binaural\n(A_tuned: bbox_area, k=1.5)",
        color=(60, 255, 160),
    ),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.50
FONT_THICK = 1
LINE_H     = 19


# ── 오디오 유틸 ───────────────────────────────────────────────────────────────

def loop_audio_to_duration(audio: np.ndarray, sr: int, target_samples: int) -> np.ndarray:
    """audio를 target_samples 길이에 맞게 루프(타일) 후 잘라 반환."""
    if len(audio) >= target_samples:
        return audio[:target_samples]
    reps = math.ceil(target_samples / len(audio))
    return np.tile(audio, reps)[:target_samples]


# ── 오디오 렌더 ───────────────────────────────────────────────────────────────

def render_baseline_pan(audio_path, traj, out_path):
    """Stereo pan only — sin-law panning, no distance effects, no HRTF."""
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    frames = traj["frames"]
    fps = float(traj.get("fps", traj.get("intrinsics", {}).get("fps", 25.0)))
    target = int(math.ceil(len(frames) / fps * sr))
    audio = loop_audio_to_duration(audio, sr, target)
    T = audio.shape[0]

    az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(
        frames, T, sr, fps=fps, gain_mode="depth_rel")

    # Pure sin-law panning, no gain, no LPF
    az = az_s[:T]
    pan = np.sin(az)           # az>0=RIGHT → sin>0 → pan_angle>pi/4 → R louder (correct)
    pan_angle = (pan + 1) * (math.pi / 4)
    L = audio * np.cos(pan_angle)
    R = audio * np.sin(pan_angle)
    stereo = np.stack([L, R], axis=0).astype(np.float32)
    peak = float(np.max(np.abs(stereo)) + 1e-9)
    if peak > 1.0:
        stereo /= (peak * 1.01)
    sf.write(out_path, stereo.T, sr, subtype="FLOAT")
    return sr, az_s, el_s, dist_s, d_rel_s, frames


def render_baseline_pan_lpf(audio_path, traj, out_path):
    """Stereo pan + distance gain/LPF — no HRTF, but same d_rel processing as A_tuned."""
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    frames = traj["frames"]
    fps = float(traj.get("fps", traj.get("intrinsics", {}).get("fps", 25.0)))
    target = int(math.ceil(len(frames) / fps * sr))
    audio = loop_audio_to_duration(audio, sr, target)
    T = audio.shape[0]
    intr = traj.get("intrinsics", {})
    img_w = int(intr.get("width",  frames[0].get("width",  1280)))
    img_h = int(intr.get("height", frames[0].get("height",  720)))

    az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(
        frames, T, sr, fps=fps, gain_mode="bbox_area",
        img_w=img_w, img_h=img_h)

    # Same gain/LPF as A_tuned
    audio_proc = apply_distance_gain_lpf(
        audio, sr, dist_s, d_rel_s,
        gain_k=HRTF_PARAMS["gain_k"],
        lpf_min_hz=HRTF_PARAMS["lpf_min"],
        lpf_max_hz=HRTF_PARAMS["lpf_max"],
        gain_min=HRTF_PARAMS["gain_min"],
    )

    # Sin-law panning: az>0=RIGHT → sin(az)>0 → pan_angle>pi/4 → R louder
    az = az_s[:T]
    pan = np.sin(az)
    pan_angle = (pan + 1) * (math.pi / 4)
    L = audio_proc * np.cos(pan_angle)
    R = audio_proc * np.sin(pan_angle)
    stereo = np.stack([L, R], axis=0).astype(np.float32)
    peak = float(np.max(np.abs(stereo)) + 1e-9)
    if peak > 1.0:
        stereo /= (peak * 1.01)
    sf.write(out_path, stereo.T, sr, subtype="FLOAT")
    return sr, az_s, el_s, dist_s, d_rel_s, frames


def render_hrtf(audio_path, traj, out_path):
    """Full HRTF binaural (A_tuned parameters). Audio looped to traj duration."""
    import tempfile
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    frames = traj["frames"]
    fps = float(traj.get("fps", traj.get("intrinsics", {}).get("fps", 25.0)))
    target = int(math.ceil(len(frames) / fps * sr))
    audio_looped = loop_audio_to_duration(audio, sr, target)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, audio_looped, sr, subtype="FLOAT")
        render_binaural_from_trajectory(
            audio_path=tmp_path,
            trajectory=traj,
            output_path=out_path,
            sofa_path=SOFA_PATH,
            gain_mode=HRTF_PARAMS["gain_mode"],
            metric_alpha=HRTF_PARAMS["metric_alpha"],
            dist_gain_k=HRTF_PARAMS["gain_k"],
            dist_lpf_min_hz=HRTF_PARAMS["lpf_min"],
            dist_lpf_max_hz=HRTF_PARAMS["lpf_max"],
            apply_reverb=True, rt60=0.4,
            use_confidence_fade=HRTF_PARAMS["confidence_fade"],
        )
    finally:
        os.remove(tmp_path)


# ── d_rel 재계산 (오버레이용) ─────────────────────────────────────────────────

def get_d_rel_per_frame(frames, gain_mode, img_w, img_h):
    """frame별 d_rel (per-clip normalized)."""
    area = np.array([f.get("w", 80) * f.get("h", 80) / (img_w * img_h)
                     for f in frames], dtype=np.float32)
    dist = np.array([f.get("dist_m", 1.0) for f in frames], dtype=np.float32)
    if gain_mode == "bbox_area":
        a_min, a_max = area.min(), area.max()
        if (a_max - a_min) < 1e-4:
            return np.full(len(frames), 0.5)
        return 1.0 - np.clip((area - a_min) / (a_max - a_min), 0, 1)
    else:  # depth_rel
        d_min, d_max = dist.min(), dist.max()
        if (d_max - d_min) < 0.1:
            return np.full(len(frames), 0.5)
        return np.clip((dist - d_min) / (d_max - d_min), 0, 1)


# ── 오버레이 비디오 ───────────────────────────────────────────────────────────

def draw_panel(cell, lines, color, y0=8):
    pad = 5
    bw = 230; bh = LINE_H * len(lines) + pad * 2
    ov = cell.copy()
    cv2.rectangle(ov, (8, y0), (8 + bw, y0 + bh), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.55, cell, 0.45, 0, cell)
    for i, line in enumerate(lines):
        y = y0 + pad + LINE_H * (i + 1) - 3
        cv2.putText(cell, line, (13, y), FONT, FONT_SCALE, color, FONT_THICK, cv2.LINE_AA)


def draw_vol_bar(cell, d_rel, color, gain_k, gain_min, lpf_min, lpf_max, show_lpf=True):
    """우측 하단: VOL + LPF 바."""
    H, W = cell.shape[:2]
    bw, bh = 16, 100
    spacing = bw + 6
    by = H - bh - 12

    actual_gain = 1.0 - (1.0 - gain_min) * (d_rel ** gain_k)

    def _bar(bx_, ratio, lbl_top, lbl_bot, bc):
        cv2.rectangle(cell, (bx_, by), (bx_ + bw, by + bh), (50, 50, 50), -1)
        fh = int(bh * max(0.0, min(1.0, ratio)))
        if fh > 0:
            cv2.rectangle(cell, (bx_, by + bh - fh), (bx_ + bw, by + bh), bc, -1)
        cv2.rectangle(cell, (bx_, by), (bx_ + bw, by + bh), (180, 180, 180), 1)
        cv2.putText(cell, lbl_bot, (bx_ - 2, by + bh + 14),
                    FONT, 0.36, bc, 1, cv2.LINE_AA)
        cv2.putText(cell, lbl_top, (bx_, by - 4),
                    FONT, 0.34, (200, 200, 200), 1, cv2.LINE_AA)

    bx = W - (spacing * (3 if show_lpf else 1)) - 10
    _bar(bx, actual_gain, "VOL", f"{actual_gain*100:.0f}%", color)

    if show_lpf:
        lp_lo = max(50., float(lpf_min)); lp_hi = max(lp_lo + 10., float(lpf_max))
        log_fc = math.log(lp_hi) - (math.log(lp_hi) - math.log(lp_lo)) * d_rel
        fc_hz  = math.exp(log_fc)
        lpf_ratio = (math.log(fc_hz) - math.log(lp_lo)) / (math.log(lp_hi) - math.log(lp_lo))
        _bar(bx + spacing, lpf_ratio, "LPF", f"{int(fc_hz)}Hz", (180, 120, 255))

        reverb_wet = 0.05 + 0.07 * d_rel
        _bar(bx + spacing * 2, reverb_wet / 0.12, "REV", f"{reverb_wet*100:.0f}%", (100, 200, 100))


def draw_bbox(cell, cx, cy, w, h, label, color):
    x1 = int(cx - w/2); y1 = int(cy - h/2)
    x2 = int(cx + w/2); y2 = int(cy + h/2)
    cv2.rectangle(cell, (x1, y1), (x2, y2), color, 2)
    r = 5; icx, icy = int(cx), int(cy)
    cv2.line(cell, (icx-r, icy), (icx+r, icy), color, 1)
    cv2.line(cell, (icx, icy-r), (icx, icy+r), color, 1)
    cv2.putText(cell, label, (max(x1, 0), max(y1-4, 14)),
                FONT, 0.48, color, 1, cv2.LINE_AA)


def make_overlay(clip, traj_frames, out_dir, fps=25.0, max_frames=0):
    img_dir = Path(LASOT_ROOT) / clip / "img"
    img_files = sorted(img_dir.glob("*.jpg"))
    if max_frames > 0:
        img_files = img_files[:max_frames]

    first = cv2.imread(str(img_files[0]))
    H, W = first.shape[:2]

    frame_map = {f["frame"]: f for f in traj_frames}

    # d_rel: baseline_pan uses depth_rel, others use bbox_area
    d_rel_depth = get_d_rel_per_frame(traj_frames, "depth_rel", W, H)
    d_rel_bbox  = get_d_rel_per_frame(traj_frames, "bbox_area",  W, H)
    dr_depth_map = {f["frame"]: d_rel_depth[i] for i, f in enumerate(traj_frames)}
    dr_bbox_map  = {f["frame"]: d_rel_bbox[i]  for i, f in enumerate(traj_frames)}

    cond_names = list(CONDITIONS.keys())
    out_w = W * len(cond_names); out_h = H
    tmp_avi = out_dir / "overlay_tmp.avi"
    out_mp4 = out_dir / "overlay.mp4"
    fourcc  = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(str(tmp_avi), fourcc, fps, (out_w, out_h))

    for fi, img_p in enumerate(img_files):
        frame0 = cv2.imread(str(img_p))
        if frame0 is None:
            continue

        cells = []
        traj_f = frame_map.get(fi)

        for cn in cond_names:
            cond  = CONDITIONS[cn]
            color = cond["color"]
            cell  = frame0.copy()

            if traj_f:
                cx     = traj_f.get("cx", W/2)
                cy     = traj_f.get("cy", H/2)
                bw_box = traj_f.get("w", 80)
                bh_box = traj_f.get("h", 80)
                dist_m = traj_f.get("dist_m", 0.0)
                az_deg = math.degrees(traj_f.get("az", 0.0))
                conf   = traj_f.get("confidence", 0.0)
                area_pct = bw_box * bh_box / (W * H) * 100.0

                if cn == "baseline_pan":
                    d_rel  = dr_depth_map.get(fi, 0.5)
                    actual_gain = 1.0  # no gain applied
                    info = [
                        "baseline_pan",
                        f"frame {fi:05d}",
                        "[Stereo pan only]",
                        f"az    : {az_deg:+6.1f} deg",
                        f"area% : {area_pct:.2f}%",
                        f"conf  : {conf:.2f}",
                        f"gain  : 100% (fixed)",
                        f"LPF   : OFF",
                        f"HRTF  : NO",
                    ]
                    bbox_label = f"{az_deg:+.1f}°"
                    draw_vol_bar(cell, d_rel, color,
                                 gain_k=1.0, gain_min=1.0,
                                 lpf_min=8000, lpf_max=8000, show_lpf=False)

                elif cn == "baseline_pan_lpf":
                    d_rel = dr_bbox_map.get(fi, 0.5)
                    actual_gain = 1.0 - (1.0 - HRTF_PARAMS["gain_min"]) * (d_rel ** HRTF_PARAMS["gain_k"])
                    lp_lo = float(HRTF_PARAMS["lpf_min"])
                    lp_hi = float(HRTF_PARAMS["lpf_max"])
                    log_fc = math.log(lp_hi) - (math.log(lp_hi) - math.log(lp_lo)) * d_rel
                    fc_hz  = math.exp(log_fc)
                    info = [
                        "baseline_pan_lpf",
                        f"frame {fi:05d}",
                        "[Stereo pan + Gain/LPF]",
                        f"az    : {az_deg:+6.1f} deg",
                        f"area% : {area_pct:.2f}%",
                        f"d_rel : {d_rel:.3f}",
                        f"gain  : {actual_gain*100:.0f}%",
                        f"LPF   : {int(fc_hz)} Hz",
                        f"HRTF  : NO",
                    ]
                    bbox_label = f"area {area_pct:.1f}%"
                    draw_vol_bar(cell, d_rel, color,
                                 gain_k=HRTF_PARAMS["gain_k"],
                                 gain_min=HRTF_PARAMS["gain_min"],
                                 lpf_min=HRTF_PARAMS["lpf_min"],
                                 lpf_max=HRTF_PARAMS["lpf_max"], show_lpf=True)

                else:  # hrtf_binaural
                    d_rel = dr_bbox_map.get(fi, 0.5)
                    actual_gain = 1.0 - (1.0 - HRTF_PARAMS["gain_min"]) * (d_rel ** HRTF_PARAMS["gain_k"])
                    lp_lo = float(HRTF_PARAMS["lpf_min"])
                    lp_hi = float(HRTF_PARAMS["lpf_max"])
                    log_fc = math.log(lp_hi) - (math.log(lp_hi) - math.log(lp_lo)) * d_rel
                    fc_hz  = math.exp(log_fc)
                    info = [
                        "hrtf_binaural (A_tuned)",
                        f"frame {fi:05d}",
                        "[HRTF + Gain/LPF + Reverb]",
                        f"az    : {az_deg:+6.1f} deg",
                        f"area% : {area_pct:.2f}%",
                        f"d_rel : {d_rel:.3f}",
                        f"gain  : {actual_gain*100:.0f}%",
                        f"LPF   : {int(fc_hz)} Hz",
                        f"HRTF  : YES (KEMAR)",
                    ]
                    bbox_label = f"area {area_pct:.1f}%"
                    draw_vol_bar(cell, d_rel, color,
                                 gain_k=HRTF_PARAMS["gain_k"],
                                 gain_min=HRTF_PARAMS["gain_min"],
                                 lpf_min=HRTF_PARAMS["lpf_min"],
                                 lpf_max=HRTF_PARAMS["lpf_max"], show_lpf=True)

                draw_bbox(cell, cx, cy, bw_box, bh_box, bbox_label, color)
            else:
                info = [cond_names[list(CONDITIONS.keys()).index(cn)],
                        f"frame {fi:05d}", "(no data)"]

            draw_panel(cell, info, color)
            cells.append(cell)

        vw.write(np.concatenate(cells, axis=1))

        if fi % 200 == 0:
            print(f"    frame {fi}/{len(img_files)}", end="\r", flush=True)

    vw.release()
    print(f"\n  written: {tmp_avi}")

    # hrtf_binaural.wav가 있으면 오디오 mux (stream_loop으로 비디오 길이에 맞게 루프)
    hrtf_wav = out_dir / "hrtf_binaural.wav"
    if hrtf_wav.exists():
        cmd = (f"{FFMPEG} -y "
               f"-stream_loop -1 -i {hrtf_wav} "
               f"-i {tmp_avi} "
               f"-map 1:v -map 0:a "
               f"-shortest "
               f"-vcodec libx264 -crf 20 -preset fast "
               f"-acodec aac -b:a 192k "
               f"-pix_fmt yuv420p {out_mp4} 2>/dev/null")
    else:
        cmd = (f"{FFMPEG} -y -i {tmp_avi} "
               f"-vcodec libx264 -crf 20 -preset fast "
               f"-pix_fmt yuv420p {out_mp4} 2>/dev/null")

    if os.system(cmd) == 0:
        os.remove(tmp_avi)
        print(f"  → {out_mp4}" + (" (with audio)" if hrtf_wav.exists() else ""))
    else:
        print(f"  [warn] ffmpeg failed, keeping {tmp_avi}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+",
                    default=list(CLIPS.keys()))
    ap.add_argument("--max_frames", type=int, default=500)
    ap.add_argument("--fps", type=float, default=0.0,
                    help="Output FPS (0=auto-detect from traj.json, default 25)")
    ap.add_argument("--skip_render", action="store_true",
                    help="Skip audio render (only regenerate overlay)")
    args = ap.parse_args()

    listen_dir = Path(OUT_ROOT) / "listen"
    listen_dir.mkdir(parents=True, exist_ok=True)

    for clip in args.clips:
        print(f"\n{'='*60}")
        print(f"[{clip}]")

        traj_p = Path(TRAJ_ROOT) / f"{clip}__{TRAJ_COND}" / "traj.json"
        if not traj_p.exists():
            print(f"  [skip] no traj: {traj_p}")
            continue
        traj = json.loads(traj_p.read_text())
        traj_frames = traj["frames"]

        audio_src = CLIPS[clip]["audio"]
        if not os.path.exists(audio_src):
            print(f"  [skip] no audio: {audio_src}")
            continue

        clip_dir = Path(OUT_ROOT) / clip
        clip_dir.mkdir(parents=True, exist_ok=True)

        # ── 오디오 렌더 ──────────────────────────────────────────────────────
        if not args.skip_render:
            print("  [audio render]")

            out_pan = clip_dir / "baseline_pan.wav"
            if not out_pan.exists():
                render_baseline_pan(audio_src, traj, str(out_pan))
                print(f"    → {out_pan.name}")
            else:
                print(f"    [skip] {out_pan.name} exists")

            out_lpf = clip_dir / "baseline_pan_lpf.wav"
            if not out_lpf.exists():
                render_baseline_pan_lpf(audio_src, traj, str(out_lpf))
                print(f"    → {out_lpf.name}")
            else:
                print(f"    [skip] {out_lpf.name} exists")

            out_hrtf = clip_dir / "hrtf_binaural.wav"
            if not out_hrtf.exists():
                render_hrtf(audio_src, traj, str(out_hrtf))
                print(f"    → {out_hrtf.name}")
            else:
                print(f"    [skip] {out_hrtf.name} exists")

            # 청취용 복사
            for fname, label in [
                ("baseline_pan.wav",     "baseline_pan"),
                ("baseline_pan_lpf.wav", "baseline_pan_lpf"),
                ("hrtf_binaural.wav",    "hrtf_binaural"),
            ]:
                src = clip_dir / fname
                dst = listen_dir / f"{clip}__{label}.wav"
                if src.exists():
                    shutil.copy2(src, dst)

        # ── 오버레이 비디오 ──────────────────────────────────────────────────
        print("  [overlay video]")
        overlay_fps = args.fps if args.fps > 0 else float(traj.get("fps", 25.0))
        make_overlay(clip, traj_frames, clip_dir,
                     fps=overlay_fps, max_frames=args.max_frames)

    print(f"\n{'='*60}")
    print("Done.")
    print(f"  Binaural files → {listen_dir}/")
    print(f"  Overlay videos → {OUT_ROOT}/{{clip}}/overlay.mp4")
    print()
    print("  Conditions compared:")
    print("    baseline_pan     : stereo sin-law pan only (no gain, no HRTF)")
    print("    baseline_pan_lpf : stereo pan + distance gain/LPF (no HRTF)")
    print("    hrtf_binaural    : HRTF + gain/LPF + reverb (A_tuned)")


if __name__ == "__main__":
    main()

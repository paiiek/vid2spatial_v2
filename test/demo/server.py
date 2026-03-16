#!/usr/bin/env python3
"""
Vid2Spatial Demo Server
- Upload video + draw init bbox → run pipeline → play binaural output
Usage:
    cd /home/seung/mmhoa/vid2spatial_v2/test/demo
    python server.py [--port 8090]
"""
import os, sys, json, uuid, threading, traceback, subprocess, shutil, time, argparse, math
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT    = SCRIPT_DIR.parent.parent
JOBS_DIR   = SCRIPT_DIR / "jobs"
SOFA_PATH  = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"
FFMPEG     = "/home/seung/miniforge3/bin/ffmpeg"
PYTHON     = "/home/seung/miniforge3/bin/python3"

sys.path.insert(0, str(V2_ROOT))

jobs = {}  # job_id → {"status": ..., "error": ..., "result": ...}

# ── OSC sender (lazy-init) ────────────────────────────────────────────────────
_osc_client = None   # pythonosc UDPClient
_osc_config  = {"enabled": False, "host": "127.0.0.1", "port": 9001, "format": "maxmsp"}

def _get_osc_client():
    global _osc_client, _osc_config
    if not _osc_config["enabled"]:
        return None
    try:
        from pythonosc.udp_client import SimpleUDPClient
        if _osc_client is None:
            _osc_client = SimpleUDPClient(_osc_config["host"], _osc_config["port"])
        return _osc_client
    except Exception:
        return None

def send_osc_trajectory(trajectory: dict, fps: float = 30.0):
    """traj.json → OSC messages at playback rate in background thread."""
    client = _get_osc_client()
    if client is None:
        return
    fmt = _osc_config.get("format", "maxmsp")
    frames = trajectory.get("frames", [])
    if not frames:
        return

    def _send():
        try:
            from pythonosc.osc_message_builder import OscMessageBuilder
            interval = 1.0 / fps
            for fr in frames:
                az_deg  = math.degrees(fr["az"])
                el_deg  = math.degrees(fr.get("el", 0.0))
                dist_m  = fr.get("dist_m", 1.0)
                d_rel   = fr.get("d_rel", 0.5)
                if fmt == "spat5":
                    # IRCAM Spat5: /source/1/aed  az el dist  (degrees, degrees, meters)
                    client.send_message("/spat5/source/1/aed",
                                        [float(az_deg), float(el_deg), float(dist_m)])
                    client.send_message("/spat5/source/1/gain",
                                        [float(1.0 - d_rel)])
                else:
                    # Max/MSP default: separate messages
                    client.send_message("/v2s/az",   float(az_deg))
                    client.send_message("/v2s/el",   float(el_deg))
                    client.send_message("/v2s/dist", float(dist_m))
                    client.send_message("/v2s/drel", float(d_rel))
                time.sleep(interval)
        except Exception as e:
            print(f"[OSC] send error: {e}")

    t = threading.Thread(target=_send, daemon=True)
    t.start()


def _render_sketch_video(frames: list, az_s, el_s, di_s, sr: int, T: int,
                         binaural, wav_path: Path, out_path: Path):
    """Render an overlay video visualizing the 3D sketch trajectory on a sphere diagram."""
    import cv2, numpy as np

    FPS   = 30
    W, H  = 960, 540
    n_vid = int((T / sr) * FPS)
    if n_vid < 2:
        return

    tmp_path = out_path.with_suffix(".tmp.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(tmp_path), fourcc, FPS, (W, H))

    # Pre-compute per-video-frame values
    for vi in range(n_vid):
        si = min(T - 1, int(vi * T / n_vid))
        az_deg  = math.degrees(float(az_s[si]))
        el_deg  = math.degrees(float(el_s[si]))
        dist_m  = float(di_s[si])

        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:] = (12, 12, 20)

        # ── Title ──
        cv2.putText(img, "Vid2Spatial — Sketch Mode", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 100, 60), 1, cv2.LINE_AA)
        cv2.putText(img, f"Frame {vi+1}/{n_vid}", (W - 140, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

        # ── Top-down circle (azimuth) ──
        cx_td, cy_td, r_td = 200, 260, 150
        cv2.circle(img, (cx_td, cy_td), r_td, (40, 40, 60), 1)
        for lbl, ang in [("F", 0), ("R", 90), ("B", 180), ("L", 270)]:
            lx = int(cx_td + (r_td + 14) * math.sin(math.radians(ang)))
            ly = int(cy_td - (r_td + 14) * math.cos(math.radians(ang)))
            cv2.putText(img, lbl, (lx - 5, ly + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 100), 1, cv2.LINE_AA)
        # Grid lines
        cv2.line(img, (cx_td - r_td, cy_td), (cx_td + r_td, cy_td), (30, 30, 50), 1)
        cv2.line(img, (cx_td, cy_td - r_td), (cx_td, cy_td + r_td), (30, 30, 50), 1)
        # All stroke points (faded)
        all_n = len(az_s)
        for fi in range(0, si, max(1, all_n // 200)):
            faz = float(az_s[fi]); fdist = float(di_s[fi])
            r_pt = int(r_td * 0.3 + r_td * 0.6 * (fdist - 1.0) / 7.0)
            px = int(cx_td + r_pt * math.sin(faz))
            py = int(cy_td - r_pt * math.cos(faz))
            cv2.circle(img, (px, py), 2, (60, 100, 80), -1)
        # Current position
        r_cur = int(r_td * 0.3 + r_td * 0.6 * (dist_m - 1.0) / 7.0)
        nx = int(cx_td + r_cur * math.sin(math.radians(az_deg)))
        ny = int(cy_td - r_cur * math.cos(math.radians(az_deg)))
        cv2.arrowedLine(img, (cx_td, cy_td), (nx, ny), (60, 200, 120), 2, tipLength=0.2)
        cv2.circle(img, (nx, ny), 8, (60, 200, 120), -1)
        cv2.putText(img, "TOP VIEW (Az)", (cx_td - 60, cy_td + r_td + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 100), 1, cv2.LINE_AA)

        # ── Side view circle (elevation) ──
        cx_sv, cy_sv, r_sv = 560, 260, 150
        cv2.circle(img, (cx_sv, cy_sv), r_sv, (40, 40, 60), 1)
        for lbl, pos in [("UP", (cx_sv, cy_sv - r_sv - 14)),
                         ("DN", (cx_sv, cy_sv + r_sv + 6)),
                         ("F",  (cx_sv + r_sv + 8, cy_sv + 5)),
                         ("B",  (cx_sv - r_sv - 18, cy_sv + 5))]:
            cv2.putText(img, lbl, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 100), 1, cv2.LINE_AA)
        cv2.line(img, (cx_sv - r_sv, cy_sv), (cx_sv + r_sv, cy_sv), (30, 30, 50), 1)
        cv2.line(img, (cx_sv, cy_sv - r_sv), (cx_sv, cy_sv + r_sv), (30, 30, 50), 1)
        # All points
        for fi in range(0, si, max(1, all_n // 200)):
            fel = float(el_s[fi]); fdist = float(di_s[fi])
            r_pt = int(r_sv * 0.3 + r_sv * 0.6 * (fdist - 1.0) / 7.0)
            faz2 = float(az_s[fi])
            px = int(cx_sv + r_pt * math.cos(faz2))
            py = int(cy_sv - r_pt * math.sin(fel))
            cv2.circle(img, (px, py), 2, (60, 100, 80), -1)
        # Current
        r_cur_s = int(r_sv * 0.3 + r_sv * 0.6 * (dist_m - 1.0) / 7.0)
        sx = int(cx_sv + r_cur_s * math.cos(math.radians(az_deg)))
        sy = int(cy_sv - r_cur_s * math.sin(math.radians(el_deg)))
        cv2.arrowedLine(img, (cx_sv, cy_sv), (sx, sy), (60, 200, 120), 2, tipLength=0.2)
        cv2.circle(img, (sx, sy), 8, (60, 200, 120), -1)
        cv2.putText(img, "SIDE VIEW (El)", (cx_sv - 65, cy_sv + r_sv + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 100), 1, cv2.LINE_AA)

        # ── HUD text ──
        hud_x, hud_y = 20, H - 120
        cv2.rectangle(img, (hud_x - 4, hud_y - 14), (400, H - 10), (20, 20, 30), -1)
        for i, (lbl, val, col) in enumerate([
            ("Az",    f"{az_deg:+.1f}°",  (100, 220, 255)),
            ("El",    f"{el_deg:+.1f}°",  (100, 200, 200)),
            ("Dist",  f"{dist_m:.2f} m",  (180, 180, 255)),
        ]):
            cv2.putText(img, f"{lbl}: {val}", (hud_x, hud_y + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 1, cv2.LINE_AA)

        # ── Az timeline bar ──
        bar_y0, bar_y1 = H - 38, H - 12
        bar_x0, bar_x1 = 20, W - 20
        cv2.rectangle(img, (bar_x0, bar_y0), (bar_x1, bar_y1), (25, 25, 35), -1)
        cv2.line(img, ((bar_x0+bar_x1)//2, bar_y0), ((bar_x0+bar_x1)//2, bar_y1), (50,50,70), 1)
        az_frac = (az_deg + 90) / 180.0
        az_bx   = int(bar_x0 + az_frac * (bar_x1 - bar_x0))
        cv2.rectangle(img, (az_bx - 3, bar_y0), (az_bx + 3, bar_y1), (60, 200, 120), -1)
        cv2.putText(img, "Az: L ←→ R", (bar_x0, bar_y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (70, 70, 90), 1, cv2.LINE_AA)

        writer.write(img)

    writer.release()

    # Mux with binaural audio using ffmpeg
    cmd = [FFMPEG, "-y", "-i", str(tmp_path), "-i", str(wav_path),
           "-c:v", "libx264", "-c:a", "aac", "-shortest",
           "-movflags", "+faststart", str(out_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        tmp_path.unlink(missing_ok=True)
    except Exception:
        # Fallback: rename tmp without audio
        shutil.move(str(tmp_path), str(out_path))


def render_overlay_video(frames_dir: Path, trajectory: dict,
                         out_path: Path, fps: float = 30.0):
    """traj.json + extracted frames → overlay.mp4 (no GT, demo mode)."""
    import cv2, numpy as np

    frames_list = trajectory.get("frames", [])
    if not frames_list:
        return False

    intr = trajectory.get("intrinsics", {})
    W    = int(intr.get("width",  640))
    H    = int(intr.get("height", 480))
    fov  = float(intr.get("fov_deg", 60.0))
    f_len = (W / 2) / math.tan(math.radians(fov / 2))

    frame_map = {fr["frame"]: fr for fr in frames_list}
    total     = max(fr["frame"] for fr in frames_list) + 1

    imgs = sorted(frames_dir.glob("*.jpg"))
    if not imgs:
        return False

    tmp_path = out_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (W, H))

    GRAPH_H = 72
    CAT_COLOR = (0, 165, 255)  # orange (demo mode)
    az_hist: list = []

    for img_p in sorted(imgs):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        # resize to traj intrinsics if needed
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H))

        idx = int(img_p.stem)  # 00000001 → 1
        frame_idx = idx - 1    # 1-indexed jpg → 0-indexed

        fr = frame_map.get(frame_idx)
        az_est = fr["az"] if fr else 0.0
        az_hist.append(az_est)

        # ── estimated bbox ──
        if fr is not None:
            bx = fr.get("cx"); by = fr.get("cy")
            bw = fr.get("w", 80); bh = fr.get("h", 80)
            conf = fr.get("confidence", 0.5)
            if bx is not None and by is not None:
                x1 = int(bx - bw/2); y1 = int(by - bh/2)
                x2 = int(bx + bw/2); y2 = int(by + bh/2)
                cv2.rectangle(img, (x1, y1), (x2, y2), CAT_COLOR, 2)
                cv2.putText(img, f"{conf:.2f}", (x1 + 2, min(y2 + 14, H - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, CAT_COLOR, 1, cv2.LINE_AA)

        # ── HUD (left panel) ──
        if fr is not None:
            az   = fr["az"]; el = fr.get("el", 0.0)
            dist = fr.get("dist_m", 1.0); conf = fr.get("confidence", 0.5)
            d_rel = fr.get("d_rel", 0.5); v_az = fr.get("v_az", 0.0)
            bknd  = fr.get("depth_backend", "midas")
            panel_w, panel_h = 178, 148
            ov = img.copy()
            cv2.rectangle(ov, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
            cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
            cv2.rectangle(img, (0, 0), (panel_w, panel_h), CAT_COLOR, 1)
            lines = [
                (f"[DEMO]  vid2spatial",          (200, 200, 200), 0.32),
                (f"Az  {math.degrees(az):+7.2f}°",(100, 220, 255), 0.40),
                (f"El  {math.degrees(el):+7.2f}°",(100, 200, 200), 0.38),
                (f"Dist  {dist:6.2f} m",          (180, 180, 255), 0.38),
                (f"d_rel  {d_rel:.3f}",           (160, 160, 200), 0.36),
                (f"v_az  {math.degrees(v_az):+.3f}°/f",(150, 200, 150), 0.34),
                (f"Conf  {conf:.3f}",             (180, 220, 100), 0.36),
                (f"Depth  {bknd}",                (150, 150, 150), 0.32),
            ]
            for i, (txt, col, sc) in enumerate(lines):
                cv2.putText(img, txt, (6, 14 + i * 17),
                            cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
            cv2.putText(img, f"{frame_idx+1:4d}/{total:4d}",
                        (panel_w + 6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.34, (160, 160, 160), 1, cv2.LINE_AA)

        # ── Spatial gauge (top-right) ──
        if fr is not None:
            r = 48; cx_ = W - r - 12; cy_ = r + 18
            cv2.circle(img, (cx_, cy_), r + 2, (20, 20, 20), -1)
            cv2.circle(img, (cx_, cy_), r,     (60, 60, 60), 1)
            cv2.line(img, (cx_-r, cy_), (cx_+r, cy_), (50,50,50), 1)
            cv2.line(img, (cx_, cy_-r), (cx_, cy_+r), (50,50,50), 1)
            for lbl, ang in [("F",0),("R",90),("B",180),("L",270)]:
                lx = int(cx_ + (r-6)*math.sin(math.radians(ang)))
                ly = int(cy_ - (r-6)*math.cos(math.radians(ang)))
                cv2.putText(img, lbl, (lx-4, ly+4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (100,100,100), 1, cv2.LINE_AA)
            nx = int(cx_ + int(r*0.85)*math.sin(math.degrees(fr["az"])*math.pi/180))
            ny = int(cy_ - int(r*0.85)*math.cos(math.degrees(fr["az"])*math.pi/180))
            cv2.arrowedLine(img, (cx_, cy_), (nx, ny), CAT_COLOR, 2, tipLength=0.25)

        # ── Az graph (bottom) ──
        if len(az_hist) >= 2:
            y0g = H - GRAPH_H - 8; y1g = y0g + GRAPH_H

            def az2y(a, _y1g=y1g, _gh=GRAPH_H):
                d = max(-60, min(60, math.degrees(a)))
                return int(_y1g - (d + 60) / 120.0 * _gh)

            ov2 = img.copy()
            cv2.rectangle(ov2, (0, y0g), (W, y1g), (20, 20, 20), -1)
            cv2.addWeighted(ov2, 0.65, img, 0.35, 0, img)
            for deg in [-30, -15, 0, 15, 30]:
                frac = (deg + 60) / 120.0
                yg = int(y1g - frac * GRAPH_H)
                cv2.line(img, (0, yg), (W, yg), (80,80,80) if deg!=0 else (120,120,120), 1)
            n = len(az_hist)
            for i in range(1, n):
                x1a = int((i-1)*W/max(n-1,1)); x2a = int(i*W/max(n-1,1))
                cv2.line(img, (x1a, az2y(az_hist[i-1])), (x2a, az2y(az_hist[i])),
                         (255, 140, 0), 2)
            xc = int((n-1)*W/max(n-1,1))
            cv2.circle(img, (xc, az2y(az_hist[-1])), 4, (255,200,50), -1)

        # ── progress bar ──
        yp = H - 4
        cv2.rectangle(img, (0, yp-4), (W, yp+4), (30,30,30), -1)
        filled = int(W * frame_idx / max(total-1, 1))
        cv2.rectangle(img, (0, yp-4), (filled, yp+4), (0,200,255), -1)

        writer.write(img)

    writer.release()

    # re-encode with ffmpeg for browser compatibility
    subprocess.run([
        FFMPEG, "-y", "-i", str(tmp_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        str(out_path), "-loglevel", "warning"
    ], capture_output=True)
    tmp_path.unlink(missing_ok=True)
    return out_path.exists()


def run_pipeline(job_id: str, video_path: str, audio_path: str,
                 bbox: tuple, out_dir: Path,
                 mode: str = "accurate",   # "realtime" | "accurate"
                 render_foa: bool = False):
    try:
        jobs[job_id]["status"] = "extracting_frames"
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Extract frames with rotation applied (ffmpeg autorotates, OpenCV does not)
        ret = subprocess.run([
            FFMPEG, "-y", "-i", video_path,
            "-vf", "scale=iw:ih",  # force transpose/rotation metadata to be baked in
            "-q:v", "2", str(frames_dir / "%08d.jpg"),
            "-loglevel", "warning"
        ], capture_output=True)
        if ret.returncode != 0:
            raise RuntimeError(f"ffmpeg frame extract failed: {ret.stderr.decode()}")

        # Check actual dimensions from extracted frames (rotation-applied)
        import cv2 as _cv2
        sample_imgs = sorted(frames_dir.glob("*.jpg"))
        if sample_imgs:
            _samp = _cv2.imread(str(sample_imgs[0]))
            actual_H, actual_W = _samp.shape[:2]
        else:
            actual_W, actual_H = 1920, 1080

        # Re-encode rotation-corrected mp4 for tracker (OpenCV reads raw video ignoring rotation)
        # Tracker needs a video where w/h matches the extracted frames
        normed_video = out_dir / "input_normed.mp4"
        subprocess.run([
            FFMPEG, "-y", "-i", video_path,
            "-vf", "scale=iw:ih",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an", str(normed_video), "-loglevel", "warning"
        ], capture_output=True)
        track_video = str(normed_video) if normed_video.exists() else video_path

        jobs[job_id]["status"] = "tracking"

        from vid2spatial_pkg.v2_spatial_tracker import V2SpatialTracker
        tracker = V2SpatialTracker(depth_backend="midas")

        # mode 선택: realtime=v1_bytetrack(빠름), accurate=yw_sam2(정확, SAM2)
        if mode == "realtime":
            # ByteTrack — COCO 클래스면 빠름, 아니면 내부에서 YW fallback
            trajectory = tracker.track(
                video_path=track_video,
                init_bbox=bbox,
                method="v1_bytetrack",
                enhance_depth=False,
            )
        else:
            # text_query="__demo__" → YOLO-World 검출 0 → SAM2가 init_bbox로 전담
            trajectory = tracker.track(
                video_path=track_video,
                init_bbox=bbox,
                method="yw_sam2",
                text_query="__demo__",
                yw_det_threshold=0.99,
                enhance_depth=False,
            )

        traj_path = out_dir / "traj.json"
        with open(traj_path, "w") as f:
            json.dump(trajectory, f, indent=2)

        jobs[job_id]["status"] = "rendering"

        # Render binaural
        from vid2spatial_pkg.foa_render import render_binaural_from_trajectory
        binaural_path = out_dir / "binaural.wav"
        render_binaural_from_trajectory(
            audio_path=audio_path,
            trajectory=trajectory,
            output_path=str(binaural_path),
            sofa_path=SOFA_PATH,
            smooth_ms=80.0,
            dist_gain_k=1.5,
            gain_min=0.10,
            dist_lpf_min_hz=1200.0,
            dist_lpf_max_hz=6000.0,
            apply_reverb=False,
            block_ms=10.0,
            gain_mode="bbox_area",
            d_rel_attack_s=0.7,
            d_rel_release_s=0.2,
        )

        # Get video fps
        fps_result = subprocess.run([
            FFMPEG, "-i", video_path, "-loglevel", "error"
        ], capture_output=True, text=True)
        import re as _re
        fps_m = _re.search(r"(\d+(?:\.\d+)?)\s*fps", fps_result.stderr)
        vid_fps = float(fps_m.group(1)) if fps_m else 30.0

        # Render FOA 4ch (optional)
        foa_path = None
        if render_foa:
            jobs[job_id]["status"] = "rendering_foa"
            from vid2spatial_pkg.foa_render import render_foa_from_trajectory
            foa_path = out_dir / "foa.wav"
            render_foa_from_trajectory(
                audio_path=audio_path,
                trajectory=trajectory,
                output_path=str(foa_path),
                smooth_ms=80.0,
                dist_gain_k=1.5,
                gain_min=0.10,
                dist_lpf_min_hz=1200.0,
                dist_lpf_max_hz=6000.0,
                apply_reverb=False,
                gain_mode="bbox_area",
                d_rel_attack_s=0.7,
                d_rel_release_s=0.2,
            )

        # Render overlay video
        jobs[job_id]["status"] = "rendering_overlay"
        overlay_path = out_dir / "overlay.mp4"
        render_overlay_video(frames_dir, trajectory, overlay_path, fps=vid_fps)

        # Mux overlay + binaural into preview mp4
        preview_path = out_dir / "preview.mp4"
        if overlay_path.exists():
            subprocess.run([
                FFMPEG, "-y",
                "-i", str(overlay_path),
                "-i", str(binaural_path),
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(preview_path), "-loglevel", "warning"
            ], capture_output=True)
        else:
            subprocess.run([
                FFMPEG, "-y",
                "-i", video_path,
                "-i", str(binaural_path),
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(preview_path), "-loglevel", "warning"
            ], capture_output=True)

        # OSC: trajectory 전송 (enabled이면)
        send_osc_trajectory(trajectory, fps=vid_fps)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "binaural": str(binaural_path),
            "foa":      str(foa_path) if foa_path and foa_path.exists() else None,
            "preview":  str(preview_path),
            "overlay":  str(overlay_path) if overlay_path.exists() else None,
            "traj":     str(traj_path),
            "n_frames": len(trajectory.get("frames", [])),
        }

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = traceback.format_exc()
        print(f"[job {job_id}] ERROR: {e}")


def parse_multipart(data: bytes, boundary: str):
    """Minimal multipart/form-data parser. Returns dict {name: value_or_bytes}."""
    sep = ("--" + boundary).encode()
    parts = {}
    chunks = data.split(sep)
    for chunk in chunks[1:]:
        if chunk.strip() in (b"", b"--", b"--\r\n"):
            continue
        header_end = chunk.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        header_raw = chunk[:header_end].decode("utf-8", errors="replace")
        body = chunk[header_end + 4:]
        if body.endswith(b"\r\n"):
            body = body[:-2]
        # parse Content-Disposition
        name = None
        filename = None
        for line in header_raw.splitlines():
            if "Content-Disposition" in line:
                for part in line.split(";"):
                    part = part.strip()
                    if part.startswith("name="):
                        name = part.split("=", 1)[1].strip('"')
                    elif part.startswith("filename="):
                        filename = part.split("=", 1)[1].strip('"')
        if name:
            parts[name] = (body, filename)
    return parts


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", ""):
            self._serve_file(SCRIPT_DIR / "index.html", "text/html")
        elif path == "/api/status":
            job_id = parse_qs(parsed.query).get("job_id", [None])[0]
            if not job_id or job_id not in jobs:
                self._json({"error": "not found"}, 404)
            else:
                self._json(jobs[job_id])
        elif path == "/api/osc_config":
            self._json(_osc_config)
        elif path == "/api/audio_sources":
            import glob
            sources = []
            base = "/home/seung/spatamb/Dataset"
            for wav in sorted(glob.glob(f"{base}/*/AuSep_*.wav"))[:80]:
                rel = wav.replace(base + "/", "")
                folder, fname = rel.split("/", 1)
                sources.append({"label": f"{folder} / {fname}", "path": wav})
            self._json({"sources": sources})
        elif path == "/api/sketch_file":
            # Serve files generated by sketch_render (binaural.wav, foa.wav, overlay.mp4)
            qs    = parse_qs(parsed.query)
            fname = qs.get("name", [""])[0]
            if not fname or "/" in fname:
                self._json({"error": "invalid"}, 404); return
            fpath = JOBS_DIR / "sketch_last" / fname
            if not fpath.exists():
                self._json({"error": "not found"}, 404); return
            ext   = fpath.suffix.lower()
            ctype = {".wav": "audio/wav", ".mp4": "video/mp4"}.get(ext, "application/octet-stream")
            self._serve_file(fpath, ctype)
        elif path.startswith("/jobs/"):
            # Serve job output files (binaural.wav, preview.mp4)
            rel = path[len("/jobs/"):]
            fpath = JOBS_DIR / rel
            if fpath.exists() and fpath.is_file():
                ext = fpath.suffix.lower()
                ctype = {".wav": "audio/wav", ".mp4": "video/mp4",
                         ".json": "application/json"}.get(ext, "application/octet-stream")
                self._serve_file(fpath, ctype)
            else:
                self._json({"error": "not found"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/upload":
            # Receive video file, save, return first frame as base64 jpeg
            ctype = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            boundary = ""
            for part in ctype.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part[9:].strip('"')

            parts = parse_multipart(body, boundary)
            if "video" not in parts:
                self._json({"error": "no video field"}, 400); return

            video_bytes, filename = parts["video"]
            job_id = str(uuid.uuid4())[:8]
            job_dir = JOBS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            ext = Path(filename or "video.mp4").suffix or ".mp4"
            video_path = job_dir / f"input{ext}"
            video_path.write_bytes(video_bytes)

            # Extract first frame for bbox drawing
            frame_path = job_dir / "frame0.jpg"
            subprocess.run([
                FFMPEG, "-y", "-i", str(video_path),
                "-vframes", "1", str(frame_path), "-loglevel", "warning"
            ], capture_output=True)

            # Get dimensions from the extracted frame (ffmpeg applies rotation metadata,
            # so frame0.jpg always has the correct display orientation)
            import cv2 as _cv2, base64
            _f = _cv2.imread(str(frame_path))
            if _f is not None:
                h, w = _f.shape[:2]
            else:
                w, h = 640, 480

            frame_b64 = base64.b64encode(frame_path.read_bytes()).decode()

            jobs[job_id] = {"status": "uploaded", "video_path": str(video_path),
                            "width": w, "height": h}
            self._json({"job_id": job_id, "frame": frame_b64, "width": w, "height": h})

        elif path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            job_id = data.get("job_id")
            bbox   = data.get("bbox")   # [x, y, w, h]
            audio_src = data.get("audio_src")  # path on server

            if not job_id or job_id not in jobs:
                self._json({"error": "invalid job_id"}, 400); return
            if not bbox or len(bbox) != 4:
                self._json({"error": "bbox must be [x,y,w,h]"}, 400); return

            job = jobs[job_id]
            if job["status"] not in ("uploaded", "done", "error"):
                self._json({"error": f"job in state {job['status']}"}); return

            video_path = job["video_path"]
            job_dir = Path(video_path).parent

            # audio source
            if not audio_src or not Path(audio_src).exists():
                # default: use first spatamb file
                import glob
                matches = glob.glob("/home/seung/spatamb/Dataset/08_Spring_fl_vn/AuSep_2_vn_08_Spring.wav")
                audio_src = matches[0] if matches else None
            if not audio_src:
                self._json({"error": "no audio source"}); return

            bbox_tuple = tuple(int(v) for v in bbox)
            mode       = data.get("mode", "accurate")        # "realtime" | "accurate"
            render_foa = bool(data.get("render_foa", False))

            jobs[job_id]["status"] = "queued"
            jobs[job_id]["result"] = None
            jobs[job_id]["error"] = None

            t = threading.Thread(
                target=run_pipeline,
                args=(job_id, video_path, audio_src, bbox_tuple, job_dir),
                kwargs={"mode": mode, "render_foa": render_foa},
                daemon=True
            )
            t.start()
            self._json({"job_id": job_id, "status": "queued"})

        elif path == "/api/upload_audio":
            # Receive mono audio file for a specific job (or "sketch" for sketch mode)
            job_id = parse_qs(parsed.query).get("job_id", [None])[0]
            if not job_id:
                self._json({"error": "missing job_id"}, 400); return
            # Allow "sketch" as a special job_id (no entry in jobs dict needed)
            if job_id != "sketch" and job_id not in jobs:
                self._json({"error": "invalid job_id"}, 400); return

            ctype = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            boundary = ""
            for part in ctype.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part[9:].strip('"')

            parts = parse_multipart(body, boundary)
            if "audio" not in parts:
                self._json({"error": "no audio field"}, 400); return

            audio_bytes, filename = parts["audio"]
            ext = Path(filename or "audio.wav").suffix or ".wav"
            job_dir = JOBS_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            audio_path = job_dir / f"input_audio{ext}"
            audio_path.write_bytes(audio_bytes)

            if job_id in jobs:
                jobs[job_id]["uploaded_audio"] = str(audio_path)
            self._json({"audio_path": str(audio_path)})

        elif path == "/api/sketch_render":
            # Sketch-mode: stroke_3d [{az,el,dist_m,t}] or legacy stroke [{x,y,t}]
            # Returns JSON {binaural_url, foa_url?, video_url?}
            # Files served from /api/sketch_file?name=xxx
            length = int(self.headers.get("Content-Length", 0))
            try:
                req          = json.loads(self.rfile.read(length))
                stroke_3d    = req.get("stroke_3d", [])
                stroke       = req.get("stroke", [])
                canvas_w     = int(req.get("canvas_w", 640))
                canvas_h     = int(req.get("canvas_h", 360))
                tone_hz      = float(req.get("tone_hz", 440.0))
                audio_src    = req.get("audio_src", None)
                render_foa   = bool(req.get("render_foa", False))
                render_video = bool(req.get("render_video", False))
                osc_enabled  = bool(req.get("osc_enabled", False))

                # ── Build frames list ──────────────────────────────────────────
                if len(stroke_3d) >= 2:
                    frames = []
                    for pt in stroke_3d:
                        az     = float(pt["az"])
                        el     = float(pt["el"])
                        dist_m = float(pt.get("dist_m", 2.0))
                        d_rel  = max(0.0, min(1.0, (dist_m - 1.0) / 7.0))
                        frames.append({"az": az, "el": el,
                                       "d_rel": d_rel, "dist_m": dist_m,
                                       "t_ms": pt["t"]})
                    duration_ms = stroke_3d[-1]["t"] - stroke_3d[0]["t"]
                    if duration_ms < 50:
                        duration_ms = len(stroke_3d) * 33.0
                elif len(stroke) >= 2:
                    fov_deg = 60.0
                    f_px = (canvas_w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
                    cx0, cy0 = canvas_w / 2.0, canvas_h / 2.0
                    frames = []
                    for i, pt in enumerate(stroke):
                        x_n = (pt["x"] - cx0) / f_px
                        y_n = (pt["y"] - cy0) / f_px
                        denom = math.sqrt(x_n**2 + y_n**2 + 1.0)
                        az = math.atan2(x_n, 1.0)
                        el = math.asin(min(1.0, max(-1.0, -y_n / denom)))
                        if i > 0:
                            dx = pt["x"] - stroke[i-1]["x"]
                            dy = pt["y"] - stroke[i-1]["y"]
                            dt = max(1.0, pt["t"] - stroke[i-1]["t"])
                            d_rel = min(1.0, math.sqrt(dx**2+dy**2) / dt / 2.0)
                        else:
                            d_rel = 0.5
                        frames.append({"az": az, "el": el,
                                       "d_rel": d_rel, "dist_m": 1.0 + 7.0 * d_rel,
                                       "t_ms": pt["t"]})
                    duration_ms = stroke[-1]["t"] - stroke[0]["t"]
                    if duration_ms < 50:
                        duration_ms = len(stroke) * 33.0
                else:
                    self._json({"error": "stroke too short (need ≥2 points)"}, 400); return

                import numpy as np
                sr = 48000
                T  = max(1, int(duration_ms / 1000.0 * sr))
                t_audio = np.arange(T, dtype=np.float32) / sr

                # ── Audio source ───────────────────────────────────────────────
                if audio_src and Path(audio_src).exists():
                    import soundfile as _sf
                    _aud, _fsr = _sf.read(audio_src, always_2d=False)
                    if _aud.ndim == 2: _aud = _aud.mean(axis=1)
                    if _fsr != sr:
                        try:
                            from scipy.signal import resample_poly
                            from math import gcd as _gcd
                            _g = _gcd(sr, _fsr)
                            _aud = resample_poly(_aud, sr//_g, _fsr//_g).astype(np.float32)
                        except Exception: pass
                    import math as _m
                    audio = np.tile(_aud.astype(np.float32),
                                    _m.ceil(T / len(_aud)))[:T]
                else:
                    audio = (0.5 * np.sin(2 * math.pi * tone_hz * t_audio)).astype(np.float32)

                # ── Interpolate trajectory → audio sample rate ─────────────────
                fr_t   = np.array([f["t_ms"]  for f in frames], dtype=np.float64)
                fr_t  -= fr_t[0]
                az_arr = np.array([f["az"]    for f in frames], dtype=np.float32)
                el_arr = np.array([f["el"]    for f in frames], dtype=np.float32)
                dr_arr = np.array([f["d_rel"] for f in frames], dtype=np.float32)
                di_arr = np.array([f["dist_m"]for f in frames], dtype=np.float32)
                t_ms_s = np.linspace(0.0, duration_ms, T, dtype=np.float64)

                az_s  = np.interp(t_ms_s, fr_t, az_arr).astype(np.float32)
                el_s  = np.interp(t_ms_s, fr_t, el_arr).astype(np.float32)
                dr_s  = np.interp(t_ms_s, fr_t, dr_arr).astype(np.float32)
                di_s  = np.interp(t_ms_s, fr_t, di_arr).astype(np.float32)

                smooth_n = max(1, int(sr * 0.08))
                if smooth_n > 1 and T > smooth_n:
                    from scipy.ndimage import uniform_filter1d
                    az_s = uniform_filter1d(az_s.astype(np.float64), size=smooth_n, mode='nearest').astype(np.float32)
                    el_s = uniform_filter1d(el_s.astype(np.float64), size=smooth_n, mode='nearest').astype(np.float32)

                from vid2spatial_pkg.foa_render import apply_distance_gain_lpf, direct_binaural_sofa
                audio_d  = apply_distance_gain_lpf(audio, sr, di_s, dr_s,
                               gain_k=1.5, lpf_min_hz=1200.0, lpf_max_hz=6000.0, gain_min=0.10)
                binaural = direct_binaural_sofa(audio_d, sr, -az_s, el_s, SOFA_PATH, block_ms=10.0)

                rms = float(np.sqrt(np.mean(binaural**2)) + 1e-9)
                tgt = 10 ** (-22.0 / 20.0)
                binaural = (binaural * (tgt / rms)).astype(np.float32)
                pk = np.max(np.abs(binaural))
                if pk > 0.99: binaural *= 0.99 / pk

                # ── Save to sketch job dir ─────────────────────────────────────
                import io, wave
                sk_dir = JOBS_DIR / "sketch_last"
                sk_dir.mkdir(parents=True, exist_ok=True)

                wav_path = sk_dir / "binaural.wav"
                with wave.open(str(wav_path), "wb") as w:
                    w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
                    L = (np.clip(binaural[0], -1, 1) * 32767).astype(np.int16)
                    R = (np.clip(binaural[1], -1, 1) * 32767).astype(np.int16)
                    iv = np.empty(2 * T, dtype=np.int16)
                    iv[0::2] = L; iv[1::2] = R
                    w.writeframes(iv.tobytes())

                result = {"binaural_url": "/api/sketch_file?name=binaural.wav",
                          "foa_url": None, "video_url": None}

                # ── FOA (optional) ─────────────────────────────────────────────
                if render_foa:
                    try:
                        from vid2spatial_pkg.foa_render import encode_mono_to_foa
                        foa = encode_mono_to_foa(audio_d, -az_s, el_s)  # [4, T]
                        foa_path = sk_dir / "foa.wav"
                        with wave.open(str(foa_path), "wb") as w:
                            w.setnchannels(4); w.setsampwidth(2); w.setframerate(sr)
                            ch = [(np.clip(foa[c], -1, 1) * 32767).astype(np.int16)
                                  for c in range(4)]
                            iv4 = np.empty(4 * T, dtype=np.int16)
                            for c in range(4): iv4[c::4] = ch[c]
                            w.writeframes(iv4.tobytes())
                        result["foa_url"] = "/api/sketch_file?name=foa.wav"
                    except Exception as e:
                        print(f"[WARN] FOA encode failed: {e}")

                # ── Overlay video (optional) ───────────────────────────────────
                if render_video:
                    try:
                        vid_path = sk_dir / "overlay.mp4"
                        _render_sketch_video(frames, az_s, el_s, di_s, sr, T,
                                             binaural, wav_path, vid_path)
                        result["video_url"] = "/api/sketch_file?name=overlay.mp4"
                    except Exception as e:
                        print(f"[WARN] Sketch video render failed: {e}")
                        traceback.print_exc()

                # ── OSC: send trajectory in background ────────────────────────
                if osc_enabled and _osc_config.get("enabled"):
                    traj_for_osc = {"frames": [
                        {"az": f["az"], "el": f["el"],
                         "dist_m": f["dist_m"], "d_rel": f["d_rel"]}
                        for f in frames]}
                    send_osc_trajectory(traj_for_osc, fps=len(frames)/max(0.001, duration_ms/1000))

                self._json(result)

            except Exception as e:
                traceback.print_exc()
                self._json({"error": str(e)}, 500)

        elif path == "/api/osc_config":
            global _osc_client
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            _osc_config["enabled"] = bool(data.get("enabled", False))
            _osc_config["host"]    = str(data.get("host", "127.0.0.1"))
            _osc_config["port"]    = int(data.get("port", 9001))
            _osc_config["format"]  = str(data.get("format", "maxmsp"))
            _osc_client = None  # reset client so it reconnects with new settings
            self._json({"ok": True, "config": _osc_config})

        else:
            self._json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_file(self, path: Path, ctype: str):
        if not path.exists():
            self.send_response(404); self.end_headers(); return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        status = args[1] if len(args) > 1 else ""
        if str(status).startswith("4") or str(status).startswith("5") or "POST" in str(args):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    JOBS_DIR.mkdir(exist_ok=True)
    server = HTTPServer((args.host, args.port), Handler)
    print(f"\n  Vid2Spatial Demo Server")
    print(f"  ========================")
    print(f"  http://localhost:{args.port}")
    print(f"  Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stopped]"); server.server_close()


if __name__ == "__main__":
    main()

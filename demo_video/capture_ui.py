#!/usr/bin/env python3
"""
Playwright-based capture of Vid2Spatial demo UI.

Produces:
  demo_video/capture/sketch_ui_recording.mp4  — 15s sketch UI interaction
  demo_video/capture/osc_terminal_recording.mp4 — 12s OSC terminal visualization
"""

import asyncio, os, sys, math, time, subprocess, json, wave, io
import numpy as np
from pathlib import Path

OUT     = Path("/home/seung/mmhoa/vid2spatial_v2/demo_video/capture")
FFMPEG  = "/home/seung/miniforge3/bin/ffmpeg"
BASE    = "/home/seung/mmhoa/vid2spatial_v2/test/e2e_final"
DEMO_DIR = "/home/seung/mmhoa/vid2spatial_v2/test/demo"
SR       = 48000

OUT.mkdir(exist_ok=True)

# ─── helpers ────────────────────────────────────────────────────────────────

def run_ff(cmd, label=""):
    print(f"  [ff:{label}] {' '.join(str(c) for c in cmd[:5])}...")
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-800:])
        raise RuntimeError(f"ffmpeg failed: {label}")

def png_to_video(png_dir, out_mp4, fps=25, dur=None):
    """Concat PNGs to video."""
    cmd = [FFMPEG, "-y",
           "-framerate", str(fps),
           "-i", f"{png_dir}/frame_%04d.png",
           "-c:v", "libx264", "-preset", "fast", "-crf", "18",
           "-pix_fmt", "yuv420p"]
    if dur:
        cmd += ["-t", str(dur)]
    cmd.append(out_mp4)
    run_ff(cmd, "png_to_video")

def mux_audio(video_mp4, wav_path, out_mp4, vol_db=6):
    run_ff([FFMPEG, "-y",
            "-i", video_mp4, "-i", wav_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-af", f"volume={vol_db}dB",
            "-c:v", "copy", "-c:a", "aac", "-ar", str(SR), "-ac", "2",
            out_mp4], "mux")

# ─── Sketch audio (same as make_demo_v2.py) ─────────────────────────────────

def make_sketch_audio(out_wav, dur_s=15.0):
    sys.path.insert(0, "/home/seung/mmhoa/vid2spatial_v2")
    try:
        from vid2spatial_pkg.foa_render import apply_distance_gain_lpf, direct_binaural_sofa
        SOFA = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"
        T = int(dur_s * SR)
        t = np.arange(T, dtype=np.float32) / SR
        audio = (0.5 * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)

        sweep_T = int(10.0 * SR)
        az_s = np.concatenate([
            np.linspace(-50, 50, sweep_T, dtype=np.float32),
            np.full(T - sweep_T, 50.0, dtype=np.float32)
        ])
        el_s = np.concatenate([
            np.zeros(int(6*SR), dtype=np.float32),
            np.linspace(0, 25, T - int(6*SR), dtype=np.float32)
        ])
        d_rel_s = np.full(T, 0.5, dtype=np.float32)
        dist_s  = np.full(T, 3.0, dtype=np.float32)

        audio_d = apply_distance_gain_lpf(
            audio, SR, dist_s, d_rel_s,
            gain_k=1.5, lpf_min_hz=1200, lpf_max_hz=6000, gain_min=0.10)
        binaural = direct_binaural_sofa(audio_d, SR, -np.radians(az_s), np.radians(el_s), SOFA)

        rms = float(np.sqrt(np.mean(binaural**2)) + 1e-9)
        binaural = (binaural * (10**(-22/20) / rms)).astype(np.float32)
        peak = np.max(np.abs(binaural))
        if peak > 0.99: binaural *= 0.99 / peak

        L = (np.clip(binaural[0], -1, 1) * 32767).astype(np.int16)
        R = (np.clip(binaural[1], -1, 1) * 32767).astype(np.int16)
        interleaved = np.empty(2*T, dtype=np.int16)
        interleaved[0::2] = L; interleaved[1::2] = R
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(interleaved.tobytes())
        print(f"  [sketch_audio] OK ({dur_s:.0f}s)")
        return True
    except Exception as e:
        print(f"  [sketch_audio] fallback silence: {e}")
        T = int(dur_s * SR)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(np.zeros(2*T, dtype=np.int16).tobytes())
        with open(str(out_wav), "wb") as f: f.write(buf.getvalue())
        return False

# ─── Playwright capture ──────────────────────────────────────────────────────

async def capture_sketch(server_port=8091):
    from playwright.async_api import async_playwright

    frames_dir = OUT / "sketch_frames"
    frames_dir.mkdir(exist_ok=True)

    print("\n  [playwright] Starting demo server...")
    srv = subprocess.Popen(
        [sys.executable, "server.py", "--port", str(server_port)],
        cwd=DEMO_DIR,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    await asyncio.sleep(2.0)

    W, H = 1280, 720
    FPS  = 25
    DUR  = 15.0   # seconds
    frame_idx = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
        ])
        ctx = await browser.new_context(
            viewport={"width": W, "height": H},
            device_scale_factor=1,
        )
        page = await ctx.new_page()

        print(f"  [playwright] Opening http://localhost:{server_port}")
        try:
            await page.goto(f"http://localhost:{server_port}",
                            wait_until="domcontentloaded", timeout=15000)
        except Exception:
            pass  # timeout ok, page may still be usable
        await asyncio.sleep(3.0)

        # Switch to sketch tab via JS (avoids CDN network blocking)
        print("  [playwright] Switching to sketch tab")
        await page.evaluate("document.getElementById('tab-sketch').click()")
        await asyncio.sleep(3.0)

        # Screenshot: initial state
        ss = await page.screenshot(full_page=False)
        with open(frames_dir / f"frame_{frame_idx:04d}.png", "wb") as f:
            f.write(ss)
        frame_idx += 1

        # Duplicate initial frame for 2s (50 frames at 25fps)
        for _ in range(49):
            with open(frames_dir / f"frame_{frame_idx:04d}.png", "wb") as f:
                f.write(ss)
            frame_idx += 1
        print(f"  [playwright] Initial state: {frame_idx} frames")

        # Find the 3D canvas (wait up to 8s for Three.js to init)
        canvas = page.locator("#sk3d-canvas")
        box = None
        for _ in range(8):
            box = await canvas.bounding_box()
            if box:
                break
            await asyncio.sleep(1.0)
        if not box:
            print("  [playwright] WARN: sk3d-canvas not found, using full page")
            box = {"x": 0, "y": 100, "width": W, "height": H - 200}

        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        r  = min(box["width"], box["height"]) * 0.38

        # Draw a smooth arc: left → right → up
        # Phase 1 (0→5s): horizontal sweep left to right
        # Phase 2 (5→10s): continue right and rise
        # Phase 3 (10→13s): hold / slight circle
        total_draw_frames = int(11.0 * FPS)  # 11s of drawing

        print("  [playwright] Starting mouse draw sequence...")
        # Mouse down at left
        start_x = cx - r * 0.85
        start_y = cy
        await page.mouse.move(start_x, start_y)
        await page.mouse.down()

        prev_ss = None
        for i in range(total_draw_frames):
            t_norm = i / total_draw_frames  # 0→1

            if t_norm < 0.45:
                # Horizontal sweep
                t2 = t_norm / 0.45
                mx = cx + r * (-0.85 + 1.7 * t2)
                my = cy + r * 0.05 * math.sin(t2 * math.pi)
            elif t_norm < 0.82:
                # Rise to upper right
                t2 = (t_norm - 0.45) / 0.37
                mx = cx + r * (0.85 - 0.15 * t2)
                my = cy + r * (-0.65 * t2)
            else:
                # Small arc back
                t2 = (t_norm - 0.82) / 0.18
                angle = math.pi * 0.3 * t2
                mx = cx + r * (0.70 + 0.10 * math.cos(angle))
                my = cy + r * (-0.65 + 0.10 * math.sin(angle))

            await page.mouse.move(mx, my)

            # Capture frame every 1/FPS
            ss = await page.screenshot(full_page=False)
            with open(frames_dir / f"frame_{frame_idx:04d}.png", "wb") as f:
                f.write(ss)
            frame_idx += 1

            # Throttle to ~25fps
            await asyncio.sleep(1.0 / FPS * 0.5)

        await page.mouse.up()
        print(f"  [playwright] Draw complete: {frame_idx} frames")

        # Hold 2s after draw
        await asyncio.sleep(0.3)
        for _ in range(int(2.0 * FPS)):
            ss = await page.screenshot(full_page=False)
            with open(frames_dir / f"frame_{frame_idx:04d}.png", "wb") as f:
                f.write(ss)
            frame_idx += 1

        await browser.close()

    srv.terminate()
    print(f"  [playwright] Captured {frame_idx} frames → {frames_dir}")

    # PNG → video (no audio yet)
    raw_mp4 = OUT / "sketch_raw.mp4"
    run_ff([FFMPEG, "-y",
            "-framerate", str(FPS),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-t", str(DUR),
            str(raw_mp4)], "sketch_png2mp4")

    return raw_mp4


async def capture_osc(server_port=8092):
    """
    OSC terminal visualization: animate a synthetic terminal log
    showing live OSC parameter values scrolling past.
    Uses playwright to render an HTML page with animated JS terminal.
    """
    from playwright.async_api import async_playwright

    frames_dir = OUT / "osc_frames"
    frames_dir.mkdir(exist_ok=True)

    # Load car-10 traj for real values
    traj_path = f"{BASE}/car-10/traj.json"
    with open(traj_path) as f:
        data = json.load(f)
    traj_frames = data if isinstance(data, list) else data.get("frames", [])
    n = len(traj_frames)

    # Build animated terminal HTML
    # Pre-compute ~300 lines of OSC log from traj
    log_lines = []
    for i in range(min(300, n)):
        fr = traj_frames[i]
        az  = math.degrees(fr.get("az", 0))
        el  = math.degrees(fr.get("el", 0))
        dr  = fr.get("d_rel", 0.5)
        if isinstance(dr, float) and dr > 1.5: dr = min(1.0, dr / 10.0)
        conf = fr.get("confidence", 0.95)
        ts   = fr.get("frame", i) / 30.0
        log_lines.append(
            f"[{ts:6.2f}s] /vid2spatial  az={az:+6.2f}  el={el:+5.2f}  "
            f"d_rel={dr:.4f}  conf={conf:.4f}"
        )

    lines_js = json.dumps(log_lines)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #050510;
    color: #e0e0e0;
    font-family: 'Courier New', monospace;
    width: 1280px; height: 720px; overflow: hidden;
  }}
  #header {{
    background: #0a0a1f;
    border-bottom: 1px solid #1a1a3a;
    padding: 14px 20px 10px;
  }}
  #title {{
    color: #00e5a0;
    font-size: 18px;
    font-weight: bold;
    letter-spacing: 1px;
  }}
  #subtitle {{
    color: #4444aa;
    font-size: 13px;
    margin-top: 3px;
  }}
  #protocol {{
    color: #00aaff;
    font-size: 12px;
    margin-top: 5px;
    font-family: monospace;
  }}
  #terminal {{
    padding: 10px 20px;
    height: 580px;
    overflow: hidden;
    position: relative;
  }}
  .line {{
    font-size: 13.5px;
    line-height: 20px;
    white-space: nowrap;
    opacity: 0;
    transition: opacity 0.1s;
  }}
  .line.green {{ color: #00ff88; }}
  .line.grey  {{ color: #666688; }}
  .line.cyan  {{ color: #00ccff; }}
  #footer {{
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: #0a0a1f;
    border-top: 1px solid #1a1a3a;
    padding: 8px 20px;
    color: #fbbf24;
    font-size: 12px;
    font-family: sans-serif;
  }}
</style>
</head>
<body>
<div id="header">
  <div id="title">(c) OSC Parameter Streaming — Vid2Spatial</div>
  <div id="subtitle">Real-time spatial parameters → UDP → any spatial audio engine</div>
  <div id="protocol">osc://localhost:9000  /vid2spatial/{{az, el, d_rel, confidence, frame, xyz}}</div>
</div>
<div id="terminal"></div>
<div id="footer">
  Verified: Reaper &nbsp;|&nbsp; Max/MSP &nbsp;|&nbsp; TouchDesigner &nbsp;|&nbsp; Unreal Engine &nbsp;|&nbsp; Pure Data
</div>
<script>
const LINES = {lines_js};
const VISIBLE = 26;
let shown = 0;

function addLine() {{
  if (shown >= LINES.length) return;
  const term = document.getElementById('terminal');
  const div = document.createElement('div');
  div.className = 'line ' + (shown % 2 === 0 ? 'green' : 'grey');
  div.textContent = LINES[shown];
  term.appendChild(div);
  requestAnimationFrame(() => {{ div.style.opacity = '1'; }});
  shown++;

  // Remove old lines to keep visible count
  while (term.children.length > VISIBLE) {{
    term.removeChild(term.firstChild);
  }}
}}

// Start scrolling: 8 lines/sec
setInterval(addLine, 125);

// First few lines appear immediately
for (let i = 0; i < 3; i++) addLine();
</script>
</body>
</html>"""

    html_path = OUT / "osc_terminal.html"
    html_path.write_text(html_content)

    W, H   = 1280, 720
    FPS    = 25
    DUR    = 12.0
    total_frames = int(DUR * FPS)

    print("\n  [playwright] Capturing OSC terminal animation...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
        ])
        ctx = await browser.new_context(
            viewport={"width": W, "height": H},
            device_scale_factor=1,
        )
        page = await ctx.new_page()
        await page.goto("file://" + str(html_path), wait_until="networkidle")
        await asyncio.sleep(0.5)

        for i in range(total_frames):
            ss = await page.screenshot(full_page=False)
            with open(frames_dir / f"frame_{i:04d}.png", "wb") as f:
                f.write(ss)
            await asyncio.sleep(1.0 / FPS * 0.7)

        await browser.close()

    print(f"  [playwright] OSC: {total_frames} frames captured")

    raw_mp4 = OUT / "osc_raw.mp4"
    run_ff([FFMPEG, "-y",
            "-framerate", str(FPS),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-t", str(DUR),
            str(raw_mp4)], "osc_png2mp4")

    return raw_mp4


async def main():
    print("=== Capture: Sketch UI ===")
    sketch_raw = await capture_sketch(server_port=8091)

    print("\n=== Generate sketch binaural audio ===")
    sketch_wav = OUT / "sketch_binaural.wav"
    make_sketch_audio(sketch_wav, dur_s=15.0)

    print("\n=== Mux sketch video + audio ===")
    sketch_final = OUT / "sketch_ui_recording.mp4"
    mux_audio(str(sketch_raw), str(sketch_wav), str(sketch_final), vol_db=6)
    print(f"  -> {sketch_final}")

    print("\n=== Capture: OSC terminal ===")
    osc_raw = await capture_osc(server_port=8092)

    print("\n=== Mux OSC video + car-10 binaural ===")
    osc_final = OUT / "osc_terminal_recording.mp4"
    mux_audio(str(osc_raw), f"{BASE}/car-10/binaural.wav", str(osc_final), vol_db=6)
    print(f"  -> {osc_final}")

    print("\n=== Done ===")
    print(f"  Sketch: {sketch_final}")
    print(f"  OSC:    {osc_final}")


if __name__ == "__main__":
    asyncio.run(main())

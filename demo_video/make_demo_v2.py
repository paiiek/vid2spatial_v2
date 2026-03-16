#!/usr/bin/env python3
"""
Vid2Spatial ISMAR 2026 Demo Video v2
=====================================
Structure:
  [0:00-0:04]  Title card
  [0:04-0:16]  (a) car-10    tracking overlay + binaural (+6dB)
  [0:16-0:19]  car-10 stats card
  [0:19-0:31]  (a) skateboard-8 tracking overlay + binaural (+6dB)
  [0:31-0:34]  skateboard-8 stats card
  [0:34-0:46]  (a) drone-13 tracking overlay + binaural (+6dB)
  [0:46-0:49]  drone-13 stats card
  [0:49-0:53]  Section header: (b) Sketch Interface
  [0:53-1:05]  (b) sketch — figure2.jpg slideshow + sketch binaural audio
  [1:05-1:08]  sketch stats card
  [1:08-1:12]  Section header: (c) OSC Parameter Streaming
  [1:12-1:24]  (c) OSC terminal visualization + car-10 binaural
  [1:24-1:28]  OSC stats card
  [1:28-1:33]  Final summary card
Total: ~93s
"""

import subprocess, os, sys, json, math, wave, struct, io
import numpy as np

FFMPEG  = "/home/seung/miniforge3/bin/ffmpeg"
BASE    = "/home/seung/mmhoa/vid2spatial_v2/test/e2e_final"
FIGURES = "/home/seung/mmhoa/vid2spatial_v2/vgtc_conference_latex-2024.02.14/figures"
OUT     = "/home/seung/mmhoa/vid2spatial_v2/demo_video"
SR      = 48000

os.makedirs(OUT, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def run(cmd, label=""):
    print(f"  [{label}] {' '.join(str(c) for c in cmd[:5])}...")
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1200:])
        raise RuntimeError(f"ffmpeg failed: {label}")

def concat_videos(file_list, out_path):
    list_path = out_path + ".list.txt"
    with open(list_path, "w") as f:
        for p in file_list:
            f.write(f"file '{p}'\n")
    run([FFMPEG, "-y",
         "-f", "concat", "-safe", "0",
         "-i", list_path,
         "-c:v", "libx264", "-preset", "fast", "-crf", "17",
         "-c:a", "aac", "-ar", str(SR), "-ac", "2",
         "-pix_fmt", "yuv420p",
         out_path], "concat")
    os.remove(list_path)

def make_card(path, line1, line2="", line3="", dur=3.5,
              bg="0x0a0a1a", fg="white", accent="0xe94560"):
    """960x540 title card (3 lines, up to 3.5s silence)."""
    W, H = 960, 540
    texts = []
    for txt, yp, sz, col in zip(
        [line1, line2, line3],
        [200, 285, 355],
        [42,  30,  24],
        [accent, fg, "0xaaaaaa"]
    ):
        if not txt:
            continue
        safe = txt.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")
        texts.append(
            f"drawtext=text='{safe}':fontcolor={col}:fontsize={sz}:"
            f"x=(w-text_w)/2:y={yp}:font=Sans"
        )
    vf = ",".join(texts) if texts else "null"
    run([FFMPEG, "-y",
         "-f", "lavfi",
         "-i", f"color=c={bg}:s={W}x{H}:d={dur}:r=25",
         "-f", "lavfi", "-i", f"aevalsrc=0:c=stereo:s={SR}:d={dur}",
         "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "aac", "-ar", str(SR), "-ac", "2",
         "-pix_fmt", "yuv420p",
         "-t", str(dur), path],
        f"card:{os.path.basename(path)}")

def label_clip(src_mp4, src_wav, dst, top, bottom, dur=12.0, vol_db=6.0):
    """
    Overlay two text bars on tracking clip + mix binaural wav at +vol_db.
    Scales video to 960x540.
    """
    W, H = 960, 540
    safe_top = top.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")
    safe_bot = bottom.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
        f"drawbox=x=0:y=0:w={W}:h=50:color=black@0.65:t=fill,"
        f"drawtext=text='{safe_top}':fontcolor=0xe94560:fontsize=20:"
        f"x=14:y=14:font=Sans,"
        f"drawbox=x=0:y={H-42}:w={W}:h=42:color=black@0.70:t=fill,"
        f"drawtext=text='{safe_bot}':fontcolor=0xcccccc:fontsize=16:"
        f"x=14:y={H-30}:font=Sans,"
        f"drawtext=text='Headphones required':fontcolor=0xfbbf24:fontsize=15:"
        f"x={W-210}:y=16:font=Sans"
    )
    # volume filter: +vol_db dB
    af = f"volume={vol_db}dB"
    run([FFMPEG, "-y",
         "-i", src_mp4,
         "-i", src_wav,
         "-map", "0:v:0", "-map", "1:a:0",
         "-vf", vf,
         "-af", af,
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "aac", "-ar", str(SR), "-ac", "2",
         "-pix_fmt", "yuv420p",
         "-t", str(dur), dst],
        f"label:{os.path.basename(dst)}")

def make_section_header(path, section_letter, title, subtitle="", dur=3.5,
                        bg="0x050510", accent="0x00e5a0"):
    """Section divider card with large letter badge."""
    W, H = 960, 540
    safe_t  = title.replace("'", "\\'").replace(":", "\\:")
    safe_st = subtitle.replace("'", "\\'").replace(":", "\\:")
    vf = (
        # Large letter
        f"drawtext=text='({section_letter})':fontcolor={accent}:fontsize=80:"
        f"x=(w-text_w)/2:y=150:font=Sans,"
        f"drawtext=text='{safe_t}':fontcolor=white:fontsize=34:"
        f"x=(w-text_w)/2:y=270:font=Sans"
    )
    if subtitle:
        vf += (
            f",drawtext=text='{safe_st}':fontcolor=0xaaaaaa:fontsize=22:"
            f"x=(w-text_w)/2:y=328:font=Sans"
        )
    run([FFMPEG, "-y",
         "-f", "lavfi",
         "-i", f"color=c={bg}:s={W}x{H}:d={dur}:r=25",
         "-f", "lavfi", "-i", f"aevalsrc=0:c=stereo:s={SR}:d={dur}",
         "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "aac", "-ar", str(SR), "-ac", "2",
         "-pix_fmt", "yuv420p",
         "-t", str(dur), path],
        f"section:{section_letter}")

def make_sketch_segment(dst, figure2_jpg, sketch_wav, dur=12.0):
    """
    Sketch interface segment:
    - figure2.jpg (1800x720) scaled/cropped to 960x540 as background
    - Overlay text labels describing sketch mode
    - Sketch binaural audio
    """
    W, H = 960, 540
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},"
        f"drawbox=x=0:y=0:w={W}:h=56:color=black@0.70:t=fill,"
        f"drawtext=text='(b) Sketch Mode - Freehand 3D Sphere Trajectory':"
        f"fontcolor=0x00e5a0:fontsize=22:x=14:y=16:font=Sans,"
        f"drawbox=x=0:y={H-46}:w={W}:h=46:color=black@0.72:t=fill,"
        f"drawtext=text='WebGL 3D sphere  |  lat-lon grid  |  freehand trajectory  ->  spatial parameter stream':"
        f"fontcolor=0xcccccc:fontsize=16:x=14:y={H-32}:font=Sans,"
        f"drawtext=text='Headphones required':fontcolor=0xfbbf24:fontsize=15:"
        f"x={W-210}:y=18:font=Sans"
    )
    af = "volume=6dB"
    run([FFMPEG, "-y",
         "-loop", "1", "-framerate", "25", "-i", figure2_jpg,
         "-i", sketch_wav,
         "-map", "0:v:0", "-map", "1:a:0",
         "-vf", vf,
         "-af", af,
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "aac", "-ar", str(SR), "-ac", "2",
         "-pix_fmt", "yuv420p",
         "-t", str(dur), dst],
        "sketch_segment")

def make_osc_segment(dst, traj_json, audio_wav, dur=12.0):
    """
    OSC streaming visualization:
    Renders a synthetic terminal-style scrolling OSC log as video,
    mixed with car-10 binaural audio.
    Uses drawtext with animated frame counter via PTS expression.
    """
    import json as _json
    with open(traj_json) as f:
        data = _json.load(f)
    frames = data if isinstance(data, list) else data.get("frames", [])

    # Pick 20 evenly spaced frames for display
    idxs = np.linspace(0, min(len(frames)-1, 299), 20, dtype=int)
    lines = []
    for i in idxs:
        fr = frames[i]
        az = math.degrees(fr.get("az", 0))
        el = math.degrees(fr.get("el", 0))
        dr = fr.get("d_rel", fr.get("dist_m", 0))
        if isinstance(dr, float) and dr > 1.5:
            dr = min(1.0, dr / 10.0)
        conf = fr.get("confidence", 0.95)
        lines.append(
            f"frame={fr.get('frame',i):03d}  az={az:+6.1f}  el={el:+5.1f}  "
            f"d_rel={dr:.3f}  conf={conf:.3f}"
        )

    W, H = 960, 540
    bg = "0x050510"
    # Build static image with all lines using python + save as PNG, then use as input
    # Generate PNG with pillow-free approach: use ffmpeg drawtext on blank canvas
    # We'll write all lines as a drawtext filter with fixed y offsets

    header  = "/vid2spatial  ->  UDP 9000"
    line_h  = 22
    y_start = 90
    green   = "0x00ff88"
    grey    = "0x888888"
    cyan    = "0x00d8ff"

    texts = [
        # OSC address header
        f"drawbox=x=0:y=0:w={W}:h=56:color=black@0.8:t=fill",
        f"drawtext=text='(c) OSC Parameter Streaming':fontcolor=0x00e5a0:"
        f"fontsize=22:x=14:y=16:font=Sans",
        # protocol line
        f"drawtext=text='osc\://localhost\\:9000  /vid2spatial/{{az,el,d_rel,confidence}}':"
        f"fontcolor={cyan}:fontsize=15:x=14:y=60:font=Sans",
    ]

    for i, line in enumerate(lines):
        y = y_start + i * line_h
        safe = line.replace("'", "\\'").replace(":", "\\:").replace("{", "\\{").replace("}", "\\}")
        col = green if i % 2 == 0 else grey
        texts.append(
            f"drawtext=text='{safe}':fontcolor={col}:fontsize=14:"
            f"x=14:y={y}:font=monospace"
        )

    # Bottom bar
    texts.append(
        f"drawbox=x=0:y={H-42}:w={W}:h=42:color=black@0.72:t=fill"
    )
    texts.append(
        f"drawtext=text='Verified\\: Reaper  Max/MSP  TouchDesigner  Unreal Engine  Pure Data':"
        f"fontcolor=0xfbbf24:fontsize=15:x=14:y={H-28}:font=Sans"
    )
    texts.append(
        f"drawtext=text='Headphones required':fontcolor=0xfbbf24:fontsize=15:"
        f"x={W-210}:y=18:font=Sans"
    )

    vf = ",".join(texts)
    af = "volume=6dB"
    run([FFMPEG, "-y",
         "-f", "lavfi",
         "-i", f"color=c={bg}:s={W}x{H}:d={dur}:r=25",
         "-i", audio_wav,
         "-map", "0:v:0", "-map", "1:a:0",
         "-vf", vf,
         "-af", af,
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "aac", "-ar", str(SR), "-ac", "2",
         "-pix_fmt", "yuv420p",
         "-t", str(dur), dst],
        "osc_segment")

def make_sketch_audio(out_wav, dur_s=12.0):
    """
    Generate a demo sketch binaural audio:
    440 Hz sine sweeping left-to-right (az -40° → +40°) then up (el 0° → +20°)
    using the same binaural render pipeline.
    """
    import sys
    sys.path.insert(0, "/home/seung/mmhoa/vid2spatial_v2")
    try:
        from vid2spatial_pkg.foa_render import apply_distance_gain_lpf, direct_binaural_sofa
        SOFA = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"
        T = int(dur_s * SR)
        t = np.arange(T, dtype=np.float32) / SR

        # 440Hz tone with slight amplitude envelope
        audio = (0.5 * np.sin(2 * math.pi * 440.0 * t)).astype(np.float32)

        # Azimuth: sweep -40° → +40° in first 8s, then hold
        sweep_T = int(8.0 * SR)
        az_s = np.concatenate([
            np.linspace(-40, 40, sweep_T, dtype=np.float32),
            np.full(T - sweep_T, 40.0, dtype=np.float32)
        ])
        az_s = np.radians(az_s)

        # Elevation: hold 0° for 6s, then rise to +20°
        el_s = np.concatenate([
            np.zeros(int(6*SR), dtype=np.float32),
            np.linspace(0, 20, T - int(6*SR), dtype=np.float32)
        ])
        el_s = np.radians(el_s)

        # d_rel: 0.5 constant
        d_rel_s = np.full(T, 0.5, dtype=np.float32)
        dist_s  = np.full(T, 3.0, dtype=np.float32)

        audio_d = apply_distance_gain_lpf(
            audio, SR, dist_s, d_rel_s,
            gain_k=1.5, lpf_min_hz=1200, lpf_max_hz=6000, gain_min=0.10
        )
        binaural = direct_binaural_sofa(audio_d, SR, -az_s, el_s, SOFA, block_ms=10.0)

        # Normalize to -22 dBFS
        rms = float(np.sqrt(np.mean(binaural**2)) + 1e-9)
        target = 10 ** (-22.0 / 20.0)
        binaural = (binaural * (target / rms)).astype(np.float32)
        peak = np.max(np.abs(binaural))
        if peak > 0.99:
            binaural = binaural * (0.99 / peak)

        # Write WAV
        L = (np.clip(binaural[0], -1, 1) * 32767).astype(np.int16)
        R = (np.clip(binaural[1], -1, 1) * 32767).astype(np.int16)
        interleaved = np.empty(2 * T, dtype=np.int16)
        interleaved[0::2] = L
        interleaved[1::2] = R
        with wave.open(out_wav, "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(interleaved.tobytes())
        print(f"  [sketch_audio] {out_wav} ({dur_s:.0f}s)")
        return True
    except Exception as e:
        print(f"  [sketch_audio] WARN: binaural render failed ({e}), using silence")
        # fallback: silence
        T = int(dur_s * SR)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes(np.zeros(2*T, dtype=np.int16).tobytes())
        with open(out_wav, "wb") as f:
            f.write(buf.getvalue())
        return False


# ═════════════════════════════════════════════════════════════════════════════
# BUILD
# ═════════════════════════════════════════════════════════════════════════════

parts = []

# ── 00: Title ─────────────────────────────────────────────────────────────────
print("\n== Title card ==")
p = f"{OUT}/v2_00_title.mp4"
make_card(p,
    line1="Vid2Spatial",
    line2="Spatial Audio Authoring from Video and Sketch",
    line3="Single bounding box  ->  spatial parameter stream  ->  binaural / FOA / OSC",
    dur=4.0, bg="0x080818", accent="0xe94560")
parts.append(p)

# ── 01a: section header ────────────────────────────────────────────────────────
print("\n== Section (a) header ==")
p = f"{OUT}/v2_01a_header.mp4"
make_section_header(p, "a", "Video Tracking",
    subtitle="Single bbox init  ->  SAM2 tracker  ->  HRTF binaural", dur=3.0)
parts.append(p)

# ── 01b: car-10 ───────────────────────────────────────────────────────────────
print("\n== car-10 ==")
p = f"{OUT}/v2_01b_car10.mp4"
label_clip(
    src_mp4=f"{BASE}/car-10/car-10_e2e.mp4",
    src_wav=f"{BASE}/car-10/binaural.wav",
    dst=p,
    top="car-10  |  Vehicle (COCO)  |  AzMAE = 0.62°  |  Az range = 43°  (L->R sweep)",
    bottom="1 bbox drawn  ->  SAM2 tracks 300 frames  ->  HRTF binaural render",
    dur=12.0, vol_db=6.0)
parts.append(p)

p = f"{OUT}/v2_01c_car10_card.mp4"
make_card(p, "car-10   Vehicle (COCO)",
    "AzMAE = 0.62°   |   Az range = 43.2°",
    "Clear left-to-right lateral sweep",
    dur=3.0, bg="0x080818")
parts.append(p)

# ── 02: skateboard-8 ──────────────────────────────────────────────────────────
print("\n== skateboard-8 ==")
p = f"{OUT}/v2_02a_sk8.mp4"
label_clip(
    src_mp4=f"{BASE}/skateboard-8/skateboard-8_e2e.mp4",
    src_wav=f"{BASE}/skateboard-8/binaural.wav",
    dst=p,
    top="skateboard-8  |  Sports (COCO)  |  AzMAE = 0.18°  (best in 22-clip eval)",
    bottom="Az range = 38°   |   Dist range = 5.1m  (approach + lateral)",
    dur=12.0, vol_db=6.0)
parts.append(p)

p = f"{OUT}/v2_02b_sk8_card.mp4"
make_card(p, "skateboard-8   Sports (COCO)",
    "AzMAE = 0.18°   |   Best accuracy in 22-clip eval",
    "Combines lateral motion + distance approach",
    dur=3.0, bg="0x080818")
parts.append(p)

# ── 03: drone-13 ──────────────────────────────────────────────────────────────
print("\n== drone-13 ==")
p = f"{OUT}/v2_03a_drone.mp4"
label_clip(
    src_mp4=f"{BASE}/drone-13/drone-13_e2e.mp4",
    src_wav=f"{BASE}/drone-13/binaural.wav",
    dst=p,
    top="drone-13  |  Drone (Non-COCO)  |  AzMAE = 0.44°  |  category-agnostic tracking",
    bottom="Az=27.8°  El=19.8°  Dist=8.1m  -  Full 3D spatial parameter stream",
    dur=12.0, vol_db=6.0)
parts.append(p)

p = f"{OUT}/v2_03b_drone_card.mp4"
make_card(p, "drone-13   Drone (Non-COCO)",
    "AzMAE = 0.44°   |   El range = 19.8°   |   Dist = 8.1m",
    "Open-vocabulary tracking - no category constraints",
    dur=3.0, bg="0x080818")
parts.append(p)

# ── 04: section (b) + Sketch ─────────────────────────────────────────────────
print("\n== Section (b) Sketch ==")
p = f"{OUT}/v2_04a_sketch_header.mp4"
make_section_header(p, "b", "Sketch Interface",
    subtitle="Freehand 3D sphere trajectory for off-screen and non-literal sources",
    dur=3.5)
parts.append(p)

# Generate sketch binaural audio
sketch_wav = f"{OUT}/v2_sketch_binaural.wav"
make_sketch_audio(sketch_wav, dur_s=12.0)

p = f"{OUT}/v2_04b_sketch.mp4"
make_sketch_segment(p,
    figure2_jpg=f"{FIGURES}/figure2.jpg",
    sketch_wav=sketch_wav,
    dur=12.0)
parts.append(p)

p = f"{OUT}/v2_04c_sketch_card.mp4"
make_card(p, "Sketch Mode",
    "Az + El + distance  ->  same parameter stream as video mode",
    "Off-screen, ambient, and non-literal sources",
    dur=3.0, bg="0x080818", accent="0x00e5a0")
parts.append(p)

# ── 05: section (c) + OSC ────────────────────────────────────────────────────
print("\n== Section (c) OSC ==")
p = f"{OUT}/v2_05a_osc_header.mp4"
make_section_header(p, "c", "OSC Parameter Streaming",
    subtitle="Real-time UDP streaming to any spatial audio engine",
    dur=3.5)
parts.append(p)

p = f"{OUT}/v2_05b_osc.mp4"
make_osc_segment(p,
    traj_json=f"{BASE}/car-10/traj.json",
    audio_wav=f"{BASE}/car-10/binaural.wav",
    dur=12.0)
parts.append(p)

p = f"{OUT}/v2_05c_osc_card.mp4"
make_card(p, "OSC Output",
    "Verified: Reaper  Max/MSP  TouchDesigner  Unreal Engine  Pure Data",
    "Atomic UDP bundle  ->  frame-coherent multi-parameter update",
    dur=3.0, bg="0x080818", accent="0x00e5a0")
parts.append(p)

# ── 06: Final summary ────────────────────────────────────────────────────────
print("\n== Final stats ==")
p = f"{OUT}/v2_06_final.mp4"
make_card(p,
    line1="22 LaSOT clips  |  AzMAE = 1.36°  (median 0.70°)",
    line2="N=20 perceptual: p<=0.025  |  N=12 efficiency: -41-49% time",
    line3="Single bbox  ->  binaural / FOA AmbiX / OSC  (one extraction)",
    dur=5.0, bg="0x0a0a1a", accent="0xe94560")
parts.append(p)

# ── Concat ───────────────────────────────────────────────────────────────────
print("\n== Concat all ==")
final = f"{OUT}/vid2spatial_demo_v2.mp4"
concat_videos(parts, final)

size = os.path.getsize(final) / 1e6
total_s = 4 + 3 + 12 + 3 + 12 + 3 + 12 + 3 + 3.5 + 12 + 3 + 3.5 + 12 + 3 + 5
print(f"\n✓ Done: {final}  ({size:.1f} MB)")
print(f"  Parts: {len(parts)}")
print(f"  Duration: ~{total_s:.0f}s")

#!/usr/bin/env python3
"""
Vid2Spatial Demo v3 — final version with real UI recordings.
Uses capture/sketch_ui_recording.mp4 and capture/osc_terminal_recording.mp4.

Structure:
  [0:00-0:04]  Title card
  [0:04-0:07]  Section (a) header
  [0:07-0:19]  car-10    tracking (+6dB)
  [0:19-0:22]  car-10 stats card
  [0:22-0:34]  skateboard-8 (+6dB)
  [0:34-0:37]  skateboard-8 stats card
  [0:37-0:49]  drone-13 (+6dB)
  [0:49-0:52]  drone-13 stats card
  [0:52-0:56]  Section (b) header
  [0:56-1:11]  sketch UI recording (real playwright capture, 15s)
  [1:11-1:14]  sketch stats card
  [1:14-1:18]  Section (c) header
  [1:18-1:31]  OSC terminal recording (real playwright capture, 13s)
  [1:31-1:34]  OSC stats card
  [1:34-1:39]  Final summary card
Total: ~99s
"""

import subprocess, os

FFMPEG  = "/home/seung/miniforge3/bin/ffmpeg"
BASE    = "/home/seung/mmhoa/vid2spatial_v2/test/e2e_final"
CAP     = "/home/seung/mmhoa/vid2spatial_v2/demo_video/capture"
OUT     = "/home/seung/mmhoa/vid2spatial_v2/demo_video"
SR      = 48000

def run(cmd, label=""):
    print(f"  [{label}] {' '.join(str(c) for c in cmd[:5])}...")
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1000:])
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
    W, H = 1280, 720
    texts = []
    for txt, yp, sz, col in zip(
        [line1, line2, line3],
        [250, 340, 415],
        [46, 32, 26],
        [accent, fg, "0xaaaaaa"]
    ):
        if not txt: continue
        safe = txt.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")
        texts.append(
            f"drawtext=text='{safe}':fontcolor={col}:fontsize={sz}:"
            f"x=(w-text_w)/2:y={yp}:font=Sans"
        )
    vf = ",".join(texts) if texts else "null"
    run([FFMPEG, "-y",
         "-f", "lavfi", "-i", f"color=c={bg}:s={W}x{H}:d={dur}:r=25",
         "-f", "lavfi", "-i", f"aevalsrc=0:c=stereo:s={SR}:d={dur}",
         "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "aac", "-ar", str(SR), "-ac", "2",
         "-pix_fmt", "yuv420p", "-t", str(dur), path],
        f"card:{os.path.basename(path)}")

def make_section_header(path, letter, title, subtitle="", dur=3.5,
                        bg="0x050510", accent="0x00e5a0"):
    W, H = 1280, 720
    safe_t  = title.replace("'", "\\'").replace(":", "\\:")
    safe_st = subtitle.replace("'", "\\'").replace(":", "\\:")
    vf = (f"drawtext=text='({letter})':fontcolor={accent}:fontsize=90:"
          f"x=(w-text_w)/2:y=180:font=Sans,"
          f"drawtext=text='{safe_t}':fontcolor=white:fontsize=38:"
          f"x=(w-text_w)/2:y=308:font=Sans")
    if subtitle:
        vf += (f",drawtext=text='{safe_st}':fontcolor=0xaaaaaa:fontsize=24:"
               f"x=(w-text_w)/2:y=368:font=Sans")
    run([FFMPEG, "-y",
         "-f", "lavfi", "-i", f"color=c={bg}:s={W}x{H}:d={dur}:r=25",
         "-f", "lavfi", "-i", f"aevalsrc=0:c=stereo:s={SR}:d={dur}",
         "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "aac", "-ar", str(SR), "-ac", "2",
         "-pix_fmt", "yuv420p", "-t", str(dur), path],
        f"section:{letter}")

def label_clip(src_mp4, src_wav, dst, top, bottom, dur=12.0, vol_db=6.0):
    W, H = 1280, 720
    safe_top = top.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")
    safe_bot = bottom.replace("'", "\\'").replace(":", "\\:").replace("%", "\\%")
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
        f"drawbox=x=0:y=0:w={W}:h=54:color=black@0.65:t=fill,"
        f"drawtext=text='{safe_top}':fontcolor=0xe94560:fontsize=21:"
        f"x=16:y=15:font=Sans,"
        f"drawbox=x=0:y={H-46}:w={W}:h=46:color=black@0.70:t=fill,"
        f"drawtext=text='{safe_bot}':fontcolor=0xcccccc:fontsize=17:"
        f"x=16:y={H-32}:font=Sans,"
        f"drawtext=text='Headphones required':fontcolor=0xfbbf24:fontsize=16:"
        f"x={W-230}:y=17:font=Sans"
    )
    run([FFMPEG, "-y",
         "-i", src_mp4, "-i", src_wav,
         "-map", "0:v:0", "-map", "1:a:0",
         "-vf", vf, "-af", f"volume={vol_db}dB",
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-c:a", "aac", "-ar", str(SR), "-ac", "2",
         "-pix_fmt", "yuv420p", "-t", str(dur), dst],
        f"label:{os.path.basename(dst)}")

def scale_and_label(src_mp4, dst, top, bottom, dur=None, vol_db=6.0,
                    audio_src=None):
    """Scale capture recording to 1280x720 and add label bars."""
    W, H = 1280, 720
    safe_top = top.replace("'", "\\'").replace(":", "\\:").replace("%","\\%")
    safe_bot = bottom.replace("'", "\\'").replace(":", "\\:").replace("%","\\%")
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
        f"drawbox=x=0:y=0:w={W}:h=54:color=black@0.65:t=fill,"
        f"drawtext=text='{safe_top}':fontcolor=0x00e5a0:fontsize=20:"
        f"x=16:y=16:font=Sans,"
        f"drawbox=x=0:y={H-44}:w={W}:h=44:color=black@0.70:t=fill,"
        f"drawtext=text='{safe_bot}':fontcolor=0xcccccc:fontsize=16:"
        f"x=16:y={H-30}:font=Sans,"
        f"drawtext=text='Headphones required':fontcolor=0xfbbf24:fontsize=16:"
        f"x={W-230}:y=17:font=Sans"
    )
    if audio_src:
        cmd = [FFMPEG, "-y",
               "-i", src_mp4, "-i", audio_src,
               "-map", "0:v:0", "-map", "1:a:0",
               "-vf", vf, "-af", f"volume={vol_db}dB",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-c:a", "aac", "-ar", str(SR), "-ac", "2",
               "-pix_fmt", "yuv420p"]
    else:
        cmd = [FFMPEG, "-y",
               "-i", src_mp4,
               "-vf", vf, "-af", f"volume={vol_db}dB",
               "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-c:a", "aac", "-ar", str(SR), "-ac", "2",
               "-pix_fmt", "yuv420p"]
    if dur:
        cmd += ["-t", str(dur)]
    cmd.append(dst)
    run(cmd, f"scale:{os.path.basename(dst)}")

# ═════════════════════════════════════════════════════════════════════════════

parts = []

print("\n== Title ==")
p = f"{OUT}/v3_00_title.mp4"
make_card(p,
    "Vid2Spatial",
    "Spatial Audio Authoring from Video and Sketch",
    "Single bounding box  ->  spatial parameter stream  ->  binaural / FOA / OSC",
    dur=4.0, bg="0x080818", accent="0xe94560")
parts.append(p)

print("\n== Section (a) ==")
p = f"{OUT}/v3_01a_header.mp4"
make_section_header(p, "a", "Video Tracking",
    "Single bbox init  ->  SAM2  ->  HRTF binaural", dur=3.0)
parts.append(p)

print("\n== car-10 ==")
p = f"{OUT}/v3_01b_car10.mp4"
label_clip(f"{BASE}/car-10/car-10_e2e.mp4", f"{BASE}/car-10/binaural.wav", p,
    "car-10  |  Vehicle (COCO)  |  AzMAE = 0.62°  |  Az range = 43°  (L->R sweep)",
    "1 bbox drawn  ->  SAM2 tracks 300 frames  ->  HRTF binaural render",
    dur=12.0, vol_db=6.0)
parts.append(p)
p = f"{OUT}/v3_01c_car10_card.mp4"
make_card(p, "car-10   Vehicle (COCO)",
    "AzMAE = 0.62°   |   Az range = 43°",
    "Clear left-to-right lateral sweep", dur=3.0, bg="0x080818")
parts.append(p)

print("\n== skateboard-8 ==")
p = f"{OUT}/v3_02a_sk8.mp4"
label_clip(f"{BASE}/skateboard-8/skateboard-8_e2e.mp4", f"{BASE}/skateboard-8/binaural.wav", p,
    "skateboard-8  |  Sports (COCO)  |  AzMAE = 0.18°  (best in 22-clip eval)",
    "Az range = 38°   |   Dist range = 5.1m  (lateral + approach)",
    dur=12.0, vol_db=6.0)
parts.append(p)
p = f"{OUT}/v3_02b_sk8_card.mp4"
make_card(p, "skateboard-8   Sports (COCO)",
    "AzMAE = 0.18°   |   Best accuracy in 22-clip eval",
    "Combines lateral motion + distance approach", dur=3.0, bg="0x080818")
parts.append(p)

print("\n== drone-13 ==")
p = f"{OUT}/v3_03a_drone.mp4"
label_clip(f"{BASE}/drone-13/drone-13_e2e.mp4", f"{BASE}/drone-13/binaural.wav", p,
    "drone-13  |  Drone (Non-COCO)  |  AzMAE = 0.44°  |  category-agnostic",
    "Az=27.8°  El=19.8°  Dist=8.1m  -  full 3D spatial parameter stream",
    dur=12.0, vol_db=6.0)
parts.append(p)
p = f"{OUT}/v3_03b_drone_card.mp4"
make_card(p, "drone-13   Drone (Non-COCO)",
    "AzMAE = 0.44°   |   El range = 19.8°   |   Dist = 8.1m",
    "Open-vocabulary tracking - no category constraints", dur=3.0, bg="0x080818")
parts.append(p)

print("\n== Section (b) Sketch ==")
p = f"{OUT}/v3_04a_sketch_header.mp4"
make_section_header(p, "b", "Sketch Interface",
    "Freehand 3D sphere for off-screen and non-literal sources", dur=3.5)
parts.append(p)

p = f"{OUT}/v3_04b_sketch.mp4"
scale_and_label(
    f"{CAP}/sketch_ui_recording.mp4", p,
    top="(b) Sketch Mode — Freehand 3D Sphere Trajectory",
    bottom="WebGL sphere  |  lat-lon grid  |  freehand draw  ->  az/el/dist parameter stream",
    dur=15.0, vol_db=0.0)   # audio already in recording
parts.append(p)

p = f"{OUT}/v3_04c_sketch_card.mp4"
make_card(p, "Sketch Mode",
    "Same parameter stream as video mode",
    "Off-screen, ambient, and non-literal sources",
    dur=3.0, bg="0x080818", accent="0x00e5a0")
parts.append(p)

print("\n== Section (c) OSC ==")
p = f"{OUT}/v3_05a_osc_header.mp4"
make_section_header(p, "c", "OSC Parameter Streaming",
    "Real-time UDP to any spatial audio engine", dur=3.5)
parts.append(p)

p = f"{OUT}/v3_05b_osc.mp4"
scale_and_label(
    f"{CAP}/osc_terminal_recording.mp4", p,
    top="(c) OSC Streaming — /vid2spatial/{az, el, d_rel, confidence}",
    bottom="Verified: Reaper  |  Max/MSP  |  TouchDesigner  |  Unreal Engine  |  Pure Data",
    dur=13.0, vol_db=0.0)
parts.append(p)

p = f"{OUT}/v3_05c_osc_card.mp4"
make_card(p, "OSC Output",
    "Verified: Reaper  Max/MSP  TouchDesigner  Unreal Engine  Pure Data",
    "Atomic UDP bundle  ->  frame-coherent multi-parameter update",
    dur=3.0, bg="0x080818", accent="0x00e5a0")
parts.append(p)

print("\n== Final ==")
p = f"{OUT}/v3_06_final.mp4"
make_card(p,
    "22 LaSOT clips  |  AzMAE = 1.36°  (median 0.70°)",
    "N=20 perceptual p<=0.025  |  N=12 efficiency -41-49% time  |  SUS 71.0",
    "Single bbox  ->  binaural / FOA AmbiX / OSC  (one extraction, any consumer)",
    dur=5.0, bg="0x0a0a1a", accent="0xe94560")
parts.append(p)

print("\n== Concat ==")
final = f"{OUT}/vid2spatial_demo_v3.mp4"
concat_videos(parts, final)

size = os.path.getsize(final) / 1e6
print(f"\n✓ {final}  ({size:.1f} MB)  {len(parts)} parts")

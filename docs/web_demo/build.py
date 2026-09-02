#!/usr/bin/env python3
"""Assemble docs/web_demo/index.html from the template + compact demo assets.

Assets (built by this script from repo data, cached in docs/web_demo/assets/):
  <clip>.mp4        640x360 H.264, LaSOT frames 1..300 at 25 fps, no audio
  <clip>.mono.m4a   12 s mono AAC source sound (same offset as the paper renders)
  <clip>.kemar.m4a  12 s stereo AAC of the paper's KEMAR binaural render
  <clip>.traj.json  [az, el, d_rel, cx, cy, w, h, conf] per frame from test/e2e_final
Everything is inlined as base64 so the page is one self-contained file
(publishable as a claude.ai artifact or on GitHub Pages).
"""
import base64, json, os, subprocess, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
ASSETS = os.path.join(ROOT, "assets")
CLIPS = {"car-10": "S01_car-10", "drone-13": "S08_drone-13", "skateboard-8": "S04_skateboard-8", "dog-1": "S02_dog-1"}

def ffmpeg():
    import shutil
    return shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()

def build_assets():
    os.makedirs(ASSETS, exist_ok=True)
    FF = ffmpeg()
    for c, stim in CLIPS.items():
        v = os.path.join(ASSETS, c + ".mp4")
        if not os.path.exists(v):
            subprocess.check_call([FF, "-y", "-loglevel", "error", "-framerate", "25", "-start_number", "1",
                "-i", f"{REPO}/data/lasot/{c}/img/%08d.jpg", "-frames:v", "300",
                "-vf", "scale=640:360:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "slow",
                "-crf", "31", "-profile:v", "main", "-movflags", "+faststart", "-an", v])
        st = f"{REPO}/test/listening_test_v3/stimuli/{stim}"
        for name, src, ac, br in (("mono", "mono.wav", "1", "56k"), ("kemar", "proposed.wav", "2", "96k")):
            a = os.path.join(ASSETS, f"{c}.{name}.m4a")
            if not os.path.exists(a):
                subprocess.check_call([FF, "-y", "-loglevel", "error", "-i", f"{st}/{src}", "-t", "12",
                                       "-ac", ac, "-c:a", "aac", "-b:a", br, a])
        t = os.path.join(ASSETS, c + ".traj.json")
        if not os.path.exists(t):
            d = json.load(open(f"{REPO}/test/e2e_final/{c}/traj.json"))
            fr = [[round(f["az"], 4), round(f["el"], 4), round(f["d_rel"], 3), round(f["cx"]), round(f["cy"]),
                   round(f["w"]), round(f["h"]), round(f["confidence"], 2)] for f in d["frames"]]
            json.dump({"fps": d["fps"], "w": 1280, "h": 720, "frames": fr}, open(t, "w"), separators=(",", ":"))

def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()

def main():
    build_assets()
    clips = {}
    for c in CLIPS:
        clips[c] = {"video": b64(f"{ASSETS}/{c}.mp4"), "mono": b64(f"{ASSETS}/{c}.mono.m4a"),
                    "kemar": b64(f"{ASSETS}/{c}.kemar.m4a"), "traj": json.load(open(f"{ASSETS}/{c}.traj.json"))}
    arch = "data:image/jpeg;base64," + b64(f"{REPO}/ISMAR2026/poster/figures/figure_arch.jpg")
    html = open(os.path.join(ROOT, "index.template.html")).read()
    html = html.replace("{{ASSETS}}", json.dumps({"clips": clips}, separators=(",", ":"))).replace("{{ARCH_IMG}}", arch)
    out = os.path.join(ROOT, "index.html")
    open(out, "w").write(html)
    print(f"wrote {out}  {os.path.getsize(out)/1e6:.2f} MB")

if __name__ == "__main__":
    main()

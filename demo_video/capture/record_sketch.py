#!/usr/bin/env python3
"""Record the playground's sketch mode (docs/web_demo/index.html) for the teaser.

Draws a scripted stroke on the WebGL sphere (left -> behind -> right, with the
distance moving in and out), presses Play, and records ~8 s at 1280x720.
Writes: capture/sketch_rec.webm, capture/sketch_traj.json, capture/sketch_marks.json
"""
import json, math, os, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.abspath(os.path.join(HERE, "..", "..", "docs", "web_demo", "index.html"))
W, H = 1280, 720

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist",
                                "--autoplay-policy=no-user-gesture-required"])
    ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=HERE,
                        record_video_size={"width": W, "height": H}, device_scale_factor=1)
    page = ctx.new_page()
    t_ctx = time.time()
    page.goto("file://" + HTML, wait_until="networkidle")
    page.wait_for_timeout(1500)
    # scroll so the sketch section's stage sits nicely in the viewport
    page.evaluate("document.querySelector('#sketch').scrollIntoView({block:'start'}); window.scrollBy(0, 290)")
    page.wait_for_timeout(800)
    box = page.locator("#skCanvas").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    r = min(box["width"], box["height"]) * 0.42
    page.wait_for_timeout(1200)
    marks = {"t_ctx": t_ctx}
    # stroke: an arc from the left, over the top-front, down to the right, ~3.6 s
    N = 90
    pts = []
    for i in range(N + 1):
        u = i / N
        ang = math.pi * (1 - u)              # pi -> 0  (left -> right)
        x = cx + r * math.cos(ang) * 0.95
        y = cy - r * 0.55 * math.sin(ang) - r * 0.25 * math.sin(2 * math.pi * u)
        pts.append((x, y))
    page.mouse.move(*pts[0])
    page.wait_for_timeout(300)
    marks["t_down"] = time.time()
    page.mouse.down()
    for i, (x, y) in enumerate(pts[1:], 1):
        page.mouse.move(x, y, steps=2)
        if i % 30 == 0:  # pull the source closer, then push it away again
            page.mouse.wheel(0, -120 if i == 30 else 240)
        page.wait_for_timeout(36)
    page.mouse.up()
    marks["t_up"] = time.time()
    page.wait_for_timeout(500)
    traj = page.evaluate("document.querySelector('#skExport').value")
    marks["t_play"] = time.time()
    page.click("#skPlay")
    page.wait_for_timeout(4600)
    marks["t_end"] = time.time()
    open(os.path.join(HERE, "sketch_traj.json"), "w").write(traj)
    json.dump(marks, open(os.path.join(HERE, "sketch_marks.json"), "w"), indent=1)
    vpath = page.video.path()
    ctx.close(); b.close()
    os.replace(vpath, os.path.join(HERE, "sketch_rec.webm"))
    print("ok", json.dumps({k: round(v - t_ctx, 2) for k, v in marks.items()}))
    print("points", len(json.loads(traj)["frames"]) if traj else 0)

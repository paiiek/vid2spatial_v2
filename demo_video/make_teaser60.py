#!/usr/bin/env python3
"""Build the ISMAR 2026 60-second fast-forward teaser (poster #5498).

Spec (ieeeismar.net/2026/resources/video-presentation-guidelines):
  <=60 s, 1920x1080 progressive 16:9, MP4/H.264, <=50 MB,
  (v2, Sep 2026: stale overlay captions patched, sketch segment re-recorded from
  the web playground, results card carries the camera-ready headline numbers)
  cover with title/authors/affiliations at the start, credits at the end,
  must still read with the sound off.

Every frame is composed inside the official ISMAR 2026 teaser template
(teaser60/Teaser_Template_ISMAR_2026.pptx, rasterised by ismar_template.py):
the demo fills the template's media placeholder and the gradient footer carries
the ISMAR/IEEE/VGTC logos plus the paper title and authors.  Source clips are the
1280x720 v5 demo segments, scaled into the 1734x975 media area.

Usage:  python3 demo_video/make_teaser60.py
Output: demo_video/teaser60/PT_5498_NA_teaservideo.mp4   (PT = poster tag)
"""
import os
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ismar_template as tpl  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "teaser60")
WORK = os.path.join(OUT, "_work")

W, H, FPS = 1920, 1080, 25
TEMPLATE = os.path.join(OUT, "Teaser_Template_ISMAR_2026.pptx")
MX, MY, MW, MH = 93, 0, 1734, 975   # media placeholder (16:9 fit inside template box)
BG = (255, 255, 255)                # template is a white slide
ACCENT = (210, 68, 95)
INK = tpl.INK                       # template text colour, used for body text
GREY = (110, 104, 104)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

TITLE = "Vid2Spatial"
SUBTITLE = "Controllable Spatial Audio Authoring by Parameter Extraction from Video and Sketch"
AUTHORS = "Seungryeol Paik¹      Kyungsu Kim¹      Kyogu Lee¹·²·³"
AUTHORS_PLAIN = "Seungryeol Paik, Kyungsu Kim, Kyogu Lee  ·  Seoul National University"
AFFIL = [
    "¹Department of Intelligence and Information      "
    "²Interdisciplinary Program in Artificial Intelligence",
    "³Artificial Intelligence Institute      Seoul National University",
]
VENUE = "IEEE ISMAR 2026  ·  Poster #5498"
FUNDING = ("Supported by the InnoCORE program of the Ministry of Science and ICT "
           "(AI Meta-Scientist, N10260110).")

# (clip stem, start seconds, duration seconds).  With cover, results card and
# credits this sums to exactly 59.0 s.
CUT = [
    ("v5_01a_header", 0.2, 2.0),   # what the system does
    ("v5_01b_car10", 1.0, 9.0),    # video mode: car
    ("v5_01c_car10_card", 0.2, 1.5),
    ("v5_03a_drone", 1.0, 8.5),    # video mode: drone (off-distribution class)
    ("v5_03b_drone_card", 0.2, 1.5),
    ("v5_04a_sketch_header", 0.2, 1.5),
    ("v6_04b_sketch", 8.5, 8.0),   # sketch mode, recorded from docs/web_demo (capture/record_sketch.py)
    ("v5_05a_osc_header", 0.2, 1.5),
    ("v5_05b_osc", 1.0, 7.5),      # OSC / runtime
]
COVER_SEC, RESULTS_SEC, CREDITS_SEC = 5.0, 7.0, 6.0

# The v5 car/drone renders carry a stale trajectory-bar caption ("... AzMAE 0.00deg",
# a placeholder from the March overlay) and a clipped "Headphones required" tag.
# Paint over that strip (source 1280x720 coords) and write the real figure.
FIX = {
    "v5_01b_car10": ("car-10   yw_sam2 (orange) vs. ground truth   AzMAE 0.62 deg", (240, 160, 48, 255)),
    "v5_03a_drone": ("drone-13   yw_sam2 (cyan) vs. ground truth   AzMAE 0.44 deg", (64, 224, 216, 255)),
    "v6_04b_sketch": ("caption", "Freehand stroke on the sphere  ->  az / el / distance stream  ->  live HRTF playback"),
}

# Camera-ready headline results (ISMAR 2026 poster #5498, Aug 2026 revision).
RESULTS = [
    ("1.36\u00b0", "mean azimuth error", "22 LaSOT clips, 5 supercategories; median 0.70\u00b0"),
    ("41\u201349%", "faster authoring", "N = 12 within-subjects vs. manual, p < 0.001"),
    ("non-inferior", "perceived quality", "0.5-point Likert margin, satisfaction held"),
    ("1.35 ms", "authoring-to-runtime", "median OSC latency, p99 9.09 ms, no drops"),
]


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


FF = ffmpeg_exe()


def run(args):
    p = subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error"] + args,
                       capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"ffmpeg failed:\n{' '.join(args)}\n{p.stderr[-2000:]}")


def centred(draw, y, text, font, fill, box=(0, W)):
    w = draw.textbbox((0, 0), text, font=font)[2]
    draw.text((box[0] + (box[1] - box[0] - w) // 2, y), text, font=font, fill=fill)


def fitted(draw, y, text, path, size, fill, margin=140, box=(0, W)):
    """Shrink until the line fits inside the box minus margins, then centre it."""
    while size > 12:
        font = ImageFont.truetype(path, size)
        if draw.textbbox((0, 0), text, font=font)[2] <= (box[1] - box[0]) - 2 * margin:
            break
        size -= 2
    centred(draw, y, text, font, fill, box)


def make_frame(path):
    """Template overlay: logos footer + title/authors in the template's text boxes."""
    im = tpl.render(TEMPLATE)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = tpl.TITLE_BOX
    fitted(d, y0 + 10, f"{TITLE}: {SUBTITLE}", FONT_B, 28, INK, margin=0, box=(x0, x1))
    x0, y0, x1, y1 = tpl.AUTHOR_BOX
    fitted(d, y0 + 2, AUTHORS_PLAIN, FONT, 20, INK, margin=0, box=(x0, x1))
    im.save(path)
    return im


def compose_card(card, frame):
    """Card (1734x975) placed in the media area, then the template on top."""
    im = Image.new("RGB", (W, H), BG)
    im.paste(card, (MX, MY))
    im.paste(frame, (0, 0), frame)
    return im


def make_cover(path, frame):
    card = Image.new("RGB", (MW, MH), (250, 247, 244))
    d = ImageDraw.Draw(card)
    B = (0, MW)
    centred(d, 200, TITLE, ImageFont.truetype(FONT_B, 108), ACCENT, B)
    fitted(d, 350, SUBTITLE, FONT, 44, INK, box=B)
    fitted(d, 500, AUTHORS, FONT, 42, INK, box=B)
    y = 600
    for line in AFFIL:
        fitted(d, y, line, FONT, 28, GREY, box=B)
        y += 44
    centred(d, 810, VENUE, ImageFont.truetype(FONT, 30), GREY, B)
    compose_card(card, frame).save(path)


def make_credits(path, frame):
    card = Image.new("RGB", (MW, MH), (250, 247, 244))
    d = ImageDraw.Draw(card)
    B = (0, MW)
    centred(d, 150, TITLE, ImageFont.truetype(FONT_B, 76), ACCENT, B)
    fitted(d, 270, SUBTITLE, FONT, 32, INK, box=B)
    fitted(d, 380, AUTHORS, FONT, 38, INK, box=B)
    y = 470
    for line in AFFIL:
        fitted(d, y, line, FONT, 26, GREY, box=B)
        y += 40
    fitted(d, 600, "Contact: paik402@snu.ac.kr", FONT, 26, GREY, box=B)
    fitted(d, 690, FUNDING, FONT, 24, GREY, box=B)
    fitted(d, 760, "Demo clips from the LaSOT benchmark (Fan et al., CVPR 2019, Creative Commons); everything else is the authors' own.", FONT, 22, GREY, box=B)
    centred(d, 850, VENUE, ImageFont.truetype(FONT, 30), GREY, B)
    compose_card(card, frame).save(path)


def make_results(path, frame):
    card = Image.new("RGB", (MW, MH), (250, 247, 244))
    d = ImageDraw.Draw(card)
    B = (0, MW)
    centred(d, 70, "What the numbers say", ImageFont.truetype(FONT_B, 54), INK, B)
    centred(d, 150, "One extraction, any consumer: binaural, FOA AmbiX, OSC", ImageFont.truetype(FONT, 28), GREY, B)
    cols, gap, top = 2, 40, 240
    cw, ch = (MW - 160 - gap) // cols, 300
    for i, (big, label, sub) in enumerate(RESULTS):
        x = 80 + (i % cols) * (cw + gap); y = top + (i // cols) * (ch + gap)
        d.rounded_rectangle((x, y, x + cw, y + ch), radius=18, fill=(255, 255, 255), outline=(225, 220, 216), width=2)
        d.rectangle((x, y, x + 10, y + ch), fill=ACCENT)
        fitted(d, y + 45, big, FONT_B, 84, ACCENT, margin=40, box=(x, x + cw))
        fitted(d, y + 165, label, FONT_B, 34, INK, margin=40, box=(x, x + cw))
        fitted(d, y + 220, sub, FONT, 24, GREY, margin=40, box=(x, x + cw))
    compose_card(card, frame).save(path)


def card_segment(png, seconds, dst):
    """Still card -> normalized segment with a silent stereo track."""
    run(["-loop", "1", "-framerate", str(FPS), "-i", png,
         "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-t", f"{seconds}", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
         "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
         "-shortest", dst])


TARGET_MEAN_DB = -13.0   # segment loudness to match
PEAK_CEILING_DB = -1.0   # never push a segment's peak above this
FLOOR_DB = -40.0         # below this a segment carries no audio; leave it alone


def measure(src, start, seconds):
    """Return (mean_dB, max_dB) for a segment, or (None, None) if unmeasurable."""
    p = subprocess.run([FF, "-hide_banner", "-ss", f"{start}", "-i", src,
                        "-t", f"{seconds}", "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    out = p.stderr
    def grab(key):
        for line in out.splitlines():
            if key in line:
                return float(line.split(key)[1].split("dB")[0].strip(": "))
        return None
    return grab("mean_volume"), grab("max_volume")


def static_gain_db(src, start, seconds):
    """One fixed gain per segment.

    A time-varying normaliser (loudnorm dynamic, compressors) would rewrite the
    d_rel distance envelope, which is exactly what this demo is showing, so the
    correction has to be a single constant per segment.
    """
    mean, peak = measure(src, start, seconds)
    if mean is None or peak is None or mean < FLOOR_DB:
        return 0.0
    gain = TARGET_MEAN_DB - mean
    return round(min(gain, PEAK_CEILING_DB - peak), 1)


def make_patch(path, text, colour):
    """1280x720 transparent PNG that covers the stale caption strip and rewrites it."""
    im = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if text == "caption":                                          # bottom-margin caption (light clips)
        f = ImageFont.truetype(FONT, 22)
        w = d.textbbox((0, 0), colour, font=f)[2]
        d.text(((1280 - w) // 2, 683), colour, font=f, fill=INK + (255,))
        im.save(path); return
    d.rectangle((95, 593, 1187, 619), fill=(28, 28, 30, 255))     # trajectory-bar caption row
    d.rectangle((1048, 14, 1092, 34), fill=(31, 31, 31, 255))     # clipped "Hea..." tag
    d.text((100, 597), text, font=ImageFont.truetype(FONT, 17), fill=colour)
    im.save(path)


def clip_segment(src, start, seconds, dst, frame_png, fix=None):
    """Source clip -> template media area -> 1920x1080 segment with the footer."""
    gain = static_gain_db(src, start, seconds)
    af = "aresample=48000" + (f",volume={gain}dB" if abs(gain) >= 0.1 else "")
    if abs(gain) >= 0.1:
        print(f"  {os.path.basename(src):26s} gain {gain:+.1f} dB")
    inputs = ["-ss", f"{start}", "-i", src, "-i", frame_png]
    pre = ""
    if fix:
        patch = dst + ".patch.png"
        make_patch(patch, *fix)
        inputs += ["-i", patch]
        pre = "[0:v][2:v]overlay=0:0[p];[p]"
    else:
        pre = "[0:v]"
    vf = (f"{pre}scale={MW}:{MH}:flags=lanczos,fps={FPS},"
          f"pad={W}:{H}:{MX}:{MY}:color=white[v];[v][1:v]overlay=0:0,format=yuv420p")
    run(inputs + ["-filter_complex", vf, "-t", f"{seconds}",
         "-c:v", "libx264", "-preset", "slow", "-crf", "20",
         "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
         "-af", af, dst])


def main():
    os.makedirs(WORK, exist_ok=True)
    parts = []

    frame_png = os.path.join(WORK, "template_frame.png")
    frame = make_frame(frame_png)
    cover_png = os.path.join(WORK, "cover.png")
    credits_png = os.path.join(WORK, "credits.png")
    make_cover(cover_png, frame)
    make_credits(credits_png, frame)

    seg = os.path.join(WORK, "00_cover.mp4")
    card_segment(cover_png, COVER_SEC, seg)
    parts.append(seg)

    for i, (stem, start, dur) in enumerate(CUT, start=1):
        src = os.path.join(ROOT, stem + ".mp4")
        if not os.path.exists(src):
            sys.exit(f"missing source clip: {src}")
        seg = os.path.join(WORK, f"{i:02d}_{stem}.mp4")
        clip_segment(src, start, dur, seg, frame_png, FIX.get(stem))
        parts.append(seg)

    results_png = os.path.join(WORK, "results.png")
    make_results(results_png, frame)
    seg = os.path.join(WORK, "98_results.mp4")
    card_segment(results_png, RESULTS_SEC, seg)
    parts.append(seg)

    seg = os.path.join(WORK, "99_credits.mp4")
    card_segment(credits_png, CREDITS_SEC, seg)
    parts.append(seg)

    listing = os.path.join(WORK, "concat.txt")
    with open(listing, "w") as fh:
        for p in parts:
            fh.write(f"file '{p}'\n")

    final = os.path.join(OUT, "PT_5498_NA_teaservideo.mp4")
    run(["-f", "concat", "-safe", "0", "-i", listing,
         "-c:v", "libx264", "-preset", "slow", "-crf", "21",
         "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart", final])

    planned = COVER_SEC + sum(d for _, _, d in CUT) + RESULTS_SEC + CREDITS_SEC
    size_mb = os.path.getsize(final) / 1024 / 1024
    print(f"wrote {final}")
    print(f"planned {planned:.1f}s, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()

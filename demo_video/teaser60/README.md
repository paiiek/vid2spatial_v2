# ISMAR 2026 fast-forward teaser — poster #5498

**File:** `PT_5498_NA_teaservideo.mp4` (PT = poster contribution tag) · **Due 4 September 2026 AoE**

Every frame sits inside the official `Teaser_Template_ISMAR_2026.pptx` (kept in this
folder): demo in the media placeholder, gradient footer with ISMAR/IEEE/CS/VGTC logos,
title and authors in the template text boxes. `demo_video/ismar_template.py` rasterises
the template (no PowerPoint on this machine).

Rebuild with `python3 demo_video/make_teaser60.py`.

## Spec compliance

Requirements from <https://www.ieeeismar.net/2026/resources/video-presentation-guidelines/>.

| Requirement | Spec | This file |
|---|---|---|
| Duration | ≤ 60 s | **59.16 s** |
| Resolution | 1920×1080 progressive | **1920×1080 progressive** |
| Aspect ratio | 16:9 | **16:9** |
| Container / codec | MP4, H.264 | **MP4, H.264 High, yuv420p** |
| File size | ≤ 50 MB | **12.1 MB** |
| Cover with title, authors, affiliations | required, template | **0:00–0:05**, inside template frame |
| Credits | required at end, template | **0:54–0:60**, inside template frame (contact, funding, LaSOT credit) |
| Readable with sound off | required | every clip is captioned on-screen |
| File name | `tag_paper-id_session_teaservideo` | `PT_5498_NA_teaservideo` (`NA` = session not yet assigned) |
| Template | cover/credits must use the provided template | footer banner + text boxes on every frame |
| Royalty-free | required | LaSOT clips (Creative Commons, credited); all other media is ours |

Audio is 48 kHz stereo AAC and carries the actual binaural render, but nothing
depends on it.

## Shot list (v2, 2 Sep 2026)

| Time | Length | Content |
|---|---|---|
| 0:00 | 5.0 s | Cover: title, three authors, three affiliations, venue |
| 0:05 | 2.0 s | Header: what the system does |
| 0:07 | 9.0 s | Video mode, car-10: one box -> SAM2 over 300 frames -> binaural, with az/el HUD |
| 0:16 | 1.5 s | car-10 result card |
| 0:17 | 8.5 s | Video mode, drone-13 (0.44 deg AzMAE, the off-distribution case) |
| 0:26 | 1.5 s | drone result card |
| 0:27 | 1.5 s | Header: sketch interface |
| 0:29 | 8.0 s | Sketch mode, recorded from the web playground (docs/web_demo) with an offline KEMAR render of the stroke |
| 0:37 | 1.5 s | Header: real-time OSC |
| 0:38 | 7.5 s | OSC streaming to a runtime (visual only, no audio in this segment) |
| 0:46 | 7.0 s | Results card: 1.36 deg AzMAE, 41-49 % faster authoring, non-inferior quality, 1.35 ms latency (camera-ready numbers) |
| 0:53 | 6.0 s | Credits: authors, affiliations, contact, funding, venue |

Changes from v1 (`_v1_PT_5498_NA_teaservideo.mp4`): the March overlays carried a
placeholder trajectory-bar caption ("AzMAE 0.00deg") and a clipped "Headphones
required" tag; both are painted over with the real per-clip figure. The sketch
segment is a fresh capture of the current playground (`capture/record_sketch.py`,
audio from `capture/render_sketch_audio.py`). The final card carries the
camera-ready headline results instead of the tracking number alone.

Sources are the 1280x720 `v5_*` demo segments (plus `v6_04b_sketch.mp4`), scaled
to the 1734x975 media area with Lanczos. The skateboard clip was dropped to fit.

`PT_5498_NA_teaservideo.pptx` is the filled ISMAR template slide with the mp4
embedded in its media placeholder (python-pptx), for PCS uploads that want the pptx.

## Audio levelling

Source segments differed by ~10 dB, so each gets **one constant gain** (drone
−7.0 dB, sketch +3.3 dB) to land at −13 dB mean with peaks under −1 dBFS.

Do **not** replace this with `loudnorm` in dynamic mode, a compressor, or any
other time-varying normaliser: those rewrite the `d_rel` distance envelope, which
is the thing the demo exists to show. A constant per-segment gain leaves the
distance dynamics and the binaural ILD/ITD cues untouched.

The OSC segment sits at −62 dB because it has no sound by design; it is left
alone rather than amplified into noise.

## Still to do

1. Watch it once end to end, on headphones.
2. Upload to Google Drive / Dropbox / Mega with **public link access**.
   **Not YouTube** — the guidelines rule it out.
3. Submit the link via PCS by **4 Sep 2026 AoE**.
4. If the session name is assigned before submission, rename `NA` to the session
   and re-upload.

A copy is in `ISMAR2026/poster/_SUBMIT_PCS/5_teaser_PT_5498_NA_teaservideo.mp4`.

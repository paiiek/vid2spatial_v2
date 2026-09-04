#!/usr/bin/env python3
"""Smallest real end-to-end pipeline.run(): video + mono audio -> FOA on disk.

This is the script docs/ISSUES.md I10 cites as the proof that run() executes on
this machine. It exists because the test suite exercises the render helpers
directly and never went through run(), which is how a librosa import error sat
in the single-source path unnoticed.

It needs a video file and a mono wav. --lasot builds both from a LaSOT clip
directory (frames in img/, first GT box from groundtruth.txt) and a synthetic
tone, so no video or audio asset has to be committed.

Usage:
    python3 tools/run_e2e_smoke.py --out DIR \
        --lasot /home/seung/mmhoa/vid2spatial_v2/data/lasot/airplane-1 --frames 60
    python3 tools/run_e2e_smoke.py --out DIR --video v.mp4 --audio a.wav \
        --init-bbox 367,101,41,16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_from_lasot(clip: Path, out: Path, n_frames: int, fps: float = 30.0):
    """An mp4 of the first n frames plus a 2 s tone, and the clip's first box."""
    import cv2
    import numpy as np
    import soundfile as sf

    frames = sorted(clip.glob("img/*.jpg"))[:n_frames]
    if not frames:
        raise SystemExit(f"no frames in {clip}/img")
    h, w = cv2.imread(str(frames[0])).shape[:2]
    video = out / "clip.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.imread(str(f)))
    vw.release()

    sr = 48000
    t = np.arange(int(sr * 2.0)) / sr
    audio = out / "tone.wav"
    sf.write(str(audio), (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype("float32"), sr)

    gt = (clip / "groundtruth.txt").read_text().splitlines()[0]
    bbox = tuple(int(float(v)) for v in gt.split(",")[:4])
    return video, audio, bbox


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="directory for inputs and outputs")
    ap.add_argument("--lasot", help="LaSOT clip dir to build the inputs from")
    ap.add_argument("--frames", type=int, default=60, help="frames to take (default 60)")
    ap.add_argument("--video")
    ap.add_argument("--audio")
    ap.add_argument("--init-bbox", help="x,y,w,h of the object in frame 0")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.lasot:
        video, audio, bbox = build_from_lasot(Path(a.lasot), out, a.frames)
    else:
        if not (a.video and a.audio and a.init_bbox):
            ap.error("give --lasot, or all of --video --audio --init-bbox")
        video, audio = Path(a.video), Path(a.audio)
        bbox = tuple(int(float(v)) for v in a.init_bbox.split(","))

    from vid2spatial_pkg.config import (DepthConfig, OutputConfig, PipelineConfig,
                                        TrackingConfig, VisionConfig)
    from vid2spatial_pkg.pipeline import SpatialAudioPipeline

    cfg = PipelineConfig(
        video_path=str(video), audio_path=str(audio),
        vision=VisionConfig(
            # KCF from a known box: no model download, no detector in the loop
            tracking=TrackingConfig(method="kcf", init_bbox=bbox),
            depth=DepthConfig(backend="none")),
        output=OutputConfig(foa_path=str(out / "out.foa.wav"),
                            trajectory_path=str(out / "out.traj.json"),
                            stereo_path=str(out / "out.stereo.wav")),
    )
    SpatialAudioPipeline(cfg).run()
    for name in ("out.traj.json", "out.foa.wav", "out.stereo.wav"):
        p = out / name
        print(f"  {name}: {p.stat().st_size} B" if p.exists() else f"  {name}: MISSING")
    return 0 if (out / "out.foa.wav").exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Re-render binaural/stereo/foa audio from existing traj.json files.
Tracker is NOT re-run — only the audio render stage.
Use this after fixing render-side bugs (e.g. az negation) to refresh outputs.

Usage:
  python test/rerender_binaural.py [--clips car-5 dog-1] [--conds AK_s1 BT_s1]
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

TEST_DIR  = Path(__file__).parent
V2_ROOT   = TEST_DIR.parent
LASOT     = Path("/home/seung/mmhoa/vid2spatial/data/lasot")
AUDIO_SRC = Path("/tmp/v2s_m3d_trajectory")
SOFA_PATH = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"
FFMPEG    = "/home/seung/miniforge3/bin/ffmpeg"
OUT_ROOT  = TEST_DIR / "tracker_compare"

sys.path.insert(0, str(V2_ROOT))

TEST_CLIPS = {
    "car-5":        dict(cls="car",        speed="fast"),
    "dog-1":        dict(cls="dog",        speed="fast"),
    "motorcycle-3": dict(cls="motorcycle", speed="medium"),
}

TRACKER_CONDITIONS = {
    "AK_s1":   dict(method="adaptive_k",   stride=1),
    "AK_s3":   dict(method="adaptive_k",   stride=3),
    "AK_s5":   dict(method="adaptive_k",   stride=5),
    "BT_s1":   dict(method="v1_bytetrack", stride=1),
    "BT_s3":   dict(method="v1_bytetrack", stride=3),
    "DINO_s1": dict(method="hybrid_dino",  stride=1),
    "DINO_s3": dict(method="hybrid_dino",  stride=3),
    "AUTO_s1": dict(method="auto",         stride=1),
}


def _find_audio(clip_name: str) -> str:
    for p in [
        AUDIO_SRC / f"{clip_name}_m3d_v2.wav",
        AUDIO_SRC / f"{clip_name}_m3d.wav",
        AUDIO_SRC / f"{clip_name}.wav",
    ]:
        if p.exists():
            return str(p)
    cls = clip_name.split("-")[0]
    foley_dir = LASOT / "foley"
    for ext in ("*.wav", "*.mp3"):
        found = list(foley_dir.glob(f"{cls}*{ext}"))
        if not found:
            found = list(foley_dir.glob(f"*{ext}"))
        if found:
            return str(found[0])
    raise FileNotFoundError(f"No audio for '{clip_name}'")


def _make_config(clip_name, clip_info, cond_info, out_dir, traj_json):
    from vid2spatial_pkg.config import (
        PipelineConfig, VisionConfig, CameraConfig, TrackingConfig,
        DepthConfig, RefinementConfig, RoomConfig, SpatialConfig,
        OutputConfig, BinauralConfig,
    )
    img_dir = LASOT / clip_name / "img"
    audio_path = _find_audio(clip_name)

    return PipelineConfig(
        video_path=str(img_dir / "%08d.jpg"),
        audio_path=audio_path,
        trajectory_json=str(traj_json),   # ← precomputed trajectory
        vision=VisionConfig(
            camera=CameraConfig(fov_deg=60.0, sample_stride=cond_info["stride"]),
            tracking=TrackingConfig(
                method=cond_info["method"],
                class_name=clip_info["cls"],
                init_bbox=None,
                fallback_center_if_no_bbox=True,
            ),
            depth=DepthConfig(backend="midas"),
            refinement=RefinementConfig(enabled=False),
        ),
        room=RoomConfig(backend="none", disabled=True),
        spatial=SpatialConfig(
            use_kalman_smoothing=False,    # already smoothed in traj.json
            angle_smooth_ms=50.0,
        ),
        output=OutputConfig(
            foa_path=str(out_dir / "output.foa.wav"),
            stereo_path=str(out_dir / "output.stereo.wav"),
            binaural_path=str(out_dir / "output.binaural.wav"),
            trajectory_path=None,          # don't overwrite traj.json
            binaural_config=BinauralConfig(mode="sofa", sofa_path=SOFA_PATH),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="+", default=list(TEST_CLIPS.keys()))
    parser.add_argument("--conds", nargs="+", default=list(TRACKER_CONDITIONS.keys()))
    args = parser.parse_args()

    clips = [c for c in args.clips if c in TEST_CLIPS]
    conds = [c for c in args.conds if c in TRACKER_CONDITIONS]

    from vid2spatial_pkg.pipeline import SpatialAudioPipeline

    total = len(clips) * len(conds)
    done  = 0

    for clip_name in clips:
        clip_info = TEST_CLIPS[clip_name]
        for cond_name in conds:
            cond_info = TRACKER_CONDITIONS[cond_name]
            done += 1
            out_dir = OUT_ROOT / f"{clip_name}__{cond_name}"
            traj_p  = out_dir / "traj.json"

            print(f"\n[{done}/{total}] {clip_name}/{cond_name}")

            if not traj_p.exists():
                print(f"    [SKIP] no traj.json — run run_tracker_compare.py first")
                continue

            t0 = time.time()
            try:
                cfg = _make_config(clip_name, clip_info, cond_info, out_dir, traj_p)
                pipeline = SpatialAudioPipeline(
                    cfg,
                    use_v2_tracker=False,      # trajectory already provided
                    use_depth_enhance=False,
                    depth_priority=("dist_m",),
                )
                result = pipeline.run()
                print(f"    [OK] t={time.time()-t0:.0f}s")
            except Exception as e:
                print(f"    [ERROR] {type(e).__name__}: {e}")
                traceback.print_exc()

    print("\nDone — all renders refreshed with az negation fix.")


if __name__ == "__main__":
    main()

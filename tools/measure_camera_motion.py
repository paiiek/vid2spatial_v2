#!/usr/bin/env python3
"""
measure_camera_motion.py — how much camera yaw is in a clip? (gap item A3)

Runs the global-motion estimator over one or more clips and reports the
accumulated yaw.  That number is the size of the error camera_frame rendering
makes on a world-static source: with no compensation, every degree of camera
yaw sweeps the whole scene through the listener's head.

Accepts video files or printf-style image sequences
(`/path/to/clip/img/%08d.jpg`, which is how LaSOT ships).

Usage:
  python tools/measure_camera_motion.py clip.mp4
  python tools/measure_camera_motion.py --lasot-root /data/lasot car-5 horse-1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vid2spatial_pkg.camera_motion import estimate_camera_motion_video  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", help="video paths, or LaSOT clip names")
    ap.add_argument("--lasot-root", default="",
                    help="if set, each clip name becomes <root>/<clip>/img/%%08d.jpg")
    ap.add_argument("--fov-deg", type=float, default=60.0)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rows = []
    for clip in args.clips:
        path = (f"{args.lasot_root.rstrip('/')}/{clip}/img/%08d.jpg"
                if args.lasot_root else clip)
        t0 = time.time()
        try:
            m = estimate_camera_motion_video(path, fov_deg=args.fov_deg,
                                             max_frames=args.max_frames)
        except OSError as exc:
            print(f"{clip:16s} [skip] {exc}")
            continue
        dt = time.time() - t0
        row = {
            "clip": clip,
            "n_frames": len(m),
            "total_yaw_deg": float(m.yaw_deg[-1]),
            "peak_abs_yaw_deg": float(np.max(np.abs(m.yaw_deg))),
            "total_pitch_deg": float(np.degrees(m.pitch_rad[-1])),
            "median_inliers": int(np.median(m.inliers[1:])) if len(m) > 1 else 0,
            "ms_per_frame": float(1000.0 * dt / max(len(m), 1)),
        }
        rows.append(row)
        print(f"{clip:16s} n={row['n_frames']:4d}  "
              f"total yaw {row['total_yaw_deg']:+8.2f} deg  "
              f"peak {row['peak_abs_yaw_deg']:7.2f} deg  "
              f"inliers {row['median_inliers']:4d}  "
              f"{row['ms_per_frame']:5.1f} ms/frame")
    if args.json and rows:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"[ok] wrote {args.json}")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
fov_sensitivity.py — how much does the assumed FOV move the answer? (gap item A2)

Pure re-projection: no re-tracking, no video, no models.  An azimuth measured
under one assumed FOV implies a pixel offset, and that pixel offset implies a
different azimuth under a different assumed FOV:

    u   = tan(az) * f_from
    az' = atan(u / f_to),        f = (W/2) / tan(fov/2)

so an existing `traj.json` can be re-scored at any FOV in milliseconds.

Two tables are produced:

1. **Trajectory sensitivity** — for each assumed FOV, the mean absolute azimuth
   shift and the change in azimuth range across the trajectories given. This is
   the size of the systematic error a wrong FOV injects into every render.
2. **Independent AzMAE** — deferred to `tools/eval_azimuth_kitti.py --fov-sweep`,
   which scores against KITTI 3D label centres rather than against a bbox.

Usage:
  python tools/fov_sensitivity.py test/e2e_final/*/traj.json
  python tools/fov_sensitivity.py --glob '/path/to/e2e_final/*/traj.json'
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vid2spatial_pkg.camera_intrinsics import reproject_azimuth  # noqa: E402


def load_traj_az(path: Path) -> Dict:
    data = json.loads(Path(path).read_text())
    frames = data.get("frames") or []
    if not frames:
        return {}
    az = np.degrees(np.array([float(f.get("az", 0.0)) for f in frames]))
    intr = data.get("intrinsics") or {}
    fov = float(intr.get("fov_deg", frames[0].get("fov_deg", 60.0)))
    return {"name": Path(path).parent.name, "az_deg": az, "fov_deg": fov,
            "n": len(frames)}


def sensitivity(trajs: List[Dict], fovs: List[float]) -> List[Dict]:
    rows = []
    for fov in fovs:
        shifts, dranges = [], []
        for t in trajs:
            az0 = t["az_deg"]
            az1 = reproject_azimuth(az0, t["fov_deg"], fov)
            shifts.append(float(np.mean(np.abs(az1 - az0))))
            dranges.append(float((az1.max() - az1.min()) - (az0.max() - az0.min())))
        rows.append({
            "fov_deg": fov,
            "mean_abs_shift_deg": float(np.mean(shifts)),
            "max_abs_shift_deg": float(np.max(shifts)),
            "mean_range_delta_deg": float(np.mean(dranges)),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="traj.json files")
    ap.add_argument("--glob", default="", help="glob for traj.json files")
    ap.add_argument("--fovs", default="40,45,50,55,60,65,70,75,80")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    paths = [Path(p) for p in args.paths]
    if args.glob:
        paths += [Path(p) for p in sorted(globmod.glob(args.glob))]
    trajs = [t for t in (load_traj_az(p) for p in paths) if t]
    if not trajs:
        print("[err] no trajectories loaded")
        return 2

    fovs = [float(v) for v in args.fovs.split(",")]
    rows = sensitivity(trajs, fovs)
    base = trajs[0]["fov_deg"]
    print(f"{len(trajs)} trajectories, "
          f"{sum(t['n'] for t in trajs)} frames, baseline FOV {base:.1f} deg")
    print("assumed_fov  mean|dAz|  max|dAz|  d(range)")
    for r in rows:
        print(f"  {r['fov_deg']:6.1f}   {r['mean_abs_shift_deg']:8.3f}  "
              f"{r['max_abs_shift_deg']:8.3f}  {r['mean_range_delta_deg']:+8.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"n_trajectories": len(trajs), "baseline_fov_deg": base,
             "clips": [t["name"] for t in trajs], "rows": rows}, indent=2) + "\n")
        print(f"[ok] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

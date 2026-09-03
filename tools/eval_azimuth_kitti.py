#!/usr/bin/env python3
"""
eval_azimuth_kitti.py — an azimuth benchmark that is NOT circular.
==================================================================

Why this exists
---------------
`test/run_quant_eval.py` derives the ground-truth azimuth from the GT 2D bbox
with *exactly* the same pinhole formula and *exactly* the same assumed FOV that
the pipeline uses for its prediction:

    az = atan2(cx_px - W/2, f),   f = W / (2 tan(fov/2)),   fov = 60 deg

Both sides therefore share the projection model, and the projection model
cancels.  The resulting number is a strictly monotone reparameterisation of the
bbox-centre pixel error — it is a *tracking* metric wearing a *spatial* metric's
clothes.  It is renamed `PixelAzMAE` in that script for exactly this reason.

This tool supplies the independent geometry check.  KITTI Tracking labels give,
per object per frame, both the 2D bbox and the object's 3D centre in *camera*
coordinates (x, y, z in metres).  So a physically independent ground-truth
azimuth is available with no projection assumption at all:

    az_gt = atan2(x_cam, z_cam)          [RIGHT = positive, same as the repo]

Three azimuths are compared against it:

  A. `repo`   az = atan2(cx_px - W/2, f_60)      -- what the pipeline computes,
              with the repo's hardcoded 60 deg FOV (config.CameraConfig.fov_deg).
  B. `truefov` same formula, but with the FOV implied by the real KITTI P2
              intrinsics (~81.6 deg for the 1242 px colour camera).  Isolates
              the FOV-assumption error.
  C. `calib`  az = atan2((cx_px - cx_P2) / fx_P2, 1) -- full true intrinsics,
              principal point included.  What is left is the irreducible
              "2D bbox centre is not the projected 3D centre" term.

AzMAE(A) is the honest deployed number.  AzMAE(A) - AzMAE(C) is the part that a
correct intrinsics estimate (gap item A2) would remove.

Data
----
Labels : data/kitti_tracking/label_02/00XX.txt   (2.2 MB,
         https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip
         -- the same source tools/build_depth_gt_kitti.py documents)
Calib  : data/kitti_tracking/calib/00XX.txt      (90 kB,
         https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip)
Neither ships images, and none are needed.

Label columns:
  frame track_id type truncated occluded alpha l t r b h w l x y z rot_y

Usage
-----
  python tools/eval_azimuth_kitti.py
  python tools/eval_azimuth_kitti.py --fov-sweep 40,50,60,70,80,81.6
  python tools/eval_azimuth_kitti.py --json out.json --markdown out.md
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# KITTI Tracking colour camera is 1242x375 for most sequences (a few are
# 1224x370 / 1238x374).  The repo's own KITTI tooling uses the same constant
# (tools/build_depth_gt_kitti.py).
DEFAULT_WIDTH = 1242
SKIP_TYPES = {"DontCare", "Misc"}


def fov_to_focal_px(fov_deg: float, width: int) -> float:
    """Horizontal FOV (degrees) -> focal length in pixels, pinhole."""
    return (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def focal_px_to_fov(focal_px: float, width: int) -> float:
    """Focal length in pixels -> horizontal FOV in degrees, pinhole."""
    return math.degrees(2.0 * math.atan((width / 2.0) / focal_px))


def read_calib(path: Path) -> Dict[str, float]:
    """Parse a KITTI tracking calib file; return P2 fx, fy, cx, cy."""
    P2: Optional[np.ndarray] = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, _, rest = line.partition(":")
        if key.strip() == "P2":
            P2 = np.array([float(v) for v in rest.split()], dtype=np.float64).reshape(3, 4)
            break
    if P2 is None:
        raise ValueError(f"no P2 row in {path}")
    return {"fx": float(P2[0, 0]), "fy": float(P2[1, 1]),
            "cx": float(P2[0, 2]), "cy": float(P2[1, 2])}


def parse_sequence(label_path: Path, *, max_z: float = 80.0, min_z: float = 1.0,
                   require_clean: bool = True) -> List[Dict]:
    """Yield per-object records from one label_02/<seq>.txt."""
    rows: List[Dict] = []
    seq = label_path.stem
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 17:
            continue
        frame = int(parts[0])
        track_id = int(parts[1])
        typ = parts[2]
        if typ in SKIP_TYPES or track_id < 0:
            continue
        truncated = float(parts[3])
        occluded = int(parts[4])
        if require_clean and (occluded != 0 or truncated != 0.0):
            continue
        left, top, right, bottom = (float(v) for v in parts[6:10])
        x_cam, y_cam, z_cam = (float(v) for v in parts[13:16])
        if not (min_z <= z_cam <= max_z):
            continue
        rows.append({
            "seq": seq, "frame": frame, "track": f"{seq}_{track_id}", "type": typ,
            "cx_px": 0.5 * (left + right), "cy_px": 0.5 * (top + bottom),
            "x_cam": x_cam, "y_cam": y_cam, "z_cam": z_cam,
        })
    return rows


def _circ_abs_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute angular difference in degrees, wrapped to [0, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def evaluate(records: List[Dict], calibs: Dict[str, Dict[str, float]],
             *, width: int, repo_fov_deg: float) -> Dict:
    cx_px = np.array([r["cx_px"] for r in records], dtype=np.float64)
    x_cam = np.array([r["x_cam"] for r in records], dtype=np.float64)
    z_cam = np.array([r["z_cam"] for r in records], dtype=np.float64)
    fx = np.array([calibs[r["seq"]]["fx"] for r in records], dtype=np.float64)
    ppx = np.array([calibs[r["seq"]]["cx"] for r in records], dtype=np.float64)

    az_gt = np.degrees(np.arctan2(x_cam, z_cam))

    f_repo = fov_to_focal_px(repo_fov_deg, width)
    az_repo = np.degrees(np.arctan2(cx_px - width / 2.0, f_repo))

    # FOV implied by the true intrinsics, principal point still assumed centred
    true_fov = np.array([focal_px_to_fov(v, width) for v in fx])
    az_truefov = np.degrees(np.arctan2(cx_px - width / 2.0, fx))

    # Full true intrinsics (fx and real principal point)
    az_calib = np.degrees(np.arctan2((cx_px - ppx) / fx, 1.0))

    out: Dict = {
        "n_records": int(len(records)),
        "n_tracks": int(len({r["track"] for r in records})),
        "n_sequences": int(len({r["seq"] for r in records})),
        "width_px": width,
        "repo_fov_deg": repo_fov_deg,
        "true_fov_deg_mean": float(np.mean(true_fov)),
        "variants": {},
    }
    for name, az in (("repo", az_repo), ("truefov", az_truefov), ("calib", az_calib)):
        err = _circ_abs_deg(az, az_gt)
        signed = (az - az_gt + 180.0) % 360.0 - 180.0
        out["variants"][name] = {
            "azmae_deg": float(np.mean(err)),
            "median_deg": float(np.median(err)),
            "p90_deg": float(np.percentile(err, 90)),
            "max_deg": float(np.max(err)),
            "signed_bias_deg": float(np.mean(signed)),
            "std_deg": float(np.std(err)),
        }

    # per-type breakdown of the deployed ("repo") variant
    by_type: Dict[str, List[float]] = defaultdict(list)
    err_repo = _circ_abs_deg(az_repo, az_gt)
    for r, e in zip(records, err_repo):
        by_type[r["type"]].append(float(e))
    out["repo_by_type"] = {
        t: {"n": len(v), "azmae_deg": float(np.mean(v))}
        for t, v in sorted(by_type.items(), key=lambda kv: -len(kv[1]))
    }
    return out


def fov_sweep(records: List[Dict], *, width: int, fovs: List[float]) -> List[Dict]:
    cx_px = np.array([r["cx_px"] for r in records], dtype=np.float64)
    x_cam = np.array([r["x_cam"] for r in records], dtype=np.float64)
    z_cam = np.array([r["z_cam"] for r in records], dtype=np.float64)
    az_gt = np.degrees(np.arctan2(x_cam, z_cam))
    rows = []
    for fov in fovs:
        f = fov_to_focal_px(fov, width)
        az = np.degrees(np.arctan2(cx_px - width / 2.0, f))
        rows.append({"fov_deg": fov, "azmae_deg": float(np.mean(_circ_abs_deg(az, az_gt)))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=str(REPO_ROOT / "data/kitti_tracking/label_02"))
    ap.add_argument("--calib", default=str(REPO_ROOT / "data/kitti_tracking/calib"))
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--repo-fov", type=float, default=60.0,
                    help="the FOV the pipeline assumes (config.CameraConfig.fov_deg)")
    ap.add_argument("--max-z", type=float, default=80.0)
    ap.add_argument("--min-z", type=float, default=1.0)
    ap.add_argument("--allow-occluded", action="store_true",
                    help="keep truncated/occluded objects (default: clean objects only)")
    ap.add_argument("--fov-sweep", default="40,45,50,55,60,65,70,75,80",
                    help="comma-separated FOVs for the sensitivity table ('' to skip)")
    ap.add_argument("--json", default=str(REPO_ROOT / "reports/azimuth_kitti_2026-09-04.json"))
    ap.add_argument("--markdown", default=str(REPO_ROOT / "reports/azimuth_kitti_2026-09-04.md"))
    args = ap.parse_args()

    label_dir = Path(args.labels)
    calib_dir = Path(args.calib)
    if not label_dir.is_dir():
        print(f"[err] labels not found: {label_dir}")
        return 2
    if not calib_dir.is_dir():
        print(f"[err] calib not found: {calib_dir}\n"
              f"      get it from "
              f"https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip")
        return 2

    records: List[Dict] = []
    calibs: Dict[str, Dict[str, float]] = {}
    for lp in sorted(label_dir.glob("*.txt")):
        cp = calib_dir / lp.name
        if not cp.exists():
            print(f"[skip] no calib for {lp.name}")
            continue
        calibs[lp.stem] = read_calib(cp)
        records.extend(parse_sequence(lp, max_z=args.max_z, min_z=args.min_z,
                                      require_clean=not args.allow_occluded))
    if not records:
        print("[err] no records after filtering")
        return 2

    res = evaluate(records, calibs, width=args.width, repo_fov_deg=args.repo_fov)
    sweep = []
    if args.fov_sweep.strip():
        sweep = fov_sweep(records, width=args.width,
                          fovs=[float(v) for v in args.fov_sweep.split(",")])
        res["fov_sweep"] = sweep

    v = res["variants"]
    print(f"records={res['n_records']}  tracks={res['n_tracks']}  seqs={res['n_sequences']}")
    print(f"true KITTI h-FOV (from P2, W={args.width}) = {res['true_fov_deg_mean']:.2f} deg")
    for name in ("repo", "truefov", "calib"):
        d = v[name]
        print(f"  {name:8s} AzMAE={d['azmae_deg']:6.3f} deg  median={d['median_deg']:6.3f}  "
              f"p90={d['p90_deg']:6.3f}  bias={d['signed_bias_deg']:+.3f}")
    for row in sweep:
        print(f"  fov={row['fov_deg']:5.1f} -> AzMAE {row['azmae_deg']:6.3f} deg")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2) + "\n")
        print(f"[ok] wrote {args.json}")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(render_markdown(res))
        print(f"[ok] wrote {args.markdown}")
    return 0


def render_markdown(res: Dict) -> str:
    v = res["variants"]
    lines = [
        "# Independent azimuth accuracy on KITTI Tracking 3D labels",
        "",
        "Generated by `tools/eval_azimuth_kitti.py` (2026-09-04).",
        "",
        "## Why the headline 1.36 deg number is not this number",
        "",
        "`test/run_quant_eval.py` builds its ground-truth azimuth from the GT **2D bbox**",
        "with the same pinhole formula and the same assumed 60 deg FOV that the pipeline",
        "uses for its prediction. The projection model cancels on both sides, so that",
        "metric is a monotone reparameterisation of bbox-centre pixel error. It is now",
        "named **PixelAzMAE** in that script and in the docs.",
        "",
        "Here the ground truth is the object's 3D centre in camera coordinates, straight",
        "from the KITTI label: `az_gt = atan2(x_cam, z_cam)`. No projection assumption.",
        "",
        "## Setup",
        "",
        f"- {res['n_records']} object detections, {res['n_tracks']} tracks, "
        f"{res['n_sequences']} sequences (KITTI Tracking `label_02`).",
        "- Clean objects only: `occluded == 0`, `truncated == 0`, depth in [1, 80] m.",
        f"- Assumed image width {res['width_px']} px; pipeline FOV "
        f"{res['repo_fov_deg']:.1f} deg (`config.CameraConfig.fov_deg`).",
        f"- True KITTI colour-camera h-FOV from the P2 intrinsics: "
        f"**{res['true_fov_deg_mean']:.2f} deg**.",
        "",
        "## Result",
        "",
        "| variant | what it assumes | AzMAE (deg) | median | p90 | signed bias |",
        "|---|---|---|---|---|---|",
    ]
    labels = {
        "repo": "bbox centre + hardcoded 60 deg FOV (**the deployed pipeline**)",
        "truefov": "bbox centre + true focal length, principal point assumed centred",
        "calib": "bbox centre + full true intrinsics (fx and real principal point)",
    }
    for name in ("repo", "truefov", "calib"):
        d = v[name]
        lines.append(
            f"| `{name}` | {labels[name]} | **{d['azmae_deg']:.3f}** | "
            f"{d['median_deg']:.3f} | {d['p90_deg']:.3f} | {d['signed_bias_deg']:+.3f} |")
    lines += [
        "",
        f"- The honest deployed error is **{v['repo']['azmae_deg']:.2f} deg**, not ~1.4 deg.",
        f"- Correcting only the FOV assumption takes it to "
        f"{v['truefov']['azmae_deg']:.2f} deg, i.e. the hardcoded 60 deg costs "
        f"{v['repo']['azmae_deg'] - v['truefov']['azmae_deg']:+.2f} deg of AzMAE.",
        f"- With full true intrinsics the residual is {v['calib']['azmae_deg']:.2f} deg. That",
        "  residual is irreducible here: it is the mismatch between the 2D bbox centre and",
        "  the projection of the 3D object centre, which no intrinsics fix removes.",
        "",
        "## AzMAE vs assumed FOV (sensitivity)",
        "",
    ]
    if res.get("fov_sweep"):
        lines += ["| assumed FOV (deg) | AzMAE (deg) |", "|---|---|"]
        for row in res["fov_sweep"]:
            lines.append(f"| {row['fov_deg']:.1f} | {row['azmae_deg']:.3f} |")
        lines.append("")
    lines += ["## Per-class breakdown (deployed `repo` variant)", "",
              "| class | n | AzMAE (deg) |", "|---|---|---|"]
    for t, d in res["repo_by_type"].items():
        lines.append(f"| {t} | {d['n']} | {d['azmae_deg']:.3f} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

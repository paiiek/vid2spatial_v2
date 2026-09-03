#!/usr/bin/env python3
"""
Build a metric-depth ground-truth file for tools/verify_depth_heuristic.py
from the KITTI Tracking benchmark label files (no images needed).

KITTI tracking labels give, per object per frame, a 2D bbox (left, top,
right, bottom in px) AND the 3D object centre in camera coordinates
(x, y, z) — z is the metric range along the optical axis. That is exactly
the (bbox area, depth_m) pair the bbox-area depth heuristic is judged on,
and because the labels are per-track over time it also exercises the
"calibrate at frame 0, then follow the track" assumption.

Source (2.2 MB, public, no login):
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip
  → unzip → training/label_02/00XX.txt   (21 sequences)
Second, non-tracking set (--format object, 5.6 MB):
  https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
  → unzip → training/label_2/00XXXX.txt  (7481 images, same 15 trailing cols)

Label columns:
  frame track_id type truncated occluded alpha l t r b h w l x y z rot_y

Filters (defaults, all overridable):
  * type not in {DontCare, Misc}
  * occluded == 0   (a partially hidden object has a wrong bbox area)
  * truncated == 0  (a bbox cut by the image border is not the object size)
  * 1 m <= z <= 80 m
  * track length after filtering >= --min-len frames (contiguous run is
    NOT required — the heuristic only needs the area sequence)

Output: JSON list of
  {"track": "<seq>_<id>", "type": str, "frame": int,
   "area": px^2, "frame_area": px^2, "depth_m": m}
ordered by track then frame. verify_depth_heuristic.py groups on "track".

Usage:
  python tools/build_depth_gt_kitti.py [--labels data/kitti_tracking/label_02]
                                        [--out test/full_eval/depth_gt.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# KITTI tracking colour cameras are 1242x375 (a few sequences are 1224x370
# or 1238x374; the <1% difference is irrelevant to an area FRACTION).
FRAME_W, FRAME_H = 1242, 375
FRAME_AREA = FRAME_W * FRAME_H

SKIP_TYPES = {"DontCare", "Misc"}


def parse_seq(path: Path, fmt: str = "tracking"):
    """Yield raw records from one label file.

    fmt="tracking": KITTI Tracking label_02/<seq>.txt — 17 cols, leading
        (frame, track_id); track = "<seq>_<id>".
    fmt="object":   KITTI Object Detection label_2/<image>.txt — 15 cols, no
        frame/id (one image per file). Each object becomes its own single-record
        "track" "<image>_<row>" at frame 0, so the per-track proxy is trivially
        exact and only the absolute area→d_rel mapping is meaningful on it.
    """
    seq = path.stem
    off = 2 if fmt == "tracking" else 0
    for i, line in enumerate(path.read_text().splitlines()):
        f = line.split()
        if len(f) < 15 + off:
            continue
        typ = f[off + 0]
        if typ in SKIP_TYPES:
            continue
        yield {
            "track": f"{seq}_{int(f[1])}" if fmt == "tracking" else f"{seq}_{i}",
            "type": typ,
            "frame": int(f[0]) if fmt == "tracking" else 0,
            "truncated": float(f[off + 1]),
            "occluded": int(f[off + 2]),
            "bbox": tuple(float(v) for v in f[off + 4:off + 8]),
            "z": float(f[off + 13]),
        }


def build(labels_dir: Path, min_len: int, max_occ: int, max_trunc: float,
          z_min: float, z_max: float, types: set[str] | None, fmt: str = "tracking"):
    tracks: dict[str, list] = defaultdict(list)
    for p in sorted(labels_dir.glob("*.txt")):
        for r in parse_seq(p, fmt):
            if r["occluded"] > max_occ or r["truncated"] > max_trunc:
                continue
            if not (z_min <= r["z"] <= z_max):
                continue
            if types and r["type"] not in types:
                continue
            x0, y0, x1, y1 = r["bbox"]
            area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
            if area <= 1.0:
                continue
            tracks[r["track"]].append({
                "track": r["track"], "type": r["type"], "frame": r["frame"],
                "area": round(area, 1), "frame_area": FRAME_AREA,
                "depth_m": round(r["z"], 3),
            })
    out = []
    for k in sorted(tracks, key=lambda s: (s.split("_")[0], int(s.split("_")[1]))):
        recs = sorted(tracks[k], key=lambda r: r["frame"])
        if len(recs) >= min_len:
            out.extend(recs)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--labels", default=str(repo / "data/kitti_tracking/label_02"))
    ap.add_argument("--out", default=str(repo / "test/full_eval/depth_gt.json"))
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-occluded", type=int, default=0)
    ap.add_argument("--max-truncated", type=float, default=0.0)
    ap.add_argument("--z-min", type=float, default=1.0)
    ap.add_argument("--z-max", type=float, default=80.0)
    ap.add_argument("--types", default="", help="comma list, e.g. Car,Pedestrian (default all)")
    ap.add_argument("--format", choices=("tracking", "object"), default="tracking",
                    help="'object' = KITTI Object Detection label_2 (per-image, no tracks; "
                         "use --min-len 1, e.g. --labels data/kitti_object/training/label_2 "
                         "--out test/full_eval/depth_gt_object.json)")
    a = ap.parse_args(argv)
    if a.format == "object" and a.min_len > 1:
        a.min_len = 1

    labels_dir = Path(a.labels)
    if not labels_dir.is_dir():
        print(f"labels dir not found: {labels_dir}\n" + __doc__.split("Source")[1].split("Label")[0])
        return 2
    types = {t for t in a.types.split(",") if t} or None
    recs = build(labels_dir, a.min_len, a.max_occluded, a.max_truncated, a.z_min, a.z_max, types, a.format)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(recs, separators=(",", ":")))
    n_tracks = len({r["track"] for r in recs})
    by_type = defaultdict(int)
    for r in recs:
        by_type[r["type"]] += 1
    print(f"{len(recs)} records / {n_tracks} tracks → {a.out}")
    print("  by type:", dict(sorted(by_type.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

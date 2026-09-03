#!/usr/bin/env python3
"""Estimate each KITTI track's calibration depth z0 with a monocular depth model.

Why
---
``tools/verify_depth_heuristic.py`` calibrates the bbox-area proxy on each
track's FIRST record using the ground-truth depth there. That isolates the
proxy term and says so honestly, but it hands the proxy a free oracle: the
deployed system has no ground truth and must estimate z0 from the image. The
number a reader actually wants is the COMPOSED error, proxy error compounded
with z0-estimation error. This tool produces the z0 estimates that
``verify_depth_heuristic.py --z0-from`` consumes.

Model and its caveat
--------------------
Depth Anything V2 **ViT-S** is used, from the checkpoint on this machine. That
checkpoint is the RELATIVE (affine-invariant inverse-depth) model, not one of
the metric variants, so it cannot produce metres on its own. A single global
affine map from predicted disparity to inverse depth,

    1 / z  ~=  a * d + b

is fitted ONCE over all tracks by least squares against the ground truth, and
then applied to every track. This is the standard affine-invariant evaluation
protocol, and it is strictly weaker than the per-track oracle it replaces --
two global numbers for the whole dataset instead of one exact depth per track.
It is NOT free metric depth: a genuinely metric backend (Metric3D v2, or a
Depth-Anything metric checkpoint) would remove the fit entirely. Any composed
result must be reported with this stated.

The adapter's ``build_depth_predictor`` min-max normalises each frame
independently, which destroys cross-frame comparability, so this tool calls
``model.infer_image`` directly and keeps the raw disparity.

Images
------
KITTI Tracking ships no images in this repo (``data/kitti_tracking`` holds
``label_02`` only) and the image zip is 15.8 GB against ~20 GB free, so the
frames are expected to be pre-extracted into ``--images`` as
``<seq>_<frame:06d>.png``.

Usage:
    python tools/estimate_z0_depth_model.py --images DIR --out z0.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_depth_gt_kitti import parse_seq  # noqa: E402

DA_ROOT = "/home/seung/Depth-Anything-V2"
CKPT = f"{DA_ROOT}/checkpoints/depth_anything_v2_vits.pth"


def first_records(labels_dir: Path, max_occ: int = 0, max_trunc: float = 0.0,
                  z_min: float = 1.0, z_max: float = 80.0, min_len: int = 20):
    """Per track, the first record that survives build_depth_gt_kitti's filters.

    The filters are duplicated from that tool so the track keys line up with
    test/full_eval/depth_gt.json exactly.
    """
    tracks: dict[str, list] = defaultdict(list)
    for p in sorted(labels_dir.glob("*.txt")):
        for r in parse_seq(p, "tracking"):
            if r["occluded"] > max_occ or r["truncated"] > max_trunc:
                continue
            if not (z_min <= r["z"] <= z_max):
                continue
            x0, y0, x1, y1 = r["bbox"]
            if max(x1 - x0, 0.0) * max(y1 - y0, 0.0) <= 1.0:
                continue
            tracks[r["track"]].append(r)
    out = {}
    for k, rs in tracks.items():
        if len(rs) < min_len:
            continue
        rs.sort(key=lambda r: r["frame"])
        out[k] = rs[0]
    return out


def build_model(device: str = "cpu"):
    if DA_ROOT not in sys.path:
        sys.path.insert(0, DA_ROOT)
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder="vits", features=64,
                            out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model


def bbox_disparity(depth: np.ndarray, bbox, shrink: float = 0.25) -> float:
    """Median raw disparity over the central part of the box.

    The central crop keeps background pixels around the object's silhouette out
    of the statistic; the median makes the rest robust.
    """
    x0, y0, x1, y1 = bbox
    h, w = depth.shape
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = (x1 - x0) * (1 - shrink), (y1 - y0) * (1 - shrink)
    a = max(int(cx - bw / 2), 0)
    b = min(int(cx + bw / 2) + 1, w)
    c = max(int(cy - bh / 2), 0)
    d = min(int(cy + bh / 2) + 1, h)
    if b <= a or d <= c:
        return float("nan")
    return float(np.median(depth[c:d, a:b]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--labels", default=str(repo / "data/kitti_tracking/label_02"))
    ap.add_argument("--images", required=True,
                    help="dir of pre-extracted <seq>_<frame:06d>.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-frames", type=int, default=200)
    a = ap.parse_args(argv)

    labels = Path(a.labels)
    if not labels.is_dir():
        for alt in (Path("/home/seung/mmhoa/vid2spatial_v2-wt-depthgt/data/kitti_tracking/label_02"),):
            if alt.is_dir():
                labels = alt
                break
    if not labels.is_dir():
        print(f"labels dir not found: {a.labels}")
        return 2

    firsts = first_records(labels)
    images = Path(a.images)
    # group tracks by the image their first record needs
    by_img: dict[str, list] = defaultdict(list)
    for k, r in firsts.items():
        seq = k.split("_")[0]
        by_img[f"{seq}_{r['frame']:06d}.png"].append((k, r))
    have = [n for n in sorted(by_img) if (images / n).exists()][:a.max_frames]
    if not have:
        print(f"no images found in {images}")
        return 2
    print(f"{len(have)} frames covering "
          f"{sum(len(by_img[n]) for n in have)} tracks", flush=True)

    import cv2
    model = build_model(a.device)

    disp, gt = {}, {}
    for i, name in enumerate(have):
        img = cv2.imread(str(images / name))
        if img is None:
            continue
        d = model.infer_image(img).astype(np.float64)  # RAW disparity, unnormalised
        for k, r in by_img[name]:
            v = bbox_disparity(d, r["bbox"])
            if np.isfinite(v):
                disp[k] = v
                gt[k] = float(r["z"])
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(have)} frames", flush=True)

    keys = sorted(disp)
    dd = np.array([disp[k] for k in keys])
    zz = np.array([gt[k] for k in keys])
    # single global affine disparity -> inverse depth, fitted once
    A = np.stack([dd, np.ones_like(dd)], axis=1)
    (a_c, b_c), *_ = np.linalg.lstsq(A, 1.0 / zz, rcond=None)
    inv = a_c * dd + b_c
    z0 = 1.0 / np.clip(inv, 1.0 / 200.0, 1.0 / 0.5)

    rel = np.abs(z0 - zz) / zz
    out = {
        "model": "Depth Anything V2 ViT-S (relative), " + CKPT,
        "alignment": {
            "form": "1/z = a*disparity + b, one global fit over all tracks",
            "a": float(a_c), "b": float(b_c),
            "note": "NOT metric depth from the model; a global affine fitted "
                    "against the GT once, the standard affine-invariant protocol. "
                    "It replaces a per-track oracle z0 with two dataset-wide numbers.",
        },
        "n_tracks": len(keys),
        "z0_abs_rel_mean": float(rel.mean()),
        "z0_abs_rel_median": float(np.median(rel)),
        "z0_delta1": float(np.mean(np.maximum(z0 / zz, zz / z0) < 1.25)),
        "z0": {k: float(v) for k, v in zip(keys, z0)},
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"z0 AbsRel mean {rel.mean():.3f} median {np.median(rel):.3f} "
          f"delta1 {out['z0_delta1']:.3f} over {len(keys)} tracks")
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

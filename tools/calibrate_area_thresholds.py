#!/usr/bin/env python3
"""
Calibrate foa_render gain_mode="bbox_area" thresholds (AREA_NEAR / AREA_FAR)
against the KITTI metric-depth GT built by tools/build_depth_gt_kitti.py.

foa_render maps bbox-area FRACTION a = (w*h)/frame_area to a distance control
    d_rel = 1 - clip((a - AREA_FAR) / (AREA_NEAR - AREA_FAR), 0, 1)
with hard-coded AREA_NEAR=0.08, AREA_FAR=0.001. Those were chosen by eye.
This tool asks: given real (area, depth_m) pairs, which thresholds make d_rel
track a perceptual (log-distance) target best, and how much of the data the
mapping saturates (d_rel pinned at 0 or 1 → no distance cue at all).

Target: d* = clip((ln z - ln z_near) / (ln z_far - ln z_near), 0, 1), the same
log-linear shape foa_render's "hybrid" mode uses for metric depth. z_near/z_far
default to the GT 5th/95th depth percentiles (override with --z-near/--z-far).

Metrics per candidate (near, far):
  * mae        mean |d_rel - d*|            (lower is better; the objective)
  * spearman   rank agreement d_rel vs z   (higher is better)
  * sat_frac   fraction of records with d_rel in {0, 1}
  * sat_near / sat_far  split of the saturation

Search: log-spaced grid over near ∈ [1e-3, 0.5], far ∈ [1e-5, 0.05], far < near.
Also reports the closed-form "quantile" calibration (far = area at z_far,
near = area at z_near, read off the GT median area per depth bin) and an
optional log-area variant of the mapping for comparison (NOT what foa_render
ships; reported so the linear-in-area design decision is quantified).

Usage:
  python tools/calibrate_area_thresholds.py [--gt test/full_eval/depth_gt.json]
        [--z-near M] [--z-far M] [--types Car,Pedestrian] [--json out.json]
Exit 0 always (this is a report, not a gate).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

CUR_NEAR, CUR_FAR = 0.08, 0.001  # foa_render.py shipping values


def d_rel_linear(a: np.ndarray, near: float, far: float) -> np.ndarray:
    return 1.0 - np.clip((a - far) / (near - far), 0.0, 1.0)


def d_rel_log(a: np.ndarray, near: float, far: float) -> np.ndarray:
    la, ln, lf = np.log(np.maximum(a, 1e-12)), np.log(near), np.log(far)
    return 1.0 - np.clip((la - lf) / (ln - lf), 0.0, 1.0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def score(d: np.ndarray, target: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    sat_near = d <= 0.0
    sat_far = d >= 1.0
    return {
        "mae": float(np.mean(np.abs(d - target))),
        "spearman": _spearman(d, z),
        "sat_frac": float(np.mean(sat_near | sat_far)),
        "sat_near": float(np.mean(sat_near)),
        "sat_far": float(np.mean(sat_far)),
    }


def grid_search(a: np.ndarray, target: np.ndarray, z: np.ndarray, fn, n: int = 60) -> Tuple[Dict, list]:
    nears = np.logspace(-3, np.log10(0.5), n)
    fars = np.logspace(-5, np.log10(0.05), n)
    best = None
    rows = []
    for near in nears:
        for far in fars:
            if far >= near * 0.5:
                continue
            s = score(fn(a, near, far), target, z)
            rows.append((float(near), float(far), s["mae"]))
            if best is None or s["mae"] < best["mae"]:
                best = {"near": float(near), "far": float(far), **s}
    return best, rows


def quantile_calibration(a: np.ndarray, z: np.ndarray, z_near: float, z_far: float) -> Dict:
    """area at the target depths = median area of records within ±10% of that depth."""
    def med_area_at(zt: float) -> float:
        m = (z > zt * 0.9) & (z < zt * 1.1)
        return float(np.median(a[m])) if m.sum() >= 20 else float("nan")
    return {"near": med_area_at(z_near), "far": med_area_at(z_far),
            "n_near_bin": int(((z > z_near * 0.9) & (z < z_near * 1.1)).sum()),
            "n_far_bin": int(((z > z_far * 0.9) & (z < z_far * 1.1)).sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--gt", default=str(repo / "test/full_eval/depth_gt.json"))
    ap.add_argument("--z-near", type=float, default=None, help="depth → d_rel=0 (default GT p5)")
    ap.add_argument("--z-far", type=float, default=None, help="depth → d_rel=1 (default GT p95)")
    ap.add_argument("--types", default="", help="comma list filter, e.g. Car,Pedestrian")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    recs = json.loads(Path(args.gt).read_text())
    if args.types:
        keep = set(args.types.split(","))
        recs = [r for r in recs if r.get("type") in keep]
    a = np.array([r["area"] / r["frame_area"] for r in recs], dtype=np.float64)
    z = np.array([r["depth_m"] for r in recs], dtype=np.float64)
    z_near = args.z_near if args.z_near else float(np.percentile(z, 5))
    z_far = args.z_far if args.z_far else float(np.percentile(z, 95))
    target = np.clip((np.log(z) - np.log(z_near)) / (np.log(z_far) - np.log(z_near)), 0.0, 1.0)

    report: Dict = {
        "gt": args.gt, "n": int(len(z)), "types": args.types or "all",
        "z_near": z_near, "z_far": z_far,
        "area_frac_percentiles": {p: float(np.percentile(a, p)) for p in (1, 5, 25, 50, 75, 95, 99)},
        "depth_percentiles": {p: float(np.percentile(z, p)) for p in (1, 5, 25, 50, 75, 95, 99)},
    }
    report["current"] = {"near": CUR_NEAR, "far": CUR_FAR, **score(d_rel_linear(a, CUR_NEAR, CUR_FAR), target, z)}
    best_lin, _ = grid_search(a, target, z, d_rel_linear)
    report["grid_best_linear"] = best_lin
    best_log, _ = grid_search(a, target, z, d_rel_log)
    report["grid_best_logarea"] = best_log
    q = quantile_calibration(a, z, z_near, z_far)
    if np.isfinite(q["near"]) and np.isfinite(q["far"]) and q["far"] < q["near"]:
        q.update(score(d_rel_linear(a, q["near"], q["far"]), target, z))
    report["quantile_linear"] = q
    # d_rel at target should be 0 at z_near, 1 at z_far: check the chosen thresholds by depth bin
    bins = [1, 2, 4, 8, 16, 32, 64, 128]
    per_bin = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (z >= lo) & (z < hi)
        if m.sum() == 0:
            continue
        per_bin.append({"z_range": [lo, hi], "n": int(m.sum()),
                        "d_cur": float(np.mean(d_rel_linear(a[m], CUR_NEAR, CUR_FAR))),
                        "d_best": float(np.mean(d_rel_linear(a[m], best_lin["near"], best_lin["far"]))),
                        "d_target": float(np.mean(target[m]))})
    report["by_depth_bin"] = per_bin
    # per-type saturation with current vs best
    by_type = {}
    for t in sorted({r.get("type") for r in recs}):
        m = np.array([r.get("type") == t for r in recs])
        by_type[t] = {"n": int(m.sum()),
                      "cur_sat": float(np.mean(np.isin(d_rel_linear(a[m], CUR_NEAR, CUR_FAR), [0.0, 1.0]))),
                      "best_sat": float(np.mean(np.isin(d_rel_linear(a[m], best_lin["near"], best_lin["far"]), [0.0, 1.0])))}
    report["by_type"] = by_type

    def fmt(d):
        return (f"near={d['near']:.5f} far={d['far']:.6f}  mae={d['mae']:.3f} "
                f"spearman={d['spearman']:.3f} sat={d['sat_frac']:.3f} (near {d['sat_near']:.3f} / far {d['sat_far']:.3f})")
    print(f"GT {args.gt}: {len(z)} records, types={args.types or 'all'}")
    print(f"target: d*=0 at {z_near:.1f} m, d*=1 at {z_far:.1f} m (log-linear)")
    print("area fraction p5/p50/p95:", {k: round(v, 5) for k, v in report['area_frac_percentiles'].items() if k in (5, 50, 95)})
    print("current  ", fmt(report["current"]))
    print("grid lin ", fmt(best_lin))
    print("grid log ", fmt(best_log), "   [log-area mapping, not shipped]")
    if "mae" in q:
        print("quantile ", fmt(q))
    print("by depth bin (mean d_rel): z_range n cur best target")
    for b in per_bin:
        print(f"  {b['z_range']!s:10} {b['n']:6d} {b['d_cur']:.2f} {b['d_best']:.2f} {b['d_target']:.2f}")
    print("saturation by type (cur → best):", {t: f"{v['cur_sat']:.2f}→{v['best_sat']:.2f}" for t, v in by_type.items()})
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print("report →", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

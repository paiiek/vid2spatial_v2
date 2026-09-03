#!/usr/bin/env python3
"""
Deterministic verification of the bbox-area depth heuristics.

Two heuristics use bbox area as a depth signal:
  (1) depth_utils.compute_bbox_scale_proxy  — z ∝ 1/sqrt(area), calibrated at
      frame 0 (used by V2SpatialTracker._enhance_depth via process_trajectory_depth)
  (2) foa_render gain_mode="bbox_area"      — d_rel = 1 - clip((A/frame - 0.001)/(0.08-0.001))

Synthetic model: pinhole camera, rigid object of physical size S at range z.
Projected bbox side = f*S/z  → area = (f*S/z)^2, so (1) should recover z EXACTLY
when the calibration depth is exact, and both must be monotone (larger bbox →
nearer). Nothing here needs a model, a GPU, or a video.

Ground truth: run with --gt <json> where the file is a list of
{"area": <px^2>, "depth_m": <m>} (optionally "track", "frame_area", "type")
to get MAE / AbsRel / delta1 / Spearman of the proxy against it, calibrated
per track on its first record. Since 2026-09-03 the repo ships
test/full_eval/depth_gt.json built from the public KITTI Tracking labels by
tools/build_depth_gt_kitti.py (2D bbox + camera-frame z per object per
frame), so the default run evaluates against real metric depth. LaSOT
(bboxes only) and FairPlay/TAU (audio) still carry no depth GT.

Usage:
    python tools/verify_depth_heuristic.py [--gt gt.json] [--json out.json]
Exit code 0 = all synthetic checks pass, 1 = a check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vid2spatial_pkg.depth_utils import compute_bbox_scale_proxy  # noqa: E402

# foa_render bbox_area constants (kept in sync with foa_render.interpolate_angles_distance)
AREA_NEAR = 0.08
AREA_FAR = 0.001

# Canonical GT locations that were checked during the 2026-09-02 audit.
_GT_CANDIDATES = (
    "test/depth_gt.json",
    "data/depth_gt.json",
    "test/full_eval/depth_gt.json",
)


def pinhole_area(z_m: np.ndarray, size_m: float = 0.5, focal_px: float = 800.0) -> np.ndarray:
    """bbox area (px^2) of a size_m object at range z under a pinhole camera."""
    return (focal_px * size_m / z_m) ** 2


def bbox_area_d_rel(area_frac: np.ndarray) -> np.ndarray:
    """foa_render gain_mode='bbox_area' mapping: area fraction → d_rel (0 near, 1 far)."""
    area_norm = np.clip((area_frac - AREA_FAR) / (AREA_NEAR - AREA_FAR), 0.0, 1.0)
    return 1.0 - area_norm


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic checks
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_checks() -> Dict[str, Dict]:
    """Return {check_name: {"pass": bool, ...metrics}}. Deterministic."""
    out: Dict[str, Dict] = {}

    # 1. Exact recovery under pinhole model with exact calibration.
    z = np.linspace(1.0, 20.0, 200)
    z0 = float(z[0])
    areas = pinhole_area(z)
    proxy = np.array(compute_bbox_scale_proxy(areas.tolist(), initial_depth_m=z0))
    err = np.abs(proxy - z)
    out["pinhole_exact_recovery"] = {
        "pass": bool(err.max() < 1e-3 * z.max()),
        "max_abs_err_m": float(err.max()),
        "abs_rel_mean": float(np.mean(err / z)),
    }

    # 2. Monotonicity: shrinking bbox → non-decreasing proxy depth (any object size).
    mono_ok = True
    for size in (0.1, 0.5, 2.0):
        a = pinhole_area(z, size_m=size)
        p = np.array(compute_bbox_scale_proxy(a.tolist(), initial_depth_m=z0))
        mono_ok &= bool(np.all(np.diff(p) >= -1e-9))
    out["proxy_monotone_in_area"] = {"pass": mono_ok}

    # 3. Range / positivity, incl. zero and tiny areas (guard clamps area≥1).
    a = np.array([0.0, 0.5, 1.0, 4.0, 1e6, 1e8])
    p = np.array(compute_bbox_scale_proxy(a.tolist(), initial_depth_m=2.0))
    out["proxy_finite_positive"] = {
        "pass": bool(np.all(np.isfinite(p)) and np.all(p > 0.0)),
        "values": p.round(4).tolist(),
    }

    # 4. Calibration sensitivity: proxy error scales LINEARLY with the frame-0
    #    metric-depth error (a 20% wrong anchor → 20% wrong everywhere).
    a = pinhole_area(z)
    p_bad = np.array(compute_bbox_scale_proxy(a.tolist(), initial_depth_m=z0 * 1.2))
    rel = p_bad / z
    out["calibration_error_propagates_linearly"] = {
        "pass": bool(np.allclose(rel, 1.2, atol=1e-6)),
        "rel_scale_min": float(rel.min()), "rel_scale_max": float(rel.max()),
    }

    # 5. Non-rigid / aspect change: same area, different aspect → identical proxy
    #    (heuristic is area-only; documents that it cannot see rotation).
    p_sq = compute_bbox_scale_proxy([100.0 * 100.0, 50.0 * 50.0], 2.0)
    p_rc = compute_bbox_scale_proxy([200.0 * 50.0, 25.0 * 100.0], 2.0)
    out["aspect_invariant"] = {"pass": bool(np.allclose(p_sq, p_rc))}

    # 6. foa_render bbox_area d_rel: in [0,1], non-increasing in area, saturates.
    frac = np.linspace(0.0, 0.2, 401)
    d = bbox_area_d_rel(frac)
    out["bbox_area_d_rel_monotone_range"] = {
        "pass": bool(np.all(d >= 0) and np.all(d <= 1)
                     and np.all(np.diff(d) <= 1e-9)
                     and d[0] == 1.0 and d[-1] == 0.0),
        "d_rel_at_far_thresh": float(bbox_area_d_rel(np.array([AREA_FAR]))[0]),
        "d_rel_at_near_thresh": float(bbox_area_d_rel(np.array([AREA_NEAR]))[0]),
    }

    # 7. Consistency between the two heuristics on a pinhole approach:
    #    both must rank frames identically (Spearman = 1 on strictly moving z).
    z2 = np.linspace(0.8, 12.0, 60)
    a2 = pinhole_area(z2, size_m=0.6, focal_px=900.0)
    frame_area = 1280.0 * 720.0
    p2 = np.array(compute_bbox_scale_proxy(a2.tolist(), initial_depth_m=float(z2[0])))
    d2 = bbox_area_d_rel(a2 / frame_area)
    # Ranks: use only the unsaturated region of d2 for a fair comparison.
    live = (d2 > 0.0) & (d2 < 1.0)
    rho = _spearman(p2[live], d2[live]) if live.sum() > 2 else float("nan")
    out["heuristics_rank_agree"] = {
        "pass": bool(rho > 0.999),
        "spearman": float(rho),
        "unsaturated_frames": int(live.sum()),
    }
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth evaluation (only if a GT file exists / is given)
# ─────────────────────────────────────────────────────────────────────────────

def find_gt(explicit: Optional[str], repo: Path) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for rel in _GT_CANDIDATES:
        p = repo / rel
        if p.exists():
            return p
    return None


def _track_metrics(areas: List[float], gt: np.ndarray, frame_areas: Optional[List[float]]) -> Dict:
    """Heuristic (1) calibrated on the track's first record; (2) from area fraction."""
    proxy = np.array(compute_bbox_scale_proxy(areas, initial_depth_m=float(gt[0])))
    err = np.abs(proxy - gt)
    out = {
        "n": int(len(gt)),
        "mae_m": float(err.mean()),
        "abs_rel": float(np.mean(err / np.maximum(gt, 1e-6))),
        "spearman_scale_proxy": _spearman(proxy, gt),
        "gt_depth_range_m": [float(gt.min()), float(gt.max())],
    }
    if frame_areas is not None:
        frac = np.array(areas) / np.array(frame_areas)
        d_rel = bbox_area_d_rel(frac)
        # d_rel is 0 near / 1 far, so it should rank WITH depth.
        out["spearman_bbox_area_drel"] = _spearman(d_rel, gt)
        out["drel_saturated_frac"] = float(np.mean((d_rel <= 0.0) | (d_rel >= 1.0)))
    return out


def evaluate_against_gt(gt_path: Path) -> Dict:
    """GT = list of {"area": px^2, "depth_m": m, ["track": str], ["frame_area": px^2]}.

    Without "track" the whole file is one track calibrated on record 0
    (legacy shape). With "track" each track is calibrated on ITS first
    record and the per-record errors are pooled; per-track medians are
    reported too so one long track cannot dominate.
    """
    recs: List[Dict] = json.loads(gt_path.read_text())
    if not recs:
        return {"n": 0, "note": "GT file empty"}
    has_frame_area = all("frame_area" in r for r in recs)
    groups: Dict[str, List[Dict]] = {}
    for r in recs:
        groups.setdefault(str(r.get("track", "_single")), []).append(r)

    per_track: Dict[str, Dict] = {}
    pooled_err: List[np.ndarray] = []
    pooled_gt: List[np.ndarray] = []
    pooled_proxy: List[np.ndarray] = []
    for name, rs in groups.items():
        areas = [float(r["area"]) for r in rs]
        gt = np.array([float(r["depth_m"]) for r in rs])
        fa = [float(r["frame_area"]) for r in rs] if has_frame_area else None
        m = _track_metrics(areas, gt, fa)
        m["type"] = rs[0].get("type")
        per_track[name] = m
        proxy = np.array(compute_bbox_scale_proxy(areas, initial_depth_m=float(gt[0])))
        pooled_err.append(np.abs(proxy - gt))
        pooled_gt.append(gt)
        pooled_proxy.append(proxy)

    err = np.concatenate(pooled_err)
    gt_all = np.concatenate(pooled_gt)
    proxy_all = np.concatenate(pooled_proxy)
    abs_rel = err / np.maximum(gt_all, 1e-6)
    tr_mae = np.array([m["mae_m"] for m in per_track.values()])
    tr_rel = np.array([m["abs_rel"] for m in per_track.values()])
    tr_sp = np.array([m["spearman_scale_proxy"] for m in per_track.values()])
    tr_sp = tr_sp[~np.isnan(tr_sp)]
    report = {
        "n": int(len(gt_all)),
        "n_tracks": len(per_track),
        # pooled = every record weighted equally
        "mae_m": float(err.mean()),
        "abs_rel": float(abs_rel.mean()),
        "delta1": float(np.mean(np.maximum(proxy_all / gt_all, gt_all / proxy_all) < 1.25)),
        "spearman": _spearman(proxy_all, gt_all),
        # per-track = every track weighted equally
        "track_median_mae_m": float(np.median(tr_mae)),
        "track_median_abs_rel": float(np.median(tr_rel)),
        "track_median_spearman": float(np.median(tr_sp)) if len(tr_sp) else float("nan"),
        "track_frac_spearman_gt_0_5": float(np.mean(tr_sp > 0.5)) if len(tr_sp) else float("nan"),
        "gt_file": str(gt_path),
    }
    if has_frame_area:
        d_rel_all = np.concatenate([
            bbox_area_d_rel(np.array([float(r["area"]) for r in rs]) / np.array([float(r["frame_area"]) for r in rs]))
            for rs in groups.values()])
        report["bbox_area_drel_spearman"] = _spearman(d_rel_all, gt_all)
        report["bbox_area_drel_saturated_frac"] = float(np.mean((d_rel_all <= 0.0) | (d_rel_all >= 1.0)))
    # by object class
    by_type: Dict[str, List[str]] = {}
    for name, m in per_track.items():
        by_type.setdefault(str(m["type"]), []).append(name)
    report["by_type"] = {
        t: {"n_tracks": len(names),
            "median_abs_rel": float(np.median([per_track[n]["abs_rel"] for n in names])),
            "median_spearman": float(np.nanmedian([per_track[n]["spearman_scale_proxy"] for n in names]))}
        for t, names in sorted(by_type.items())
    }
    report["per_track"] = per_track
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gt", default=None, help="GT json [{area, depth_m}, ...]")
    ap.add_argument("--json", default=None, help="write full report here")
    a = ap.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    report = {"synthetic": synthetic_checks()}
    all_pass = all(v["pass"] for v in report["synthetic"].values())

    print("bbox-area depth heuristic — synthetic checks")
    for name, res in report["synthetic"].items():
        extra = {k: v for k, v in res.items() if k != "pass"}
        print(f"  [{'PASS' if res['pass'] else 'FAIL'}] {name}  {extra}")

    gt_path = find_gt(a.gt, repo)
    if gt_path is None:
        report["ground_truth"] = {
            "available": False,
            "note": ("no metric-depth ground truth in repo (searched "
                     + ", ".join(_GT_CANDIDATES) + "); LaSOT has bboxes only, "
                     "FairPlay/TAU are audio datasets. Synthetic checks only."),
        }
        print("\nground truth: NONE FOUND — " + report["ground_truth"]["note"])
    else:
        report["ground_truth"] = {"available": True, **evaluate_against_gt(gt_path)}
        summary = {k: v for k, v in report["ground_truth"].items() if k != "per_track"}
        print(f"\nground truth: {json.dumps(summary, indent=2)}")

    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2))
        print(f"report → {a.json}")
    if not all_pass:
        verdict = "SYNTHETIC CHECK FAILED"
    elif gt_path is None:
        verdict = ("SYNTHETIC CONSISTENCY CHECKS PASS (not a metric-accuracy validation; "
                   "the pinhole model is the heuristic's own inverse — supply --gt for real error)")
    else:
        g = report["ground_truth"]
        verdict = (f"SYNTHETIC PASS + METRIC GT: AbsRel {g['abs_rel']:.3f}, delta1 {g['delta1']:.3f}, "
                   f"Spearman {g['spearman']:.3f} over {g['n']} records / {g['n_tracks']} tracks")
    print("\nRESULT:", verdict)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

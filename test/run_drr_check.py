#!/usr/bin/env python3
"""
run_drr_check.py — does the rendered direct-to-reverberant ratio follow distance?
================================================================================
Gap item A7 verification.

Auditory distance is carried jointly by intensity (about -6 dB per doubling for
the direct path) and by the direct-to-reverberant ratio, and DRR is the cue that
survives when intensity is unreliable.  The legacy renderer mapped d_rel
linearly into a gain range of [0.3, 1.0] -- a 10.5 dB span across an entire
scene -- and ramped reverb wetness on a separate linear curve that was not tied
to the direct path's falloff, so DRR did not follow distance.

`gain_mode="physical"` uses `1/max(d, d_ref)` for the direct path plus ISO
9613-1 air absorption, and holds the reverb send constant in ABSOLUTE level, so
DRR falls as 1/d by construction.

This script renders a broadband burst at a series of fixed distances in each
mode and measures, on the rendered signal, the direct energy against the
reverberant energy.

Usage: python test/run_drr_check.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vid2spatial_pkg.foa_render import (  # noqa: E402
    apply_distance_gain_lpf, apply_physical_distance,
    build_physical_wet_curve, build_wet_curve_from_dist_occ,
)
from vid2spatial_pkg.irgen import fft_convolve, schroeder_ir  # noqa: E402

DISTANCES_M = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def render_at_distance(dist_m: float, mode: str, sr: int = 48000,
                       dur_s: float = 2.0, rt60: float = 0.6, seed: int = 11):
    """Return (direct, reverberant) signals for a burst at a fixed distance."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * sr)
    x = (rng.standard_normal(n) * 0.2).astype(np.float32)
    dist_s = np.full(n, dist_m, dtype=np.float32)

    if mode.startswith("physical"):
        dry = apply_physical_distance(x, sr, dist_s, d_ref_m=1.0, gain_floor=0.005,
                                      air_absorption=(mode != "physical_no_air"))
        wet = build_physical_wet_curve(dist_s, d_ref_m=1.0, gain_floor=0.005,
                                       reverb_send=0.02)
    else:
        # legacy: d_rel normalised over a 0.5-20 m scene, the shipped default path
        d_rel = np.clip((dist_s - 0.5) / (20.0 - 0.5), 0.0, 1.0)
        dry = apply_distance_gain_lpf(x, sr, dist_s, d_rel)
        wet = build_wet_curve_from_dist_occ(d_rel)

    ir = schroeder_ir(sr, rt60=rt60).astype(np.float32)
    rev = fft_convolve(dry, ir)[:n].astype(np.float32)
    direct = (1.0 - wet[:n]) * dry[:n]
    reverberant = wet[:n] * rev
    return direct, reverberant


def drr_db(direct: np.ndarray, reverberant: np.ndarray) -> float:
    e_d = float(np.mean(direct.astype(np.float64) ** 2)) + 1e-20
    e_r = float(np.mean(reverberant.astype(np.float64) ** 2)) + 1e-20
    return 10.0 * math.log10(e_d / e_r)


def direct_level_db(direct: np.ndarray) -> float:
    return 10.0 * math.log10(float(np.mean(direct.astype(np.float64) ** 2)) + 1e-20)


def run(distances=DISTANCES_M,
        modes=("legacy", "physical", "physical_no_air")) -> dict:
    out = {"distances_m": list(distances), "modes": {}}
    for mode in modes:
        rows = []
        for d in distances:
            direct, rev = render_at_distance(d, mode)
            rows.append({"dist_m": d, "drr_db": drr_db(direct, rev),
                         "direct_db": direct_level_db(direct)})
        drrs = np.array([r["drr_db"] for r in rows])
        lvls = np.array([r["direct_db"] for r in rows])
        # slope per doubling of distance
        oct_ = np.log2(np.array(distances))
        drr_slope = float(np.polyfit(oct_, drrs, 1)[0])
        lvl_slope = float(np.polyfit(oct_, lvls, 1)[0])
        out["modes"][mode] = {
            "rows": rows,
            "drr_span_db": float(drrs[0] - drrs[-1]),
            "direct_span_db": float(lvls[0] - lvls[-1]),
            "drr_db_per_doubling": drr_slope,
            # near field only (first four distances): away from the wet cap,
            # where the law should sit on the textbook -6 dB per doubling
            "drr_db_per_doubling_near": float(np.polyfit(oct_[:4], drrs[:4], 1)[0]),
            "direct_db_per_doubling": lvl_slope,
            "drr_monotonic": bool(np.all(np.diff(drrs) < 0)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO_ROOT / "reports/drr_physical_2026-09-04.json"))
    args = ap.parse_args()
    res = run()
    for mode, m in res["modes"].items():
        print(f"{mode}:")
        for r in m["rows"]:
            print(f"   d={r['dist_m']:5.1f} m   DRR={r['drr_db']:7.2f} dB   "
                  f"direct={r['direct_db']:7.2f} dB")
        print(f"   DRR span {m['drr_span_db']:.2f} dB "
              f"({m['drr_db_per_doubling']:+.2f} dB/doubling), "
              f"direct span {m['direct_span_db']:.2f} dB "
              f"({m['direct_db_per_doubling']:+.2f} dB/doubling), "
              f"monotonic={m['drr_monotonic']}")
        print(f"   DRR near-field slope {m['drr_db_per_doubling_near']:+.2f} dB/doubling")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=2) + "\n")
        print(f"[ok] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

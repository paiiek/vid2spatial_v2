#!/usr/bin/env python3
"""
make_render_golden.py — freeze the default render path against a known commit.
==============================================================================
The 2026-09-04 geometry/render work added new options to the renderers. The
promise attached to that work is that a DEFAULT render is unchanged, so this
builds a fully deterministic stimulus and hashes what the default path produces.

Everything is synthesised in-process: a seeded noise burst, a fixed trajectory
that sweeps azimuth, dips in distance, and carries a `confidence` field that
drops below 0.5 for a run of frames (the case that made the confidence gate a
behaviour change rather than a no-op). No models, no video, no network.

Run it on the reference commit and on the working tree; the arrays must match.

  git worktree add /tmp/base 09289ea
  cp test/make_render_golden.py /tmp/base/test/
  python /tmp/base/test/make_render_golden.py --out golden_base.json
  python test/make_render_golden.py --out golden_head.json
  python test/make_render_golden.py --check golden_base.json

The committed reference lives at test/render_golden_09289ea.json and is checked
by test_geometry_render.py::TestDefaultRenderInvariance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SR = 16000
FPS = 25.0
N_FRAMES = 50
DUR_S = N_FRAMES / FPS
SEED = 20260904

# The SOFA used for the binaural golden. Absent on a machine without it, in
# which case that section is simply omitted from the golden.
SOFA_CANDIDATES = [
    "/home/seung/mmhoa/spatial_engine-proto/.localdeps/libmysofa-v1.3.2/share/"
    "MIT_KEMAR_normal_pinna.sofa",
    "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa",
]


def make_audio() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n = int(DUR_S * SR)
    return (rng.standard_normal(n) * 0.2).astype(np.float32)


def make_trajectory() -> dict:
    """A trajectory that exercises azimuth motion, a distance dip, and a
    low-confidence run of frames."""
    frames = []
    for i in range(N_FRAMES):
        # confidence dips under 0.5 for frames 18..27 inclusive
        conf = 0.25 if 18 <= i <= 27 else 0.85
        frames.append({
            "frame": i,
            "az": math.radians(-30.0 + 1.2 * i),
            "el": math.radians(2.0 * math.sin(i / 7.0)),
            "dist_m": 3.0 + 5.0 * (0.5 - 0.5 * math.cos(i / 8.0)),
            "w": 120 + (i % 5) * 4,
            "h": 90 + (i % 3) * 4,
            "confidence": conf,
            "width": 1280,
            "height": 720,
        })
    return {"frames": frames, "fps": FPS,
            "intrinsics": {"width": 1280, "height": 720, "fov_deg": 60.0}}


def _digest(x: np.ndarray) -> dict:
    a = np.asarray(x, dtype=np.float32)
    return {
        "shape": list(a.shape),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        "sum": float(np.sum(a.astype(np.float64))),
        "abs_max": float(np.max(np.abs(a))) if a.size else 0.0,
        # a coarse fingerprint that survives trivial float noise, for the
        # human reading a diff
        "head": [round(float(v), 7) for v in a.reshape(-1)[:8]],
    }


def build(sofa_path: str = "") -> dict:
    from vid2spatial_pkg.foa_render import (
        _load_and_prepare, encode_mono_to_foa, render_foa_from_trajectory,
    )

    traj = make_trajectory()
    audio = make_audio()
    out: dict = {"sr": SR, "fps": FPS, "n_frames": N_FRAMES, "seed": SEED,
                 "sections": {}}

    with tempfile.TemporaryDirectory() as td:
        wav = str(Path(td) / "in.wav")
        sf.write(wav, audio, SR, subtype="FLOAT")

        # 1. the shared prep, which is where the confidence gate lives
        for mode in ("depth_rel", "bbox_area", "bbox_area_log", "hybrid"):
            proc, sr, az_s, el_s, dist_s, d_rel_s, _ = _load_and_prepare(
                wav, traj, gain_mode=mode)
            out["sections"][f"prep::{mode}::audio"] = _digest(proc)
            out["sections"][f"prep::{mode}::az"] = _digest(az_s)
            out["sections"][f"prep::{mode}::d_rel"] = _digest(d_rel_s)

        # 2. the full FOA render, defaults, with and without reverb
        for reverb in (False, True):
            foa_out = str(Path(td) / f"foa_{int(reverb)}.wav")
            render_foa_from_trajectory(wav, traj, foa_out, apply_reverb=reverb)
            data, _ = sf.read(foa_out, dtype="float32", always_2d=True)
            out["sections"][f"foa::reverb={reverb}"] = _digest(data)

        # 3. direct FOA encode of a fixed angle pair (no file IO)
        proc, sr, az_s, el_s, _, _, _ = _load_and_prepare(wav, traj)
        out["sections"]["encode"] = _digest(encode_mono_to_foa(proc, -az_s, el_s))

        # 4. binaural, when a SOFA is available
        if sofa_path:
            from vid2spatial_pkg.foa_render import render_binaural_from_trajectory
            bin_out = str(Path(td) / "bin.wav")
            render_binaural_from_trajectory(wav, traj, bin_out, sofa_path)
            data, _ = sf.read(bin_out, dtype="float32", always_2d=True)
            out["sections"]["binaural"] = _digest(data)
            out["sofa"] = sofa_path
    return out


def find_sofa() -> str:
    for p in SOFA_CANDIDATES:
        if Path(p).exists():
            return p
    return ""


def compare(a: dict, b: dict, allow_missing: tuple = ()) -> list:
    """Return a list of human-readable mismatches.

    `allow_missing` names section prefixes that may legitimately be absent from
    one side -- the binaural section needs a SOFA file that is not present on
    every machine, and its absence is not drift.
    """
    bad = []
    keys = sorted(set(a["sections"]) & set(b["sections"]))
    only = sorted(set(a["sections"]) ^ set(b["sections"]))
    for k in only:
        if any(k.startswith(pre) for pre in allow_missing):
            continue
        bad.append(f"{k}: present in only one golden")
    for k in keys:
        da, db = a["sections"][k], b["sections"][k]
        if da["sha256"] != db["sha256"]:
            bad.append(f"{k}: sha256 differs (sum {da['sum']:.9g} vs "
                       f"{db['sum']:.9g}, abs_max {da['abs_max']:.9g} vs "
                       f"{db['abs_max']:.9g})")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--check", default="",
                    help="compare against this golden JSON and exit nonzero on drift")
    ap.add_argument("--no-sofa", action="store_true")
    args = ap.parse_args()

    sofa = "" if args.no_sofa else find_sofa()
    res = build(sofa)
    print(f"built {len(res['sections'])} sections"
          f"{' (with SOFA)' if sofa else ' (no SOFA)'}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        print(f"[ok] wrote {args.out}")

    if args.check:
        ref = json.loads(Path(args.check).read_text())
        bad = compare(ref, res, allow_missing=("binaural",) if not sofa else ())
        if bad:
            print(f"[FAIL] {len(bad)} section(s) drifted from {args.check}:")
            for line in bad:
                print(f"   {line}")
            return 1
        print(f"[ok] every section matches {args.check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

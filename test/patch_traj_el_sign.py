#!/usr/bin/env python3
"""
patch_traj_el_sign.py
=====================
Fix elevation sign in existing e2e_final traj.json files.

Bug: vision.py ray_to_angles() returned el = arcsin(y) where y is image-down
     → objects below horizon had positive el (wrong).
Fix: el_correct = -el_stored (negate all stored el values).

Also fixes the 3D y,z fields that depend on el:
  x = dist * cos(el) * sin(az)
  y = dist * sin(el)          ← sign-corrected
  z = dist * cos(el) * cos(az)

Usage:
    cd /home/seung/mmhoa/vid2spatial_v2
    python test/patch_traj_el_sign.py [--dry_run]
"""
import argparse, json, math
from pathlib import Path

E2E_ROOT = Path(__file__).parent / "e2e_final"

CLIPS = [
    "car-5", "car-10", "car-13",
    "dog-1", "dog-3", "dog-9", "dog-14",
    "horse-1", "horse-3", "horse-11",
    "motorcycle-1", "motorcycle-3", "motorcycle-6",
    "skateboard-8", "skateboard-11", "skateboard-17",
    "train-16", "train-17",
    "drone-2", "drone-6", "drone-13",
    "guitar-9",
]


def patch_traj(traj_path: Path, dry_run: bool = False) -> dict:
    with open(traj_path) as f:
        traj = json.load(f)

    frames = traj.get("frames", [])
    if not frames:
        return {"status": "empty"}

    # Check first frame: does it have el stored?
    if "el" not in frames[0]:
        return {"status": "no_el"}

    n_patched = 0
    for fr in frames:
        el_old = float(fr["el"])
        el_new = -el_old
        fr["el"] = el_new

        # Update 3D y field if present (y = dist * sin(el))
        if "y" in fr and "dist_m" in fr:
            dist = float(fr["dist_m"])
            fr["y"] = dist * math.sin(el_new)

        # Update z if present (z = dist * cos(el) * cos(az))
        # cos(el) vs cos(-el) = same, so z unchanged
        # Same for x: x = dist * cos(el) * sin(az) → unchanged

        n_patched += 1

    if not dry_run:
        traj_path.write_text(json.dumps(traj, indent=2))

    return {"status": "ok", "n_patched": n_patched}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"Patching el sign in {len(CLIPS)} traj.json files")
    print(f"  dry_run={args.dry_run}\n")

    ok, skip, err = 0, 0, 0
    for clip in CLIPS:
        traj_path = E2E_ROOT / clip / "traj.json"
        if not traj_path.exists():
            print(f"  {clip:20s} [SKIP] no traj.json")
            skip += 1
            continue
        r = patch_traj(traj_path, dry_run=args.dry_run)
        if r["status"] == "ok":
            print(f"  {clip:20s} patched {r['n_patched']} frames")
            ok += 1
        else:
            print(f"  {clip:20s} [{r['status']}]")
            err += 1

    print(f"\nDone: {ok} patched, {skip} skipped, {err} errors")
    if args.dry_run:
        print("(dry_run — no files written)")


if __name__ == "__main__":
    main()

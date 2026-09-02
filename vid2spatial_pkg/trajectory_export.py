"""
Offline automation export for a tracked trajectory.

The live path (osc_sender.py → spatial_engine/bridge/vid2spatial_osc.py) is
streaming-only. This module writes the same per-frame data to a file so a
trajectory can be replayed, inspected, or imported without the bridge running.

Engine-side status (audit 2026-09-02): spatial_engine-proto's only offline
loader is core/src/scene/TimelineJson.cpp, which transports SCENE SNAPSHOT
keyframes ({t_ms, scene_name, crossfade_ms}); it has no per-object trajectory
or automation-curve reader. So this export is a documented CSV + JSON whose
keys mirror the ADM-OSC bridge contract, not a native engine format.

Row schema (one row per tracked frame):
    frame        int    source video frame index
    t_s          float  frame / fps
    object_id    int    ADM object number N (→ /adm/obj/N/aed)
    az_deg       float  pipeline azimuth, RIGHT = +  (degrees)
    el_deg       float  elevation, UP = +            (degrees)
    dist_m       float  metric distance              (meters)
    dist_norm    float  vid2spatial wire distance, 1 = near, 0 = far
                        (= 1 - min(dist_m / distance_max_m, 1), as osc_sender)
    gain_lin     float  linear gain from the renderer's inverse-square curve
                        (foa_render.apply_distance_gain_lpf hardcoded branch)
    az_adm_deg   float  = -az_deg     (bridge OscTranslator.az_pipeline_to_adm)
    el_adm_deg   float  = el_deg
    dist_adm     float  = 1 - dist_norm  (bridge OscTranslator.dist_v2s_to_adm, the
                        /vid2spatial/distance path, 10 m). CAVEAT: send_frame emits
                        /vid2spatial/spatial LAST and the bridge's _handle_spatial
                        normalises dist_m with 20 m and overwrites the track, so a
                        live bridge currently forwards dist_m/20 = half of this
                        column until the bridge is unified on 10 m (engine-repo item).
    confidence   float  tracker confidence (0-1), 1.0 if absent

CSV: header row + rows above.  JSON: {"format": "vid2spatial-automation",
"version": 1, "fps", "object_id", "distance_max_m", "frames": [row, ...]}.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

FORMAT_NAME = "vid2spatial-automation"
FORMAT_VERSION = 1

COLUMNS: Sequence[str] = (
    "frame", "t_s", "object_id",
    "az_deg", "el_deg", "dist_m", "dist_norm", "gain_lin",
    "az_adm_deg", "el_adm_deg", "dist_adm",
    "confidence",
)

# Mirrors foa_render.apply_distance_gain_lpf (hardcoded ISL branch).
_R_NEAR = 1.0
_R_FAR = 8.0
_D_REL_MIN_M = 0.5
_D_REL_MAX_M = 10.0


def distance_to_norm(dist_m: float, distance_max_m: float = 10.0) -> float:
    """vid2spatial wire distance: 1 = near, 0 = far (same as OSCSpatialSender)."""
    return 1.0 - min(max(float(dist_m), 0.0) / float(distance_max_m), 1.0)


def distance_to_gain(dist_m: float, gain_min: float = 0.3, gain_max: float = 1.0) -> float:
    """Linear gain via the renderer's inverse-square curve on d_rel."""
    d_rel = min(max((float(dist_m) - _D_REL_MIN_M) / (_D_REL_MAX_M - _D_REL_MIN_M), 0.0), 1.0)
    r = _R_NEAR + (_R_FAR - _R_NEAR) * d_rel
    g_isl = min(max((_R_NEAR / r) ** 2, 0.0), 1.0)
    return float(gain_min + (gain_max - gain_min) * g_isl)


def frame_to_row(
    fr: Dict,
    idx: int,
    fps: float,
    object_id: int = 1,
    distance_max_m: float = 10.0,
) -> Dict[str, float]:
    """Convert one tracker frame dict (az/el in radians) to an export row."""
    frame = int(fr.get("frame", idx))
    az_deg = math.degrees(float(fr.get("az", 0.0)))
    el_deg = math.degrees(float(fr.get("el", 0.0)))
    dist_m = float(fr.get("dist_m", 2.0))
    dist_norm = distance_to_norm(dist_m, distance_max_m)
    return {
        "frame": frame,
        "t_s": frame / float(fps),
        "object_id": int(object_id),
        "az_deg": az_deg,
        "el_deg": el_deg,
        "dist_m": dist_m,
        "dist_norm": dist_norm,
        "gain_lin": distance_to_gain(dist_m),
        "az_adm_deg": -az_deg,
        "el_adm_deg": el_deg,
        "dist_adm": 1.0 - dist_norm,
        "confidence": float(fr.get("confidence", 1.0)),
    }


def trajectory_to_rows(
    frames: Iterable[Dict],
    fps: float = 30.0,
    object_id: int = 1,
    distance_max_m: float = 10.0,
) -> List[Dict[str, float]]:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return [frame_to_row(fr, i, fps, object_id, distance_max_m)
            for i, fr in enumerate(frames)]


def _frames_of(trajectory: Union[Dict, List[Dict]]) -> List[Dict]:
    if isinstance(trajectory, dict):
        return list(trajectory.get("frames", []))
    return list(trajectory)


def export_trajectory_csv(
    trajectory: Union[Dict, List[Dict]],
    path: Union[str, Path],
    fps: float = 30.0,
    object_id: int = 1,
    distance_max_m: float = 10.0,
) -> Path:
    """Write the trajectory as CSV (header = COLUMNS). Returns the path."""
    rows = trajectory_to_rows(_frames_of(trajectory), fps, object_id, distance_max_m)
    path = Path(path)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    return path


def export_trajectory_json(
    trajectory: Union[Dict, List[Dict]],
    path: Union[str, Path],
    fps: float = 30.0,
    object_id: int = 1,
    distance_max_m: float = 10.0,
) -> Path:
    """Write the trajectory as ADM-OSC-shaped JSON. Returns the path."""
    rows = trajectory_to_rows(_frames_of(trajectory), fps, object_id, distance_max_m)
    doc = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "fps": float(fps),
        "object_id": int(object_id),
        "distance_max_m": float(distance_max_m),
        "osc_address": f"/adm/obj/{int(object_id)}/aed",
        "columns": list(COLUMNS),
        "frames": rows,
    }
    path = Path(path)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
    return path


def export_trajectory(
    trajectory: Union[Dict, List[Dict]],
    path: Union[str, Path],
    fmt: Optional[str] = None,
    **kw,
) -> Path:
    """Dispatch on `fmt` ("csv" | "json") or the path suffix."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt == "csv":
        return export_trajectory_csv(trajectory, path, **kw)
    if fmt == "json":
        return export_trajectory_json(trajectory, path, **kw)
    raise ValueError(f"unsupported export format '{fmt}' (use csv or json)")


def read_automation_csv(path: Union[str, Path]) -> List[Dict[str, float]]:
    """Read a CSV written by export_trajectory_csv back into typed rows."""
    out: List[Dict[str, float]] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({
                k: (int(v) if k in ("frame", "object_id") else float(v))
                for k, v in r.items()
            })
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Export a vid2spatial trajectory JSON to an automation file (CSV/JSON).")
    p.add_argument("trajectory_json", help="trajectory JSON ({'frames': [...]})")
    p.add_argument("out", help="output path (.csv or .json)")
    p.add_argument("--format", choices=["csv", "json"], default=None,
                   help="override format inferred from the output suffix")
    p.add_argument("--fps", type=float, default=None,
                   help="frame rate; default = the trajectory's own 'fps' (else 30)")
    p.add_argument("--object-id", type=int, default=1, help="ADM object number N")
    p.add_argument("--distance-max-m", type=float, default=10.0,
                   help="meters that map to dist_norm=0 (osc_sender default 10)")
    a = p.parse_args(argv)
    with open(a.trajectory_json) as fh:
        traj = json.load(fh)
    fps = a.fps if a.fps else float(traj.get("fps") or 30.0)
    out = export_trajectory(traj, a.out, a.format, fps=fps,
                            object_id=a.object_id, distance_max_m=a.distance_max_m)
    n = len(_frames_of(traj))
    print(f"[export] {n} frames → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Camera field-of-view resolution from container metadata (gap item A2).

Why
---
Before this module the horizontal FOV was hardcoded to 60 deg
(`config.CameraConfig.fov_deg`) and never read from the file.  A 27 mm phone
lens is ~67 deg h-FOV and a 50 mm is ~40 deg, so assuming 60 deg on a 40 deg
clip inflates every azimuth by ~1.5x — a systematic error larger than the whole
reported PixelAzMAE.  `tools/eval_azimuth_kitti.py` measures the cost directly:
on KITTI (true h-FOV 81.7 deg) the 60 deg assumption costs +2.77 deg of AzMAE.

What it does
------------
`resolve_fov()` tries, in order:

  1. an explicit `--fov-deg` given by the user            -> source "cli:fov"
  2. an explicit `--focal-35mm` given by the user         -> source "cli:focal_35mm"
  3. a sidecar `<video>.fov.json` with {"fov_deg": ...}   -> source "sidecar"
  4. `exiftool -j <video>` (EXIF / XMP, phone footage)    -> source "metadata:exiftool"
  5. `ffprobe -show_streams -show_format <video>`          -> source "metadata:ffprobe"
  6. the 60 deg default, with a LOUD warning              -> source "default"

The provenance is returned alongside the value and is written into the
trajectory JSON (`intrinsics.fov_source`) so a downstream result is auditable.

Design note: the subprocess call and the parsing are separate functions.  The
parsers take already-decoded JSON, so the unit tests exercise the real parsing
logic without needing ffprobe or exiftool installed.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Full-frame 35 mm still format is 36 mm wide; "focal length in 35 mm format"
# is defined against that width.
SENSOR_35MM_WIDTH_MM = 36.0
DEFAULT_FOV_DEG = 60.0

# Sane bounds — anything outside is a parse error, not a lens.
MIN_FOV_DEG = 5.0
MAX_FOV_DEG = 179.0


@dataclass(frozen=True)
class FovInfo:
    """A resolved horizontal field of view plus where it came from."""
    fov_deg: float
    source: str          # "cli:fov" | "cli:focal_35mm" | "sidecar" |
                         # "metadata:exiftool" | "metadata:ffprobe" | "default"
    detail: str = ""     # the raw tag/value the number was derived from

    @property
    def is_default(self) -> bool:
        return self.source == "default"

    def to_dict(self) -> Dict[str, Any]:
        return {"fov_deg": self.fov_deg, "fov_source": self.source,
                "fov_detail": self.detail}


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------
def fov_from_focal_35mm(focal_mm: float) -> float:
    """Horizontal FOV in degrees from a 35 mm-equivalent focal length."""
    if focal_mm <= 0:
        raise ValueError("focal_mm must be positive")
    return math.degrees(2.0 * math.atan((SENSOR_35MM_WIDTH_MM / 2.0) / focal_mm))


def focal_35mm_from_fov(fov_deg: float) -> float:
    """Inverse of :func:`fov_from_focal_35mm`."""
    return (SENSOR_35MM_WIDTH_MM / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def fov_from_focal_px(focal_px: float, width_px: int) -> float:
    """Horizontal FOV in degrees from a focal length in pixels."""
    if focal_px <= 0 or width_px <= 0:
        raise ValueError("focal_px and width_px must be positive")
    return math.degrees(2.0 * math.atan((width_px / 2.0) / focal_px))


def focal_px_from_fov(fov_deg: float, width_px: int) -> float:
    """Focal length in pixels from a horizontal FOV. Matches vision.py."""
    return (width_px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def _valid(fov: Optional[float]) -> bool:
    return fov is not None and MIN_FOV_DEG <= fov <= MAX_FOV_DEG


# ---------------------------------------------------------------------------
# parsers (pure — take decoded JSON, no subprocess)
# ---------------------------------------------------------------------------
def parse_exiftool_json(payload: Any) -> Optional[Tuple[float, str]]:
    """Extract (fov_deg, detail) from `exiftool -j` output.

    Recognises, in priority order:
      FOV                              e.g. "63.4 deg"
      FocalLengthIn35mmFormat          e.g. "27 mm"
      FocalLength35efl                 e.g. "4.2 mm (35 mm equivalent: 27.0 mm)"
    """
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        return None

    raw = payload.get("FOV")
    fov = _first_float(raw)
    if _valid(fov):
        return float(fov), f"exif FOV={raw!r}"

    for key in ("FocalLengthIn35mmFormat", "FocalLength35efl"):
        raw = payload.get(key)
        if raw is None:
            continue
        if key == "FocalLength35efl" and isinstance(raw, str) and "equivalent" in raw:
            # "4.2 mm (35 mm equivalent: 27.0 mm)" -> take the equivalent
            f35 = _first_float(raw.split("equivalent")[-1])
        else:
            f35 = _first_float(raw)
        if f35 and f35 > 0:
            fov = fov_from_focal_35mm(f35)
            if _valid(fov):
                return fov, f"exif {key}={raw!r}"
    return None


def parse_ffprobe_json(payload: Any) -> Optional[Tuple[float, str]]:
    """Extract (fov_deg, detail) from `ffprobe -print_format json` output.

    Containers rarely carry FOV directly, so this looks for, in order:
      * a stream/format tag literally naming a field of view
      * a 35 mm-equivalent focal length tag
      * a focal length in pixels plus the coded width (some drone/action-cam
        containers write this)
    """
    if not isinstance(payload, dict):
        return None
    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}

    tag_sets: List[Tuple[Dict[str, Any], int]] = []
    for st in streams:
        if not isinstance(st, dict):
            continue
        if st.get("codec_type") not in (None, "video"):
            continue
        width = int(st.get("width") or st.get("coded_width") or 0)
        tag_sets.append((_lower_keys(st.get("tags") or {}), width))
    tag_sets.append((_lower_keys(fmt.get("tags") or {}), 0))

    for tags, width in tag_sets:
        for key, raw in tags.items():
            if "fov" in key or "field_of_view" in key or "fieldofview" in key:
                fov = _first_float(raw)
                if _valid(fov):
                    return float(fov), f"ffprobe tag {key}={raw!r}"

    for tags, width in tag_sets:
        for key, raw in tags.items():
            if "35mm" in key or "35_mm" in key or "focallengthin35" in key:
                f35 = _first_float(raw)
                if f35 and f35 > 0:
                    fov = fov_from_focal_35mm(f35)
                    if _valid(fov):
                        return fov, f"ffprobe tag {key}={raw!r}"

    for tags, width in tag_sets:
        if width <= 0:
            continue
        for key, raw in tags.items():
            if key in ("focal_length_px", "focal_length_pixels", "fx"):
                fpx = _first_float(raw)
                if fpx and fpx > 0:
                    fov = fov_from_focal_px(fpx, width)
                    if _valid(fov):
                        return fov, f"ffprobe tag {key}={raw!r} @ width={width}"
    return None


def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).lower(): v for k, v in d.items()}


def _first_float(value: Any) -> Optional[float]:
    """First float in a string, or the number itself. None if there is none."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    num = ""
    for ch in text:
        if ch.isdigit() or ch == "." or (ch == "-" and not num):
            num += ch
        elif num:
            break
    try:
        return float(num) if num else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# probes (subprocess)
# ---------------------------------------------------------------------------
def _run_json(cmd: List[str], timeout: float = 10.0) -> Optional[Any]:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, *cmd[1:]], capture_output=True, timeout=timeout,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    try:
        return json.loads(out.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


def probe_exiftool(video_path: str) -> Optional[Tuple[float, str]]:
    payload = _run_json(["exiftool", "-j", "-n", str(video_path)])
    return parse_exiftool_json(payload) if payload is not None else None


def probe_ffprobe(video_path: str) -> Optional[Tuple[float, str]]:
    payload = _run_json(["ffprobe", "-v", "quiet", "-print_format", "json",
                         "-show_streams", "-show_format", str(video_path)])
    return parse_ffprobe_json(payload) if payload is not None else None


def read_sidecar(video_path: str) -> Optional[Tuple[float, str]]:
    """`<video>.fov.json` containing {"fov_deg": ...} or {"focal_35mm": ...}."""
    p = Path(str(video_path) + ".fov.json")
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    fov = _first_float(data.get("fov_deg"))
    if _valid(fov):
        return float(fov), f"sidecar {p.name}"
    f35 = _first_float(data.get("focal_35mm"))
    if f35 and f35 > 0:
        fov = fov_from_focal_35mm(f35)
        if _valid(fov):
            return fov, f"sidecar {p.name} focal_35mm={f35}"
    return None


# ---------------------------------------------------------------------------
# the resolver
# ---------------------------------------------------------------------------
def resolve_fov(video_path: Optional[str] = None,
                *,
                explicit_fov_deg: Optional[float] = None,
                focal_35mm: Optional[float] = None,
                default_fov_deg: float = DEFAULT_FOV_DEG,
                use_metadata: bool = True,
                warn: bool = True,
                probes: Optional[List[Any]] = None) -> FovInfo:
    """Resolve the horizontal FOV to use, with provenance.

    Args:
        video_path:       the clip; may be None when only CLI values are given.
        explicit_fov_deg: a user-supplied FOV that is NOT the library default.
                          Pass None to let metadata win.
        focal_35mm:       user-supplied 35 mm-equivalent focal length (--focal-35mm).
        default_fov_deg:  fallback when nothing else is available.
        use_metadata:     set False to skip the sidecar/exiftool/ffprobe probes.
        warn:             print the loud fallback warning.
        probes:           override the probe chain (testing). Each entry is
                          (source_name, callable(video_path) -> (fov, detail) | None).

    Returns:
        FovInfo. Never raises for a missing tool or an unreadable file.
    """
    if _valid(explicit_fov_deg):
        return FovInfo(float(explicit_fov_deg), "cli:fov", f"--fov-deg={explicit_fov_deg}")

    if focal_35mm is not None and float(focal_35mm) > 0:
        fov = fov_from_focal_35mm(float(focal_35mm))
        if _valid(fov):
            return FovInfo(fov, "cli:focal_35mm", f"--focal-35mm={focal_35mm}")

    if use_metadata and video_path:
        chain = probes if probes is not None else [
            ("sidecar", read_sidecar),
            ("metadata:exiftool", probe_exiftool),
            ("metadata:ffprobe", probe_ffprobe),
        ]
        for source, fn in chain:
            try:
                got = fn(video_path)
            except Exception:  # a broken probe must never break a render
                got = None
            if got is None:
                continue
            fov, detail = got
            if _valid(fov):
                return FovInfo(float(fov), source, detail)

    if warn:
        print(f"[WARN] camera FOV not found in metadata for "
              f"{video_path if video_path else '<no video>'} — assuming "
              f"{default_fov_deg:.1f} deg. Every azimuth is scaled by this "
              f"assumption; pass --fov-deg or --focal-35mm if you know the lens. "
              f"See reports/azimuth_kitti_2026-09-04.md for what a wrong FOV costs.")
    return FovInfo(float(default_fov_deg), "default", "hardcoded fallback")


def reproject_azimuth(az_deg, from_fov_deg: float, to_fov_deg: float, ):
    """Re-project an azimuth series measured under one assumed FOV to another.

    Pure geometry on the pixel offset the azimuth implies, so an existing eval
    result can be re-scored under a different FOV assumption with no re-tracking:

        u = tan(az) * f_from      (pixel offset from the principal point)
        az' = atan(u / f_to)

    The image width cancels, so none is needed.
    """
    import numpy as np
    az = np.asarray(az_deg, dtype=np.float64)
    f_from = 1.0 / math.tan(math.radians(from_fov_deg) / 2.0)
    f_to = 1.0 / math.tan(math.radians(to_fov_deg) / 2.0)
    u = np.tan(np.radians(az)) * f_from
    return np.degrees(np.arctan(u / f_to))


__all__ = [
    "FovInfo", "DEFAULT_FOV_DEG", "SENSOR_35MM_WIDTH_MM",
    "fov_from_focal_35mm", "focal_35mm_from_fov",
    "fov_from_focal_px", "focal_px_from_fov",
    "parse_exiftool_json", "parse_ffprobe_json",
    "probe_exiftool", "probe_ffprobe", "read_sidecar",
    "resolve_fov", "reproject_azimuth",
]

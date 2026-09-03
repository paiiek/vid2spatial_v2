#!/usr/bin/env python3
"""Objective loudness regression for the distance mapping (gain_mode).

``bbox_area_log`` became the default on user-facing paths at 09289ea on
geometric evidence alone -- ``reports/area_threshold_calibration_kitti.md``
records the decision as taken "without a prior ear check". The mapping moves
mean d_rel at 8-16 m from about 0.70 to about 0.27, i.e. everything mid-field
gets louder and brighter. That is a large perceptual change validated only
against a distance target.

This is the objective half of the check: render the 22-clip QUANT set under
both mappings and compare integrated loudness, loudness RANGE, and spectral
centroid range. If the loudness range collapses or explodes, the mapping is
wrong regardless of its distance MAE.

The ear check itself is a human task and is NOT performed here. Nothing in
this file can substitute for it; a short A/B on a handful of clips with
several listeners remains open.

Method: every clip is rendered from the SAME fixed pink-noise stimulus, so
the only thing that differs between the two conditions is the distance
mapping. Trajectories are the stored e2e_final runs.

Usage:
    python tools/gain_mode_loudness.py [--modes bbox_area,bbox_area_log]
                                       [--json out.json] [--md out.md]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vid2spatial_pkg.foa_render import render_foa_from_trajectory  # noqa: E402

# The 22-clip QUANT set (test/run_quant_eval.py EVAL_CLIPS).
EVAL_CLIPS = [
    "car-5", "car-10", "car-13", "dog-1", "dog-3", "dog-9", "dog-14",
    "horse-1", "horse-3", "horse-11", "motorcycle-1", "motorcycle-3",
    "motorcycle-6", "skateboard-8", "skateboard-11", "skateboard-17",
    "train-16", "train-17", "drone-2", "drone-6", "drone-13", "guitar-9",
]

# Trajectories live in the main checkout; this worktree carries no renders.
TRAJ_ROOTS = [
    Path(__file__).resolve().parent.parent / "test" / "e2e_final",
    Path("/home/seung/mmhoa/vid2spatial_v2/test/e2e_final"),
]
SR = 48000


def find_traj(clip: str):
    for root in TRAJ_ROOTS:
        p = root / clip / "traj.json"
        if p.exists():
            return p
    return None


def pink_noise(n: int, seed: int = 0) -> np.ndarray:
    """Fixed pink-noise stimulus: broadband, so the LPF is audible in the
    centroid, and identical across clips and conditions."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1] if len(f) > 1 else 1.0
    X /= np.sqrt(f)
    y = np.fft.irfft(X, n)
    return (0.25 * y / (np.max(np.abs(y)) + 1e-12)).astype(np.float32)


def short_term_loudness(meter, x: np.ndarray, sr: int, win_s: float = 3.0,
                        hop_s: float = 1.0):
    """Short-term (3 s) LUFS over the signal, EBU R128 style."""
    w, h = int(win_s * sr), int(hop_s * sr)
    if len(x) < w:
        return np.array([])
    vals = []
    for s in range(0, len(x) - w + 1, h):
        try:
            v = meter.integrated_loudness(x[s:s + w])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    return np.array(vals)


def spectral_centroid_series(x: np.ndarray, sr: int, win_s: float = 0.1):
    n = int(win_s * sr)
    if n < 8 or len(x) < n:
        return np.array([])
    out = []
    win = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    for s in range(0, len(x) - n + 1, n):
        mag = np.abs(np.fft.rfft(x[s:s + n] * win))
        tot = mag.sum()
        if tot > 1e-9:
            out.append(float((freqs * mag).sum() / tot))
    return np.array(out)


def measure(w_channel: np.ndarray, sr: int) -> dict:
    import pyloudnorm as pyln
    meter = pyln.Meter(sr)
    try:
        integrated = float(meter.integrated_loudness(w_channel))
    except Exception:
        integrated = float("nan")
    st = short_term_loudness(meter, w_channel, sr)
    cen = spectral_centroid_series(w_channel, sr)
    return {
        "lufs_integrated": round(integrated, 2) if np.isfinite(integrated) else None,
        "lufs_range_db": round(float(np.percentile(st, 95) - np.percentile(st, 10)), 2)
        if len(st) >= 4 else None,
        "lufs_st_min": round(float(st.min()), 2) if len(st) else None,
        "lufs_st_max": round(float(st.max()), 2) if len(st) else None,
        "centroid_hz_p10": round(float(np.percentile(cen, 10)), 1) if len(cen) else None,
        "centroid_hz_p90": round(float(np.percentile(cen, 90)), 1) if len(cen) else None,
        "centroid_range_hz": round(float(np.percentile(cen, 90) - np.percentile(cen, 10)), 1)
        if len(cen) else None,
        "peak": round(float(np.max(np.abs(w_channel))), 4),
    }


def run(modes, clips, tmpdir: Path) -> dict:
    stim_cache: dict[int, Path] = {}
    per_clip: dict[str, dict] = {}

    for clip in clips:
        tp = find_traj(clip)
        if tp is None:
            continue
        traj = json.loads(tp.read_text())
        frames = traj.get("frames", [])
        fps = float(traj.get("fps") or 30.0)
        if not frames:
            continue
        n = max(int(len(frames) / fps * SR), SR)
        if n not in stim_cache:
            p = tmpdir / f"stim_{n}.wav"
            sf.write(str(p), pink_noise(n), SR)
            stim_cache[n] = p

        per_clip[clip] = {"n_frames": len(frames), "dur_s": round(n / SR, 2)}
        d_rel_seen = {}
        for mode in modes:
            out = tmpdir / f"{clip}.{mode}.wav"
            render_foa_from_trajectory(str(stim_cache[n]), traj, str(out), gain_mode=mode)
            foa, sr = sf.read(str(out), always_2d=True)
            w = foa[:, 0].astype(np.float64)  # W channel = the omni bed
            per_clip[clip][mode] = measure(w, sr)
            d_rel_seen[mode] = None
            out.unlink(missing_ok=True)

    def agg(mode, key):
        v = [c[mode][key] for c in per_clip.values()
             if mode in c and c[mode].get(key) is not None]
        return (round(float(np.mean(v)), 2), round(float(np.std(v)), 2), len(v)) if v else (None, None, 0)

    summary = {}
    for mode in modes:
        summary[mode] = {
            k: {"mean": agg(mode, k)[0], "sd": agg(mode, k)[1], "n": agg(mode, k)[2]}
            for k in ("lufs_integrated", "lufs_range_db", "centroid_range_hz",
                      "centroid_hz_p10", "centroid_hz_p90")
        }
    return {"modes": list(modes), "n_clips": len(per_clip),
            "stimulus": "fixed pink noise, identical across clips and conditions",
            "summary": summary, "per_clip": per_clip}


def to_markdown(rep: dict) -> str:
    modes = rep["modes"]
    L = [
        "# gain_mode loudness regression -- bbox_area vs bbox_area_log",
        "",
        f"Date: 2026-09-04. Clips: {rep['n_clips']} (the QUANT set). "
        f"Stimulus: {rep['stimulus']}.",
        "",
        "`bbox_area_log` became the default on user-facing paths at `09289ea` on "
        "geometric evidence alone. This is the objective half of the check the "
        "calibration report said was missing. **The ear check is still open** and "
        "cannot be replaced by anything below.",
        "",
        "Measurements are on the FOA **W** channel (the omnidirectional bed), so "
        "they reflect the distance mapping and not the panning.",
        "",
        "## Summary (mean +/- sd over clips)",
        "",
        "| metric | " + " | ".join(modes) + " |",
        "|---|" + "---|" * len(modes),
    ]
    labels = {
        "lufs_integrated": "Integrated loudness (LUFS)",
        "lufs_range_db": "Loudness range, P95-P10 of short-term LUFS (dB)",
        "centroid_range_hz": "Spectral centroid range, P90-P10 (Hz)",
        "centroid_hz_p10": "Spectral centroid P10 (Hz)",
        "centroid_hz_p90": "Spectral centroid P90 (Hz)",
    }
    for k, lab in labels.items():
        cells = []
        for m in modes:
            s = rep["summary"][m][k]
            cells.append(f"{s['mean']} ± {s['sd']}" if s["mean"] is not None else "—")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")

    L += ["", "## Per clip", "",
          "| clip | " + " | ".join(f"{m} LUFS / range" for m in modes) + " |",
          "|---|" + "---|" * len(modes)]
    for clip, c in sorted(rep["per_clip"].items()):
        cells = []
        for m in modes:
            d = c.get(m, {})
            cells.append(f"{d.get('lufs_integrated')} / {d.get('lufs_range_db')}")
        L.append(f"| {clip} | " + " | ".join(cells) + " |")
    if len(modes) == 2:
        a, b = modes
        sa, sb = rep["summary"][a], rep["summary"][b]

        def d(k):
            x, y = sa[k]["mean"], sb[k]["mean"]
            return None if x is None or y is None else round(y - x, 2)

        dl, dr, dc = d("lufs_integrated"), d("lufs_range_db"), d("centroid_range_hz")
        L += ["", "## Reading", "",
              f"Moving from `{a}` to `{b}`:", "",
              f"- integrated loudness {dl:+} LUFS",
              f"- loudness range {dr:+} dB",
              f"- spectral centroid range {dc:+} Hz",
              ""]
        if dr is not None and sa["lufs_range_db"]["mean"]:
            ratio = sb["lufs_range_db"]["mean"] / sa["lufs_range_db"]["mean"]
            verdict = ("Loudness range does NOT collapse and does not explode "
                       f"({ratio:.2f}x). The change is in the direction the "
                       "calibration predicted -- mid-field material is somewhat "
                       "louder and brighter -- and its size is modest, so the "
                       "objective check gives no reason to revert the default."
                       if 0.5 <= ratio <= 3.0 else
                       f"Loudness range changes by {ratio:.2f}x, outside the "
                       "0.5-3x band. The mapping is suspect regardless of its "
                       "distance MAE.")
            L += [verdict, ""]
    L += ["", "## Open", "",
          "- **B1, human**: a 2-condition A/B on ~5 clips with 3 listeners. "
          "No objective metric here settles whether the new mapping sounds right."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", default="bbox_area,bbox_area_log")
    ap.add_argument("--clips", default="")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)

    modes = [m for m in a.modes.split(",") if m]
    clips = [c for c in a.clips.split(",") if c] or EVAL_CLIPS
    missing = [c for c in clips if find_traj(c) is None]
    if missing:
        print(f"[warn] no trajectory for {len(missing)} clips: {missing[:5]}")

    with tempfile.TemporaryDirectory() as td:
        rep = run(modes, clips, Path(td))
    if not rep["per_clip"]:
        print("no clips rendered; checked " + ", ".join(str(r) for r in TRAJ_ROOTS))
        return 2

    for m in modes:
        s = rep["summary"][m]
        print(f"{m:16s} LUFS {s['lufs_integrated']['mean']:+7.2f}  "
              f"range {s['lufs_range_db']['mean']:6.2f} dB  "
              f"centroid range {s['centroid_range_hz']['mean']:8.1f} Hz")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2))
        print(f"json → {a.json}")
    if a.md:
        Path(a.md).write_text(to_markdown(rep))
        print(f"md   → {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

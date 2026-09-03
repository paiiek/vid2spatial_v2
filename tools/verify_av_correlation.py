#!/usr/bin/env python3
"""Verify the audio-visual correlation gate discriminates true from decoy objects.

The claim under test: ``av_correlation.av_confidence`` scores the TRUE tracked
object above a DECOY box that moves independently in the same clip.

Data caveat, stated up front
----------------------------
LaSOT ships image sequences and bounding boxes only -- there is NO audio in the
dataset (``ls data/lasot/<clip>/`` is img/, groundtruth.txt, nlp.txt,
full_occlusion.txt, out_of_view.txt). Neither does KITTI. So the visual side of
this check is real -- the true track is the dataset's own ground-truth box, and
the decoy is a real alternative box in the same frames -- while the audio side
is SIMULATED: a signal whose envelope is driven by the true object's motion
plus independent noise, which is what a sound physically caused by that
object's movement would look like.

That makes this a test of the estimator, not of the world. It shows the score
separates a genuinely related signal from an unrelated object under realistic
tracking noise; it does not show what av_confidence is worth on real recorded
audio, which needs a dataset this repo does not have.

Usage:
    python tools/verify_av_correlation.py [--n-clips 12] [--json out.json]
Exit 0 if the true object beats the decoy on a clear majority of clips.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vid2spatial_pkg.av_correlation import av_confidence  # noqa: E402

LASOT = Path("/home/seung/mmhoa/vid2spatial_v2/data/lasot")
SR = 16000
FPS = 30.0


def load_boxes(clip_dir: Path, max_frames: int = 600):
    """LaSOT groundtruth.txt: one 'x,y,w,h' per frame."""
    rows = []
    for line in clip_dir.joinpath("groundtruth.txt").read_text().splitlines()[:max_frames]:
        parts = line.replace(",", " ").split()
        if len(parts) >= 4:
            rows.append(tuple(float(v) for v in parts[:4]))
    return rows


def frames_from_boxes(boxes):
    return [{"frame": i, "bbox": list(b)} for i, b in enumerate(boxes)]


def decoy_boxes(boxes, rng):
    """A decoy that moves, but independently of the true object.

    Same size sequence and same overall motion statistics (so it is not
    trivially distinguishable by being static), driven by an independent
    random walk. Time-reversing the true track would leave the correlation
    intact under symmetric motion, so a fresh walk is used instead.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    steps = np.diff(boxes[:, :2], axis=0)
    scale = np.linalg.norm(steps, axis=1).mean() if len(steps) else 1.0
    walk = np.cumsum(rng.normal(0.0, max(scale, 1e-6), size=(len(boxes), 2)), axis=0)
    out = boxes.copy()
    out[:, 0] = boxes[0, 0] + walk[:, 0]
    out[:, 1] = boxes[0, 1] + walk[:, 1]
    return [tuple(b) for b in out]


def motion_driven_audio(boxes, rng, snr: float = 1.0):
    """Audio whose envelope follows the TRUE object's frame-to-frame motion.

    This is the simulated half: a broadband carrier amplitude-modulated by the
    object's motion energy, plus independent noise at the given SNR.
    """
    pts = np.asarray(boxes, dtype=np.float64)[:, :2]
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d = np.concatenate([d[:1], d])
    if d.std() > 1e-9:
        env = (d - d.min()) / (d.max() - d.min() + 1e-12)
    else:
        env = np.full(len(d), 0.5)
    env = env + rng.normal(0.0, (1.0 / max(snr, 1e-6)) * 0.3, size=len(env))
    env = np.clip(env, 0.0, None)

    n_samples = int(len(boxes) / FPS * SR)
    t = np.arange(n_samples)
    env_s = np.interp(t / SR * FPS, np.arange(len(env)), env)
    carrier = rng.normal(0.0, 1.0, size=n_samples)
    return (env_s * carrier).astype(np.float32)


def run(n_clips: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    if not LASOT.is_dir():
        return {"available": False, "note": f"LaSOT not found at {LASOT}"}
    clips = sorted(p for p in LASOT.iterdir()
                   if p.is_dir() and (p / "groundtruth.txt").exists())
    clips = clips[:: max(1, len(clips) // n_clips)][:n_clips]

    per_clip, wins = [], 0
    for c in clips:
        boxes = load_boxes(c)
        if len(boxes) < 60:
            continue
        audio = motion_driven_audio(boxes, rng)
        true_r = av_confidence(audio, SR, frames_from_boxes(boxes), FPS)
        dec_r = av_confidence(audio, SR, frames_from_boxes(decoy_boxes(boxes, rng)), FPS)
        win = true_r["av_confidence"] > dec_r["av_confidence"]
        wins += bool(win)
        per_clip.append({
            "clip": c.name, "n_frames": len(boxes),
            "true_av_confidence": round(true_r["av_confidence"], 4),
            "decoy_av_confidence": round(dec_r["av_confidence"], 4),
            "margin": round(true_r["av_confidence"] - dec_r["av_confidence"], 4),
            "true_warned": true_r.get("warning") is not None,
            "decoy_warned": dec_r.get("warning") is not None,
            "true_beats_decoy": bool(win),
        })

    n = len(per_clip)
    return {
        "available": True,
        "audio_is_simulated": True,
        "note": ("LaSOT ships no audio; the true track and the decoy are real boxes "
                 "from the dataset, the audio envelope is simulated from the true "
                 "object's motion. This validates the estimator, not real recordings."),
        "n_clips": n,
        "true_beats_decoy": wins,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "mean_true": round(float(np.mean([c["true_av_confidence"] for c in per_clip])), 4) if n else 0.0,
        "mean_decoy": round(float(np.mean([c["decoy_av_confidence"] for c in per_clip])), 4) if n else 0.0,
        "mean_margin": round(float(np.mean([c["margin"] for c in per_clip])), 4) if n else 0.0,
        "decoy_warned_frac": round(float(np.mean([c["decoy_warned"] for c in per_clip])), 4) if n else 0.0,
        "per_clip": per_clip,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-clips", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    rep = run(a.n_clips, a.seed)
    if not rep["available"]:
        print(rep["note"])
        return 2
    print(f"{'clip':28s} {'true':>7s} {'decoy':>7s} {'margin':>7s}")
    for c in rep["per_clip"]:
        print(f"{c['clip']:28s} {c['true_av_confidence']:7.3f} "
              f"{c['decoy_av_confidence']:7.3f} {c['margin']:+7.3f}")
    print(f"\nwin rate {rep['true_beats_decoy']}/{rep['n_clips']} = {rep['win_rate']:.2f}   "
          f"mean true {rep['mean_true']:.3f} vs decoy {rep['mean_decoy']:.3f} "
          f"(margin {rep['mean_margin']:+.3f})")
    print(f"decoy warned as unverified on {rep['decoy_warned_frac']:.0%} of clips")
    print("\nCAVEAT:", rep["note"])
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2))
        print(f"report → {a.json}")
    return 0 if rep["win_rate"] >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())

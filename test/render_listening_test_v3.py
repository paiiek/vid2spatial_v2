#!/usr/bin/env python3
"""
render_listening_test_v3.py
===========================
Vid2Spatial v3 청취평가 stimuli 렌더 (ISMAR 2026)

4 conditions × 10 clips + 1 practice clip:
  - mono      : dual-mono anchor (공개)
  - baseline  : stereo pan (sin law) + bbox_area gain/LPF
  - nodepth   : HRTF binaural + flat gain (no distance modulation)
  - proposed  : HRTF binaural + bbox_area gain/LPF  ← full system

Output: test/listening_test_v3/stimuli/{S01_car-10,...}/
  proposed.wav  baseline.wav  nodepth.wav  mono.wav  video.mp4

Usage:
    cd /home/seung/mmhoa/vid2spatial_v2
    python test/render_listening_test_v3.py [--clips all] [--dry_run]
"""

import argparse, json, math, os, sys, shutil
from pathlib import Path

import numpy as np

TEST_DIR  = Path(__file__).parent
V2_ROOT   = TEST_DIR.parent
TRAJ_ROOT = TEST_DIR / "e2e_final"          # yw_sam2 threshold=0.99 traj
OUT_ROOT  = TEST_DIR / "listening_test_v3" / "stimuli"
SOFA_PATH = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"
FFMPEG    = "/home/seung/miniforge3/bin/ffmpeg"
SPATAMB_BASE = Path("/home/seung/spatamb/Dataset")

sys.path.insert(0, str(V2_ROOT))

def spatamb_sep(folder: str, filename: str) -> str:
    """Return full path to a specific AuSep file in a spatamb folder."""
    path = str(SPATAMB_BASE / folder / filename)
    if not Path(path).exists():
        raise FileNotFoundError(f"AuSep not found: {path}")
    return path

# ── 10 eval clips + 1 practice ────────────────────────────────────────────────
# audio_src: single-instrument AuSep mono wav
# audio_offset: best 12s active window (auto-computed, no silent stretches)
EVAL_CLIPS = {
    "S01_car-10":        {"clip": "car-10",        "cat": "vehicle",    "audio_src": spatamb_sep("10_March_tpt_sax",              "AuSep_1_tpt_10_March.wav"),        "audio_offset": 48.5},
    "S02_dog-1":         {"clip": "dog-1",          "cat": "animal",     "audio_src": spatamb_sep("08_Spring_fl_vn",               "AuSep_2_vn_08_Spring.wav"),        "audio_offset": 11.0},
    "S03_motorcycle-6":  {"clip": "motorcycle-6",   "cat": "vehicle",    "audio_src": spatamb_sep("07_GString_tpt_tbn",            "AuSep_2_tbn_07_GString.wav"),      "audio_offset": 236.5},
    "S04_skateboard-8":  {"clip": "skateboard-8",   "cat": "sports",     "audio_src": spatamb_sep("24_Pirates_vn_vn_va_vc",       "AuSep_1_vn_24_Pirates.wav"),       "audio_offset": 24.0},
    "S05_skateboard-11": {"clip": "skateboard-11",  "cat": "sports",     "audio_src": spatamb_sep("24_Pirates_vn_vn_va_vc",       "AuSep_3_va_24_Pirates.wav"),       "audio_offset": 36.5},
    "S06_horse-1":       {"clip": "horse-1",        "cat": "animal",     "audio_src": spatamb_sep("36_Rondeau_vn_vn_va_vc",       "AuSep_1_vn_36_Rondeau.wav"),       "audio_offset": 28.5},
    "S07_dog-14":        {"clip": "dog-14",         "cat": "animal",     "audio_src": spatamb_sep("14_Waltz_fl_fl_cl",             "AuSep_1_fl_14_Waltz.wav"),         "audio_offset": 78.0},
    "S08_drone-13":      {"clip": "drone-13",       "cat": "drone",      "audio_src": spatamb_sep("04_Allegro_fl_fl",              "AuSep_2_fl_04_Allegro.wav"),       "audio_offset": 132.0},
    "S09_train-16":      {"clip": "train-16",       "cat": "vehicle",    "audio_src": spatamb_sep("43_Chorale_tpt_tpt_hn_tbn_tba","AuSep_1_tpt_43_Chorale.wav"),      "audio_offset": 27.0},
    "S10_guitar-9":      {"clip": "guitar-9",       "cat": "instrument", "audio_src": spatamb_sep("17_Nocturne_vn_fl_cl",          "AuSep_1_vn_17_Nocturne.wav"),      "audio_offset": 39.0},
}

PRACTICE_CLIP = {
    "practice": {"clip": "car-5", "cat": "vehicle", "audio_src": spatamb_sep("06_Entertainer_sax_sax", "AuSep_1_sax_06_Entertainer.wav"), "audio_offset": 40.5}
}

TARGET_RMS_DBFS = -22.0   # ITU-R BS.1770-4
SR = 48000


# ── Audio utilities ────────────────────────────────────────────────────────────
def load_source_audio(offset_sec: float, duration_sec: float, audio_src: str = None) -> np.ndarray:
    """Load mono float32 chunk of duration_sec starting at offset_sec from audio_src."""
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    path = audio_src
    src, sr_src = sf.read(str(path), dtype="float32")
    if src.ndim == 2:
        src = src.mean(axis=1)
    if sr_src != SR:
        g = gcd(SR, sr_src)
        src = resample_poly(src, SR // g, sr_src // g)
    start = int(offset_sec * SR)
    n_need = int(duration_sec * SR) + SR  # +1s buffer
    chunk = src[start: start + n_need]
    while len(chunk) < n_need:
        chunk = np.concatenate([chunk, src])
    return chunk[:n_need].astype(np.float32)


def rms_normalize(audio: np.ndarray, target_dbfs: float = -22.0) -> np.ndarray:
    """Normalize to target RMS dBFS."""
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    target_rms = 10 ** (target_dbfs / 20.0)
    return audio * (target_rms / rms)


def to_dual_mono(mono: np.ndarray) -> np.ndarray:
    """[T] → [T, 2] dual-mono stereo."""
    return np.stack([mono, mono], axis=1)


def write_wav(path: Path, audio: np.ndarray, sr: int = SR):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def save_tmp_mono(mono: np.ndarray, tmp_path: Path):
    import soundfile as sf
    sf.write(str(tmp_path), mono, SR, subtype="FLOAT")


# ── Per-condition renderers ────────────────────────────────────────────────────
def render_mono(mono: np.ndarray) -> np.ndarray:
    """Dual-mono anchor."""
    return rms_normalize(to_dual_mono(mono))


def render_baseline(mono: np.ndarray, trajectory: dict, tmp_wav: Path) -> np.ndarray:
    """Stereo pan (sin law) + bbox_area gain/LPF."""
    from vid2spatial_pkg.foa_render import _load_and_prepare, render_stereo_pan_baseline
    save_tmp_mono(mono, tmp_wav)
    audio_proc, sr, az_s, el_s, dist_s, d_rel_s, _ = _load_and_prepare(
        str(tmp_wav), trajectory,
        smooth_ms=50.0,
        dist_gain_k=2.5, lpf_min=2000.0, lpf_max=8000.0,
        gain_mode="bbox_area_log",
    )
    stereo = render_stereo_pan_baseline(audio_proc, sr, az_s, dist_s, d_rel_s, apply_gain=False)
    # [2, T] → [T, 2]
    out = rms_normalize(stereo.T)
    return out


def render_nodepth(mono: np.ndarray, trajectory: dict, tmp_wav: Path,
                   out_wav: Path) -> np.ndarray:
    """HRTF binaural + flat gain (no distance modulation). C-NODEP.
    Bypasses apply_distance_gain_lpf entirely: raw audio → HRTF only.
    """
    from vid2spatial_pkg.foa_render import (
        interpolate_angles_distance, smooth_limit_angles, direct_binaural_sofa,
    )
    frames = trajectory.get("frames", [])
    fps    = float(trajectory.get("fps", 25.0))
    T      = len(mono)

    az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(
        frames, T, SR, fps=fps, gain_mode="bbox_area_log",
    )
    az_s, el_s = smooth_limit_angles(az_s, el_s, SR, smooth_ms=50.0)

    # Negate az: pipeline az>0=RIGHT but SOFA az>0=LEFT
    az_sofa = -az_s

    # Direct HRTF — no gain/LPF applied (flat distance response)
    stereo = direct_binaural_sofa(mono, SR, az_sofa, el_s, SOFA_PATH, block_ms=10.0)
    # [2, T] → [T, 2]
    out = rms_normalize(stereo.T)
    return out


def render_proposed(mono: np.ndarray, trajectory: dict, tmp_wav: Path,
                    out_wav: Path) -> np.ndarray:
    """HRTF binaural + bbox_area gain/LPF. C-PROP (full system)."""
    import soundfile as sf
    from vid2spatial_pkg.foa_render import render_binaural_from_trajectory
    save_tmp_mono(mono, tmp_wav)
    render_binaural_from_trajectory(
        audio_path=str(tmp_wav),
        trajectory=trajectory,
        output_path=str(out_wav),
        sofa_path=SOFA_PATH,
        smooth_ms=50.0,
        dist_gain_k=2.5,
        dist_lpf_min_hz=2000.0,
        dist_lpf_max_hz=8000.0,
        apply_reverb=False,
        block_ms=10.0,
        gain_mode="bbox_area_log",
    )
    audio, _ = sf.read(str(out_wav), dtype="float32")
    return rms_normalize(audio)


# ── Video: muted overlay ───────────────────────────────────────────────────────
def make_muted_video(clip: str, out_dir: Path):
    """Copy e2e_final overlay video, strip audio track."""
    src_mp4 = TRAJ_ROOT / clip / f"{clip}_e2e.mp4"
    dst_mp4 = out_dir / "video.mp4"
    if not src_mp4.exists():
        print(f"  [video] WARNING: {src_mp4} not found")
        return
    cmd = (f"{FFMPEG} -y -i {src_mp4} -an -c:v copy {dst_mp4} -loglevel warning")
    ret = os.system(cmd)
    if ret != 0:
        # fallback: copy as-is
        shutil.copy2(src_mp4, dst_mp4)


# ── Per-clip render ────────────────────────────────────────────────────────────
def render_clip(sid: str, info: dict, dry_run: bool = False):
    clip = info["clip"]
    out_dir = OUT_ROOT / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path = TRAJ_ROOT / clip / "traj.json"
    if not traj_path.exists():
        print(f"  [SKIP] traj not found: {traj_path}")
        return False

    with open(traj_path) as f:
        trajectory = json.load(f)

    frames = trajectory.get("frames", [])
    fps    = float(trajectory.get("fps", 25.0))
    dur    = len(frames) / fps

    if dry_run:
        print(f"  [dry_run] {sid}: dur={dur:.1f}s, frames={len(frames)}")
        return True

    # Load source audio (clip-specific)
    mono = load_source_audio(info["audio_offset"], dur, audio_src=info["audio_src"])
    mono = mono[:int(dur * SR)]   # trim to exact duration

    tmp_wav  = out_dir / "_tmp_mono.wav"
    tmp_out  = out_dir / "_tmp_out.wav"

    try:
        # ── Mono ──────────────────────────────────────────────────────────────
        mono_out = render_mono(mono)
        write_wav(out_dir / "mono.wav", mono_out)
        print(f"  [mono]     ✓")

        # ── Baseline ──────────────────────────────────────────────────────────
        baseline_out = render_baseline(mono, trajectory, tmp_wav)
        write_wav(out_dir / "baseline.wav", baseline_out)
        print(f"  [baseline] ✓")

        # ── No-depth ──────────────────────────────────────────────────────────
        nodepth_out = render_nodepth(mono, trajectory, tmp_wav, tmp_out)
        write_wav(out_dir / "nodepth.wav", nodepth_out)
        print(f"  [nodepth]  ✓")

        # ── Proposed ──────────────────────────────────────────────────────────
        proposed_out = render_proposed(mono, trajectory, tmp_wav, tmp_out)
        write_wav(out_dir / "proposed.wav", proposed_out)
        print(f"  [proposed] ✓")

        # ── Video (muted) ─────────────────────────────────────────────────────
        make_muted_video(clip, out_dir)
        print(f"  [video]    ✓")

    finally:
        for p in [tmp_wav, tmp_out]:
            if p.exists():
                p.unlink()

    return True


# ── Config JSON ───────────────────────────────────────────────────────────────
def write_config(clips_done: list):
    all_clips = EVAL_CLIPS
    clip_entries = []
    for sid, info in all_clips.items():
        if sid not in clips_done:
            continue
        clip_entries.append({
            "id": sid,
            "name": info["clip"],
            "cat": info["cat"],
        })

    config = {
        "version": "v3_ismar2026",
        "date": "2026-03-04",
        "n_clips": len(clip_entries),
        "n_conditions": 4,
        "n_questions": 4,
        "target_rms_dbfs": TARGET_RMS_DBFS,
        "duration_sec": 12.0,
        "scale": "5-point MOS (1=Bad, 5=Excellent)",
        "clips": clip_entries,
        "conditions": [
            {"id": "proposed", "file": "proposed.wav",
             "blind_label": None,
             "description": "Proposed: KEMAR HRTF binaural + bbox_area gain/LPF"},
            {"id": "baseline", "file": "baseline.wav",
             "blind_label": None,
             "description": "Baseline: Stereo pan (sin law) + bbox_area gain/LPF"},
            {"id": "nodepth",  "file": "nodepth.wav",
             "blind_label": None,
             "description": "Proposed w/o Depth: KEMAR HRTF binaural, flat gain"},
            {"id": "mono",     "file": "mono.wav",
             "blind_label": "Mono Reference",
             "description": "Mono anchor (dual-mono, no spatialization)"},
        ],
        "questions": [
            {"id": "Q1", "dimension": "Spatial Alignment",
             "text": "How well does the sound position match the on-screen object?",
             "anchors": {"1": "No match", "5": "Perfect match"}},
            {"id": "Q2", "dimension": "Motion Smoothness",
             "text": "How smooth and artifact-free is the spatial audio movement?",
             "anchors": {"1": "Severe artifacts", "5": "Completely smooth"}},
            {"id": "Q3", "dimension": "Depth Perception",
             "text": "How convincing is the sense of distance (near/far)?",
             "anchors": {"1": "No depth", "5": "Very convincing"}},
            {"id": "Q4", "dimension": "Overall Quality",
             "text": "Overall, how would you rate the spatial audio quality?",
             "anchors": {"1": "Bad", "5": "Excellent"}},
        ],
    }
    cfg_path = OUT_ROOT / "config.json"
    cfg_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"[config] → {cfg_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", nargs="*", default=["all"])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    all_clips = {**PRACTICE_CLIP, **EVAL_CLIPS}
    if "all" in args.clips:
        targets = list(all_clips.keys())
    else:
        targets = args.clips

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Listening Test v3 Stimuli Render")
    print(f"  clips={len(targets)}, conditions=4, out={OUT_ROOT}")
    print(f"  dry_run={args.dry_run}")

    done = []
    for sid in targets:
        info = all_clips.get(sid)
        if not info:
            print(f"[skip] unknown sid: {sid}")
            continue
        print(f"\n{'='*50}")
        print(f"Clip: {sid}  [{info['clip']}]")
        ok = render_clip(sid, info, dry_run=args.dry_run)
        if ok:
            done.append(sid)

    if not args.dry_run:
        write_config([s for s in done if s in EVAL_CLIPS])

    print(f"\n{'='*50}")
    print(f"Done: {len(done)}/{len(targets)}")
    print(f"Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()

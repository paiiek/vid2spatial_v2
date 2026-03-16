#!/usr/bin/env python3
"""
거리 감쇠 파라미터 비교 렌더러
usage:
    python compare_distance.py --job 0b0f3687 --audio /path/to/mono.wav
    python compare_distance.py --traj /path/to/traj.json --audio /path/to/mono.wav
    python compare_distance.py --job 0b0f3687  # spatamb 기본 오디오 사용
"""
import sys, json, argparse
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
V2_ROOT  = DEMO_DIR.parent.parent
sys.path.insert(0, str(V2_ROOT))

SOFA_PATH = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"
DEFAULT_AUDIO = "/home/seung/spatamb/Dataset/08_Spring_fl_vn/AuSep_2_vn_08_Spring.wav"

# ── 비교할 조합들 ──────────────────────────────────────────────────────────────
# (name, gain_mode, gain_k, gain_min, lpf_min, lpf_max, pre_delay_ms, air_abs_slope)
#   gain_mode: "bbox_area" | "hybrid" | "depth_rel"
#   gain_k:    ISL exponent (1=standard ISL, 1.5=더 급격)
#   gain_min:  최소 gain (0.3=현재, 0.05=극적)
#   lpf_min/max: 고주파 차단 범위 Hz
#   pre_delay_ms: 거리 비례 최대 pre-delay ms
#   air_abs_slope: spectral tilt dB/octave at d_rel=1 (0=없음, 6=강함)
VARIANTS = [
    # ── 0: 레퍼런스 모노 (HRTF 없음, gain 없음) ──
    ("00_mono_ref",   "bbox_area", 1.0, 1.0,  8000, 8000,  0,  0),

    # ── gain_mode 비교 (파라미터 고정, mode만 바꿈) ──
    ("01_bbox_full",  "bbox_area", 1.5, 0.05, 1200, 6000,  0,  6),
    ("02_hybrid_full","hybrid",    1.5, 0.05, 1200, 6000,  0,  6),
    ("03_depth_full", "depth_rel", 1.5, 0.05, 1200, 6000,  0,  6),

    # ── bbox_area 기반 파라미터 비교 ──
    ("04_bbox_baseline", "bbox_area", 1.0, 0.30, 2000, 8000,  0,  0),
    ("05_bbox_lowgain",  "bbox_area", 1.0, 0.05, 2000, 8000,  0,  0),
    ("06_bbox_isl",      "bbox_area", 1.5, 0.05, 2000, 8000,  0,  0),
    ("07_bbox_isl_air",  "bbox_area", 1.5, 0.05, 1200, 6000,  0,  6),
    ("08_bbox_airabs",   "bbox_area", 1.0, 0.10, 1200, 6000,  0,  6),
    ("09_bbox_extreme",  "bbox_area", 2.0, 0.03,  800, 5000,  0,  9),
]


def apply_pre_delay(x, sr, d_rel_s, max_delay_ms=40.0):
    """거리 비례 pre-delay. 선형 보간으로 위상 연속성 보장 (클릭 없음).
    d_rel=0→0ms, d_rel=1→max_delay_ms.
    """
    import numpy as np
    max_samp = float(sr * max_delay_ms / 1000.0)
    if max_samp < 1:
        return x
    T = len(x)
    # d_rel을 먼저 300ms 스무딩 (delay 변화를 매우 완만하게)
    from scipy.ndimage import uniform_filter1d
    d = np.clip(d_rel_s[:T].astype(np.float64), 0.0, 1.0)
    smooth_n = max(1, int(sr * 0.3))
    d = uniform_filter1d(d, size=smooth_n, mode='nearest')

    delay_samp = d * max_samp   # 각 출력 샘플의 delay (실수)
    src_pos = np.arange(T, dtype=np.float64) - delay_samp  # 읽어올 소스 위치

    # 선형 보간 (fractional delay)
    idx0 = np.floor(src_pos).astype(np.int64)
    idx1 = idx0 + 1
    frac = (src_pos - idx0).astype(np.float32)

    xp = np.concatenate([[0.0], x.astype(np.float64), [0.0]])  # pad
    offset = 1  # xp[i+offset] = x[i]

    i0 = np.clip(idx0 + offset, 0, len(xp) - 1)
    i1 = np.clip(idx1 + offset, 0, len(xp) - 1)
    out = (1.0 - frac) * xp[i0] + frac * xp[i1]
    return out.astype(np.float32)


def apply_air_absorption(x, sr, d_rel_s, slope_db_per_oct=6.0):
    """거리별 spectral tilt: 멀수록 고주파 감쇠.
    time-varying one-pole LPF (sample-wise) → 블록 경계 클릭 없음.
    """
    import numpy as np
    import math

    if slope_db_per_oct == 0:
        return x

    T = len(x)
    from scipy.ndimage import uniform_filter1d
    d = np.clip(d_rel_s[:T].astype(np.float32), 0.0, 1.0)
    # d_rel 300ms 스무딩
    smooth_n = max(1, int(sr * 0.3))
    d = uniform_filter1d(d, size=smooth_n, mode='nearest').astype(np.float32)

    # d_rel → fc: near(d=0)→8000Hz, far(d=1)→1200Hz
    fc = np.clip(8000.0 * (1.0 - d * 0.85), 1200.0, sr * 0.45).astype(np.float32)
    # fc 추가 스무딩 (500ms)
    fc = uniform_filter1d(fc, size=max(1, int(sr * 0.5)), mode='nearest').astype(np.float32)

    # blend: slope_db_per_oct에 비례 (6→full, 0→dry)
    blend = np.clip(d * slope_db_per_oct / 6.0, 0.0, 1.0).astype(np.float32)

    # sample-wise one-pole LPF (y[n] = y[n-1] + a*(x[n]-y[n-1]))
    two_pi = 2.0 * math.pi
    xf = x.astype(np.float32)
    y_lp = np.zeros(T, dtype=np.float32)
    prev = 0.0
    for i in range(T):
        a = (two_pi * fc[i]) / (two_pi * fc[i] + sr)
        prev = prev + a * (xf[i] - prev)
        y_lp[i] = prev

    return ((1.0 - blend) * xf + blend * y_lp).astype(np.float32)


def render_variant(traj, audio_path, out_path, gain_mode, gain_k, gain_min,
                   lpf_min, lpf_max, pre_delay_ms, air_abs_slope):
    import numpy as np
    import soundfile as sf
    from vid2spatial_pkg.foa_render import (
        interpolate_angles_distance, smooth_limit_angles,
        apply_distance_gain_lpf, direct_binaural_sofa,
    )

    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    T = audio.shape[0]

    frames = traj.get("frames", [])
    fps    = float(traj.get("fps", 30.0))
    intr   = traj.get("intrinsics", {})
    img_w  = int(intr.get("width", 1280))
    img_h  = int(intr.get("height", 720))

    az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(
        frames, T, sr, fps=fps,
        gain_mode=gain_mode, img_w=img_w, img_h=img_h,
    )
    az_s, el_s = smooth_limit_angles(az_s, el_s, sr, smooth_ms=80.0)

    # d_rel 추가 스무딩: 음량/LPF 변화가 훅훅 튀지 않도록 (500ms)
    from scipy.ndimage import uniform_filter1d
    smooth_drel = max(1, int(sr * 0.5))
    d_rel_s = uniform_filter1d(d_rel_s, size=smooth_drel, mode='nearest').astype(d_rel_s.dtype)

    # ── 00_mono_ref: HRTF 없이 그냥 모노 원본 저장 ──
    if gain_min >= 1.0 and gain_k == 1.0 and pre_delay_ms == 0 and air_abs_slope == 0:
        mono = audio[:T].astype(np.float32)
        peak = float(np.max(np.abs(mono)) + 1e-9)
        if peak > 1.0:
            mono = mono / (peak * 1.01)
        sf.write(str(out_path), mono, sr, subtype='PCM_24')
        print(f"  → {out_path.name}  [mono ref]")
        return

    # 1. gain + LPF (nonlinear ISL)
    audio_proc = apply_distance_gain_lpf(
        audio, sr, dist_s, d_rel_s,
        gain_k=gain_k, gain_min=gain_min,
        lpf_min_hz=lpf_min, lpf_max_hz=lpf_max,
    )

    # 2. pre-delay
    if pre_delay_ms > 0:
        audio_proc = apply_pre_delay(audio_proc, sr, d_rel_s, pre_delay_ms)

    # 3. air absorption
    if air_abs_slope > 0:
        audio_proc = apply_air_absorption(audio_proc, sr, d_rel_s, air_abs_slope)

    # 4. HRTF binaural
    az_ambiX = -az_s
    binaural = direct_binaural_sofa(audio_proc, sr, az_ambiX, el_s, SOFA_PATH)

    peak = float(np.max(np.abs(binaural)) + 1e-9)
    if peak > 1.0:
        binaural = binaural / (peak * 1.01)

    sf.write(str(out_path), binaural.T, sr, subtype='PCM_24')
    print(f"  → {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--job", help="demo job id (e.g. 0b0f3687)")
    grp.add_argument("--traj", help="직접 traj.json 경로 지정")
    parser.add_argument("--audio", default=DEFAULT_AUDIO)
    parser.add_argument("--variants", nargs="+", help="특정 variant만 (e.g. 01_bbox_full 02_hybrid_full)")
    parser.add_argument("--out", help="출력 디렉토리 (기본: traj 옆 compare/)")
    args = parser.parse_args()

    if args.traj:
        traj_path = Path(args.traj)
        out_dir   = traj_path.parent / "compare"
        label     = traj_path.parent.name
    else:
        job_dir   = DEMO_DIR / "jobs" / args.job
        traj_path = job_dir / "traj.json"
        out_dir   = job_dir / "compare"
        label     = args.job

    if not traj_path.exists():
        print(f"ERROR: traj.json not found: {traj_path}"); sys.exit(1)

    if args.out:
        out_dir = Path(args.out)

    audio_path = args.audio
    if not Path(audio_path).exists():
        print(f"ERROR: audio not found: {audio_path}"); sys.exit(1)

    with open(traj_path) as f:
        traj = json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    variants = VARIANTS
    if args.variants:
        variants = [v for v in VARIANTS if v[0] in args.variants]

    # traj 통계 출력
    import math
    frames = traj.get("frames", [])
    az_deg = [math.degrees(f["az"]) for f in frames]
    intr   = traj.get("intrinsics", {})
    img_area = intr.get("width", 1280) * intr.get("height", 720)
    areas  = [f.get("w", 80) * f.get("h", 80) / img_area * 100 for f in frames]
    print(f"Traj: {label}  |  {len(frames)} frames")
    print(f"  az:   {min(az_deg):.1f}° ~ {max(az_deg):.1f}°  (range {max(az_deg)-min(az_deg):.1f}°)")
    print(f"  bbox: {min(areas):.2f}% ~ {max(areas):.2f}%  (frame area)")
    print(f"Output: {out_dir}\n")

    for name, gain_mode, gain_k, gain_min, lpf_min, lpf_max, pre_delay, air_abs in variants:
        print(f"[{name}] mode={gain_mode} gain_k={gain_k} gain_min={gain_min} "
              f"lpf={lpf_min}-{lpf_max}Hz predelay={pre_delay}ms air={air_abs}dB/oct")
        out_path = out_dir / f"{name}.wav"
        try:
            render_variant(traj, audio_path, out_path,
                           gain_mode, gain_k, gain_min, lpf_min, lpf_max, pre_delay, air_abs)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\n완료. {out_dir} 에서 확인하세요.")
    print("파일 목록:")
    for f in sorted(out_dir.glob("*.wav")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb}KB)")


if __name__ == "__main__":
    main()

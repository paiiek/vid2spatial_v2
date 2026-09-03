import math
import numpy as np
import soundfile as sf
from typing import List, Dict, Tuple
from typing import Optional


SQ2 = math.sqrt(2.0)
SQ3_2 = math.sqrt(3.0 / 2.0)  # SN3D scaling for first order


def dir_to_foa_acn_sn3d_gains(az: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Compute FOA gains in AmbiX (ACN/SN3D) channel order [W, Y, Z, X].

    x = cos(az)cos(el), y = sin(az)cos(el), z = sin(el)
    W = 1/sqrt(2)
    X = sqrt(3/2)*x, Y = sqrt(3/2)*y, Z = sqrt(3/2)*z
    Returned gains: [W, Y, Z, X] with shape [4, T]
    """
    x = np.cos(az) * np.cos(el)
    y = np.sin(az) * np.cos(el)
    z = np.sin(el)
    W = np.full_like(x, 1.0 / SQ2)
    X = SQ3_2 * x
    Y = SQ3_2 * y
    Z = SQ3_2 * z
    return np.stack([W, Y, Z, X], axis=0).astype(np.float32)


def interpolate_angles(frames: List[Dict], T: int, sr: int,
                       fps: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of (az, el) over audio sample grid [0..T-1].

    Maps frame indices to audio samples using fps so that the trajectory
    covers [frame_start, frame_start + T/sr*fps] frames — i.e., only the
    portion of the video that corresponds to the audio duration is used.

    Without fps-aware mapping, a long video clip (e.g. 157s) would have its
    full trajectory compressed into a 10s audio window, causing ~15× faster
    apparent motion and audible 'swishing' artifacts.
    """
    if not frames:
        raise ValueError("Empty frames for interpolation")
    idx = np.array([f["frame"] for f in frames], dtype=np.float32)
    az = np.array([f["az"] for f in frames], dtype=np.float32)
    el = np.array([f["el"] for f in frames], dtype=np.float32)
    if len(idx) == 1:
        az_s = np.full((T,), float(az[0]), np.float32)
        el_s = np.full((T,), float(el[0]), np.float32)
        return az_s, el_s
    # Unwrap azimuth to prevent linear interpolation across the ±π discontinuity.
    # e.g. az = [-170°, +170°] should interpolate through -180°, not through 0°.
    az = np.unwrap(az.astype(np.float64)).astype(np.float32)
    # Convert frame indices to audio sample positions using fps.
    # sample_pos = frame_idx * (sr / fps)
    spf = float(sr) / float(fps)   # samples per frame
    idx_samples = idx * spf        # frame index → audio sample position
    s = np.arange(T, dtype=np.float32)
    az_s = np.interp(s, idx_samples, az).astype(np.float32)
    el_s = np.interp(s, idx_samples, el).astype(np.float32)
    return az_s, el_s


def _asymmetric_smooth_d_rel(x: np.ndarray, sr: int,
                              attack_s: float, release_s: float) -> np.ndarray:
    """One-pole IIR asymmetric smoother for d_rel.

    d_rel decreasing (object nearer → gain going UP) → slow, controlled by attack_s.
    d_rel increasing (object farther → gain going DOWN) → fast, controlled by release_s.

    This prevents sudden loudness jumps when an object rushes toward the camera,
    while allowing natural fade-out when it moves away.
    """
    alpha_atk = float(1.0 - np.exp(-1.0 / (attack_s  * sr)))
    alpha_rel = float(1.0 - np.exp(-1.0 / (release_s * sr)))
    out  = np.empty_like(x, dtype=np.float32)
    prev = float(x[0])
    for i in range(len(x)):
        v = float(x[i])
        alpha = alpha_atk if v < prev else alpha_rel
        prev  = prev + alpha * (v - prev)
        out[i] = prev
    return out


def interpolate_angles_distance(frames: List[Dict], T: int, sr: int,
                                fps: float = 30.0,
                                depth_keys: Optional[tuple] = None,
                                gain_mode: str = "depth_rel",
                                img_w: int = 1280,
                                img_h: int = 720,
                                metric_alpha: float = 0.5,
                                use_confidence_fade: bool = False,
                                conf_fade_strength: float = 0.6,
                                d_rel_attack_s: float = 0.0,
                                d_rel_release_s: float = 0.0,
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate (az, el, dist, d_rel) per audio sample.

    Priority for distance (default):
    1. depth_render (explicit render value from RTS smoothing)
    2. depth_blended (from depth enhancement)
    3. dist_m_raw (raw metric depth)
    4. dist_m (may be smoothed)
    5. 1.0 (fallback)

    Args:
        depth_keys:    Override depth lookup order. None = default priority.
        gain_mode:     How d_rel (gain driver) is computed:
                       "depth_rel"   – current default: per-clip normalised dist_m (baseline)
                       "bbox_area"   – (A) bbox area / frame area → 0=tiny/far, 1=full-frame/near
                                       d_rel = 1 - area_rel  so near(big)=0 → loud, far(small)=1 → quiet
                       "bbox_area_log" – (A-log) same thresholds, linear in log(area) so d_rel is
                                       linear in log-distance (area ∝ 1/z²). KITTI-GT calibrated.
                       "hybrid"      – (B) metric_alpha * depth_metric + (1-alpha) * bbox
                                       depth component uses absolute dist_m (Metric3D-style),
                                       clipped to [0.5, 20]m then inverted log-scale.
        img_w, img_h:      Frame dimensions used for bbox area normalisation.
        metric_alpha:      Blend weight for metric depth in "hybrid" mode (0=bbox only, 1=depth only).
        use_confidence_fade: If True, push d_rel toward 1 (quiet) when tracker confidence is low.
                             Handles off-screen / detection-miss frames naturally.
        conf_fade_strength:  Max d_rel increase when conf=0. 0.6 → conf=0.3 raises d_rel by ~0.42.
        d_rel_attack_s:  One-pole attack time (seconds) for d_rel asymmetric smoothing.
                         Controls how slowly d_rel can decrease (object approaching → gain rising).
                         0.0 = disabled (use uniform smoothing only). Recommended: 1.0s.
        d_rel_release_s: One-pole release time (seconds) for d_rel asymmetric smoothing.
                         Controls how quickly d_rel can increase (object receding → gain falling).
                         0.0 = disabled. Recommended: 0.3s.
                         Both attack and release must be > 0 to enable asymmetric smoothing.

    Returns (az_s, el_s, dist_s, d_rel_s) arrays of length T.
    d_rel_s ∈ [0,1]: 0 = near/loud, 1 = far/quiet.
    """
    az_s, el_s = interpolate_angles(frames, T, sr, fps=fps)
    idx = np.array([f["frame"] for f in frames], dtype=np.float32)
    spf = float(sr) / float(fps)
    idx_samples = idx * spf
    s = np.arange(T, dtype=np.float32)

    # ── depth series (always computed for dist_s return value) ───────────────
    _depth_keys = depth_keys if depth_keys is not None else ("depth_render", "depth_blended", "dist_m_raw", "dist_m")
    def _get_depth(f):
        for k in _depth_keys:
            if k in f:
                return float(f[k])
        return 1.0
    dist = np.array([_get_depth(f) for f in frames], dtype=np.float32)
    dist_s = np.interp(s, idx_samples, dist).astype(np.float32)

    smooth_samples = max(1, int(sr * 0.1))
    if smooth_samples > 1 and len(dist_s) > smooth_samples:
        from scipy.ndimage import uniform_filter1d
        dist_s = uniform_filter1d(dist_s, size=smooth_samples, mode='nearest').astype(np.float32)

    # ── d_rel computation by gain_mode ────────────────────────────────────────
    frame_area = float(img_w * img_h)

    if gain_mode in ("bbox_area", "bbox_area_log"):
        # (A) bbox area → d_rel using absolute thresholds (not per-clip normalize).
        # Per-clip normalize was causing bbox variation to always map to full [0,1]
        # regardless of absolute object size, killing perceptual distance effect.
        #   area >= AREA_NEAR (8% of frame)  → d_rel=0 → loud + bright
        #   area <= AREA_FAR  (0.1% of frame) → d_rel=1 → quiet + dark
        # "bbox_area"     : linear in area fraction (original, chosen by eye).
        # "bbox_area_log" : linear in log(area). Since area ∝ 1/z², log-area is
        #   linear in log z — the same log-distance shape "hybrid" uses. On the
        #   KITTI depth GT (tools/calibrate_area_thresholds.py, 2026-09-03) the
        #   same thresholds give MAE vs log-distance target 0.309 → 0.116 with
        #   unchanged saturation (~9 %); the linear form pushes most 4–16 m
        #   objects to d_rel 0.4–0.7 (too far).
        AREA_NEAR = 0.08   # 8% of frame → "near" (loud, bright)
        AREA_FAR  = 0.001  # 0.1% of frame → "far"  (quiet, dark)
        def _bbox_area(f):
            w = float(f.get("w", 80))
            h = float(f.get("h", 80))
            return (w * h) / frame_area
        area = np.array([_bbox_area(f) for f in frames], dtype=np.float32)
        area_s = np.interp(s, idx_samples, area).astype(np.float32)
        if smooth_samples > 1:
            from scipy.ndimage import uniform_filter1d
            area_s = uniform_filter1d(area_s, size=smooth_samples, mode='nearest').astype(np.float32)
        if gain_mode == "bbox_area_log":
            la = np.log(np.maximum(area_s, 1e-12))
            area_norm = np.clip((la - math.log(AREA_FAR)) / (math.log(AREA_NEAR) - math.log(AREA_FAR)), 0.0, 1.0)
        else:
            area_norm = np.clip((area_s - AREA_FAR) / (AREA_NEAR - AREA_FAR), 0.0, 1.0)
        d_rel_s = 1.0 - area_norm   # large area → d_rel=0 → loud

    elif gain_mode == "hybrid":
        # (B) blend bbox_area component + metric depth component
        # bbox component (same as mode A)
        def _bbox_area(f):
            w = float(f.get("w", 80))
            h = float(f.get("h", 80))
            return (w * h) / frame_area
        area = np.array([_bbox_area(f) for f in frames], dtype=np.float32)
        area_s = np.interp(s, idx_samples, area).astype(np.float32)
        if smooth_samples > 1:
            from scipy.ndimage import uniform_filter1d
            area_s = uniform_filter1d(area_s, size=smooth_samples, mode='nearest').astype(np.float32)
        a_min, a_max = area_s.min(), area_s.max()
        if (a_max - a_min) < 1e-4:
            bbox_d_rel = np.full_like(dist_s, 0.5)
        else:
            area_norm = np.clip((area_s - a_min) / (a_max - a_min), 0.0, 1.0)
            bbox_d_rel = 1.0 - area_norm

        # metric depth component: absolute dist_m, log-inverted to [0,1]
        # near(0.5m)→0, far(20m)→1  using log scale for perceptual linearity
        log_near = math.log(0.5)
        log_far  = math.log(20.0)
        dist_clip = np.clip(dist_s, 0.5, 20.0).astype(np.float32)
        depth_d_rel = np.clip((np.log(dist_clip) - log_near) / (log_far - log_near), 0.0, 1.0)

        alpha = float(np.clip(metric_alpha, 0.0, 1.0))
        d_rel_s = alpha * depth_d_rel + (1.0 - alpha) * bbox_d_rel

    else:
        # baseline "depth_rel": per-clip normalised dist_m (original behaviour)
        d_min = float(np.min(dist_s))
        d_max = float(np.max(dist_s))
        d_range = d_max - d_min
        if d_range < 0.1:
            d_rel_s = np.full_like(dist_s, 0.5)
        else:
            d_rel_s = np.clip((dist_s - d_min) / d_range, 0.0, 1.0)

    # ── asymmetric d_rel smoothing (optional) ────────────────────────────────
    # Prevents sudden loudness jumps when object rushes toward camera (attack),
    # while allowing natural fade-out when it moves away (release).
    if d_rel_attack_s > 0.0 and d_rel_release_s > 0.0:
        d_rel_s = _asymmetric_smooth_d_rel(
            d_rel_s.astype(np.float32), sr,
            attack_s=d_rel_attack_s, release_s=d_rel_release_s,
        )

    # ── confidence fade (off-screen / detection-miss) ─────────────────────────
    # When tracker loses the object, confidence drops (e.g. v1_bytetrack sets
    # conf=0.3 on miss frames).  Push d_rel toward 1 (quiet/muffled) proportionally.
    # conf=1.0 → no change; conf=0.0 → d_rel += conf_fade_strength (clamped to 1).
    # 0.1s smoothing prevents clicks at detection boundaries.
    if use_confidence_fade:
        conf_raw = np.array([float(f.get("confidence", 1.0)) for f in frames],
                            dtype=np.float32)
        # clamp to sane range; ByteTrack miss = 0.3, good det = 0.6-0.9
        conf_raw = np.clip(conf_raw, 0.0, 1.0)
        conf_s   = np.interp(s, idx_samples, conf_raw).astype(np.float32)
        if smooth_samples > 1:
            from scipy.ndimage import uniform_filter1d
            conf_s = uniform_filter1d(conf_s, size=smooth_samples,
                                      mode='nearest').astype(np.float32)
        # penalty: 0 when conf=1, conf_fade_strength when conf=0
        fade_penalty = float(conf_fade_strength) * (1.0 - conf_s)
        d_rel_s = np.clip(d_rel_s + fade_penalty, 0.0, 1.0)

    return az_s, el_s, dist_s, d_rel_s


def smooth_limit_angles(
    az_s: np.ndarray,
    el_s: np.ndarray,
    sr: int,
    *,
    smooth_ms: float = 50.0,
    max_deg_per_s: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply moving-average smoothing and optional per-second delta limiting to az/el series.
    - smooth_ms: moving average window (milliseconds)
    - max_deg_per_s: if set, clamp per-sample delta to this rate
    """
    az = az_s.astype(np.float32).copy()
    el = el_s.astype(np.float32).copy()
    # moving average smoothing
    win = max(int(sr * (float(smooth_ms) / 1000.0)), 1)
    if win > 1:
        def movavg(x: np.ndarray, w: int) -> np.ndarray:
            k = np.ones((w,), dtype=np.float32) / float(w)
            y = np.convolve(x, k, mode='same')
            return y.astype(np.float32)
        az = movavg(az, win)
        el = movavg(el, win)
    # delta limit (radians per sample)
    if max_deg_per_s is not None and float(max_deg_per_s) > 0:
        max_rad_per_s = float(max_deg_per_s) * (np.pi / 180.0)
        thr = max_rad_per_s / float(sr)
        def clamp_delta(x: np.ndarray, thr: float) -> np.ndarray:
            y = x.copy()
            for i in range(1, y.shape[0]):
                d = y[i] - y[i-1]
                if d > thr:
                    y[i] = y[i-1] + thr
                elif d < -thr:
                    y[i] = y[i-1] - thr
            return y
        az = clamp_delta(az, thr)
        el = clamp_delta(el, thr)
    return az.astype(np.float32), el.astype(np.float32)


def apply_distance_gain_lpf(x: np.ndarray, sr: int, dist_s: np.ndarray,
                            d_rel_s: np.ndarray = None,
                            *, gain_k: float = 1.0,
                            lpf_min_hz: float = 800.0,
                            lpf_max_hz: float = 8000.0,
                            gain_min: float = 0.3,
                            gain_max: float = 1.0,
                            use_learned_mapping: bool = False,
                            learned_model_path: str = None,
                            ) -> np.ndarray:
    """Apply distance-based gain and 1st-order low-pass filter to mono signal.

    All mappings use d_rel (normalized 0-1) for consistent perceptual results
    regardless of whether distance comes from metric depth or bbox proxy.
    - gain: linear from d_rel (0=near=loud, 1=far=quiet)
    - LPF cutoff: log-scaled from d_rel (0=near=bright, 1=far=muffled)

    Args:
        x: mono audio signal
        sr: sample rate
        dist_s: distance in meters (used as fallback to compute d_rel)
        d_rel_s: normalized distance 0-1. If None, computed from dist_s.
        gain_k: gain curve exponent (1.0 = linear, >1 = more aggressive)
        lpf_min_hz: LPF cutoff for far objects
        lpf_max_hz: LPF cutoff for near objects
        gain_min: minimum gain at d_rel=1 (far)
        gain_max: maximum gain at d_rel=0 (near)
        use_learned_mapping: if True, use data-driven curves instead of hardcoded
        learned_model_path: path to .npz weights (None = default)
    """
    T = x.shape[0]
    d = dist_s[:T].astype(np.float32)

    # Compute d_rel if not provided
    if d_rel_s is not None:
        nd = np.clip(d_rel_s[:T].astype(np.float32), 0.0, 1.0)
    else:
        nd = np.clip((d - 0.5) / (10.0 - 0.5), 0.0, 1.0)

    if use_learned_mapping:
        from .distance_model import get_distance_model
        model = get_distance_model(learned_model_path)
        if model is not None:
            g = model.predict_gain(nd)
            fc = model.predict_lpf(nd)
        else:
            print("[warn] distance model weights not found, using hardcoded")
            use_learned_mapping = False

    if not use_learned_mapping:
        # Nonlinear gain: inverse-square law mapped through d_rel
        # d_rel=0 (near) → gain_max, d_rel=1 (far) → gain_min
        # ISL: gain ∝ 1/r² → map d_rel to virtual distance r=[r_near, r_far]
        # then normalize so r_near→1.0, r_far→gain_min
        r_near = 1.0   # virtual reference distance
        r_far  = 8.0   # virtual max distance (tune for perceptual range)
        r = r_near + (r_far - r_near) * nd          # linear d_rel → virtual r
        g_isl = np.clip((r_near / r) ** 2, 0.0, 1.0)  # inverse-square
        # re-scale to [gain_min, gain_max] range
        g = gain_min + (gain_max - gain_min) * g_isl
        # optional extra shaping via gain_k (>1 = more aggressive rolloff)
        if float(gain_k) != 1.0:
            g = gain_min + (gain_max - gain_min) * (g_isl ** float(gain_k))

        lp_min = max(50.0, float(lpf_min_hz))
        lp_max = max(lp_min + 10.0, float(lpf_max_hz))
        log_min = math.log(lp_min)
        log_max = math.log(lp_max)
        # LPF also nonlinear: square root curve → more HF rolloff near, gentler far
        log_fc = log_max - (log_max - log_min) * np.sqrt(nd)
        fc = np.exp(log_fc)
        # Smooth fc trajectory to avoid zipper noise (500ms window)
        from scipy.ndimage import uniform_filter1d
        fc_smooth_samples = max(1, int(sr * 0.5))
        fc = uniform_filter1d(fc, size=fc_smooth_samples, mode='nearest').astype(np.float32)

    y = (x.astype(np.float32) * g).astype(np.float32)

    # One-pole LPF with time-varying alpha
    two_pi = 2.0 * np.pi
    y_lp = np.zeros_like(y)
    prev = 0.0
    for i in range(T):
        a = (two_pi * fc[i]) / (two_pi * fc[i] + sr)
        prev = prev + a * (y[i] - prev)
        y_lp[i] = prev

    # Clip-protect only — do NOT unconditionally normalize.
    # Full normalization here would cancel the distance gain we just applied
    # (far object at gain=0.1 would be boosted back to 1.0).
    # Only clamp if we'd clip the output.
    peak = float(np.max(np.abs(y_lp)) + 1e-9)
    if peak > 1.0:
        y_lp = y_lp / (peak * 1.01)
    return y_lp.astype(np.float32)


# ISO 9613-1 atmospheric absorption at 20 degC, 50 % RH, 101.325 kPa.
# Octave-band attenuation in dB per kilometre; divided by 1000 for dB/m.
ISO9613_FREQ_HZ = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0,
                            4000.0, 8000.0, 16000.0])
ISO9613_ALPHA_DB_PER_KM = np.array([0.4, 1.1, 2.7, 6.0, 15.3, 44.2, 140.0, 440.0])
SPEED_OF_SOUND_M_S = 343.0


def air_absorption_cutoff_hz(dist_m: np.ndarray, *, drop_db: float = 3.0,
                             f_min: float = 500.0,
                             f_max: float = 18000.0) -> np.ndarray:
    """Frequency at which ISO 9613-1 air absorption reaches `drop_db` (gap A7).

    Used as the cutoff of a one-pole low-pass, so the rendered HF roll-off
    follows the published atmospheric attenuation instead of an invented curve.
    Interpolation is log-log, which is where the ISO table is close to straight.
    """
    d = np.maximum(np.asarray(dist_m, dtype=np.float64), 1e-3)
    alpha_target = float(drop_db) / d          # dB/m needed to lose drop_db
    log_f = np.log(ISO9613_FREQ_HZ)
    log_a = np.log(ISO9613_ALPHA_DB_PER_KM / 1000.0)
    fc = np.exp(np.interp(np.log(alpha_target), log_a, log_f))
    return np.clip(fc, f_min, f_max).astype(np.float32)


def apply_physical_distance(x: np.ndarray, sr: int, dist_s: np.ndarray, *,
                            d_ref_m: float = 1.0, gain_floor: float = 0.02,
                            air_absorption: bool = True) -> np.ndarray:
    """Direct-path attenuation by the actual inverse-distance law (gap item A7).

    The legacy curve maps d_rel linearly into [gain_min, gain_max] = [0.3, 1.0],
    a 10.5 dB span across an entire scene — roughly a third of the physical
    range over 6-60 m, and untethered from any real distance.  Here the direct
    gain is `d_ref / max(d, d_ref)`, i.e. the textbook -6 dB per doubling, with
    a floor so a far source does not vanish, plus ISO 9613-1 air absorption as a
    distance-dependent one-pole low-pass.

    The reverb half of the physical law lives in build_physical_wet_curve():
    holding the reverb send constant in absolute level is what makes the
    direct-to-reverberant ratio fall as 1/d, which is the distance cue that
    survives when intensity is unreliable.
    """
    T = x.shape[0]
    d = np.maximum(np.asarray(dist_s[:T], dtype=np.float64), 1e-6)
    g = np.clip(float(d_ref_m) / np.maximum(d, float(d_ref_m)),
                float(gain_floor), 1.0).astype(np.float32)
    y = (x[:T].astype(np.float32) * g).astype(np.float32)

    if not air_absorption:
        return y

    fc = air_absorption_cutoff_hz(d)
    from scipy.ndimage import uniform_filter1d
    fc = uniform_filter1d(fc, size=max(1, int(sr * 0.5)), mode="nearest").astype(np.float32)

    two_pi = 2.0 * np.pi
    out = np.zeros_like(y)
    prev = 0.0
    for i in range(T):
        a = (two_pi * fc[i]) / (two_pi * fc[i] + sr)
        prev = prev + a * (y[i] - prev)
        out[i] = prev
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 1.0:
        out = out / (peak * 1.01)
    return out.astype(np.float32)


def physical_direct_gain(dist_m, *, d_ref_m: float = 1.0,
                         gain_floor: float = 0.02) -> np.ndarray:
    """The A7 direct-path gain curve on its own (for tests and plots)."""
    d = np.maximum(np.asarray(dist_m, dtype=np.float64), 1e-6)
    return np.clip(float(d_ref_m) / np.maximum(d, float(d_ref_m)),
                   float(gain_floor), 1.0).astype(np.float32)


def build_physical_wet_curve(dist_s: np.ndarray, *, d_ref_m: float = 1.0,
                             gain_floor: float = 0.02,
                             reverb_send: float = 0.02,
                             wet_cap: float = 0.9) -> np.ndarray:
    """Wet mix that holds the reverb ABSOLUTE level constant (gap item A7).

    The renderer mixes `out = (1 - wet) * dry_attenuated + wet * reverb(dry_attenuated)`,
    and the reverb of an attenuated dry is itself attenuated by the same g. So to
    keep the reverberant field at a constant absolute level K the wet mix must be

        wet(d) = K / g(d)

    which makes the direct term (1 - K/g) * g = g - K and therefore

        DRR(d) = (g(d) - K) / K   ~   1/d

    i.e. the direct-to-reverberant ratio falls at the physical rate by
    construction rather than by a separately-tuned ramp.
    """
    g = physical_direct_gain(dist_s, d_ref_m=d_ref_m, gain_floor=gain_floor)
    wet = np.clip(float(reverb_send) / np.maximum(g, 1e-6), 0.0, float(wet_cap))
    return wet.astype(np.float32)


def apply_doppler(x: np.ndarray, sr: int, dist_s: np.ndarray, *,
                  max_ratio: float = 0.25,
                  c_m_s: float = SPEED_OF_SOUND_M_S) -> np.ndarray:
    """Resample by the radial velocity implied by the distance track (gap A14).

    The RTS smoother already carries `v_dist` in its state and then discards it;
    the same quantity is recoverable from the distance series the renderer
    already has.  For a source moving at radial velocity v (positive = receding)
    the received frequency is f * c / (c + v), so the read pointer advances at

        rate[n] = 1 - v[n] / c        (first order in v/c)

    Receding therefore reads the source slower and drops the pitch. The rate is
    clamped to [1 - max_ratio, 1 + max_ratio] so a depth glitch cannot produce a
    siren. Opt-in: `doppler=True` on the render entry points.
    """
    T = x.shape[0]
    d = np.asarray(dist_s[:T], dtype=np.float64)
    if T < 2:
        return x.astype(np.float32)
    v = np.gradient(d) * float(sr)                     # metres per second
    rate = np.clip(1.0 - v / float(c_m_s),
                   1.0 - float(max_ratio), 1.0 + float(max_ratio))
    read_pos = np.cumsum(rate) - rate[0]
    read_pos = np.clip(read_pos, 0.0, T - 1)
    out = np.interp(read_pos, np.arange(T, dtype=np.float64),
                    x[:T].astype(np.float64))
    return out.astype(np.float32)


def build_wet_curve_from_dist_occ(d_rel_s: np.ndarray,
                                  occ_s: np.ndarray | None = None,
                                  *,
                                  wet_min: float = 0.05,
                                  wet_max: float = 0.12,
                                  occ_boost: float = 0.10,
                                  use_learned_mapping: bool = False,
                                  learned_model_path: str = None,
                                  ) -> np.ndarray:
    """Map d_rel/occlusion to reverb wetness curve in [0,1].

    Uses d_rel (pre-normalized 0-1) for consistent reverb mapping across tracks.
    - d_rel=0 (near) → wet_min (less reverb)
    - d_rel=1 (far) → wet_max (more reverb)
    - occlusion adds additional wetness

    Args:
        d_rel_s: normalized distance [0,1] from depth_utils
        occ_s: occlusion values [0,1], optional
        wet_min: reverb wetness for near objects
        wet_max: reverb wetness for far objects
        occ_boost: additional wetness for occluded objects
        use_learned_mapping: if True, use data-driven wet curve
        learned_model_path: path to .npz weights (None = default)
    """
    d_norm = np.clip(d_rel_s.astype(np.float32), 0.0, 1.0)

    if use_learned_mapping:
        from .distance_model import get_distance_model
        model = get_distance_model(learned_model_path)
        if model is not None:
            wet = model.predict_wet(d_norm)
        else:
            print("[warn] distance model weights not found, using hardcoded wet")
            wet = wet_min + (wet_max - wet_min) * d_norm
    else:
        wet = wet_min + (wet_max - wet_min) * d_norm

    if occ_s is not None:
        occ = np.clip(occ_s.astype(np.float32), 0.0, 1.0)
        wet = wet + occ_boost * occ
    return np.clip(wet, 0.0, 1.0).astype(np.float32)


def apply_timevarying_reverb_mono(x: np.ndarray, sr: int, wet_curve: np.ndarray, rt60: float = 0.6) -> np.ndarray:
    """Apply time-varying wet mix using a single mono Schroeder IR.
    y = (1-wet)*x + wet*conv(x, ir)
    """
    from .irgen import schroeder_ir, fft_convolve
    T = x.shape[0]
    wet = wet_curve[:T].astype(np.float32)
    ir = schroeder_ir(sr, rt60=rt60).astype(np.float32)
    y_rev = fft_convolve(x.astype(np.float32), ir)[:T]
    y = (1.0 - wet) * x.astype(np.float32) + wet * y_rev
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / (peak * 1.01)
    return y.astype(np.float32)


def apply_timevarying_reverb_foa(foa: np.ndarray, sr: int, wet_curve: np.ndarray, air_foa: np.ndarray | None = None, rt60: float = 0.6) -> np.ndarray:
    """Apply time-varying wet mix to FOA.
    - If air_foa(4ch) provided: wet * (foa * air) + (1-wet)*foa
    - Else: per-channel Schroeder IR convolution
    """
    T = foa.shape[1]
    wet = wet_curve[:T].astype(np.float32)
    if air_foa is not None:
        from .irgen import convolve_foa_with_air
        wet_foa = convolve_foa_with_air(foa, air_foa)
    else:
        from .irgen import schroeder_ir, fft_convolve
        ir = schroeder_ir(sr, rt60=rt60).astype(np.float32)
        wet_foa = np.vstack([fft_convolve(foa[c], ir)[:T] for c in range(4)]).astype(np.float32)
    y = (1.0 - wet)[None, :] * foa.astype(np.float32) + wet[None, :] * wet_foa
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = y / (peak * 1.01)
    return y.astype(np.float32)


def encode_mono_to_foa(mono: np.ndarray,
                       az_series: np.ndarray,
                       el_series: np.ndarray) -> np.ndarray:
    """Time-varying FOA encoding of mono signal using per-sample az/el.
    Returns FOA array [4, T] in AmbiX (ACN/SN3D) [W, Y, Z, X]."""
    assert mono.ndim == 1
    T = mono.shape[0]
    assert az_series.shape[0] == T and el_series.shape[0] == T
    gains = dir_to_foa_acn_sn3d_gains(az_series, el_series)  # [4,T]
    foa = gains * mono[None, :]
    # peak normalization to avoid clipping
    peak = float(np.max(np.abs(foa)))
    if peak > 1.0:
        foa /= (peak * 1.01)
    return foa.astype(np.float32)


def write_foa_wav(path: str, foa: np.ndarray, sr: int) -> None:
    if foa.shape[0] != 4:
        raise ValueError("FOA must have 4 channels [W,Y,Z,X] in AmbiX order")
    # apply gentle soft limiter to prevent inter-sample clipping
    y = foa.astype(np.float32)
    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 0.98:
        # soft clipper: y = tanh(a*y)/tanh(a)
        a = 1.5
        y = np.tanh(a * y) / math.tanh(a)
    sf.write(path, y.T, sr, subtype="FLOAT")


def encode_many_to_foa(monolist: List[np.ndarray], az_list: List[np.ndarray], el_list: List[np.ndarray]) -> np.ndarray:
    """Sum multiple mono sources into one FOA.
    All sequences must be length-matched; returns [4, T]."""
    assert len(monolist) == len(az_list) == len(el_list)
    if len(monolist) == 0:
        raise ValueError("empty sources")
    T = monolist[0].shape[0]
    acc = np.zeros((4, T), np.float32)
    for x, az, el in zip(monolist, az_list, el_list):
        assert x.shape[0] == T and az.shape[0] == T and el.shape[0] == T
        acc += encode_mono_to_foa(x, az, el)
    peak = float(np.max(np.abs(acc)))
    if peak > 1.0:
        acc /= (peak * 1.01)
    return acc.astype(np.float32)


LOST_CONF_THRESHOLD = 0.5
LOST_DUCK_DB = -9.0
LOST_DIFFUSE_BOOST = 0.35
LOST_FADE_MS = 80.0


def freeze_lost_frames(frames: List[Dict], conf_threshold: float = LOST_CONF_THRESHOLD):
    """Hold az/el at the last confident value across lost episodes (gap item A16).

    FAILURE_MODE_ANALYSIS.json puts frac_lost at 4.0 % over 55 episodes, with
    azimuth error 10.06 deg while lost against 0.757 deg while good.  The
    renderer had no notion of "lost": it interpolated straight through and
    panned confidently to a wrong place.  Freezing is the conservative choice --
    a source that stops moving is far less objectionable than one that lunges.

    Frames before the first confident frame are back-filled from it.  A frame
    with no "confidence" key counts as confident (1.0), so a trajectory that
    never recorded confidence is passed through untouched.

    Returns (frames_out, stats) where frames_out is a new list of shallow
    copies and stats has n_lost / n_episodes / frac_lost.
    """
    conf = np.array([float(f.get("confidence", 1.0)) for f in frames], dtype=np.float64)
    lost = conf < float(conf_threshold)
    stats = {
        "n_frames": int(len(frames)),
        "n_lost": int(lost.sum()),
        "frac_lost": float(lost.mean()) if len(frames) else 0.0,
        "n_episodes": int(np.count_nonzero(np.diff(np.concatenate(([False], lost))) & lost)),
        "conf_threshold": float(conf_threshold),
    }
    if not lost.any():
        return list(frames), stats

    out = [dict(f) for f in frames]
    first_ok = int(np.argmax(~lost)) if (~lost).any() else 0
    hold_az = float(out[first_ok].get("az", 0.0))
    hold_el = float(out[first_ok].get("el", 0.0))
    for i, f in enumerate(out):
        if lost[i]:
            f["az_measured"] = float(f.get("az", 0.0))
            f["el_measured"] = float(f.get("el", 0.0))
            f["az"] = hold_az
            f["el"] = hold_el
            f["lost"] = True
        else:
            hold_az = float(f.get("az", hold_az))
            hold_el = float(f.get("el", hold_el))
            f["lost"] = False
    return out, stats


def build_lost_curves(frames: List[Dict], T: int, sr: int, fps: float,
                      *, conf_threshold: float = LOST_CONF_THRESHOLD,
                      duck_db: float = LOST_DUCK_DB,
                      diffuse_boost: float = LOST_DIFFUSE_BOOST,
                      fade_ms: float = LOST_FADE_MS):
    """Per-sample (duck_gain, diffuse_boost) curves for lost episodes (A16).

    duck_gain falls to 10**(duck_db/20) while lost and returns to 1.0 when the
    track is recovered; diffuse rises to `diffuse_boost` over the same window so
    the source is pushed toward the reverberant field rather than being panned
    confidently to a wrong direction.  Both are box-smoothed over `fade_ms` to
    keep the transitions click-free.
    """
    conf = np.array([float(f.get("confidence", 1.0)) for f in frames], dtype=np.float32)
    lost = (conf < float(conf_threshold)).astype(np.float32)
    if not lost.any() or T <= 0:
        return np.ones(max(T, 0), dtype=np.float32), np.zeros(max(T, 0), dtype=np.float32)

    idx = np.array([f["frame"] for f in frames], dtype=np.float32)
    idx_samples = idx * (float(sr) / float(fps))
    s = np.arange(T, dtype=np.float32)
    lost_s = np.interp(s, idx_samples, lost).astype(np.float32)

    fade_samples = max(1, int(sr * float(fade_ms) / 1000.0))
    if fade_samples > 1:
        from scipy.ndimage import uniform_filter1d
        lost_s = uniform_filter1d(lost_s, size=fade_samples, mode="nearest").astype(np.float32)
    lost_s = np.clip(lost_s, 0.0, 1.0)

    duck_floor = float(10.0 ** (float(duck_db) / 20.0))
    duck = (1.0 - lost_s) + duck_floor * lost_s
    diffuse = float(diffuse_boost) * lost_s
    return duck.astype(np.float32), diffuse.astype(np.float32)


def _load_and_prepare(audio_path: str, trajectory: Dict, smooth_ms: float = 50.0,
                      dist_gain_k: float = 1.0, lpf_min: float = 800.0, lpf_max: float = 8000.0,
                      gain_min: float = 0.3,
                      use_learned_mapping: bool = False, learned_model_path: str = None,
                      gain_mode: str = "depth_rel", metric_alpha: float = 0.5,
                      use_confidence_fade: bool = False, conf_fade_strength: float = 0.6,
                      d_rel_attack_s: float = 0.0, d_rel_release_s: float = 0.0,
                      confidence_gate: bool = True,
                      conf_threshold: float = LOST_CONF_THRESHOLD,
                      lost_duck_db: float = LOST_DUCK_DB,
                      lost_diffuse_boost: float = LOST_DIFFUSE_BOOST,
                      d_ref_m: float = 1.0, air_absorption: bool = True,
                      doppler: bool = False, doppler_max_ratio: float = 0.25):
    """Shared prep for FOA and binaural renderers: load audio, interpolate trajectory, apply distance FX.

    gain_mode: "depth_rel" (baseline), "bbox_area" (A), "bbox_area_log" (A-log),
               "hybrid" (B), "physical" (A7 -- 1/d direct gain, constant absolute
               reverb send, ISO 9613-1 air absorption).
    metric_alpha: blend weight for depth component in "hybrid" mode.
    use_confidence_fade: fade out when tracker confidence is low (off-screen handling).
    conf_fade_strength: max d_rel increase at conf=0 (0.6 recommended).
    confidence_gate: freeze azimuth and duck toward diffuse during lost episodes (A16).
                     A trajectory with no "confidence" field is unaffected.
    doppler: resample by the radial velocity in the trajectory (A14, opt-in).

    Returns (audio_proc, sr, az_s, el_s, dist_s, d_rel_s, frames). Callers that
    need the lost-episode diffuse curve rebuild it with build_lost_curves().
    """
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    T = audio.shape[0]
    frames = trajectory.get("frames", [])
    if not frames:
        raise ValueError("Empty trajectory frames")
    fps = float(trajectory.get("fps", trajectory.get("intrinsics", {}).get("fps", 30.0)))
    intr = trajectory.get("intrinsics", {})
    img_w = int(intr.get("width",  frames[0].get("width",  1280)))
    img_h = int(intr.get("height", frames[0].get("height",  720)))

    # ── A16: freeze azimuth across lost episodes before interpolating ────────
    duck_s = None
    if confidence_gate:
        frames, lost_stats = freeze_lost_frames(frames, conf_threshold=conf_threshold)
        if lost_stats["n_lost"]:
            print(f"[info] confidence gate: {lost_stats['n_lost']}/{lost_stats['n_frames']} "
                  f"frames lost in {lost_stats['n_episodes']} episodes — azimuth frozen, "
                  f"ducking {lost_duck_db:+.1f} dB toward diffuse")
            duck_s, _ = build_lost_curves(
                frames, T, sr, fps, conf_threshold=conf_threshold,
                duck_db=lost_duck_db, diffuse_boost=lost_diffuse_boost)

    az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(
        frames, T, sr, fps=fps,
        gain_mode=gain_mode, img_w=img_w, img_h=img_h, metric_alpha=metric_alpha,
        use_confidence_fade=use_confidence_fade, conf_fade_strength=conf_fade_strength,
        d_rel_attack_s=d_rel_attack_s, d_rel_release_s=d_rel_release_s,
    )
    az_s, el_s = smooth_limit_angles(az_s, el_s, sr, smooth_ms=smooth_ms)

    # ── A14: Doppler from the radial velocity the smoother already computes ──
    if doppler:
        audio = apply_doppler(audio, sr, dist_s, max_ratio=doppler_max_ratio)

    if gain_mode == "physical":
        # A7: 1/d direct gain + ISO 9613-1 air absorption. The reverb side is
        # handled by the caller through build_physical_wet_curve().
        audio_proc = apply_physical_distance(
            audio, sr, dist_s, d_ref_m=d_ref_m, gain_floor=gain_min,
            air_absorption=air_absorption)
    else:
        audio_proc = apply_distance_gain_lpf(audio, sr, dist_s, d_rel_s,
                                             gain_k=dist_gain_k, lpf_min_hz=lpf_min, lpf_max_hz=lpf_max,
                                             gain_min=gain_min,
                                             use_learned_mapping=use_learned_mapping,
                                             learned_model_path=learned_model_path)

    if duck_s is not None:
        n = min(len(audio_proc), len(duck_s))
        audio_proc = audio_proc.copy()
        audio_proc[:n] = (audio_proc[:n] * duck_s[:n]).astype(np.float32)

    return audio_proc, sr, az_s, el_s, dist_s, d_rel_s, frames


def _build_render_wet_curve(frames, T, sr, trajectory, d_rel_s, dist_s, *,
                            gain_mode, d_ref_m, gain_min, reverb_send,
                            confidence_gate, conf_threshold, lost_diffuse_boost,
                            use_learned_mapping, learned_model_path):
    """Reverb wet curve for a render: physical law (A7) + lost-episode duck (A16)."""
    if gain_mode == "physical":
        wet = build_physical_wet_curve(dist_s[:T], d_ref_m=d_ref_m,
                                       gain_floor=gain_min, reverb_send=reverb_send)
    else:
        wet = build_wet_curve_from_dist_occ(
            d_rel_s, use_learned_mapping=use_learned_mapping,
            learned_model_path=learned_model_path)
    if confidence_gate:
        fps = float(trajectory.get("fps",
                                   trajectory.get("intrinsics", {}).get("fps", 30.0)))
        _, diffuse = build_lost_curves(frames, T, sr, fps,
                                       conf_threshold=conf_threshold,
                                       diffuse_boost=lost_diffuse_boost)
        n = min(len(wet), len(diffuse))
        wet = wet.copy()
        wet[:n] = np.clip(wet[:n] + diffuse[:n], 0.0, 1.0)
    return wet.astype(np.float32)


def render_foa_from_trajectory(
    audio_path: str,
    trajectory: Dict,
    output_path: str,
    *,
    smooth_ms: float = 50.0,
    dist_gain_k: float = 1.0,
    gain_min: float = 0.3,
    dist_lpf_min_hz: float = 800.0,
    dist_lpf_max_hz: float = 8000.0,
    apply_reverb: bool = False,
    rt60: float = 0.5,
    use_learned_mapping: bool = False,
    learned_model_path: str = None,
    gain_mode: str = "depth_rel",
    metric_alpha: float = 0.5,
    use_confidence_fade: bool = False,
    conf_fade_strength: float = 0.6,
    d_rel_attack_s: float = 0.0,
    d_rel_release_s: float = 0.0,
    confidence_gate: bool = True,
    conf_threshold: float = LOST_CONF_THRESHOLD,
    lost_duck_db: float = LOST_DUCK_DB,
    lost_diffuse_boost: float = LOST_DIFFUSE_BOOST,
    d_ref_m: float = 1.0,
    air_absorption: bool = True,
    reverb_send: float = 0.02,
    doppler: bool = False,
    doppler_max_ratio: float = 0.25,
) -> Dict:
    """Render mono audio to FOA (4-channel AmbiX) using trajectory.

    gain_mode: "depth_rel" (baseline) | "bbox_area" (A) | "bbox_area_log" (A-log) | "hybrid" (B)
    use_confidence_fade: fade out when tracker confidence is low (off-screen).
    This produces the FOA output only.  For binaural headphone output,
    use render_binaural_from_trajectory() instead.
    """
    audio_proc, sr, az_s, el_s, dist_s, d_rel_s, frames = _load_and_prepare(
        audio_path, trajectory, smooth_ms, dist_gain_k, dist_lpf_min_hz, dist_lpf_max_hz,
        gain_min=gain_min,
        use_learned_mapping=use_learned_mapping, learned_model_path=learned_model_path,
        gain_mode=gain_mode, metric_alpha=metric_alpha,
        use_confidence_fade=use_confidence_fade, conf_fade_strength=conf_fade_strength,
        d_rel_attack_s=d_rel_attack_s, d_rel_release_s=d_rel_release_s,
        confidence_gate=confidence_gate, conf_threshold=conf_threshold,
        lost_duck_db=lost_duck_db, lost_diffuse_boost=lost_diffuse_boost,
        d_ref_m=d_ref_m, air_absorption=air_absorption,
        doppler=doppler, doppler_max_ratio=doppler_max_ratio)
    T = audio_proc.shape[0]

    # Convert pipeline azimuth to AmbiX convention (negate)
    az_ambiX = -az_s

    # Encode to FOA
    foa = encode_mono_to_foa(audio_proc, az_ambiX, el_s)

    # Apply reverb if requested
    if apply_reverb:
        wet_curve = _build_render_wet_curve(
            frames, T, sr, trajectory, d_rel_s, dist_s,
            gain_mode=gain_mode, d_ref_m=d_ref_m, gain_min=gain_min,
            reverb_send=reverb_send,
            confidence_gate=confidence_gate, conf_threshold=conf_threshold,
            lost_diffuse_boost=lost_diffuse_boost,
            use_learned_mapping=use_learned_mapping,
            learned_model_path=learned_model_path)
        foa = apply_timevarying_reverb_foa(foa, sr, wet_curve, rt60=rt60)

    write_foa_wav(output_path, foa, sr)

    return {
        "foa_path": output_path,
        "sample_rate": sr,
        "duration_sec": T / sr,
        "num_frames": len(frames),
    }


def render_binaural_from_trajectory(
    audio_path: str,
    trajectory: Dict,
    output_path: str,
    sofa_path: str,
    *,
    smooth_ms: float = 50.0,
    dist_gain_k: float = 1.0,
    gain_min: float = 0.3,
    dist_lpf_min_hz: float = 800.0,
    dist_lpf_max_hz: float = 8000.0,
    apply_reverb: bool = False,
    rt60: float = 0.5,
    block_ms: float = 10.0,
    hrir_interp: str = "barycentric",
    use_learned_mapping: bool = False,
    learned_model_path: str = None,
    gain_mode: str = "depth_rel",
    metric_alpha: float = 0.5,
    use_confidence_fade: bool = False,
    conf_fade_strength: float = 0.6,
    d_rel_attack_s: float = 0.0,
    d_rel_release_s: float = 0.0,
    confidence_gate: bool = True,
    conf_threshold: float = LOST_CONF_THRESHOLD,
    lost_duck_db: float = LOST_DUCK_DB,
    lost_diffuse_boost: float = LOST_DIFFUSE_BOOST,
    d_ref_m: float = 1.0,
    air_absorption: bool = True,
    reverb_send: float = 0.02,
    doppler: bool = False,
    doppler_max_ratio: float = 0.25,
) -> Dict:
    """Render mono audio to HRTF binaural stereo using trajectory.

    gain_mode: "depth_rel" (baseline) | "bbox_area" (A) | "bbox_area_log" (A-log) | "hybrid" (B)
    use_confidence_fade: fade out when tracker confidence is low (off-screen).
    block_ms: HRTF update interval — 10ms (default) balances smoothness vs cost.
              50ms caused ~1.5-frame lag on 30fps video; 10ms is perceptually tight.
    Uses direct HRIR convolution (per-block nearest-neighbor lookup from SOFA)
    for full-bandwidth spatial cues (ILD, ITD, pinna).
    Independent from the FOA path.
    """
    audio_proc, sr, az_s, el_s, dist_s, d_rel_s, frames = _load_and_prepare(
        audio_path, trajectory, smooth_ms, dist_gain_k, dist_lpf_min_hz, dist_lpf_max_hz,
        gain_min=gain_min,
        use_learned_mapping=use_learned_mapping, learned_model_path=learned_model_path,
        gain_mode=gain_mode, metric_alpha=metric_alpha,
        use_confidence_fade=use_confidence_fade, conf_fade_strength=conf_fade_strength,
        d_rel_attack_s=d_rel_attack_s, d_rel_release_s=d_rel_release_s,
        confidence_gate=confidence_gate, conf_threshold=conf_threshold,
        lost_duck_db=lost_duck_db, lost_diffuse_boost=lost_diffuse_boost,
        d_ref_m=d_ref_m, air_absorption=air_absorption,
        doppler=doppler, doppler_max_ratio=doppler_max_ratio)
    T = audio_proc.shape[0]

    # Convert pipeline azimuth to AmbiX/SOFA convention (negate)
    az_sofa = -az_s

    # Direct HRTF binaural render (before reverb, so spatial cues are crisp)
    stereo = direct_binaural_sofa(audio_proc, sr, az_sofa, el_s, sofa_path,
                                  block_ms=block_ms, hrir_interp=hrir_interp)

    # Apply reverb to stereo AFTER HRTF (reverb is diffuse, shouldn't mask spatial cues)
    if apply_reverb:
        wet_curve = _build_render_wet_curve(
            frames, T, sr, trajectory, d_rel_s, dist_s,
            gain_mode=gain_mode, d_ref_m=d_ref_m, gain_min=gain_min,
            reverb_send=reverb_send,
            confidence_gate=confidence_gate, conf_threshold=conf_threshold,
            lost_diffuse_boost=lost_diffuse_boost,
            use_learned_mapping=use_learned_mapping,
            learned_model_path=learned_model_path)
        from .irgen import schroeder_ir, fft_convolve
        ir = schroeder_ir(sr, rt60=rt60).astype(np.float32)
        wet = wet_curve[:T].astype(np.float32)
        for ch in range(2):
            dry = stereo[ch, :T].astype(np.float32)
            rev = fft_convolve(dry, ir)[:T]
            stereo[ch, :T] = (1.0 - wet) * dry + wet * rev

    # Peak normalize
    peak = float(np.max(np.abs(stereo)) + 1e-9)
    if peak > 1.0:
        stereo = stereo / (peak * 1.01)

    sf.write(output_path, stereo.T, sr, subtype="FLOAT")

    return {
        "binaural_path": output_path,
        "binaural_method": "hrtf_direct",
        "sample_rate": sr,
        "duration_sec": T / sr,
        "num_frames": len(frames),
    }


def foa_to_stereo(foa_sn3d: np.ndarray, sr: int, az_deg_L: float = +30.0, az_deg_R: float = -30.0) -> np.ndarray:
    """Decode FOA (AmbiX ACN/SN3D [W,Y,Z,X]) to stereo at ±az degrees on horizon.
    Prefer spaudiopy; fall back to a pure‑NumPy first‑order decoder if unavailable.
    Returns [2, T] float32.
    """
    if foa_sn3d.shape[0] != 4:
        raise ValueError("FOA must be [4,T] in AmbiX [W,Y,Z,X]")
    try:
        import spaudiopy as spa  # type: ignore
        foa_n3d = foa_sn3d.copy()
        foa_n3d[1:4, :] *= np.sqrt(3.0)
        az = np.deg2rad(np.array([az_deg_L, az_deg_R], np.float32))
        zen = np.deg2rad(np.array([90.0, 90.0], np.float32))
        Y = spa.sph.sh_matrix(1, az, zen, sh_type='real')  # [2, 4]
        stereo = (Y @ foa_n3d).astype(np.float32)
    except Exception:
        # Pure‑NumPy fallback (ACN/N3D, real, order‑1, elevation 0)
        foa_n3d = foa_sn3d.copy()
        foa_n3d[1:4, :] *= np.sqrt(3.0)
        az = np.deg2rad(np.array([az_deg_L, az_deg_R], np.float32))
        # For el=0: Y00=1, Y1-1=sqrt(3)sin(az), Y10=0, Y11=sqrt(3)cos(az)
        Y = np.stack([
            np.array([1.0, np.sqrt(3.0)*np.sin(az[0]), 0.0, np.sqrt(3.0)*np.cos(az[0])], dtype=np.float32),
            np.array([1.0, np.sqrt(3.0)*np.sin(az[1]), 0.0, np.sqrt(3.0)*np.cos(az[1])], dtype=np.float32),
        ], axis=0)
        stereo = (Y @ foa_n3d).astype(np.float32)
    peak = float(np.max(np.abs(stereo)))
    if peak > 1.0:
        stereo /= (peak * 1.01)
    return stereo


def foa_to_binaural(foa_sn3d: np.ndarray, sr: int) -> np.ndarray:
    """Convenience: FOA -> stereo(±30°) -> simple crossfeed binaural.
    Returns [2, T] float32.
    """
    st = foa_to_stereo(foa_sn3d, sr)
    # simple crossfeed
    L, R = st[0], st[1]
    d = max(1, int(0.0003 * sr))
    Lp = np.pad(L, (0, 0))
    Rp = np.pad(R, (0, 0))
    Ld = np.pad(Lp, (d, 0))[: L.shape[0]]
    Rd = np.pad(Rp, (d, 0))[: R.shape[0]]
    Lo = L + 0.22 * Rd
    Ro = R + 0.22 * Ld
    peak = float(max(np.max(np.abs(Lo)), np.max(np.abs(Ro))))
    if peak > 1.0:
        Lo /= (peak * 1.01)
        Ro /= (peak * 1.01)
    return np.stack([Lo.astype(np.float32), Ro.astype(np.float32)], 0)


HRIR_INTERP_MODES = ("nearest", "barycentric")


def hrir_barycentric_weights(src_cart: np.ndarray, q: np.ndarray):
    """Barycentric weights over the 3 nearest SOFA directions (gap item A8).

    A KEMAR SOFA grid is typically 5 deg in azimuth, so nearest-neighbour lookup
    snaps a smoothly moving source between HRIRs: audible zipper artefacts, and
    an effective angular resolution floor of ~5 deg -- far coarser than the
    trajectory being fed to it.

    Solve `A w = q` where the columns of A are the three nearest unit
    measurement directions, then normalise so the weights sum to 1. That is the
    exact barycentric coordinate of the query direction inside the spherical
    triangle. If the query falls outside the triangle (a negative weight) or the
    triangle is degenerate, fall back to inverse-angular-distance weighting over
    the same three points, which is always non-negative and still continuous.

    Args:
        src_cart: (M, 3) unit vectors of the measurement grid.
        q:        (3,) unit vector of the query direction.

    Returns:
        (idx, w) with idx an int array of 3 indices into src_cart and w a
        float64 array of 3 non-negative weights summing to 1.
    """
    dots = src_cart @ q
    # the 3 closest directions, ordered closest first
    idx = np.argpartition(-dots, 3)[:3]
    idx = idx[np.argsort(-dots[idx])]
    P = src_cart[idx]                      # (3, 3), rows are directions

    w = None
    try:
        w = np.linalg.solve(P.T, q)        # columns of P.T are the directions
    except np.linalg.LinAlgError:
        w = None
    if w is None or not np.all(np.isfinite(w)) or np.any(w < -1e-9):
        # Outside the triangle, or degenerate -- which is the normal case on a
        # planar grid, e.g. a horizontal-only measurement ring, where the three
        # nearest directions are coplanar with the query and the solve is
        # singular. Fall back to a linear blend of the two bracketing
        # directions, weighted by angular distance. On a ring that reproduces
        # exact angular interpolation; elsewhere it is still continuous.
        ang = np.arccos(np.clip(dots[idx], -1.0, 1.0))
        if float(ang[0]) < 1e-9:
            w = np.array([1.0, 0.0, 0.0])
        else:
            a0, a1 = float(ang[0]), float(ang[1])
            denom = a0 + a1
            w = (np.array([a1 / denom, a0 / denom, 0.0]) if denom > 1e-12
                 else np.array([1.0, 0.0, 0.0]))
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= 0.0:
        w = np.array([1.0, 0.0, 0.0])
        total = 1.0
    return idx, (w / total)


def make_hrir_lookup(src_cart: np.ndarray, ir_data: np.ndarray,
                     mode: str = "barycentric"):
    """Build an (az, el) -> HRIR (2, L) lookup in the requested mode.

    mode="nearest":     argmax of the dot product (pre-2026-09-04 behaviour,
                        kept for stimulus reproducibility).
    mode="barycentric": weighted sum over the 3 nearest measurement directions.
    """
    if mode not in HRIR_INTERP_MODES:
        raise ValueError(f"unknown hrir_interp {mode!r}; expected one of {HRIR_INTERP_MODES}")

    def _q(az: float, el: float) -> np.ndarray:
        return np.array([math.cos(el) * math.cos(az),
                         math.cos(el) * math.sin(az),
                         math.sin(el)], dtype=np.float64)

    if mode == "nearest":
        def lookup(az: float, el: float) -> np.ndarray:
            return ir_data[int(np.argmax(src_cart @ _q(az, el)))]
    else:
        def lookup(az: float, el: float) -> np.ndarray:
            idx, w = hrir_barycentric_weights(src_cart, _q(az, el))
            return np.tensordot(w, ir_data[idx], axes=(0, 0)).astype(ir_data.dtype)

    return lookup


def direct_binaural_sofa(mono: np.ndarray, sr: int, az_s: np.ndarray,
                         el_s: np.ndarray, sofa_path: str,
                         block_ms: float = 50.0,
                         hrir_interp: str = "barycentric") -> np.ndarray:
    """Render mono to binaural using direct HRTF lookup per time block.

    Instead of FOA virtual-speaker decode (which washes out high-frequency
    spatial cues), this directly convolves mono with the nearest HRIR for
    the source direction, updated every block_ms milliseconds.

    Args:
        mono: Mono audio [T]
        sr: Sample rate
        az_s: Azimuth per sample (AmbiX convention: +az = left)
        el_s: Elevation per sample
        sofa_path: Path to SOFA HRTF file
        block_ms: HRTF update interval in ms (trades smoothness vs compute)

    Returns:
        Binaural [2, T] float32
    """
    import h5py
    from scipy.signal import fftconvolve, resample_poly

    T = mono.shape[0]

    with h5py.File(sofa_path, 'r') as sofa:
        ir_data = sofa['Data.IR'][:]          # (M, 2, N)
        fs_hrir = float(sofa['Data.SamplingRate'][0])
        src_pos = sofa['SourcePosition'][:]   # (M, 3)

    src_az = np.radians(src_pos[:, 0])
    src_el = np.radians(src_pos[:, 1])
    src_cart = np.stack([
        np.cos(src_el) * np.cos(src_az),
        np.cos(src_el) * np.sin(src_az),
        np.sin(src_el),
    ], axis=1)  # (M, 3)

    # Resample HRIRs if needed
    if int(fs_hrir) != int(sr):
        gcd = np.gcd(int(sr), int(fs_hrir))
        up, down = int(sr) // gcd, int(fs_hrir) // gcd
        M, _, L = ir_data.shape
        L_out = int(np.ceil(L * sr / fs_hrir))
        ir_resampled = np.zeros((M, 2, L_out), dtype=np.float32)
        for i in range(M):
            for ch in range(2):
                ir_resampled[i, ch] = resample_poly(
                    ir_data[i, ch].astype(np.float32), up, down
                )[:L_out]
        ir_data = ir_resampled

    hrir_len = ir_data.shape[2]
    # Use 50% overlap-add with Hann window for smooth HRIR transitions.
    # block_ms controls the HRIR update interval; the actual analysis frame
    # is 2× block_ms with 50% hop, ensuring perfect reconstruction.
    hop = max(1, int(sr * block_ms / 1000.0))
    win_len = hop * 2  # 50% overlap
    window = np.hanning(win_len).astype(np.float64)

    hrir_lookup = make_hrir_lookup(src_cart, ir_data, mode=hrir_interp)

    out = np.zeros((2, T), dtype=np.float64)

    # Overlap-add: each frame is win_len samples, hopped by hop.
    # The Hann window with 50% overlap sums to 1.0 (perfect reconstruction).
    n_frames = max(1, (T - win_len) // hop + 1)
    # Handle tail: add one extra frame if needed
    if (n_frames - 1) * hop + win_len < T:
        n_frames += 1

    for fi in range(n_frames):
        start = fi * hop
        end = min(start + win_len, T)
        flen = end - start

        # HRIR lookup at frame center
        az_med = float(np.median(az_s[start:min(start + hop, T)]))
        el_med = float(np.median(el_s[start:min(start + hop, T)]))
        hrir = hrir_lookup(az_med, el_med)

        # Extended input for correct convolution at [start, end)
        ext_start = max(0, start - hrir_len + 1)
        ext_mono = mono[ext_start:end].astype(np.float64)
        offset = start - ext_start

        conv_L = fftconvolve(ext_mono, hrir[0])
        conv_R = fftconvolve(ext_mono, hrir[1])

        block_L = conv_L[offset:offset + flen]
        block_R = conv_R[offset:offset + flen]

        # Apply Hann window (truncated for last frame)
        w = window[:flen] if flen == win_len else np.hanning(flen).astype(np.float64)
        out[0, start:end] += w * block_L
        out[1, start:end] += w * block_R

    # Clip-protect only (do NOT unconditionally normalize — preserve distance attenuation)
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 1.0:
        out = out / (peak * 1.01)
    return out.astype(np.float32)


def foa_to_binaural_sofa(foa_sn3d: np.ndarray, sr: int, sofa_path: str,
                         hrir_interp: str = "barycentric") -> np.ndarray:
    """Decode FOA (AmbiX ACN/SN3D [W,Y,Z,X]) to binaural using SOFA HRTF.

    Uses direct HRIR convolution via h5py (no spaudiopy dependency).
    FOA is decoded to virtual speakers, each convolved with the nearest HRIR.
    Returns [2, T] float32. Falls back to simple crossfeed on error.
    """
    if foa_sn3d.shape[0] != 4:
        raise ValueError("FOA must be [4,T] in AmbiX [W,Y,Z,X]")
    try:
        import h5py
        from scipy.signal import fftconvolve, resample_poly

        # --- Load SOFA HRTF ---
        with h5py.File(sofa_path, 'r') as sofa:
            ir_data = sofa['Data.IR'][:]          # (M, 2, N) — M measurements, 2 ears, N taps
            fs_hrir = float(sofa['Data.SamplingRate'][0])
            src_pos = sofa['SourcePosition'][:]   # (M, 3) — [az_deg, el_deg, dist]

        src_az = np.radians(src_pos[:, 0])   # azimuth in radians
        src_el = np.radians(src_pos[:, 1])   # elevation in radians

        # Precompute unit vectors for fast nearest-neighbor lookup
        src_x = np.cos(src_el) * np.cos(src_az)
        src_y = np.cos(src_el) * np.sin(src_az)
        src_z = np.sin(src_el)
        src_cart = np.stack([src_x, src_y, src_z], axis=1)  # (M, 3)

        # Resample HRIRs if sample rates differ
        if int(fs_hrir) != int(sr):
            gcd = np.gcd(int(sr), int(fs_hrir))
            up, down = int(sr) // gcd, int(fs_hrir) // gcd
            M, _, L = ir_data.shape
            L_out = int(np.ceil(L * sr / fs_hrir))
            ir_resampled = np.zeros((M, 2, L_out), dtype=np.float32)
            for i in range(M):
                for ch in range(2):
                    ir_resampled[i, ch] = resample_poly(
                        ir_data[i, ch].astype(np.float32), up, down
                    )[:L_out]
            ir_data = ir_resampled

        hrir_lookup = make_hrir_lookup(src_cart, ir_data, mode=hrir_interp)

        # --- Virtual speaker decode (8 speakers on cube for order-1) ---
        # Speaker directions: front, back, left, right, up-front, up-back, down-front, down-back
        vspk_dirs = [
            (0.0, 0.0),                    # front
            (np.pi, 0.0),                  # back
            (np.pi/2, 0.0),                # left
            (-np.pi/2, 0.0),               # right
            (np.pi/4, np.pi/4),            # upper-front-left
            (-np.pi/4, np.pi/4),           # upper-front-right
            (np.pi/4, -np.pi/4),           # lower-front-left
            (-np.pi/4, -np.pi/4),          # lower-front-right
        ]

        T = foa_sn3d.shape[1]
        binlr = np.zeros((2, T + ir_data.shape[2] - 1), dtype=np.float64)

        for az_spk, el_spk in vspk_dirs:
            # AmbiX ACN/SN3D decode weights for order-1
            # W=1, Y=sin(az)*cos(el), Z=sin(el), X=cos(az)*cos(el)
            w_W = 1.0
            w_Y = np.sin(az_spk) * np.cos(el_spk)
            w_Z = np.sin(el_spk)
            w_X = np.cos(az_spk) * np.cos(el_spk)
            weights = np.array([w_W, w_Y, w_Z, w_X], dtype=np.float32)

            # Decode: speaker signal = sum of weighted FOA channels
            spk_signal = weights @ foa_sn3d  # (T,)

            # Get HRIR for this speaker direction
            # SOFA convention: az=0 is front, positive=counterclockwise
            hrir = hrir_lookup(az_spk, el_spk)  # (2, L)

            # Convolve
            for ch in range(2):
                binlr[ch] += fftconvolve(spk_signal, hrir[ch], mode='full')[:binlr.shape[1]]

        # Trim to input length
        binlr = binlr[:, :T]

        # Peak normalize (preserve spatial balance between L/R)
        peak = float(np.max(np.abs(binlr)) + 1e-9)
        binlr = binlr / (peak * 1.01)  # always normalize to use full dynamic range
        return binlr.astype(np.float32)
    except Exception:
        import traceback
        traceback.print_exc()
        return foa_to_binaural(foa_sn3d, sr)


# =============================================================================
# BASELINE RENDERERS FOR COMPARISON
# =============================================================================

def render_stereo_pan_baseline(
    audio: np.ndarray,
    sr: int,
    az_series: np.ndarray,
    dist_s: np.ndarray = None,
    d_rel_s: np.ndarray = None,
    apply_gain: bool = True,
) -> np.ndarray:
    """
    Simple stereo panning baseline (no FOA).

    This is the simplest spatial audio approach:
    pan = sin(azimuth)
    L = audio * (1 - pan) / 2
    R = audio * (1 + pan) / 2

    Args:
        audio: Mono audio signal
        sr: Sample rate
        az_series: Azimuth in radians per sample
        dist_s: Distance in meters (optional, for gain)
        d_rel_s: Normalized distance 0-1 (optional, for gain)
        apply_gain: Apply distance-based gain attenuation

    Returns:
        Stereo [2, T] array
    """
    T = len(audio)
    audio = audio.astype(np.float32)

    # Apply distance gain from d_rel (relative distance)
    if apply_gain and (d_rel_s is not None or dist_s is not None):
        if d_rel_s is not None:
            nd = np.clip(d_rel_s[:T].astype(np.float32), 0.0, 1.0)
        else:
            nd = np.clip((dist_s[:T] - 0.5) / 9.5, 0.0, 1.0)
        gain = 1.0 - 0.7 * nd  # near=1.0, far=0.3
        audio = audio * gain

    # Constant-power panning
    # Pipeline: az > 0 = RIGHT side of image (atan2(x, z), x > 0 when pixel right of center)
    # pan: +1 = full right, -1 = full left
    # sin(az) > 0 when az > 0 (right) → pan > 0 → R louder ✓
    az = az_series[:T].astype(np.float32)
    pan = np.sin(az)

    pan_angle = (pan + 1) * (np.pi / 4)  # 0 to pi/2
    L_gain = np.cos(pan_angle)
    R_gain = np.sin(pan_angle)

    L = audio * L_gain
    R = audio * R_gain

    stereo = np.stack([L, R], axis=0).astype(np.float32)

    # Normalize
    peak = float(np.max(np.abs(stereo)) + 1e-9)
    if peak > 1.0:
        stereo = stereo / (peak * 1.01)

    return stereo


def render_stereo_pan_reverb_baseline(
    audio: np.ndarray,
    sr: int,
    az_series: np.ndarray,
    dist_s: np.ndarray = None,
    d_rel_s: np.ndarray = None,
    apply_lpf: bool = True,
    rt60: float = 0.5,
) -> np.ndarray:
    """
    Stereo panning + distance-based reverb baseline.

    This is a more sophisticated baseline that adds:
    1. Simple stereo panning (sin law)
    2. Distance-based gain (1/r)
    3. Distance-based LPF
    4. Distance-based reverb

    Args:
        audio: Mono audio signal
        sr: Sample rate
        az_series: Azimuth in radians per sample
        dist_s: Distance in meters
        d_rel_s: Normalized distance 0-1
        apply_lpf: Apply distance-based low-pass filter
        rt60: Reverb time

    Returns:
        Stereo [2, T] array
    """
    T = len(audio)
    audio = audio.astype(np.float32)

    # Apply distance processing (gain + LPF)
    if dist_s is not None:
        if d_rel_s is None:
            d_rel_s = np.clip((dist_s - 0.5) / 9.5, 0.0, 1.0)

        if apply_lpf:
            audio = apply_distance_gain_lpf(audio, sr, dist_s, d_rel_s)
        else:
            # Just gain from d_rel
            nd = np.clip(d_rel_s[:T].astype(np.float32), 0.0, 1.0)
            gain = 1.0 - 0.7 * nd  # near=1.0, far=0.3
            audio = audio * gain

    # Apply panning (same convention as render_stereo_pan_baseline)
    # Pipeline: az > 0 = RIGHT → sin(az) > 0 → pan > 0 → R louder ✓
    az = az_series[:T].astype(np.float32)
    pan = np.sin(az)
    pan_angle = (pan + 1) * (np.pi / 4)
    L_gain = np.cos(pan_angle)
    R_gain = np.sin(pan_angle)

    L = audio * L_gain
    R = audio * R_gain

    stereo = np.stack([L, R], axis=0).astype(np.float32)

    # Apply reverb based on distance
    if d_rel_s is not None:
        wet_curve = build_wet_curve_from_dist_occ(d_rel_s[:T])
        stereo = _apply_stereo_reverb(stereo, sr, wet_curve, rt60)

    # Normalize
    peak = float(np.max(np.abs(stereo)) + 1e-9)
    if peak > 1.0:
        stereo = stereo / (peak * 1.01)

    return stereo


def _apply_stereo_reverb(
    stereo: np.ndarray,
    sr: int,
    wet_curve: np.ndarray,
    rt60: float = 0.5,
) -> np.ndarray:
    """Apply time-varying reverb to stereo signal."""
    from .irgen import schroeder_ir, fft_convolve

    T = stereo.shape[1]
    wet = wet_curve[:T].astype(np.float32)

    ir = schroeder_ir(sr, rt60=rt60).astype(np.float32)

    # Convolve each channel
    L_rev = fft_convolve(stereo[0], ir)[:T]
    R_rev = fft_convolve(stereo[1], ir)[:T]

    # Time-varying mix
    L_out = (1.0 - wet) * stereo[0] + wet * L_rev
    R_out = (1.0 - wet) * stereo[1] + wet * R_rev

    result = np.stack([L_out, R_out], axis=0).astype(np.float32)

    # Normalize
    peak = float(np.max(np.abs(result)) + 1e-9)
    if peak > 1.0:
        result = result / (peak * 1.01)

    return result


def render_baselines_from_trajectory(
    audio_path: str,
    trajectory: Dict,
    output_dir: str,
    prefix: str = "baseline",
) -> Dict[str, str]:
    """
    Render all baseline versions from a trajectory for comparison.

    Generates:
    1. stereo_pan: Simple stereo panning only
    2. stereo_pan_reverb: Stereo panning + distance reverb
    3. foa: Full FOA rendering (for reference)

    Args:
        audio_path: Path to mono audio file
        trajectory: Trajectory dict with 'frames'
        output_dir: Output directory
        prefix: Output filename prefix

    Returns:
        Dict mapping baseline name to output path
    """
    import os
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    T = audio.shape[0]

    frames = trajectory.get("frames", [])
    if not frames:
        raise ValueError("Empty trajectory frames")

    # Interpolate to audio sample rate
    az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, sr)

    os.makedirs(output_dir, exist_ok=True)
    outputs = {}

    # 1. Simple stereo panning
    stereo_pan = render_stereo_pan_baseline(audio, sr, az_s, dist_s, d_rel_s)
    path_pan = os.path.join(output_dir, f"{prefix}_stereo_pan.wav")
    sf.write(path_pan, stereo_pan.T, sr, subtype="FLOAT")
    outputs["stereo_pan"] = path_pan

    # 2. Stereo panning + reverb
    stereo_reverb = render_stereo_pan_reverb_baseline(audio, sr, az_s, dist_s, d_rel_s)
    path_reverb = os.path.join(output_dir, f"{prefix}_stereo_reverb.wav")
    sf.write(path_reverb, stereo_reverb.T, sr, subtype="FLOAT")
    outputs["stereo_pan_reverb"] = path_reverb

    # 3. FOA reference (using existing function)
    foa_path = os.path.join(output_dir, f"{prefix}_foa.wav")
    render_foa_from_trajectory(
        audio_path, trajectory, foa_path,
        apply_reverb=True,
        output_stereo=False,
    )
    outputs["foa"] = foa_path

    # 4. FOA decoded to stereo (for direct A/B comparison)
    foa_data, _ = sf.read(foa_path, dtype='float32')
    foa_data = foa_data.T  # [4, T]
    foa_stereo = foa_to_stereo(foa_data, sr)
    path_foa_stereo = os.path.join(output_dir, f"{prefix}_foa_stereo.wav")
    sf.write(path_foa_stereo, foa_stereo.T, sr, subtype="FLOAT")
    outputs["foa_stereo"] = path_foa_stereo

    print(f"[baseline] Generated {len(outputs)} baseline renders in {output_dir}")
    return outputs


# Update __all__
__all__ = [
    "dir_to_foa_acn_sn3d_gains",
    "interpolate_angles",
    "interpolate_angles_distance",
    "encode_mono_to_foa",
    "write_foa_wav",
    "encode_many_to_foa",
    "render_foa_from_trajectory",
    "render_binaural_from_trajectory",
    "direct_binaural_sofa",
    "foa_to_stereo",
    "foa_to_binaural",
    "foa_to_binaural_sofa",
    # Baselines
    "render_stereo_pan_baseline",
    "render_stereo_pan_reverb_baseline",
    "render_baselines_from_trajectory",
    # Distance processing
    "freeze_lost_frames",
    "build_lost_curves",
    "apply_physical_distance",
    "physical_direct_gain",
    "build_physical_wet_curve",
    "apply_doppler",
    "air_absorption_cutoff_hz",
    "apply_distance_gain_lpf",
    "make_hrir_lookup",
    "hrir_barycentric_weights",
    "HRIR_INTERP_MODES",
    "build_wet_curve_from_dist_occ",
]

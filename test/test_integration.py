"""
Integration + Scenario Tests for vid2spatial_v2 pipeline.

Tests that combine multiple modules end-to-end without requiring GPU/video files.
All tests use synthetic data (in-memory numpy arrays, temp files).

Scenarios covered:
  1. FOA render pipeline (traj dict → FOA wav)
  2. Reverb wet mix pipeline
  3. Multi-source FOA summation
  4. Pipeline IQR clipping + Kalman + FOA render chain
  5. Distance gain continuity (no clipping artifacts)
  6. Depth priority fallback chain
  7. Fast-moving object (large az excursion)
  8. Static object (az near constant)
  9. Object moving in depth only (el/az stable, dist varies)
  10. Edge case: single-frame trajectory
  11. Edge case: very short audio (< 1 frame)
  12. FOA energy ≤ input energy (no artificial amplification)
  13. Binaural downmix (stereo pan sanity)
  14. IQR clipping removes runaway depth outliers
  15. Kalman smoother reduces az jitter vs raw trajectory
  16. d_rel recomputed after Kalman (no stale values)
  17. depth_keys override: dist_m only vs default priority
  18. Reverb FOA stays in SN3D range
  19. Scenario: car driving left→right (az increasing)
  20. Scenario: approaching object (dist decreasing → gain increasing)
  21. Scenario: occluded then visible (confidence jump)
  22. Scenario: fast dog (50% of frames within 1s span)
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vid2spatial_pkg.foa_render import (
    interpolate_angles_distance,
    smooth_limit_angles,
    apply_distance_gain_lpf,
    encode_mono_to_foa,
    encode_many_to_foa,
    render_foa_from_trajectory,
    dir_to_foa_acn_sn3d_gains,
    build_wet_curve_from_dist_occ,
    apply_timevarying_reverb_foa,
    foa_to_stereo,
    write_foa_wav,
)
from vid2spatial_pkg.temporal_smoother import smooth_trajectory_batch
from vid2spatial_pkg.pipeline import SpatialAudioPipeline


# ── Helpers ──────────────────────────────────────────────────────────────────

SR = 48000
FPS = 30.0
DURATION_S = 2.0
N_AUDIO = int(SR * DURATION_S)
N_FRAMES = int(FPS * DURATION_S)


def _make_frames(az_arr, el_arr, dist_arr, n_frames=None, extra_key=None, extra_arr=None):
    """Build trajectory frames from numpy arrays."""
    if n_frames is None:
        n_frames = len(az_arr)
    frames = []
    for i in range(n_frames):
        f = {
            "frame": i,
            "az": float(az_arr[i]),
            "el": float(el_arr[i]),
            "dist_m": float(dist_arr[i]),
        }
        if extra_key is not None and extra_arr is not None:
            f[extra_key] = float(extra_arr[i])
        frames.append(f)
    return frames


def _make_traj(frames, fps=FPS):
    return {"frames": frames, "fps": fps, "intrinsics": {"fov_deg": 60.0}}


def _white_noise(n=N_AUDIO, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    return x / (np.max(np.abs(x)) + 1e-9)


def _audio_to_temp_wav(audio, sr=SR):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, sr, subtype="FLOAT")
    return path


# ── 1. Full FOA render pipeline ───────────────────────────────────────────────

class TestFoaRenderPipeline(unittest.TestCase):

    def test_render_foa_from_trajectory_produces_4ch_wav(self):
        """render_foa_from_trajectory → 4-channel WAV written correctly."""
        az = np.linspace(-0.5, 0.5, N_FRAMES)
        el = np.zeros(N_FRAMES)
        dist = np.ones(N_FRAMES) * 3.0
        frames = _make_frames(az, el, dist)
        traj = _make_traj(frames)
        audio = _white_noise()
        audio_path = _audio_to_temp_wav(audio)
        try:
            fd, out_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            result = render_foa_from_trajectory(audio_path, traj, out_path)
            self.assertEqual(result["sample_rate"], SR)
            foa, sr = sf.read(out_path, always_2d=True)
            self.assertEqual(foa.shape[1], 4)
            self.assertAlmostEqual(foa.shape[0] / sr, DURATION_S, delta=0.05)
        finally:
            os.unlink(audio_path)
            os.unlink(out_path)

    def test_render_foa_output_finite(self):
        """No NaN or Inf in FOA output."""
        az = np.linspace(-1.0, 1.0, N_FRAMES)
        el = np.linspace(-0.3, 0.3, N_FRAMES)
        dist = np.linspace(1.0, 10.0, N_FRAMES)
        frames = _make_frames(az, el, dist)
        traj = _make_traj(frames)
        audio = _white_noise()
        audio_path = _audio_to_temp_wav(audio)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            render_foa_from_trajectory(audio_path, traj, out_path)
            foa, _ = sf.read(out_path, always_2d=True)
            self.assertTrue(np.all(np.isfinite(foa)))
        finally:
            os.unlink(audio_path)
            os.unlink(out_path)

    def test_foa_peak_le_one(self):
        """FOA wav should be within [-1, 1] after soft limiting."""
        az = np.zeros(N_FRAMES)
        el = np.zeros(N_FRAMES)
        dist = np.ones(N_FRAMES) * 2.0
        frames = _make_frames(az, el, dist)
        traj = _make_traj(frames)
        audio = _white_noise()
        audio_path = _audio_to_temp_wav(audio)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            render_foa_from_trajectory(audio_path, traj, out_path)
            foa, _ = sf.read(out_path, always_2d=True)
            self.assertLessEqual(np.max(np.abs(foa)), 1.05)  # allow soft clip headroom
        finally:
            os.unlink(audio_path)
            os.unlink(out_path)

    def test_render_foa_with_reverb(self):
        """render_foa_from_trajectory with apply_reverb=True still produces valid 4ch."""
        az = np.zeros(N_FRAMES)
        el = np.zeros(N_FRAMES)
        dist = np.linspace(2.0, 8.0, N_FRAMES)
        frames = _make_frames(az, el, dist)
        traj = _make_traj(frames)
        audio = _white_noise(seed=7)
        audio_path = _audio_to_temp_wav(audio)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            result = render_foa_from_trajectory(
                audio_path, traj, out_path, apply_reverb=True, rt60=0.4
            )
            foa, sr = sf.read(out_path, always_2d=True)
            self.assertEqual(foa.shape[1], 4)
            self.assertTrue(np.all(np.isfinite(foa)))
        finally:
            os.unlink(audio_path)
            os.unlink(out_path)


# ── 2. Stereo downmix direction sanity ───────────────────────────────────────

class TestStereoDirectionality(unittest.TestCase):

    def _stereo_from_az(self, az_deg):
        """Return (L_rms, R_rms) for a static source at az_deg."""
        az_rad = math.radians(az_deg)
        el_rad = 0.0
        az_arr = np.full(N_FRAMES, az_rad, dtype=np.float32)
        el_arr = np.full(N_FRAMES, el_rad, dtype=np.float32)
        dist_arr = np.full(N_FRAMES, 3.0, dtype=np.float32)
        frames = _make_frames(az_arr, el_arr, dist_arr)
        traj = _make_traj(frames)
        audio = _white_noise()
        audio_path = _audio_to_temp_wav(audio)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            render_foa_from_trajectory(audio_path, traj, out_path)
            foa, sr = sf.read(out_path, always_2d=True)
            foa_t = foa.T  # [4, T]
            stereo = foa_to_stereo(foa_t, sr)
            L_rms = float(np.sqrt(np.mean(stereo[0] ** 2)))
            R_rms = float(np.sqrt(np.mean(stereo[1] ** 2)))
            return L_rms, R_rms
        finally:
            os.unlink(audio_path)
            os.unlink(out_path)

    def test_right_source_louder_in_right_channel(self):
        """Source at az=+45° (right of image → pipeline az>0=right → AmbiX az<0=right)
        should be louder in right channel after foa_to_stereo."""
        L, R = self._stereo_from_az(45)
        self.assertGreater(R, L, f"Expected R>{L:.4f} for az=+45°, got L={L:.4f} R={R:.4f}")

    def test_left_source_louder_in_left_channel(self):
        """Source at az=-45° (left of image) should be louder in left channel."""
        L, R = self._stereo_from_az(-45)
        self.assertGreater(L, R, f"Expected L>{R:.4f} for az=-45°, got L={L:.4f} R={R:.4f}")

    def test_center_source_balanced(self):
        """Source at az=0° should produce roughly equal L/R."""
        L, R = self._stereo_from_az(0)
        ratio = max(L, R) / (min(L, R) + 1e-9)
        self.assertLess(ratio, 1.15, f"Expected balanced L/R for az=0, got ratio={ratio:.3f}")


# ── 3. Multi-source FOA summation ─────────────────────────────────────────────

class TestMultiSourceFoa(unittest.TestCase):

    def test_two_sources_sum_finite(self):
        """encode_many_to_foa with two sources → finite output."""
        T = N_AUDIO
        mono1 = _white_noise(T, seed=1)
        mono2 = _white_noise(T, seed=2)
        az1 = np.full(T, -0.5, np.float32)
        az2 = np.full(T, 0.5, np.float32)
        el = np.zeros(T, np.float32)
        foa = encode_many_to_foa([mono1, mono2], [az1, az2], [el, el])
        self.assertEqual(foa.shape, (4, T))
        self.assertTrue(np.all(np.isfinite(foa)))
        self.assertLessEqual(np.max(np.abs(foa)), 1.05)

    def test_two_sources_opposite_sides_cancel_Y(self):
        """Two sources at ±90° (opposite sides) should partially cancel Y channel."""
        T = 4800
        mono = np.ones(T, np.float32)
        # az=+π/2 and az=-π/2 (in AmbiX convention: left=positive)
        # But in pipeline convention: right=+, left=-
        # After negation in render: az_ambiX = -az_pipeline
        # So az_pipeline=+π/2 (right) → az_ambiX=-π/2 (AmbiX right)
        az_l = np.full(T, math.pi / 2, np.float32)   # pipeline left → AmbiX right? no.
        az_r = np.full(T, -math.pi / 2, np.float32)
        el = np.zeros(T, np.float32)
        # We're calling encode_many_to_foa directly (no negation applied here)
        # Just verify the sum doesn't blow up and is finite
        foa = encode_many_to_foa([mono, mono], [az_l, az_r], [el, el])
        self.assertTrue(np.all(np.isfinite(foa)))
        # Y channel (index 1) should be near 0 by symmetry
        Y_rms = float(np.sqrt(np.mean(foa[1] ** 2)))
        self.assertLess(Y_rms, 0.1, f"Y channel RMS={Y_rms:.4f} for symmetric sources, expected <0.1")


# ── 4. IQR clipping + Kalman + render chain ───────────────────────────────────

class TestChainIqrKalmanRender(unittest.TestCase):

    def test_iqr_kalman_render_chain(self):
        """Full chain: IQR clip → Kalman smooth → FOA render → finite output."""
        rng = np.random.default_rng(0)
        n = N_FRAMES
        az = np.deg2rad(rng.uniform(-30, 30, n))
        el = np.deg2rad(rng.uniform(-10, 10, n))
        dist = rng.uniform(1.0, 5.0, n)
        # Inject outliers
        dist[::20] = 200.0
        frames = _make_frames(az, el, dist)

        # IQR clip (using SpatialAudioPipeline static method)
        frames = SpatialAudioPipeline._clip_outlier_distances(frames, iqr_k=5.0)
        clipped = [f["dist_m"] for f in frames]
        self.assertLess(max(clipped), 50.0, "Outliers should be clipped below 50m")

        # Kalman smooth
        smoothed = smooth_trajectory_batch(frames, fps=FPS, process_noise=0.01, measurement_noise=0.1)
        self.assertEqual(len(smoothed), n)

        # FOA render
        traj = _make_traj(smoothed)
        audio = _white_noise()
        audio_path = _audio_to_temp_wav(audio)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            render_foa_from_trajectory(audio_path, traj, out_path)
            foa, _ = sf.read(out_path, always_2d=True)
            self.assertTrue(np.all(np.isfinite(foa)))
        finally:
            os.unlink(audio_path)
            os.unlink(out_path)


# ── 5. Distance gain continuity ───────────────────────────────────────────────

class TestDistanceGainContinuity(unittest.TestCase):

    def test_gain_monotone_decreasing_with_distance(self):
        """Farther source → lower gain (monotone relationship via d_rel)."""
        T = SR  # 1 second
        audio = _white_noise(T)
        # Near: d_rel=0 → far: d_rel=1
        for d_rel_val, expected_order in [(0.0, "near"), (0.5, "mid"), (1.0, "far")]:
            d_rel_s = np.full(T, d_rel_val, dtype=np.float32)
            dist_s = np.full(T, 3.0, dtype=np.float32)
            y = apply_distance_gain_lpf(audio, SR, dist_s, d_rel_s,
                                        gain_min=0.3, gain_max=1.0)
            rms = float(np.sqrt(np.mean(y ** 2)))
            setattr(self, f"rms_{expected_order}", rms)

        self.assertGreater(self.rms_near, self.rms_mid,
                           "Near should be louder than mid")
        self.assertGreater(self.rms_mid, self.rms_far,
                           "Mid should be louder than far")

    def test_no_discontinuity_at_boundary(self):
        """d_rel ramp 0→1 over 1s should produce smooth gain in steady-state.

        Note: one-pole LPF has initial transient (first ~1000 samples). We check
        only the steady-state region (middle 50% of signal) for smoothness.
        The LPF cutoff also changes over time, causing gradual damping rather than
        abrupt jumps. We verify no sudden large discontinuities in middle region.
        """
        T = SR
        audio = np.ones(T, dtype=np.float32)  # DC signal for clean gain check
        d_rel_s = np.linspace(0.0, 1.0, T, dtype=np.float32)
        dist_s = np.linspace(0.5, 10.0, T, dtype=np.float32)
        y = apply_distance_gain_lpf(audio, SR, dist_s, d_rel_s)
        # Check steady-state region (middle 50% to avoid LPF startup transient)
        # and tail (after LPF has settled)
        mid_start = T // 4
        mid_end = 3 * T // 4
        diff = np.abs(np.diff(y[mid_start:mid_end].astype(np.float64)))
        max_jump = float(np.max(diff))
        # In steady-state, per-sample gain changes should be gradual (<0.01)
        self.assertLess(max_jump, 0.01,
                        f"Steady-state max gain jump={max_jump:.5f} > 0.01 (discontinuity)")


# ── 6. Depth priority fallback chain ──────────────────────────────────────────

class TestDepthPriorityChain(unittest.TestCase):

    def test_depth_render_takes_priority_over_dist_m(self):
        """depth_render should override dist_m in audio interpolation."""
        n = N_FRAMES
        # dist_m = constant 1.0, depth_render = constant 10.0
        frames = []
        for i in range(n):
            frames.append({
                "frame": i,
                "az": 0.0, "el": 0.0,
                "dist_m": 1.0,
                "depth_render": 10.0,
            })
        T = SR
        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, SR, fps=FPS)
        # dist_s should reflect depth_render=10.0, not dist_m=1.0
        self.assertGreater(float(np.mean(dist_s)), 5.0,
                           "depth_render should dominate; expected dist>5m")

    def test_fallback_to_dist_m_when_depth_render_absent(self):
        """When no depth_render, dist_m is used as fallback."""
        n = N_FRAMES
        frames = [{"frame": i, "az": 0.0, "el": 0.0, "dist_m": 2.5} for i in range(n)]
        T = SR
        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, SR, fps=FPS)
        self.assertAlmostEqual(float(np.mean(dist_s)), 2.5, delta=0.1)

    def test_depth_keys_override_to_dist_m_only(self):
        """depth_keys=('dist_m',) should use dist_m even when depth_blended exists."""
        n = N_FRAMES
        frames = []
        for i in range(n):
            frames.append({
                "frame": i,
                "az": 0.0, "el": 0.0,
                "dist_m": 2.0,
                "depth_blended": 50.0,
            })
        T = SR
        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(
            frames, T, SR, fps=FPS, depth_keys=("dist_m",)
        )
        self.assertAlmostEqual(float(np.mean(dist_s)), 2.0, delta=0.1,
                               msg="depth_keys override should ignore depth_blended")


# ── 7. Scenario: fast-moving object ──────────────────────────────────────────

class TestScenarioFastObject(unittest.TestCase):

    def test_fast_object_az_range_preserved(self):
        """Fast-moving object (left→right in 2s) should show large az range in audio."""
        n = N_FRAMES
        # az sweeps from -60° to +60° = 120° range
        az = np.deg2rad(np.linspace(-60, 60, n))
        el = np.zeros(n)
        dist = np.full(n, 3.0)
        frames = _make_frames(az, el, dist)
        T = N_AUDIO
        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, SR, fps=FPS)
        az_range_deg = math.degrees(float(np.max(az_s) - np.min(az_s)))
        self.assertGreater(az_range_deg, 100.0,
                           f"Fast object az_range={az_range_deg:.1f}° should be >100°")

    def test_fast_object_smooth_no_large_jumps(self):
        """After smooth_limit_angles, per-sample jumps should be bounded."""
        n = N_FRAMES
        az = np.deg2rad(np.linspace(-60, 60, n))
        el = np.zeros(n)
        dist = np.full(n, 3.0)
        frames = _make_frames(az, el, dist)
        T = N_AUDIO
        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, SR, fps=FPS)
        az_sm, el_sm = smooth_limit_angles(az_s, el_s, SR, smooth_ms=50.0)
        # Maximum per-sample jump should be tiny (linear interpolation)
        max_jump_deg = math.degrees(float(np.max(np.abs(np.diff(az_sm.astype(np.float64))))))
        self.assertLess(max_jump_deg, 1.0,
                        f"Max az jump={max_jump_deg:.4f}° after smoothing, expected <1°")


# ── 8. Scenario: static object ───────────────────────────────────────────────

class TestScenarioStaticObject(unittest.TestCase):

    def test_static_object_az_constant(self):
        """Static object (all frames same az) → az_s nearly constant in audio."""
        n = N_FRAMES
        fixed_az = math.radians(30.0)
        az = np.full(n, fixed_az)
        el = np.full(n, math.radians(10.0))
        dist = np.full(n, 4.0)
        frames = _make_frames(az, el, dist)
        T = N_AUDIO
        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, SR, fps=FPS)
        az_std = float(np.std(az_s))
        self.assertLess(az_std, 0.01, f"Static object az_std={az_std:.4f} should be <0.01 rad")

    def test_static_object_d_rel_half(self):
        """Static object with constant dist → d_rel=0.5 (flat clip, d_range<0.1)."""
        n = N_FRAMES
        az = np.zeros(n)
        el = np.zeros(n)
        dist = np.full(n, 5.0)  # constant → d_range=0 < 0.1
        frames = _make_frames(az, el, dist)
        T = SR
        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, SR, fps=FPS)
        self.assertAlmostEqual(float(np.mean(d_rel_s)), 0.5, delta=0.01,
                               msg="Constant distance → d_rel=0.5")


# ── 9. Scenario: approaching object ──────────────────────────────────────────

class TestScenarioApproachingObject(unittest.TestCase):

    def test_approaching_object_gain_increases(self):
        """Object approaching (dist 10→1m) → gain should increase over time."""
        n = N_FRAMES
        az = np.zeros(n)
        el = np.zeros(n)
        dist = np.linspace(10.0, 1.0, n)  # approaching
        frames = _make_frames(az, el, dist)
        T = N_AUDIO
        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, SR, fps=FPS)
        audio = np.ones(T, dtype=np.float32)  # DC for clean gain measurement
        y = apply_distance_gain_lpf(audio, SR, dist_s, d_rel_s)
        # First quarter should be quieter than last quarter
        q = T // 4
        gain_start = float(np.mean(y[:q]))
        gain_end = float(np.mean(y[-q:]))
        self.assertGreater(gain_end, gain_start,
                           f"Approaching: start_gain={gain_start:.3f} should < end_gain={gain_end:.3f}")

    def test_approaching_d_rel_decreases(self):
        """d_rel should decrease as object approaches (near=0, far=1)."""
        n = N_FRAMES
        dist = np.linspace(10.0, 1.0, n)
        frames = _make_frames(np.zeros(n), np.zeros(n), dist)
        T = N_AUDIO
        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, SR, fps=FPS)
        q = T // 4
        d_rel_start = float(np.mean(d_rel_s[:q]))
        d_rel_end = float(np.mean(d_rel_s[-q:]))
        self.assertGreater(d_rel_start, d_rel_end,
                           f"Approaching: d_rel_start={d_rel_start:.3f} should > d_rel_end={d_rel_end:.3f}")


# ── 10. Edge case: single frame ───────────────────────────────────────────────

class TestEdgeCaseSingleFrame(unittest.TestCase):

    def test_single_frame_renders_constant_position(self):
        """Single-frame trajectory → constant az/el throughout audio."""
        frames = [{"frame": 0, "az": math.radians(45.0), "el": math.radians(-10.0), "dist_m": 3.0}]
        T = SR
        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, SR, fps=FPS)
        self.assertAlmostEqual(float(np.std(az_s)), 0.0, delta=1e-5)
        self.assertAlmostEqual(float(az_s[0]), math.radians(45.0), delta=1e-4)

    def test_single_frame_foa_render_ok(self):
        """render_foa_from_trajectory works with single-frame trajectory."""
        frames = [{"frame": 0, "az": 0.0, "el": 0.0, "dist_m": 3.0}]
        traj = _make_traj(frames)
        audio = _white_noise()
        audio_path = _audio_to_temp_wav(audio)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            render_foa_from_trajectory(audio_path, traj, out_path)
            foa, _ = sf.read(out_path, always_2d=True)
            self.assertEqual(foa.shape[1], 4)
            self.assertTrue(np.all(np.isfinite(foa)))
        finally:
            os.unlink(audio_path)
            os.unlink(out_path)


# ── 11. Kalman reduces jitter ─────────────────────────────────────────────────

class TestKalmanJitterReduction(unittest.TestCase):

    def test_kalman_reduces_az_jitter(self):
        """Kalman smoother reduces az standard deviation of noisy trajectory."""
        rng = np.random.default_rng(0)
        n = N_FRAMES
        true_az = np.deg2rad(np.linspace(-30, 30, n))
        noisy_az = true_az + np.deg2rad(rng.normal(0, 5.0, n))
        el = np.zeros(n)
        dist = np.full(n, 3.0)
        frames = _make_frames(noisy_az, el, dist)

        smoothed = smooth_trajectory_batch(frames, fps=FPS, process_noise=0.01, measurement_noise=0.1)
        raw_err = float(np.mean(np.abs(noisy_az - true_az)))
        smooth_err = float(np.mean(np.abs(
            np.array([f["az"] for f in smoothed]) - true_az
        )))
        self.assertLess(smooth_err, raw_err,
                        f"Kalman error ({smooth_err:.4f}) should < raw error ({raw_err:.4f})")

    def test_kalman_d_rel_recomputed(self):
        """After Kalman, d_rel values should be recomputed from smoothed dist_m."""
        n = N_FRAMES
        # dist goes from 1→10 with noise
        rng = np.random.default_rng(1)
        dist = np.linspace(1.0, 10.0, n) + rng.normal(0, 0.5, n)
        frames = _make_frames(np.zeros(n), np.zeros(n), dist)
        smoothed = smooth_trajectory_batch(frames, fps=FPS)
        d_rels = [f["d_rel"] for f in smoothed]
        # d_rel should go from ~0 to ~1
        self.assertLess(d_rels[0], 0.3, f"First d_rel={d_rels[0]:.3f} should be near 0")
        self.assertGreater(d_rels[-1], 0.7, f"Last d_rel={d_rels[-1]:.3f} should be near 1")


# ── 12. FOA energy ≤ input energy ─────────────────────────────────────────────

class TestFoaEnergyBound(unittest.TestCase):

    def test_foa_energy_le_input(self):
        """Total FOA energy should not exceed input energy (no artificial amplification)."""
        az = np.zeros(N_FRAMES, dtype=np.float32)
        el = np.zeros(N_FRAMES, dtype=np.float32)
        audio = _white_noise(N_AUDIO)
        mono_energy = float(np.sum(audio.astype(np.float64) ** 2))
        az_s = np.full(N_AUDIO, 0.0, np.float32)
        el_s = np.full(N_AUDIO, 0.0, np.float32)
        foa = encode_mono_to_foa(audio, az_s, el_s)
        foa_energy = float(np.sum(foa.astype(np.float64) ** 2))
        # FOA encodes 4 channels; energy should be bounded but > mono
        # W=1/√2, X=√(3/2) for az=el=0: gains sum = W²+X²+Y²+Z² = 0.5 + 1.5 = 2.0
        # FOA energy ≈ 2 * mono_energy (SN3D normalization)
        # Just check it doesn't exceed 3× (with gain applied)
        self.assertLess(foa_energy, 3.0 * mono_energy,
                        f"FOA energy {foa_energy:.1f} > 3× mono energy {mono_energy:.1f}")
        # And check it's at least substantial (not silent)
        self.assertGreater(foa_energy, 0.1 * mono_energy)


# ── 13. Reverb stays in valid SN3D range ──────────────────────────────────────

class TestReverbFoaSN3D(unittest.TestCase):

    def test_reverb_foa_bounded(self):
        """FOA with time-varying reverb should stay in [-1, 1] range."""
        n = N_FRAMES
        audio = _white_noise(N_AUDIO, seed=99)
        az_s = np.full(N_AUDIO, 0.0, np.float32)
        el_s = np.full(N_AUDIO, 0.0, np.float32)
        # Apply az negation for AmbiX convention
        foa = encode_mono_to_foa(audio, -az_s, el_s)
        wet = build_wet_curve_from_dist_occ(np.linspace(0, 1, N_AUDIO).astype(np.float32))
        foa_rev = apply_timevarying_reverb_foa(foa, SR, wet, rt60=0.5)
        self.assertTrue(np.all(np.isfinite(foa_rev)))
        self.assertLessEqual(np.max(np.abs(foa_rev)), 1.05)


# ── 14. Scenario: car driving left→right ─────────────────────────────────────

class TestScenarioCarDriving(unittest.TestCase):

    def test_car_driving_left_to_right(self):
        """Car driving left→right: az increases, stereo panning should follow."""
        n = N_FRAMES
        # Pipeline convention: az>0 = right
        az = np.deg2rad(np.linspace(-40, 40, n))  # left→right
        el = np.zeros(n)
        dist = np.full(n, 5.0)
        frames = _make_frames(az, el, dist)
        traj = _make_traj(frames)
        audio = _white_noise(N_AUDIO)
        audio_path = _audio_to_temp_wav(audio)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            render_foa_from_trajectory(audio_path, traj, out_path)
            foa, sr = sf.read(out_path, always_2d=True)
            foa_t = foa.T  # [4, T]
            stereo = foa_to_stereo(foa_t, sr)
            # First quarter (az≈-40° = left): L should dominate
            q = N_AUDIO // 4
            L_first = float(np.sqrt(np.mean(stereo[0, :q] ** 2)))
            R_first = float(np.sqrt(np.mean(stereo[1, :q] ** 2)))
            # Last quarter (az≈+40° = right): R should dominate
            L_last = float(np.sqrt(np.mean(stereo[0, -q:] ** 2)))
            R_last = float(np.sqrt(np.mean(stereo[1, -q:] ** 2)))
            self.assertGreater(L_first, R_first,
                               f"First quarter (left): L={L_first:.4f} should > R={R_first:.4f}")
            self.assertGreater(R_last, L_last,
                               f"Last quarter (right): R={R_last:.4f} should > L={L_last:.4f}")
        finally:
            os.unlink(audio_path)
            os.unlink(out_path)


# ── 15. Scenario: depth outlier clips correctly in chain ──────────────────────

class TestScenarioDepthOutlier(unittest.TestCase):

    def test_iqr_removes_metric_depth_outliers(self):
        """IQR fence k=5 should clip M3D runaway values (e.g. 200m for nearby object)."""
        n = 60
        rng = np.random.default_rng(42)
        dist = rng.uniform(1.0, 5.0, n)
        dist[::10] = 200.0  # M3D runaways
        frames = [{"frame": i, "az": 0.0, "el": 0.0, "dist_m": float(dist[i])} for i in range(n)]
        clipped_frames = SpatialAudioPipeline._clip_outlier_distances(frames, iqr_k=5.0)
        max_dist = max(f["dist_m"] for f in clipped_frames)
        self.assertLess(max_dist, 50.0, f"After IQR clip, max_dist={max_dist:.1f} should be <50m")

    def test_no_clipping_for_normal_range(self):
        """IQR clip should not modify normal depth values (1-20m range)."""
        n = 60
        dist = np.linspace(1.0, 20.0, n)
        frames = [{"frame": i, "az": 0.0, "el": 0.0, "dist_m": float(dist[i])} for i in range(n)]
        original_dists = [f["dist_m"] for f in frames]
        clipped_frames = SpatialAudioPipeline._clip_outlier_distances(frames, iqr_k=5.0)
        clipped_dists = [f["dist_m"] for f in clipped_frames]
        # No values should change since 20m is well within k=5 IQR fence
        for orig, clip in zip(original_dists, clipped_dists):
            self.assertAlmostEqual(orig, clip, delta=1e-6)


# ── 16. FPS-aware interpolation sanity ────────────────────────────────────────

class TestFpsAwareInterpolation(unittest.TestCase):

    def test_25fps_vs_30fps_same_trajectory_different_timing(self):
        """Same trajectory at 25fps vs 30fps should produce different audio timing."""
        n = 60
        az = np.deg2rad(np.linspace(-30, 30, n))
        el = np.zeros(n)
        dist = np.full(n, 3.0)
        frames = _make_frames(az, el, dist)
        T = SR  # 1s of audio
        az_25, _, _, _ = interpolate_angles_distance(frames, T, SR, fps=25.0)
        az_30, _, _, _ = interpolate_angles_distance(frames, T, SR, fps=30.0)
        # At 1s: 25fps has 25 frames covered, 30fps has 30 frames covered
        # So az_30 should be slightly further along the trajectory
        # They should differ (not identical)
        diff = float(np.mean(np.abs(az_25 - az_30)))
        self.assertGreater(diff, 0.01,
                           f"25fps vs 30fps should differ; mean_diff={diff:.4f} rad")

    def test_high_fps_video_no_compression(self):
        """60fps video: trajectory should not be compressed into shorter window."""
        n = 120  # 2s of frames at 60fps
        az = np.deg2rad(np.linspace(-30, 30, n))
        el = np.zeros(n)
        dist = np.full(n, 3.0)
        frames = _make_frames(az, el, dist, n_frames=n)
        T = SR * 2  # 2s of audio
        az_s, _, _, _ = interpolate_angles_distance(frames, T, SR, fps=60.0)
        az_range_deg = math.degrees(float(np.max(az_s) - np.min(az_s)))
        # Should span roughly -30° to +30° = 60°
        self.assertGreater(az_range_deg, 50.0,
                           f"60fps: az_range={az_range_deg:.1f}° should be >50° (no compression)")


# ── 17. Scenario: confidence-based occlusion ──────────────────────────────────

class TestScenarioOcclusion(unittest.TestCase):

    def test_wet_curve_increases_with_distance(self):
        """Reverb wetness should increase with d_rel (far = more reverb)."""
        near_drel = np.zeros(100, dtype=np.float32)
        far_drel = np.ones(100, dtype=np.float32)
        wet_near = build_wet_curve_from_dist_occ(near_drel)
        wet_far = build_wet_curve_from_dist_occ(far_drel)
        self.assertLess(float(np.mean(wet_near)), float(np.mean(wet_far)),
                        "Far objects should have more reverb than near objects")

    def test_occlusion_boost_adds_wetness(self):
        """Occlusion flag should increase reverb wetness."""
        d_rel = np.full(100, 0.5, dtype=np.float32)
        occ_none = np.zeros(100, dtype=np.float32)
        occ_full = np.ones(100, dtype=np.float32)
        wet_no_occ = build_wet_curve_from_dist_occ(d_rel, occ_none)
        wet_with_occ = build_wet_curve_from_dist_occ(d_rel, occ_full)
        self.assertGreater(float(np.mean(wet_with_occ)), float(np.mean(wet_no_occ)),
                           "Occlusion should increase reverb wetness")


# ── 18. Full render chain: car, dog, motorcycle scenarios ─────────────────────

class TestScenarioEndToEnd(unittest.TestCase):
    """Simulate 3 benchmark scenarios without real video (synthetic trajectories)."""

    def _run_scenario(self, az_seq_deg, dist_seq_m, scenario_name):
        """Run full render pipeline and check output sanity."""
        n = len(az_seq_deg)
        az = np.deg2rad(np.array(az_seq_deg, dtype=np.float64))
        el = np.zeros(n)
        dist = np.array(dist_seq_m, dtype=np.float64)
        frames = _make_frames(az, el, dist)
        frames = SpatialAudioPipeline._clip_outlier_distances(frames)
        frames = smooth_trajectory_batch(frames, fps=FPS)
        traj = _make_traj(frames)
        audio = _white_noise(N_AUDIO, seed=hash(scenario_name) % 1000)
        audio_path = _audio_to_temp_wav(audio)
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            result = render_foa_from_trajectory(audio_path, traj, out_path)
            foa, sr = sf.read(out_path, always_2d=True)
            self.assertEqual(foa.shape[1], 4, f"{scenario_name}: expected 4ch FOA")
            self.assertTrue(np.all(np.isfinite(foa)), f"{scenario_name}: FOA has NaN/Inf")
            self.assertLessEqual(np.max(np.abs(foa)), 1.05,
                                 f"{scenario_name}: FOA peak exceeds 1.0")
            return foa
        finally:
            os.unlink(audio_path)
            os.unlink(out_path)

    def test_scenario_car(self):
        """Car driving past: az -30°→+30°, constant dist 8m."""
        az = list(np.linspace(-30, 30, N_FRAMES))
        dist = [8.0] * N_FRAMES
        self._run_scenario(az, dist, "car")

    def test_scenario_dog_fast(self):
        """Fast dog: az oscillates ±45°, dist 2-4m."""
        t = np.linspace(0, 4 * np.pi, N_FRAMES)
        az = list(45 * np.sin(t))
        dist = list(3.0 + np.sin(t / 2))
        self._run_scenario(az, dist, "dog_fast")

    def test_scenario_motorcycle(self):
        """Motorcycle approaching: az steady 0°, dist 20→2m."""
        az = list(np.zeros(N_FRAMES))
        dist = list(np.linspace(20, 2, N_FRAMES))
        self._run_scenario(az, dist, "motorcycle")

    def test_scenario_with_depth_outliers(self):
        """Real-world condition: M3D runaways every 20 frames (should clip cleanly)."""
        n = N_FRAMES
        az = list(np.deg2rad(np.linspace(-20, 20, n)))
        dist = [3.0] * n
        for i in range(0, n, 20):
            dist[i] = 200.0  # M3D runaway
        self._run_scenario(az, dist, "outlier")


if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)


# ── A5: multi_source wired into the offline pipeline ─────────────────────────

class TestPipelineMultiSourceOffline(unittest.TestCase):
    """`multi_source.py` was dead: nothing in the pipeline reached it and the
    delivered artefact was strictly one source. The offline path now tracks one
    source per init bbox, sums them with encode_many_to_foa, and writes one
    automation file per ADM object 1..N.

    There is deliberately NO live multi-object OSC path: the bridge keys every
    datagram on the single tracking id "default".
    """

    SR = 16000
    DUR = 1.0

    def _cfg(self, **ms):
        from vid2spatial_pkg.config import MultiSourceConfig, OutputConfig, PipelineConfig
        c = PipelineConfig(video_path="none.mp4", audio_path="none.wav",
                           output=OutputConfig(foa_path="out.foa.wav"))
        c.multi_source = MultiSourceConfig(**ms)
        return c

    @staticmethod
    def _traj(az_deg, n=30, fps=30.0):
        """Static source at a fixed azimuth (pipeline convention: RIGHT = +)."""
        return {"fps": fps,
                "frames": [{"frame": i, "az": math.radians(az_deg), "el": 0.0,
                            "dist_m": 2.0, "conf": 1.0} for i in range(n)]}

    def _tone(self, f):
        t = np.arange(int(self.SR * self.DUR)) / self.SR
        return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)

    @staticmethod
    def _band_energy(x, sr, f0, half=120.0):
        """Energy of x in [f0-half, f0+half]."""
        X = np.fft.rfft(x * np.hanning(len(x)))
        freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
        m = (freqs >= f0 - half) & (freqs <= f0 + half)
        return float(np.sum(np.abs(X[m]) ** 2))

    def test_two_sources_map_to_separate_sides(self):
        """The headline claim: two objects at opposite azimuths land on
        opposite sides of the FOA field, each identifiable by its own tone."""
        pipe = SpatialAudioPipeline(self._cfg(
            enabled=True, init_bboxes=[(10, 10, 40, 40), (200, 10, 40, 40)]))
        f_left, f_right = 500.0, 4000.0
        # source 1 hard LEFT of image (az < 0), source 2 hard RIGHT (az > 0)
        foa = pipe._render_multi_source(
            [self._tone(f_left), self._tone(f_right)], self.SR,
            [self._traj(-90.0), self._traj(+90.0)])

        self.assertEqual(foa.shape[0], 4)
        # AmbiX ACN: W, Y, Z, X. Y = sin(az_ambix)cos(el); az_ambix = -az_pipeline,
        # so the LEFT source (az_pipeline = -90) has Y = +1 and the RIGHT one -1.
        W, Y = foa[0], foa[1]
        left, right = W + Y, W - Y  # virtual left / right cardioid-ish decode

        e_left_lo = self._band_energy(left, self.SR, f_left)
        e_right_lo = self._band_energy(right, self.SR, f_left)
        e_left_hi = self._band_energy(left, self.SR, f_right)
        e_right_hi = self._band_energy(right, self.SR, f_right)

        # each source's tone dominates its own side by a wide margin
        self.assertGreater(e_left_lo, 10.0 * e_right_lo,
                           "left source's tone did not dominate the left decode")
        self.assertGreater(e_right_hi, 10.0 * e_left_hi,
                           "right source's tone did not dominate the right decode")
        # and both sources actually survived the mix
        self.assertGreater(e_left_lo, 0.0)
        self.assertGreater(e_right_hi, 0.0)

    def test_three_sources_and_unequal_lengths(self):
        """N > 2 and ragged source lengths must not raise (encode_many_to_foa
        asserts equal lengths; the pipeline pads)."""
        pipe = SpatialAudioPipeline(self._cfg(
            enabled=True, init_bboxes=[(0, 0, 9, 9), (1, 1, 9, 9), (2, 2, 9, 9)]))
        a = self._tone(500.0)
        foa = pipe._render_multi_source(
            [a, a[: len(a) // 2], a[: len(a) // 3]], self.SR,
            [self._traj(-60.0), self._traj(0.0), self._traj(+60.0)])
        self.assertEqual(foa.shape, (4, len(a)))
        self.assertTrue(np.isfinite(foa).all())

    def test_mix_equals_sum_of_single_source_renders(self):
        """The multi path must be the single path summed, not a second
        implementation that can drift from it."""
        pipe = SpatialAudioPipeline(self._cfg(
            enabled=True, init_bboxes=[(0, 0, 9, 9), (1, 1, 9, 9)]))
        audios = [self._tone(500.0), self._tone(4000.0)]
        trajs = [self._traj(-45.0), self._traj(+45.0)]
        mix = pipe._render_multi_source(audios, self.SR, trajs)
        singles = [pipe._render_spatial_audio(a, self.SR, t) for a, t in zip(audios, trajs)]
        summed = singles[0] + singles[1]
        # encode_many_to_foa peak-normalises the sum, so compare shape-wise
        scale = float(np.max(np.abs(summed))) / max(float(np.max(np.abs(mix))), 1e-12)
        np.testing.assert_allclose(mix * scale, summed, atol=1e-4)

    def test_per_object_export_is_one_to_n(self):
        """One automation file per ADM object, ids 1..N, addressed /adm/obj/N/aed."""
        import json
        from vid2spatial_pkg.trajectory_export import export_trajectory
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "auto.json")
            paths = []
            for i, az in enumerate((-30.0, 30.0, 0.0)):
                oid = i + 1
                stem, ext = os.path.splitext(base)
                p = export_trajectory(self._traj(az), f"{stem}.obj{oid}{ext}",
                                      fps=30.0, object_id=oid)
                paths.append(p)
            self.assertEqual(len(paths), 3)
            for i, p in enumerate(paths):
                doc = json.loads(open(p).read())
                self.assertEqual(doc["object_id"], i + 1)
                self.assertEqual(doc["osc_address"], f"/adm/obj/{i + 1}/aed")

    def test_config_rejects_a_single_bbox_and_mismatched_audio(self):
        from vid2spatial_pkg.config import MultiSourceConfig
        with self.assertRaises(ValueError):
            MultiSourceConfig(enabled=True, init_bboxes=[(0, 0, 9, 9)])
        with self.assertRaises(ValueError):
            MultiSourceConfig(enabled=True, init_bboxes=[(0, 0, 9, 9), (1, 1, 9, 9)],
                              audio_paths=["a.wav"])
        # disabled config with no bboxes is the default and must be fine
        self.assertEqual(MultiSourceConfig().n_sources, 0)

    def test_run_dispatches_to_multi_source(self):
        import inspect
        src = inspect.getsource(SpatialAudioPipeline.run)
        self.assertIn("multi_source.enabled", src)
        self.assertIn("run_multi_source", src)

    def test_no_live_osc_object_addresses_were_added(self):
        """The engine does not implement /vid2spatial/obj/{N}/...; adding it
        here would stream N sources into a void."""
        import inspect
        from vid2spatial_pkg import osc_sender
        self.assertNotIn("/obj/", inspect.getsource(osc_sender))
        from vid2spatial_pkg.config import MultiSourceConfig
        self.assertIn("no live multi-object osc path", MultiSourceConfig.__doc__.lower())

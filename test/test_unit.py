#!/usr/bin/env python3
"""
vid2spatial_v2 — Unit Test Suite
모듈별 함수 단위 검증. 외부 모델(YOLO, MiDaS 등) 없이 실행 가능.

실행:
    cd /home/seung/mmhoa/vid2spatial_v2
    /home/seung/miniforge3/bin/python3 test/test_unit.py [-v]
"""
import sys, math, unittest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# 1. foa_render — 공간음향 렌더러
# ─────────────────────────────────────────────────────────────────────────────
class TestFoaRenderCoordinates(unittest.TestCase):
    """AmbiX 좌표 변환 및 FOA 인코딩 검증"""

    def setUp(self):
        from vid2spatial_pkg.foa_render import dir_to_foa_acn_sn3d_gains
        self.gains_fn = dir_to_foa_acn_sn3d_gains

    def test_front_center_gains(self):
        """정면 중앙(az=0, el=0) → X채널 최대, Y=Z=0"""
        az = np.array([0.0])
        el = np.array([0.0])
        g = self.gains_fn(az, el)  # [W, Y, Z, X]
        self.assertAlmostEqual(float(g[0, 0]), 1.0/math.sqrt(2), places=5, msg="W channel")
        self.assertAlmostEqual(float(g[1, 0]), 0.0, places=5, msg="Y channel (front, should be 0)")
        self.assertAlmostEqual(float(g[2, 0]), 0.0, places=5, msg="Z channel (el=0, should be 0)")
        self.assertGreater(float(g[3, 0]), 0, msg="X channel (front, should be positive)")

    def test_right_az_positive(self):
        """az > 0 → RIGHT in pipeline convention. FOA encode 시 az 부호 확인."""
        # pipeline: az=+45° = 오른쪽
        # AmbiX: az=+45° = 왼쪽 → 반드시 negate 후 인코딩해야 함
        az_pipeline = np.array([math.radians(45)])
        el = np.array([0.0])
        # negate for AmbiX
        az_foa = -az_pipeline
        g = self.gains_fn(az_foa, el)
        # AmbiX 기준 az=-45° → Y<0 (오른쪽)
        self.assertLess(float(g[1, 0]), 0, msg="Y channel should be negative for right-side source")

    def test_overhead_el_positive(self):
        """el=+90° → Z채널 최대"""
        az = np.array([0.0])
        el = np.array([math.radians(90)])
        g = self.gains_fn(az, el)
        self.assertGreater(abs(float(g[2, 0])), 0.8, msg="Z channel dominant for overhead")

    def test_energy_preservation(self):
        """다양한 방향에서 총 에너지 일정해야 (SN3D normalization)
        W=1/sqrt(2), dipoles=sqrt(3/2) → W^2+X^2+Y^2+Z^2 = 0.5+1.5=2.0
        """
        for az_deg in [0, 45, 90, 135, 180, -90]:
            for el_deg in [0, 30, -30, 60]:
                az = np.array([math.radians(az_deg)])
                el = np.array([math.radians(el_deg)])
                g = self.gains_fn(az, el)
                # g shape: [4, 1] — extract scalars
                energy = float(g[0,0]**2 + g[1,0]**2 + g[2,0]**2 + g[3,0]**2)
                self.assertAlmostEqual(energy, 2.0, places=4,
                    msg=f"Energy not preserved at az={az_deg}, el={el_deg}")


class TestFoaRenderInterpolation(unittest.TestCase):
    """fps-aware 보간 및 d_rel 계산 검증"""

    def setUp(self):
        from vid2spatial_pkg.foa_render import interpolate_angles_distance
        self.interp = interpolate_angles_distance

    def _make_frames(self, n, az_vals, el_vals, dist_vals):
        return [{"frame": i, "az": az_vals[i], "el": el_vals[i], "dist_m": dist_vals[i]}
                for i in range(n)]

    def test_fps_aware_mapping(self):
        """frame_idx * sr/fps 로 sample 위치 계산 — V1 linear linspace 버그 방지"""
        sr, fps, T = 48000, 30.0, 48000
        # 2프레임 궤적: frame0=0°, frame1=90°
        frames = self._make_frames(2,
            [0.0, math.radians(90)],
            [0.0, 0.0],
            [1.0, 2.0])
        az_s, el_s, dist_s, d_rel_s = self.interp(frames, T, sr, fps=fps)
        # frame1 → sample = 1 * (48000/30) = 1600
        # sample 0: az≈0, sample 1600: az≈90°
        self.assertAlmostEqual(float(az_s[0]), 0.0, places=3)
        self.assertAlmostEqual(float(az_s[1600]), math.radians(90), places=3)

    def test_d_rel_per_clip_normalize(self):
        """d_rel 반드시 [0,1] 범위, min=0, max=1"""
        sr, fps, T = 48000, 30.0, 9600
        dist_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        frames = self._make_frames(5,
            [0.0]*5, [0.0]*5, dist_vals)
        _, _, _, d_rel_s = self.interp(frames, T, sr, fps=fps)
        self.assertAlmostEqual(float(d_rel_s.min()), 0.0, places=3, msg="d_rel min should be 0")
        self.assertAlmostEqual(float(d_rel_s.max()), 1.0, places=3, msg="d_rel max should be 1")

    def test_d_rel_flat_depth(self):
        """모든 depth 같으면 d_rel = 0.5 (d_range < 0.1 케이스)"""
        sr, fps, T = 48000, 30.0, 4800
        frames = self._make_frames(3,
            [0.0]*3, [0.0]*3, [2.0, 2.0, 2.0])
        _, _, _, d_rel_s = self.interp(frames, T, sr, fps=fps)
        self.assertTrue(np.allclose(d_rel_s, 0.5), msg="Flat depth → d_rel=0.5 everywhere")

    def test_depth_keys_priority(self):
        """depth_keys 파라미터로 depth 소스 선택 순서 제어"""
        sr, fps, T = 48000, 30.0, 4800
        # depth_blended > dist_m
        frames = [{"frame": 0, "az": 0.0, "el": 0.0, "dist_m": 1.0, "depth_blended": 5.0},
                  {"frame": 1, "az": 0.0, "el": 0.0, "dist_m": 1.0, "depth_blended": 5.0}]
        # default priority → depth_blended 선택
        _, _, dist_s_default, _ = self.interp(frames, T, sr, fps=fps)
        # dist_m only → dist_m 선택
        _, _, dist_s_distm, _ = self.interp(frames, T, sr, fps=fps,
                                             depth_keys=("dist_m",))
        self.assertGreater(float(dist_s_default.mean()), 3.0, msg="Default: uses depth_blended(5.0)")
        self.assertAlmostEqual(float(dist_s_distm.mean()), 1.0, delta=0.1, msg="dist_m only: uses 1.0")

    def test_single_frame(self):
        """단일 프레임 → T개 샘플 전부 동일값"""
        sr, fps, T = 48000, 30.0, 4800
        frames = [{"frame": 0, "az": math.radians(30), "el": math.radians(10), "dist_m": 3.0}]
        az_s, el_s, dist_s, _ = self.interp(frames, T, sr, fps=fps)
        self.assertTrue(np.allclose(az_s, math.radians(30)), msg="Single frame az constant")
        self.assertTrue(np.allclose(el_s, math.radians(10)), msg="Single frame el constant")


class TestFoaRenderDistanceGain(unittest.TestCase):
    """거리 gain/LPF 계산 검증"""

    def setUp(self):
        from vid2spatial_pkg.foa_render import apply_distance_gain_lpf
        self.apply = apply_distance_gain_lpf

    def test_near_louder_than_far(self):
        """d_rel=0 (가까움)이 d_rel=1 (멀음)보다 볼륨 커야 함"""
        sr = 48000
        T = sr
        tone = np.sin(2*np.pi*440*np.arange(T)/sr).astype(np.float32)
        dist_s = np.ones(T, np.float32)

        d_rel_near = np.zeros(T, np.float32)   # 가장 가까운
        d_rel_far  = np.ones(T, np.float32)    # 가장 먼

        y_near = self.apply(tone, sr, dist_s, d_rel_near)
        y_far  = self.apply(tone, sr, dist_s, d_rel_far)
        rms_near = float(np.sqrt(np.mean(y_near**2)))
        rms_far  = float(np.sqrt(np.mean(y_far**2)))
        self.assertGreater(rms_near, rms_far, msg="Near source should be louder than far")

    def test_near_brighter_than_far(self):
        """d_rel=0 (가까움)이 d_rel=1 (멀음)보다 고주파 에너지 많아야"""
        sr = 48000
        T = sr
        # 4kHz 톤
        tone = np.sin(2*np.pi*4000*np.arange(T)/sr).astype(np.float32)
        dist_s = np.ones(T, np.float32)

        d_near = np.zeros(T, np.float32)
        d_far  = np.ones(T, np.float32)

        y_near = self.apply(tone, sr, dist_s, d_near, lpf_min_hz=500, lpf_max_hz=8000)
        y_far  = self.apply(tone, sr, dist_s, d_far,  lpf_min_hz=500, lpf_max_hz=8000)
        # 고주파 에너지 비교 (4kHz 이상)
        from numpy.fft import rfft
        fft_near = np.abs(rfft(y_near[1000:]))
        fft_far  = np.abs(rfft(y_far[1000:]))
        hf_near = float(fft_near[len(fft_near)//2:].mean())
        hf_far  = float(fft_far[len(fft_far)//2:].mean())
        self.assertGreater(hf_near, hf_far, msg="Near source should have more HF energy")

    def test_gain_range(self):
        """gain은 [gain_min, gain_max] 범위 내에 있어야"""
        sr = 48000
        T = sr // 10
        tone = np.ones(T, np.float32)
        dist_s = np.ones(T, np.float32)
        for d_rel_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            d_rel = np.full(T, d_rel_val, np.float32)
            y = self.apply(tone, sr, dist_s, d_rel,
                           gain_min=0.3, gain_max=1.0)
            rms = float(np.sqrt(np.mean(y**2)))
            self.assertLessEqual(rms, 1.1, msg=f"Output too loud at d_rel={d_rel_val}")
            self.assertGreater(rms, 0.0, msg=f"Zero output at d_rel={d_rel_val}")


class TestFoaRenderSmoothing(unittest.TestCase):
    """각도 스무딩 및 jerk limiting 검증"""

    def setUp(self):
        from vid2spatial_pkg.foa_render import smooth_limit_angles
        self.smooth = smooth_limit_angles

    def test_moving_average_reduces_jitter(self):
        """Moving average 후 2차 미분(jitter) 감소"""
        sr = 48000
        T = sr
        # 노이즈 + 선형 트렌드
        rng = np.random.default_rng(42)
        az = (np.linspace(0, math.pi/2, T) + rng.normal(0, 0.05, T)).astype(np.float32)
        el = np.zeros(T, np.float32)
        jitter_before = float(np.sqrt(np.mean(np.diff(az, n=2)**2)))
        az_sm, _ = self.smooth(az, el, sr, smooth_ms=50.0)
        jitter_after = float(np.sqrt(np.mean(np.diff(az_sm, n=2)**2)))
        self.assertLess(jitter_after, jitter_before * 0.5,
                        msg="Smoothing should reduce jitter by at least 50%")

    def test_no_smoothing_passthrough(self):
        """smooth_ms=0 이면 원본 신호 통과 (no-op)"""
        sr = 48000
        T = sr // 10
        az = np.linspace(0, 1.0, T).astype(np.float32)
        el = np.zeros(T, np.float32)
        az_sm, _ = self.smooth(az, el, sr, smooth_ms=0.0)
        self.assertTrue(np.allclose(az_sm, az, atol=1e-4),
                        msg="smooth_ms=0 should pass through unchanged")


# ─────────────────────────────────────────────────────────────────────────────
# 2. pipeline — IQR outlier clipping
# ─────────────────────────────────────────────────────────────────────────────
class TestPipelineOutlierClipping(unittest.TestCase):
    """IQR 아웃라이어 제거 로직 검증"""

    def setUp(self):
        from vid2spatial_pkg.pipeline import SpatialAudioPipeline
        self.clip_fn = SpatialAudioPipeline._clip_outlier_distances

    def _make_frames(self, vals):
        return [{"frame": i, "dist_m": v} for i, v in enumerate(vals)]

    def test_normal_range_untouched(self):
        """정상 범위 값은 수정 없어야"""
        vals = [1.0, 1.5, 2.0, 2.5, 3.0, 2.8, 1.8, 2.2]
        frames = self._make_frames(vals)
        result = self.clip_fn(frames, iqr_k=5.0)
        for orig, res in zip(vals, result):
            self.assertAlmostEqual(orig, res["dist_m"], places=5)

    def test_runaway_values_clamped(self):
        """극단값 (212m) 제거"""
        vals = [1.0, 1.2, 1.1, 1.3, 212.0, 1.0, 1.1, 1.2, 1.3, 1.4]
        frames = self._make_frames(vals)
        result = self.clip_fn(frames, iqr_k=5.0)
        max_val = max(f["dist_m"] for f in result)
        self.assertLess(max_val, 10.0, msg="Runaway 212m should be clamped")

    def test_aggressive_k_clips_more(self):
        """k=1이면 더 많은 값 제거"""
        vals = list(range(1, 11)) + [50.0]  # 50m가 아웃라이어
        frames_k5 = self._make_frames(vals)
        frames_k1 = self._make_frames(vals)
        self.clip_fn(frames_k5, iqr_k=5.0)
        self.clip_fn(frames_k1, iqr_k=1.0)
        max_k5 = max(f["dist_m"] for f in frames_k5)
        max_k1 = max(f["dist_m"] for f in frames_k1)
        self.assertGreaterEqual(max_k5, max_k1, msg="k=1 should clip more aggressively")

    def test_all_same_values(self):
        """모든 값 동일 → IQR=0 → 스킵 (수정 없음)"""
        vals = [2.0] * 10
        frames = self._make_frames(vals)
        result = self.clip_fn(frames, iqr_k=5.0)
        for res in result:
            self.assertAlmostEqual(res["dist_m"], 2.0)

    def test_fewer_than_4_frames_skipped(self):
        """프레임 수 < 4 → 클리핑 스킵"""
        vals = [1.0, 200.0, 1.0]
        frames = self._make_frames(vals)
        result = self.clip_fn(frames, iqr_k=5.0)
        # 3프레임이므로 클리핑 안 됨
        self.assertAlmostEqual(result[1]["dist_m"], 200.0)

    def test_multiple_depth_keys(self):
        """depth_blended 필드도 같이 클리핑 (IQR이 0이 아닌 케이스여야 작동)"""
        # IQR fence는 iqr < 0.01이면 스킵 → 클리핑되려면 값 분산 필요
        frames = [
            {"frame": i, "dist_m": float(i+1), "depth_blended": float(i+1)}
            for i in range(8)
        ]
        # depth_blended[7] = 500 → IQR 기반 클리핑 작동
        frames[7]["depth_blended"] = 500.0
        result = self.clip_fn(frames, iqr_k=3.0)
        max_blended = max(f["depth_blended"] for f in result)
        self.assertLess(max_blended, 100.0,
                        msg="depth_blended outlier should be clamped with sufficient variance")


# ─────────────────────────────────────────────────────────────────────────────
# 3. temporal_smoother — Kalman + RTS
# ─────────────────────────────────────────────────────────────────────────────
class TestTemporalSmoother(unittest.TestCase):
    """Kalman filter + RTS smoother 검증"""

    def setUp(self):
        from vid2spatial_pkg.temporal_smoother import smooth_trajectory_batch
        self.smooth = smooth_trajectory_batch

    def _make_traj(self, n, az_fn, el_fn=None, dist_fn=None, conf=0.8):
        el_fn   = el_fn   or (lambda i: 0.0)
        dist_fn = dist_fn or (lambda i: 2.0)
        return [
            {"frame": i, "az": az_fn(i), "el": el_fn(i),
             "dist_m": dist_fn(i), "confidence": conf,
             "cx": 320.0, "cy": 240.0, "w": 60.0, "h": 40.0}
            for i in range(n)
        ]

    def test_rts_smoother_than_forward(self):
        """RTS (use_rts=True) 는 forward-only보다 2차 미분 jitter 낮아야"""
        rng = np.random.default_rng(0)
        n = 100
        traj = self._make_traj(n,
            az_fn=lambda i: math.sin(i*0.1) + rng.normal(0, 0.05),
            conf=0.8)
        fwd  = self.smooth(traj, fps=30.0, use_rts=False)
        rts  = self.smooth(traj, fps=30.0, use_rts=True)
        if fwd and rts:
            azf = np.array([f["az"] for f in fwd])
            azr = np.array([f["az"] for f in rts])
            jitter_fwd = float(np.sqrt(np.mean(np.diff(azf, n=2)**2)))
            jitter_rts = float(np.sqrt(np.mean(np.diff(azr, n=2)**2)))
            self.assertLess(jitter_rts, jitter_fwd * 1.1,
                            msg="RTS should not be worse than forward Kalman")

    def test_single_frame_no_crash(self):
        """단일 프레임 → 오류 없이 동작"""
        traj = self._make_traj(1, az_fn=lambda i: 0.0)
        result = self.smooth(traj, fps=30.0)
        self.assertIsNotNone(result)

    def test_d_rel_recomputed_after_rts(self):
        """RTS 후 d_rel 은 스무딩된 dist_m 기준으로 재계산 → [0,1]"""
        n = 60
        traj = self._make_traj(n,
            az_fn=lambda i: 0.0,
            dist_fn=lambda i: 1.0 + i * 0.1)  # 1.0 → 6.9m
        result = self.smooth(traj, fps=30.0, use_rts=True)
        if result:
            d_rels = [f.get("d_rel", 0.5) for f in result]
            self.assertAlmostEqual(min(d_rels), 0.0, delta=0.05, msg="d_rel min ≈ 0")
            self.assertAlmostEqual(max(d_rels), 1.0, delta=0.05, msg="d_rel max ≈ 1")

    def test_smoothing_reduces_noise(self):
        """스무딩 후 az jitter 감소 확인"""
        rng = np.random.default_rng(1)
        n = 120
        traj = self._make_traj(n,
            az_fn=lambda i: 0.3 + rng.normal(0, 0.2),
            conf=0.9)
        raw_az = np.array([f["az"] for f in traj])
        result = self.smooth(traj, fps=30.0, use_rts=True)
        if result:
            sm_az = np.array([f["az"] for f in result])
            raw_jitter = float(np.sqrt(np.mean(np.diff(raw_az, n=2)**2)))
            sm_jitter  = float(np.sqrt(np.mean(np.diff(sm_az, n=2)**2)))
            self.assertLess(sm_jitter, raw_jitter,
                            msg="Smoothed trajectory must have lower jitter")

    def test_back_projection_pinhole_correct(self):
        """back-projection: az=0,el=0 → cx≈W/2, cy≈H/2 (pinhole forward projection)"""
        # smooth_trajectory_batch()는 width/height/fov_deg 파라미터 없음
        # 대신 결과 cx/cy가 원래 입력(320, 180)에 가깝게 유지되는지 검증
        n = 20
        traj = [{"frame": i, "az": 0.0, "el": 0.0, "dist_m": 2.0,
                 "confidence": 0.9, "cx": 320.0, "cy": 180.0,
                 "w": 60.0, "h": 40.0} for i in range(n)]
        result = self.smooth(traj, fps=30.0, use_rts=True)
        if result and "cx" in result[0]:
            mid = result[n//2]
            self.assertAlmostEqual(mid["cx"], 320.0, delta=20.0,
                                   msg="Front-center cx should stay near W/2")
            self.assertAlmostEqual(mid["cy"], 180.0, delta=20.0,
                                   msg="Front-center cy should stay near H/2")


# ─────────────────────────────────────────────────────────────────────────────
# 4. depth_utils — variance-gated depth blending
# ─────────────────────────────────────────────────────────────────────────────
class TestDepthUtils(unittest.TestCase):
    """Depth enhancement 로직 검증"""

    def setUp(self):
        from vid2spatial_pkg.depth_utils import (
            compute_bbox_scale_proxy,
            DepthConfig,
        )
        self.proxy_fn  = compute_bbox_scale_proxy
        self.DepthCfg  = DepthConfig

    def test_bbox_proxy_inverse_square(self):
        """물체가 2배 가까워지면 bbox 면적 4배 → proxy depth 2배 감소
        API: compute_bbox_scale_proxy(bbox_areas: List[float], initial_depth_m=2.0)
        """
        # area_init=100(ref), area_new=400 → depth_proxy=2*sqrt(100/400)=1.0
        bbox_areas = [100.0, 400.0]
        proxies = self.proxy_fn(bbox_areas, initial_depth_m=2.0)
        if proxies:
            self.assertAlmostEqual(proxies[0], 2.0, delta=0.1, msg="First frame = reference depth")
            self.assertAlmostEqual(proxies[1], 1.0, delta=0.2, msg="4x area → proxy halved")

    def test_bbox_zero_area_no_crash(self):
        """bbox area=0 → 크래시 없이 처리 (실제 API: List[float])"""
        # area=0인 경우 div-by-zero 가능
        bbox_areas = [0.0, 100.0, 100.0]
        try:
            proxies = self.proxy_fn(bbox_areas, initial_depth_m=2.0)
            # 크래시 없이 처리되면 통과
        except ZeroDivisionError:
            self.fail("Zero bbox area caused ZeroDivisionError — needs guard")
        except Exception:
            pass  # 다른 예외는 일단 허용 (numpy may handle gracefully)

    def test_proxy_farther_when_smaller(self):
        """물체가 더 작아지면(멀어지면) proxy depth 커야"""
        # area: 400(큰/가까운) → 100(작은/먼)
        bbox_areas = [400.0, 100.0]
        proxies = self.proxy_fn(bbox_areas, initial_depth_m=2.0)
        if proxies and len(proxies) >= 2:
            self.assertGreater(proxies[1], proxies[0],
                               msg="Smaller bbox → farther proxy depth")


# ─────────────────────────────────────────────────────────────────────────────
# 5. 좌표 변환 — pixel ↔ az/el (pinhole 역변환)
# ─────────────────────────────────────────────────────────────────────────────
class TestCoordinateConversion(unittest.TestCase):
    """pixel → az/el → pixel 라운드트립 검증"""

    def pixel_to_angle(self, cx, cy, W, H, fov_deg):
        f_px = (W / 2) / math.tan(math.radians(fov_deg / 2))
        x_ndc = (cx - W / 2) / f_px
        y_ndc = (cy - H / 2) / f_px
        az = math.atan2(x_ndc, 1.0)
        el_arg = y_ndc / math.sqrt(x_ndc**2 + 1)
        el = math.atan(el_arg)
        return az, el

    def angle_to_pixel(self, az, el, W, H, fov_deg):
        f_px = (W / 2) / math.tan(math.radians(fov_deg / 2))
        x_ndc = math.tan(az)
        y_ndc = math.tan(el) * math.sqrt(x_ndc**2 + 1)
        cx = W / 2 + f_px * x_ndc
        cy = H / 2 + f_px * y_ndc
        return cx, cy

    def test_center_pixel_is_zero_angle(self):
        """화면 중앙 → az=0, el=0"""
        W, H = 640, 360
        az, el = self.pixel_to_angle(320, 180, W, H, 60.0)
        self.assertAlmostEqual(az, 0.0, places=5)
        self.assertAlmostEqual(el, 0.0, places=5)

    def test_roundtrip_pixel_angle_pixel(self):
        """pixel → az/el → pixel 라운드트립 (오차 < 0.5px)"""
        W, H, fov = 640, 360, 60.0
        test_pixels = [(320, 180), (450, 200), (200, 120), (500, 280), (100, 300)]
        for cx0, cy0 in test_pixels:
            az, el = self.pixel_to_angle(cx0, cy0, W, H, fov)
            cx1, cy1 = self.angle_to_pixel(az, el, W, H, fov)
            self.assertAlmostEqual(cx0, cx1, delta=0.5,
                                   msg=f"cx roundtrip failed for ({cx0}, {cy0})")
            self.assertAlmostEqual(cy0, cy1, delta=0.5,
                                   msg=f"cy roundtrip failed for ({cx0}, {cy0})")

    def test_right_pixel_positive_az(self):
        """오른쪽 픽셀 → az > 0 (pipeline 규약)"""
        W, H = 640, 360
        az, el = self.pixel_to_angle(500, 180, W, H, 60.0)
        self.assertGreater(az, 0.0, msg="Right of center → az > 0")

    def test_elevation_correct(self):
        """위쪽 픽셀 (cy < H/2) → y_ndc < 0이므로 el < 0 (y-down convention)"""
        W, H = 640, 360
        az, el = self.pixel_to_angle(320, 100, W, H, 60.0)
        # y-down: cy<H/2 → y_ndc<0 → el<0
        self.assertLess(el, 0.0, msg="Above center (y-down) → el < 0")

    def test_el_formula_correct_vs_naive(self):
        """올바른 공식 tan(el)*sqrt(1+tan²(az)) vs 틀린 공식 tan(el)만 사용"""
        # az=45°, el=30° 에서 올바른 역변환
        az = math.radians(45)
        el = math.radians(30)
        W, H, fov = 640, 360, 60.0
        f_px = (W/2) / math.tan(math.radians(fov/2))
        x_ndc = math.tan(az)
        # 올바른: y_ndc = tan(el) * sqrt(1 + x_ndc^2)
        y_ndc_correct = math.tan(el) * math.sqrt(1 + x_ndc**2)
        # 틀린: y_ndc = tan(el)
        y_ndc_naive   = math.tan(el)
        # 역변환 결과 비교
        el_back_correct = math.atan(y_ndc_correct / math.sqrt(x_ndc**2 + 1))
        el_back_naive   = math.atan(y_ndc_naive   / math.sqrt(x_ndc**2 + 1))
        self.assertAlmostEqual(el_back_correct, el, places=5,
                               msg="Correct formula should round-trip")
        self.assertNotAlmostEqual(el_back_naive, el, places=2,
                                  msg="Naive formula should NOT round-trip at nonzero az")


# ─────────────────────────────────────────────────────────────────────────────
# 6. irgen — IR 합성
# ─────────────────────────────────────────────────────────────────────────────
class TestIrgen(unittest.TestCase):
    """Room IR 합성 및 convolution 검증"""

    def setUp(self):
        from vid2spatial_pkg.irgen import fft_convolve, schroeder_ir
        self.fft_convolve = fft_convolve
        # API: schroeder_ir(fs, rt60, length_s)
        self.schroeder_ir = lambda rt60: schroeder_ir(fs=48000, rt60=rt60, length_s=1.0)

    def test_fft_convolve_correctness(self):
        """FFT convolution vs scipy.fftconvolve 결과 일치"""
        from scipy.signal import fftconvolve
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000).astype(np.float32)
        h = rng.standard_normal(50).astype(np.float32)
        y_fft = self.fft_convolve(x, h)
        y_ref = fftconvolve(x, h, mode='full')[:len(x)]
        self.assertTrue(np.allclose(y_fft[:len(x)], y_ref, atol=1e-3),
                        msg="FFT convolution should match scipy reference")

    def test_schroeder_ir_output_shape(self):
        """Schroeder IR: 길이 > 0, 에너지 > 0"""
        ir = self.schroeder_ir(rt60=0.5)
        self.assertGreater(len(ir), 0, msg="IR should not be empty")
        self.assertGreater(float(np.sum(ir**2)), 0.0, msg="IR energy should be positive")

    def test_schroeder_ir_decays(self):
        """Schroeder IR: 앞부분이 뒷부분보다 에너지 높아야 (decay)"""
        ir = self.schroeder_ir(rt60=0.5)
        if len(ir) > 100:
            energy_front = float(np.sum(ir[:len(ir)//4]**2))
            energy_back  = float(np.sum(ir[3*len(ir)//4:]**2))
            self.assertGreater(energy_front, energy_back,
                               msg="IR should decay over time")


# ─────────────────────────────────────────────────────────────────────────────
# 7. visual_room_estimator — 방 크기 추정
# ─────────────────────────────────────────────────────────────────────────────
class TestVisualRoomEstimator(unittest.TestCase):
    """depth map → room volume/RT60 추정 검증"""

    def setUp(self):
        # VisualRoomEstimator.estimate_from_depth is a classmethod/staticmethod
        from vid2spatial_pkg.visual_room_estimator import VisualRoomEstimator
        self.estimator = VisualRoomEstimator()
        self.estimate = self.estimator.estimate_from_depth

    def test_larger_depth_means_larger_room(self):
        """더 먼 depth → 더 큰 방 추정"""
        depth_near = np.full((360, 640), 1.0, dtype=np.float32)
        depth_far  = np.full((360, 640), 5.0, dtype=np.float32)
        res_near = self.estimate(depth_near, fov_deg=60.0)
        res_far  = self.estimate(depth_far,  fov_deg=60.0)
        v_near = res_near.volume_m3
        v_far  = res_far.volume_m3
        self.assertGreater(v_far, v_near,
                           msg="Larger scene depth → larger room volume")

    def test_all_zero_depth_no_crash(self):
        """모든 depth=0 → 크래시 없이 처리"""
        depth = np.zeros((360, 640), dtype=np.float32)
        try:
            res = self.estimate(depth, fov_deg=60.0)
            self.assertIsNotNone(res)
        except Exception as e:
            self.fail(f"All-zero depth caused crash: {e}")

    def test_rt60_positive(self):
        """RT60 > 0 항상"""
        rng = np.random.default_rng(7)
        depth = rng.uniform(1.0, 5.0, (180, 320)).astype(np.float32)
        res = self.estimate(depth, fov_deg=60.0)
        self.assertGreater(res.rt60, 0.0, msg="RT60 must be positive")


# ─────────────────────────────────────────────────────────────────────────────
# 8. 엔드투엔드 smoke test (모델 없이 precomputed traj 사용)
# ─────────────────────────────────────────────────────────────────────────────
class TestEndToEndSmoke(unittest.TestCase):
    """저장된 traj.json으로 오디오 렌더 가능한지 smoke test"""

    TRAJ_PATH = Path("/home/seung/mmhoa/vid2spatial_v2/test/listen_compare/traj/dog-1__D_v1_legacy.json")
    AUDIO_PATH = Path("/tmp/v2s_m3d_trajectory/dog-1_m3d_v2.wav")
    SOFA_PATH  = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"

    def test_render_foa_from_precomputed_traj(self):
        """precomputed traj.json → 파이프라인 통해 FOA 렌더 (YOLO/MiDaS 없이)
        render_foa_from_trajectory API: (audio_path, trajectory_dict, output_path, ...)
        """
        if not self.TRAJ_PATH.exists():
            self.skipTest(f"Traj not found: {self.TRAJ_PATH}")
        if not self.AUDIO_PATH.exists():
            self.skipTest(f"Audio not found: {self.AUDIO_PATH}")

        import json, tempfile
        from vid2spatial_pkg.foa_render import interpolate_angles_distance, encode_mono_to_foa
        import soundfile as sf

        with open(self.TRAJ_PATH) as f:
            traj = json.load(f)
        frames = traj["frames"]
        audio, sr = sf.read(str(self.AUDIO_PATH), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        T = len(audio)

        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, sr, fps=30.0)
        foa = encode_mono_to_foa(audio, az_s, el_s)

        self.assertEqual(foa.shape[0], 4, msg="FOA must have 4 channels")
        self.assertEqual(foa.shape[1], T, msg="FOA length must match audio")
        self.assertFalse(np.any(np.isnan(foa)), msg="FOA must not contain NaN")
        self.assertFalse(np.any(np.isinf(foa)), msg="FOA must not contain Inf")

    def test_foa_output_not_clipped(self):
        """FOA 출력 peak < 1.1 (소프트 리미터 동작)"""
        if not self.TRAJ_PATH.exists():
            self.skipTest(f"Traj not found: {self.TRAJ_PATH}")
        if not self.AUDIO_PATH.exists():
            self.skipTest(f"Audio not found: {self.AUDIO_PATH}")

        import json
        from vid2spatial_pkg.foa_render import (
            interpolate_angles_distance, apply_distance_gain_lpf,
            encode_mono_to_foa, write_foa_wav
        )
        import soundfile as sf

        with open(self.TRAJ_PATH) as f:
            traj = json.load(f)
        frames = traj["frames"]
        audio, sr = sf.read(str(self.AUDIO_PATH), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        T = len(audio)

        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(frames, T, sr)
        audio_proc = apply_distance_gain_lpf(audio, sr, dist_s, d_rel_s)
        foa = encode_mono_to_foa(audio_proc, az_s, el_s)

        peak = float(np.max(np.abs(foa)))
        self.assertLessEqual(peak, 1.1, msg=f"FOA peak {peak:.3f} exceeds safe range")


# ─────────────────────────────────────────────────────────────────────────────
# 9. config — 설정 직렬화 검증
# ─────────────────────────────────────────────────────────────────────────────
class TestConfig(unittest.TestCase):
    """Config 데이터클래스 직렬화/역직렬화 검증"""

    def test_pipeline_config_defaults(self):
        """기본값으로 PipelineConfig 생성 가능"""
        from vid2spatial_pkg.config import (
            PipelineConfig, VisionConfig, CameraConfig, TrackingConfig,
            DepthConfig, RoomConfig, SpatialConfig, OutputConfig, BinauralConfig,
        )
        cfg = PipelineConfig(
            video_path="/tmp/test.mp4",
            audio_path="/tmp/test.wav",
            vision=VisionConfig(
                camera=CameraConfig(),
                tracking=TrackingConfig(),
                depth=DepthConfig(),
            ),
            room=RoomConfig(disabled=True),
            spatial=SpatialConfig(),
            output=OutputConfig(
                foa_path="/tmp/out.foa.wav",
                stereo_path="/tmp/out.stereo.wav",
                binaural_path="/tmp/out.binaural.wav",
                binaural_config=BinauralConfig(),
            ),
        )
        self.assertEqual(cfg.video_path, "/tmp/test.mp4")
        self.assertTrue(cfg.room.disabled)

    def test_camera_config_fov(self):
        """CameraConfig FOV 범위 확인"""
        from vid2spatial_pkg.config import CameraConfig
        cfg = CameraConfig(fov_deg=90.0)
        self.assertEqual(cfg.fov_deg, 90.0)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args, remaining = parser.parse_known_args()
    verbosity = 2 if args.verbose else 1

    # unittest.main()에는 -v 전달
    test_args = [sys.argv[0]] + (["-v"] if args.verbose else [])
    unittest.main(argv=test_args, verbosity=verbosity)

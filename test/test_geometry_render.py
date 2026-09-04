"""
Tests for the geometry + render lane added 2026-09-04 (gap items A1, A2, A3,
A7, A8, A14, A16).

Everything here is CPU-only, model-free and deterministic.  Anything that needs
a real SOFA file or the KITTI labels skips cleanly when they are absent.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# A1 — independent azimuth evaluation on KITTI 3D labels
# ---------------------------------------------------------------------------
class TestEvalAzimuthKitti:
    def test_fov_focal_roundtrip(self):
        ek = _load_tool("eval_azimuth_kitti")
        for fov in (40.0, 60.0, 81.7):
            f = ek.fov_to_focal_px(fov, 1242)
            assert ek.focal_px_to_fov(f, 1242) == pytest.approx(fov, abs=1e-9)

    def test_kitti_p2_implies_82_deg_not_60(self):
        """The repo assumes 60 deg; the real KITTI colour camera is ~81.7 deg."""
        ek = _load_tool("eval_azimuth_kitti")
        fx = 721.5377  # KITTI P2 focal length, published constant
        assert ek.focal_px_to_fov(fx, 1242) == pytest.approx(81.4, abs=0.2)

    def test_gt_azimuth_is_projection_free(self):
        """az_gt = atan2(x_cam, z_cam) must not depend on any assumed FOV."""
        ek = _load_tool("eval_azimuth_kitti")
        records = [{"seq": "0000", "frame": 0, "track": "0000_0", "type": "Car",
                    "cx_px": 800.0, "cy_px": 180.0,
                    "x_cam": 3.0, "y_cam": 1.6, "z_cam": 12.0}]
        calibs = {"0000": {"fx": 721.5377, "fy": 721.5377, "cx": 609.5593, "cy": 172.854}}
        a = ek.evaluate(records, calibs, width=1242, repo_fov_deg=60.0)
        b = ek.evaluate(records, calibs, width=1242, repo_fov_deg=80.0)
        # GT is identical, so the only thing that moves is the "repo" variant.
        assert a["variants"]["calib"]["azmae_deg"] == pytest.approx(
            b["variants"]["calib"]["azmae_deg"])
        assert a["variants"]["repo"]["azmae_deg"] != pytest.approx(
            b["variants"]["repo"]["azmae_deg"])

    def test_circular_difference_wraps(self):
        ek = _load_tool("eval_azimuth_kitti")
        d = ek._circ_abs_deg(np.array([179.0]), np.array([-179.0]))
        assert d[0] == pytest.approx(2.0)

    def test_report_reproduces_on_real_labels(self):
        """Run the real thing when the KITTI labels + calib are on disk."""
        ek = _load_tool("eval_azimuth_kitti")
        labels = REPO_ROOT / "data/kitti_tracking/label_02"
        calib = REPO_ROOT / "data/kitti_tracking/calib"
        if not labels.is_dir() or not calib.is_dir():
            pytest.skip("KITTI tracking labels/calib not on disk")
        records, calibs = [], {}
        for lp in sorted(labels.glob("*.txt"))[:3]:
            cp = calib / lp.name
            if not cp.exists():
                continue
            calibs[lp.stem] = ek.read_calib(cp)
            records.extend(ek.parse_sequence(lp))
        assert records, "no KITTI records parsed"
        res = ek.evaluate(records, calibs, width=1242, repo_fov_deg=60.0)
        v = res["variants"]
        # The whole point of A1: the honest number is materially worse than the
        # circular PixelAzMAE, and true intrinsics recover most of the gap.
        assert v["repo"]["azmae_deg"] > 2.0
        assert v["calib"]["azmae_deg"] < v["repo"]["azmae_deg"]
        assert res["true_fov_deg_mean"] > 75.0


# ---------------------------------------------------------------------------
# A2 — FOV from container metadata, with provenance
# ---------------------------------------------------------------------------
class TestCameraIntrinsics:
    def test_focal_35mm_roundtrip_and_known_lenses(self):
        from vid2spatial_pkg import camera_intrinsics as ci
        # A 27 mm phone lens is ~67 deg; a 50 mm is ~40 deg.
        assert ci.fov_from_focal_35mm(27.0) == pytest.approx(67.4, abs=0.3)
        assert ci.fov_from_focal_35mm(50.0) == pytest.approx(39.6, abs=0.3)
        for f in (18.0, 27.0, 35.0, 50.0):
            assert ci.focal_35mm_from_fov(ci.fov_from_focal_35mm(f)) == pytest.approx(f)

    def test_focal_px_matches_vision_module(self):
        """camera_intrinsics and vision.CameraIntrinsics must agree exactly."""
        from vid2spatial_pkg import camera_intrinsics as ci
        from vid2spatial_pkg.vision import CameraIntrinsics
        K = CameraIntrinsics(width=1280, height=720, fov_deg=60.0)
        assert ci.focal_px_from_fov(60.0, 1280) == pytest.approx(K.focal_length)
        assert ci.fov_from_focal_px(K.focal_length, 1280) == pytest.approx(60.0)

    def test_parse_exiftool_fov_tag(self):
        from vid2spatial_pkg import camera_intrinsics as ci
        got = ci.parse_exiftool_json([{"FOV": "63.4 deg"}])
        assert got is not None and got[0] == pytest.approx(63.4)

    def test_parse_exiftool_focal_35mm(self):
        from vid2spatial_pkg import camera_intrinsics as ci
        got = ci.parse_exiftool_json([{"FocalLengthIn35mmFormat": "27 mm"}])
        assert got is not None and got[0] == pytest.approx(67.38, abs=0.05)

    def test_parse_exiftool_35efl_string(self):
        from vid2spatial_pkg import camera_intrinsics as ci
        got = ci.parse_exiftool_json(
            [{"FocalLength35efl": "4.2 mm (35 mm equivalent: 27.0 mm)"}])
        assert got is not None and got[0] == pytest.approx(67.38, abs=0.05)

    def test_parse_ffprobe_stream_tag(self):
        from vid2spatial_pkg import camera_intrinsics as ci
        payload = {"streams": [{"codec_type": "video", "width": 1920,
                                "tags": {"FieldOfView": "94.4"}}], "format": {}}
        got = ci.parse_ffprobe_json(payload)
        assert got is not None and got[0] == pytest.approx(94.4)

    def test_parse_ffprobe_focal_px_tag(self):
        from vid2spatial_pkg import camera_intrinsics as ci
        payload = {"streams": [{"codec_type": "video", "width": 1242,
                                "tags": {"focal_length_px": "721.5377"}}]}
        got = ci.parse_ffprobe_json(payload)
        assert got is not None and got[0] == pytest.approx(81.4, abs=0.2)

    def test_parse_rejects_nonsense(self):
        from vid2spatial_pkg import camera_intrinsics as ci
        assert ci.parse_ffprobe_json({"streams": [{"tags": {"fov": "0"}}]}) is None
        assert ci.parse_exiftool_json([{"FOV": "400 deg"}]) is None
        assert ci.parse_exiftool_json([{}]) is None

    def test_sidecar_roundtrip(self, tmp_path):
        import json
        from vid2spatial_pkg import camera_intrinsics as ci
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"not a real video")
        (tmp_path / "clip.mp4.fov.json").write_text(json.dumps({"fov_deg": 72.5}))
        info = ci.resolve_fov(str(video), warn=False)
        assert info.fov_deg == pytest.approx(72.5)
        assert info.source == "sidecar"
        assert not info.is_default

    def test_default_is_60_and_loud(self, tmp_path, capsys):
        from vid2spatial_pkg import camera_intrinsics as ci
        video = tmp_path / "no_metadata.mp4"
        video.write_bytes(b"nope")
        info = ci.resolve_fov(str(video), probes=[])
        assert info.fov_deg == pytest.approx(60.0)
        assert info.source == "default"
        assert "WARN" in capsys.readouterr().out

    def test_explicit_cli_wins_over_metadata(self, tmp_path):
        import json
        from vid2spatial_pkg import camera_intrinsics as ci
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        (tmp_path / "clip.mp4.fov.json").write_text(json.dumps({"fov_deg": 72.5}))
        info = ci.resolve_fov(str(video), explicit_fov_deg=45.0, warn=False)
        assert info.fov_deg == pytest.approx(45.0) and info.source == "cli:fov"

    def test_focal_35mm_flag(self):
        from vid2spatial_pkg import camera_intrinsics as ci
        info = ci.resolve_fov(None, focal_35mm=27.0, warn=False)
        assert info.source == "cli:focal_35mm"
        assert info.fov_deg == pytest.approx(67.38, abs=0.05)

    def test_broken_probe_never_breaks_the_render(self):
        from vid2spatial_pkg import camera_intrinsics as ci

        def boom(_path):
            raise RuntimeError("exiftool exploded")

        info = ci.resolve_fov("whatever.mp4", probes=[("metadata:exiftool", boom)],
                              warn=False)
        assert info.source == "default" and info.fov_deg == pytest.approx(60.0)

    def test_reprojection_is_exact_geometry(self):
        from vid2spatial_pkg import camera_intrinsics as ci
        # 10 deg measured under a 60 deg assumption, re-scored at 80 deg
        az = np.array([0.0, 10.0, -25.0])
        out = ci.reproject_azimuth(az, 60.0, 80.0)
        assert out[0] == pytest.approx(0.0)
        # a wider assumed FOV means each pixel spans more angle, so the same
        # pixel offset re-scores to a LARGER azimuth
        assert abs(out[1]) > abs(az[1])
        assert ci.reproject_azimuth(out, 80.0, 60.0) == pytest.approx(az)

    def test_config_carries_provenance_fields(self):
        from vid2spatial_pkg.config import CameraConfig
        cfg = CameraConfig()
        assert cfg.fov_deg == 60.0                    # default unchanged
        assert cfg.fov_source == "default"
        # opt-in: reading the container can change the FOV, hence every azimuth
        assert cfg.fov_from_metadata is False
        assert cfg.focal_35mm is None
        assert cfg.motion_mode == "camera_frame"      # default unchanged


# ---------------------------------------------------------------------------
# A3 — camera-motion compensation
# ---------------------------------------------------------------------------
def _textured_frame(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    # blur a little so corner detection has real structure, not pure noise
    import cv2
    return cv2.GaussianBlur(img, (5, 5), 1.2)


def _pan_sequence(n_frames=12, shift_px_per_frame=4.0, h=240, w=640):
    """A wide textured canvas cropped with a moving window = a synthetic pan."""
    import cv2
    canvas = _textured_frame(h, w * 2, seed=7)
    out = []
    for i in range(n_frames):
        x0 = int(round(i * shift_px_per_frame))
        out.append(np.ascontiguousarray(canvas[:, x0:x0 + w]))
    return out, cv2


class TestCameraMotion:
    def test_estimate_global_affine_recovers_known_shift(self):
        from vid2spatial_pkg.camera_motion import estimate_global_affine
        frames, _ = _pan_sequence(n_frames=2, shift_px_per_frame=6.0)
        est = estimate_global_affine(frames[0], frames[1])
        assert est["inliers"] >= 8
        # cropping the window rightwards moves content LEFT by 6 px
        assert est["dx"] == pytest.approx(-6.0, abs=0.5)
        assert est["dy"] == pytest.approx(0.0, abs=0.5)

    def test_yaw_accumulates_with_the_pan(self):
        from vid2spatial_pkg.camera_motion import (
            estimate_camera_motion_frames, focal_px_from_fov)
        frames, _ = _pan_sequence(n_frames=10, shift_px_per_frame=5.0)
        f = focal_px_from_fov(60.0, 640)
        m = estimate_camera_motion_frames(frames, focal_px=f)
        expected_deg = np.degrees(9 * 5.0 / f)   # 9 increments of +5 px pan
        assert m.yaw_deg[-1] == pytest.approx(expected_deg, abs=0.15)
        assert np.all(np.diff(m.yaw_deg) > 0)    # monotone during a steady pan

    def test_world_frame_stays_flat_camera_frame_tracks_the_pan(self):
        """The A3 acceptance test: a source that is static in the world."""
        from vid2spatial_pkg.camera_motion import (
            estimate_camera_motion_frames, apply_motion_mode, focal_px_from_fov,
            MODE_CAMERA_FRAME, MODE_WORLD_FRAME)
        n, shift, w = 10, 5.0, 640
        frames, _ = _pan_sequence(n_frames=n, shift_px_per_frame=shift)
        f = focal_px_from_fov(60.0, w)
        m = estimate_camera_motion_frames(frames, focal_px=f)

        # A world-static source: its image x moves LEFT by `shift` per frame,
        # exactly as the background does, so its image azimuth sweeps.
        cx0 = w / 2.0
        az_img = np.arctan2(np.array([cx0 - i * shift for i in range(n)]) - w / 2.0, f)
        el_img = np.zeros(n)

        az_cam, _ = apply_motion_mode(az_img, el_img, m, mode=MODE_CAMERA_FRAME)
        az_world, _ = apply_motion_mode(az_img, el_img, m, mode=MODE_WORLD_FRAME)

        cam_range = np.degrees(az_cam.max() - az_cam.min())
        world_range = np.degrees(az_world.max() - az_world.min())
        assert cam_range > 3.0                 # camera_frame tracks the pan
        assert world_range < 0.2               # world_frame stays flat
        assert world_range < cam_range / 10.0

    def test_camera_frame_is_the_identity(self):
        from vid2spatial_pkg.camera_motion import (
            CameraMotion, apply_motion_mode, MODE_CAMERA_FRAME)
        az = np.array([0.1, 0.2, 0.3])
        m = CameraMotion(yaw_rad=np.array([0.0, 0.5, 1.0]),
                         pitch_rad=np.zeros(3))
        out_az, out_el = apply_motion_mode(az, np.zeros(3), m, MODE_CAMERA_FRAME)
        assert out_az == pytest.approx(az)
        assert out_el == pytest.approx(np.zeros(3))

    def test_unknown_mode_raises(self):
        from vid2spatial_pkg.camera_motion import CameraMotion, apply_motion_mode
        m = CameraMotion(yaw_rad=np.zeros(3), pitch_rad=np.zeros(3))
        with pytest.raises(ValueError):
            apply_motion_mode(np.zeros(3), np.zeros(3), m, mode="head_frame")

    def test_low_texture_pair_holds_the_accumulator(self):
        from vid2spatial_pkg.camera_motion import estimate_camera_motion_frames
        flat = [np.zeros((120, 160), dtype=np.uint8) for _ in range(4)]
        m = estimate_camera_motion_frames(flat, focal_px=500.0)
        assert np.allclose(m.yaw_rad, 0.0)
        assert np.all(m.inliers[1:] == 0)

    def test_annotate_trajectory_records_yaw_without_changing_camera_frame(self, monkeypatch):
        from vid2spatial_pkg import camera_motion as cm
        traj = {"frames": [{"frame": i, "az": 0.1 * i, "el": 0.0} for i in range(5)]}
        fake = cm.CameraMotion(yaw_rad=np.linspace(0, 0.2, 5),
                               pitch_rad=np.zeros(5))
        monkeypatch.setattr(cm, "estimate_camera_motion_video",
                            lambda *a, **k: fake)
        out = cm.annotate_trajectory_with_camera_motion(
            dict(traj), "fake.mp4", mode=cm.MODE_CAMERA_FRAME)
        assert [f["az"] for f in out["frames"]] == [0.0, 0.1, 0.2, 0.30000000000000004, 0.4]
        assert out["camera_motion"]["mode"] == "camera_frame"
        assert out["frames"][-1]["camera_yaw_deg"] == pytest.approx(np.degrees(0.2))

    def test_annotate_trajectory_world_frame_shifts_and_keeps_the_original(self, monkeypatch):
        from vid2spatial_pkg import camera_motion as cm
        traj = {"frames": [{"frame": i, "az": 0.0, "el": 0.0} for i in range(5)]}
        fake = cm.CameraMotion(yaw_rad=np.linspace(0, 0.2, 5), pitch_rad=np.zeros(5))
        monkeypatch.setattr(cm, "estimate_camera_motion_video", lambda *a, **k: fake)
        out = cm.annotate_trajectory_with_camera_motion(
            traj, "fake.mp4", mode=cm.MODE_WORLD_FRAME)
        assert out["frames"][-1]["az"] == pytest.approx(0.2)
        assert out["frames"][-1]["az_camera_frame"] == pytest.approx(0.0)

    def test_estimation_failure_degrades_to_camera_frame(self, monkeypatch):
        from vid2spatial_pkg import camera_motion as cm

        def boom(*a, **k):
            raise OSError("cannot open video")

        monkeypatch.setattr(cm, "estimate_camera_motion_video", boom)
        traj = {"frames": [{"frame": 0, "az": 0.3, "el": 0.0}]}
        out = cm.annotate_trajectory_with_camera_motion(traj, "missing.mp4",
                                                        mode=cm.MODE_WORLD_FRAME)
        assert out["frames"][0]["az"] == pytest.approx(0.3)
        assert out["camera_motion"]["method"] == "failed"


# ---------------------------------------------------------------------------
# A8 — HRTF interpolation instead of nearest-neighbour
# ---------------------------------------------------------------------------
def _ring_grid(step_deg=5.0):
    """A horizontal-only measurement ring: the KEMAR-like degenerate case."""
    az = np.radians(np.arange(0.0, 360.0, step_deg))
    src_cart = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=1)
    # a 1-tap "HRIR" whose taps encode cos/sin of the direction, so the
    # interpolated value can be checked against the exact angle
    ir = np.zeros((len(az), 2, 4), dtype=np.float64)
    ir[:, 0, 0] = np.cos(az)
    ir[:, 1, 0] = np.sin(az)
    return src_cart, ir


class TestHrirInterpolation:
    def test_weights_sum_to_one_and_are_nonnegative(self):
        from vid2spatial_pkg.foa_render import hrir_barycentric_weights
        rng = np.random.default_rng(0)
        src = rng.standard_normal((200, 3))
        src /= np.linalg.norm(src, axis=1, keepdims=True)
        for _ in range(50):
            q = rng.standard_normal(3)
            q /= np.linalg.norm(q)
            idx, w = hrir_barycentric_weights(src, q)
            assert idx.shape == (3,) and w.shape == (3,)
            assert w.sum() == pytest.approx(1.0)
            assert np.all(w >= 0.0)

    def test_exact_grid_point_returns_that_hrir(self):
        from vid2spatial_pkg.foa_render import make_hrir_lookup
        src, ir = _ring_grid()
        for mode in ("nearest", "barycentric"):
            lk = make_hrir_lookup(src, ir, mode=mode)
            h = lk(math.radians(35.0), 0.0)
            assert h[0, 0] == pytest.approx(math.cos(math.radians(35.0)), abs=1e-6)
            assert h[1, 0] == pytest.approx(math.sin(math.radians(35.0)), abs=1e-6)

    def test_interpolation_tracks_the_true_angle_between_grid_points(self):
        from vid2spatial_pkg.foa_render import make_hrir_lookup
        src, ir = _ring_grid(step_deg=5.0)
        near = make_hrir_lookup(src, ir, mode="nearest")
        bary = make_hrir_lookup(src, ir, mode="barycentric")
        for a in (1.0, 2.5, 4.0, 12.5, 47.5):
            truth = math.sin(math.radians(a))
            # a linear blend of two grid directions is a chord, not an arc, so
            # it undershoots the true value by O(spacing^2) -- ~1e-3 at 5 deg
            assert bary(math.radians(a), 0.0)[1, 0] == pytest.approx(truth, abs=1.5e-3)
        # nearest snaps to the grid: at 2.5 deg it is stuck on a grid point
        assert near(math.radians(2.5), 0.0)[1, 0] in (
            pytest.approx(math.sin(math.radians(0.0)), abs=1e-9),
            pytest.approx(math.sin(math.radians(5.0)), abs=1e-9),
        )

    def test_nearest_is_a_staircase_and_barycentric_is_not(self):
        from vid2spatial_pkg.foa_render import make_hrir_lookup
        src, ir = _ring_grid(step_deg=5.0)
        angles = np.radians(np.linspace(0.0, 20.0, 201))
        for mode, expect_unique in (("nearest", 6), ("barycentric", 150)):
            lk = make_hrir_lookup(src, ir, mode=mode)
            vals = np.array([lk(a, 0.0)[1, 0] for a in angles])
            n_unique = len(np.unique(np.round(vals, 9)))
            if mode == "nearest":
                assert n_unique <= expect_unique     # one value per 5 deg cell
            else:
                assert n_unique >= expect_unique     # continuously varying

    def test_unknown_mode_rejected(self):
        from vid2spatial_pkg.foa_render import make_hrir_lookup
        src, ir = _ring_grid()
        with pytest.raises(ValueError):
            make_hrir_lookup(src, ir, mode="cubic")

    def test_render_paths_expose_the_switch(self):
        import inspect
        from vid2spatial_pkg.foa_render import (
            direct_binaural_sofa, foa_to_binaural_sofa,
            render_binaural_from_trajectory)
        for fn in (direct_binaural_sofa, foa_to_binaural_sofa,
                   render_binaural_from_trajectory):
            p = inspect.signature(fn).parameters
            assert "hrir_interp" in p
            # nearest is the DEFAULT: it is what every shipped stimulus was
            # rendered with, so interpolation has to be opted into (HIGH 1 of
            # the 2026-09-04 review). See TestDefaultRenderInvariance.
            assert p["hrir_interp"].default == "nearest"

    def test_itd_staircase_measured_on_a_real_sofa(self):
        """The A8 acceptance measurement, on the real 5 deg KEMAR grid."""
        harness = REPO_ROOT / "test" / "run_hrtf_interp_check.py"
        spec = importlib.util.spec_from_file_location("hrtf_interp_check", harness)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sofa = mod.find_sofa()
        if not sofa or mod.grid_spacing_deg(sofa) < 2.0:
            pytest.skip("no sparse SOFA grid available")
        res = mod.run(sofa, arc_deg=90.0, dur_s=4.0)
        near = res["modes"]["nearest"]
        bary = res["modes"]["barycentric"]
        # both must still sweep the full arc
        assert bary["recovered_range_deg"] > 60.0
        # interpolation must remove the big discrete jumps
        assert bary["step_max_deg"] < near["step_max_deg"] * 0.6
        assert bary["mean_abs_second_diff_deg"] < near["mean_abs_second_diff_deg"]


# ---------------------------------------------------------------------------
# A16 — confidence-aware rendering during lost episodes
# ---------------------------------------------------------------------------
def _traj_with_lost(n=60, lost_slice=slice(20, 35), fps=30.0):
    frames = []
    for i in range(n):
        conf = 0.2 if lost_slice.start <= i < lost_slice.stop else 0.9
        # azimuth ramps steadily; during the lost run it lunges off to the side
        az = math.radians(i * 0.5)
        if lost_slice.start <= i < lost_slice.stop:
            az = math.radians(60.0)
        frames.append({"frame": i, "az": az, "el": 0.0, "confidence": conf,
                       "dist_m": 5.0, "w": 80, "h": 80})
    return {"frames": frames, "fps": fps, "intrinsics": {"width": 1280, "height": 720}}


class TestConfidenceAwareRender:
    def test_freeze_holds_the_last_confident_azimuth(self):
        from vid2spatial_pkg.foa_render import freeze_lost_frames
        traj = _traj_with_lost()
        out, stats = freeze_lost_frames(traj["frames"])
        assert stats["n_lost"] == 15
        assert stats["n_episodes"] == 1
        held = math.radians(19 * 0.5)
        for i in range(20, 35):
            assert out[i]["az"] == pytest.approx(held)
            assert out[i]["lost"] is True
            # the bad measurement is preserved, not destroyed
            assert out[i]["az_measured"] == pytest.approx(math.radians(60.0))
        assert out[35]["az"] == pytest.approx(math.radians(35 * 0.5))

    def test_azimuth_is_flat_across_the_lost_episode(self):
        """The A16 acceptance test."""
        from vid2spatial_pkg.foa_render import freeze_lost_frames
        traj = _traj_with_lost()
        raw = np.degrees([f["az"] for f in traj["frames"]])
        out, _ = freeze_lost_frames(traj["frames"])
        fixed = np.degrees([f["az"] for f in out])
        lost = slice(20, 35)
        assert float(np.ptp(raw[lost])) == pytest.approx(0.0)     # the lunge itself is flat
        # the damage is the JUMP into and out of the episode
        assert abs(raw[20] - raw[19]) > 40.0
        assert abs(fixed[20] - fixed[19]) < 1e-9
        assert float(np.ptp(fixed[lost])) == pytest.approx(0.0)

    def test_no_confidence_field_is_a_no_op(self):
        from vid2spatial_pkg.foa_render import freeze_lost_frames
        frames = [{"frame": i, "az": 0.1 * i, "el": 0.0} for i in range(10)]
        out, stats = freeze_lost_frames(frames)
        assert stats["n_lost"] == 0
        assert [f["az"] for f in out] == [f["az"] for f in frames]
        assert "lost" not in out[0]        # untouched objects, not rewritten

    def test_duck_and_diffuse_curves(self):
        from vid2spatial_pkg.foa_render import build_lost_curves
        traj = _traj_with_lost()
        sr, fps = 8000, 30.0
        T = int(len(traj["frames"]) / fps * sr)
        duck, diffuse = build_lost_curves(traj["frames"], T, sr, fps,
                                          duck_db=-9.0, diffuse_boost=0.35)
        assert duck.shape == (T,) and diffuse.shape == (T,)
        assert duck.max() == pytest.approx(1.0, abs=1e-3)
        assert duck.min() < 0.45           # -9 dB is 0.355
        assert diffuse.max() > 0.30
        assert diffuse.min() == pytest.approx(0.0, abs=1e-3)
        # ducked exactly where the track is lost
        mid = int((27 / fps) * sr)
        assert duck[mid] < 0.4 and diffuse[mid] > 0.3

    def test_curves_are_flat_when_nothing_is_lost(self):
        from vid2spatial_pkg.foa_render import build_lost_curves
        frames = [{"frame": i, "az": 0.0, "el": 0.0, "confidence": 1.0}
                  for i in range(30)]
        duck, diffuse = build_lost_curves(frames, 8000, 8000, 30.0)
        assert np.allclose(duck, 1.0)
        assert np.allclose(diffuse, 0.0)

    def test_confidence_gate_is_off_by_default_in_the_renderers(self):
        import inspect
        from vid2spatial_pkg.foa_render import (
            render_foa_from_trajectory, render_binaural_from_trajectory)
        for fn in (render_foa_from_trajectory, render_binaural_from_trajectory):
            p = inspect.signature(fn).parameters
            # OFF by default: trajectory_export.py writes a confidence field on
            # every row, so an on-by-default gate would silently change real
            # renders (HIGH 2 of the 2026-09-04 review).
            assert p["confidence_gate"].default is False
            assert p["conf_threshold"].default == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# A7 — physical distance law with distance-coupled DRR
# ---------------------------------------------------------------------------
class TestPhysicalDistanceLaw:
    def test_direct_gain_is_inverse_distance(self):
        from vid2spatial_pkg.foa_render import physical_direct_gain
        d = np.array([1.0, 2.0, 4.0, 8.0])
        g = physical_direct_gain(d, d_ref_m=1.0, gain_floor=1e-6)
        db = 20 * np.log10(g)
        # -6 dB per doubling, exactly
        assert np.allclose(np.diff(db), -6.0206, atol=1e-3)

    def test_gain_floor_and_reference_distance(self):
        from vid2spatial_pkg.foa_render import physical_direct_gain
        g = physical_direct_gain(np.array([0.1, 0.5, 1.0]), d_ref_m=1.0)
        assert np.allclose(g, 1.0)          # never louder than the reference
        far = physical_direct_gain(np.array([1e6]), gain_floor=0.02)
        assert far[0] == pytest.approx(0.02)

    def test_air_absorption_cutoff_falls_with_distance(self):
        from vid2spatial_pkg.foa_render import air_absorption_cutoff_hz
        fc = air_absorption_cutoff_hz(np.array([1.0, 10.0, 100.0, 1000.0]))
        assert np.all(np.diff(fc) <= 0)
        # ISO 9613-1: 3 dB of loss at ~100 m lands in the low kHz
        assert 1000.0 < fc[2] < 6000.0

    def test_physical_wet_curve_holds_reverb_absolute(self):
        from vid2spatial_pkg.foa_render import (
            build_physical_wet_curve, physical_direct_gain)
        d = np.array([1.0, 2.0, 4.0, 8.0])
        g = physical_direct_gain(d, gain_floor=0.005)
        wet = build_physical_wet_curve(d, gain_floor=0.005, reverb_send=0.02)
        absolute_reverb = wet * g            # must be constant by construction
        assert np.allclose(absolute_reverb, absolute_reverb[0], rtol=1e-5)
        assert np.all(np.diff(wet) > 0)      # wetter with distance

    def test_existing_modes_are_untouched(self):
        """A7 must not perturb any shipped stimulus."""
        from vid2spatial_pkg.foa_render import apply_distance_gain_lpf
        rng = np.random.default_rng(2)
        x = rng.standard_normal(4000).astype(np.float32)
        d = np.linspace(1.0, 10.0, 4000).astype(np.float32)
        d_rel = np.linspace(0.0, 1.0, 4000).astype(np.float32)
        a = apply_distance_gain_lpf(x, 16000, d, d_rel)
        b = apply_distance_gain_lpf(x, 16000, d, d_rel)
        assert np.array_equal(a, b)
        # the legacy span really is the documented ~10.5 dB
        span_db = 20 * math.log10(1.0 / 0.3)
        assert span_db == pytest.approx(10.46, abs=0.05)

    def test_drr_falls_monotonically_at_roughly_the_physical_rate(self):
        """The A7 acceptance measurement."""
        harness = REPO_ROOT / "test" / "run_drr_check.py"
        spec = importlib.util.spec_from_file_location("drr_check", harness)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        res = mod.run(distances=[1.0, 2.0, 4.0, 8.0], modes=("legacy", "physical"))
        phys = res["modes"]["physical"]
        leg = res["modes"]["legacy"]
        assert phys["drr_monotonic"]
        assert phys["drr_db_per_doubling"] == pytest.approx(-6.0, abs=1.0)
        # and the legacy curve does NOT follow distance anywhere near that rate
        assert abs(leg["drr_db_per_doubling"]) < 2.5


# ---------------------------------------------------------------------------
# A14 — Doppler from the radial velocity
# ---------------------------------------------------------------------------
def _dominant_hz(x: np.ndarray, sr: int) -> float:
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w))
    return float(np.fft.rfftfreq(len(x), 1.0 / sr)[int(np.argmax(spec))])


class TestDoppler:
    def test_static_source_is_a_no_op(self):
        from vid2spatial_pkg.foa_render import apply_doppler
        sr = 16000
        t = np.arange(sr, dtype=np.float64) / sr
        x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        y = apply_doppler(x, sr, np.full(sr, 7.0, dtype=np.float32))
        assert _dominant_hz(y, sr) == pytest.approx(_dominant_hz(x, sr), abs=2.0)

    def test_approaching_source_shifts_pitch_up(self):
        """The A14 acceptance test: synthetic approach, measured shift."""
        from vid2spatial_pkg.foa_render import apply_doppler, SPEED_OF_SOUND_M_S
        sr, f0, dur = 16000, 1000.0, 2.0
        n = int(sr * dur)
        t = np.arange(n) / sr
        x = np.sin(2 * np.pi * f0 * t).astype(np.float32)
        v = -34.3                                   # approaching at 0.1 c
        dist = (100.0 + v * t).astype(np.float32)
        y = apply_doppler(x, sr, dist)
        seg = slice(sr // 4, sr // 4 + sr // 2)
        expected = f0 * (1.0 - v / SPEED_OF_SOUND_M_S)
        assert _dominant_hz(y[seg], sr) == pytest.approx(expected, rel=0.02)
        assert _dominant_hz(y[seg], sr) > f0

    def test_receding_source_shifts_pitch_down(self):
        from vid2spatial_pkg.foa_render import apply_doppler
        sr, f0, dur = 16000, 1000.0, 2.0
        n = int(sr * dur)
        t = np.arange(n) / sr
        x = np.sin(2 * np.pi * f0 * t).astype(np.float32)
        dist = (10.0 + 34.3 * t).astype(np.float32)
        y = apply_doppler(x, sr, dist)
        seg = slice(sr // 4, sr // 4 + sr // 2)
        assert _dominant_hz(y[seg], sr) < f0 - 50.0

    def test_rate_is_clamped_against_depth_glitches(self):
        from vid2spatial_pkg.foa_render import apply_doppler
        sr = 8000
        x = np.random.default_rng(1).standard_normal(sr).astype(np.float32)
        dist = np.full(sr, 5.0, dtype=np.float32)
        dist[sr // 2] = 5000.0                      # a one-sample depth runaway
        y = apply_doppler(x, sr, dist, max_ratio=0.25)
        assert np.all(np.isfinite(y))
        assert y.shape == x.shape

    def test_doppler_is_opt_in(self):
        import inspect
        from vid2spatial_pkg.foa_render import (
            render_foa_from_trajectory, render_binaural_from_trajectory)
        for fn in (render_foa_from_trajectory, render_binaural_from_trajectory):
            assert inspect.signature(fn).parameters["doppler"].default is False


# ---------------------------------------------------------------------------
# Default-render invariance against the pre-change commit
# ---------------------------------------------------------------------------
class TestDefaultRenderInvariance:
    """A default render must be byte-identical to commit 09289ea.

    test_existing_modes_are_untouched only proves the code is deterministic,
    which says nothing about whether it still agrees with what shipped. This
    compares real rendered arrays against a golden built by running
    test/make_render_golden.py inside a worktree checked out at 09289ea.
    """

    def _harness(self):
        path = REPO_ROOT / "test" / "make_render_golden.py"
        spec = importlib.util.spec_from_file_location("make_render_golden", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_golden_file_is_present_and_sane(self):
        import json
        golden = REPO_ROOT / "test" / "render_golden_09289ea.json"
        assert golden.exists(), "golden reference missing"
        data = json.loads(golden.read_text())
        assert data["sections"], "golden has no sections"
        # every mode that ships must be covered
        for mode in ("depth_rel", "bbox_area", "bbox_area_log", "hybrid"):
            assert f"prep::{mode}::audio" in data["sections"]
        assert "foa::reverb=False" in data["sections"]
        assert "foa::reverb=True" in data["sections"]

    def test_default_render_matches_09289ea(self):
        import json
        mod = self._harness()
        golden = REPO_ROOT / "test" / "render_golden_09289ea.json"
        ref = json.loads(golden.read_text())
        sofa = mod.find_sofa()
        res = mod.build(sofa)
        bad = mod.compare(ref, res, allow_missing=() if sofa else ("binaural",))
        assert not bad, "default render drifted from 09289ea:\n  " + "\n  ".join(bad)

    def test_the_golden_actually_detects_a_default_change(self):
        """Guard the guard: flipping a default must make the check fail."""
        import json
        mod = self._harness()
        ref = json.loads((REPO_ROOT / "test" / "render_golden_09289ea.json").read_text())
        traj = mod.make_trajectory()
        assert any(f["confidence"] < 0.5 for f in traj["frames"]), \
            "the golden trajectory must contain a lost episode to be a real test"

        import tempfile
        import soundfile as sf
        from vid2spatial_pkg.foa_render import _load_and_prepare
        with tempfile.TemporaryDirectory() as td:
            wav = str(Path(td) / "in.wav")
            sf.write(wav, mod.make_audio(), mod.SR, subtype="FLOAT")
            off, *_ = _load_and_prepare(wav, traj, confidence_gate=False)
            on, *_ = _load_and_prepare(wav, traj, confidence_gate=True)
        assert not np.array_equal(off, on), \
            "the confidence gate must actually change this stimulus"
        assert float(np.max(np.abs(off - on))) > 1e-3
        # and the default is the one that matches the golden
        assert ref["sections"]["prep::depth_rel::audio"]["abs_max"] == pytest.approx(
            float(np.max(np.abs(off))), rel=1e-6)


# ---------------------------------------------------------------------------
# 2026-09-04 review follow-ups
# ---------------------------------------------------------------------------
class TestReviewFollowUps:
    def test_kitti_per_sequence_widths(self):
        ek = _load_tool("eval_azimuth_kitti")
        assert ek.width_for("0000") == 1242
        assert ek.width_for("0014") == 1224
        assert ek.width_for("0018") == 1238
        assert ek.width_for("0020") == 1241
        assert ek.width_for("9999") == ek.DEFAULT_WIDTH
        assert len(ek.SEQ_WIDTH) == 21

    def test_per_sequence_width_changes_the_answer(self):
        ek = _load_tool("eval_azimuth_kitti")
        labels = REPO_ROOT / "data/kitti_tracking/label_02"
        calib = REPO_ROOT / "data/kitti_tracking/calib"
        if not labels.is_dir() or not calib.is_dir():
            pytest.skip("KITTI tracking labels/calib not on disk")
        records, calibs = [], {}
        for lp in sorted(labels.glob("*.txt")):
            cp = calib / lp.name
            if not cp.exists():
                continue
            calibs[lp.stem] = ek.read_calib(cp)
            records.extend(ek.parse_sequence(lp))
        per = ek.evaluate(records, calibs, width=1242, repo_fov_deg=60.0,
                          per_sequence_width=True)
        one = ek.evaluate(records, calibs, width=1242, repo_fov_deg=60.0,
                          per_sequence_width=False)
        assert per["widths_used"] == [1224, 1238, 1241, 1242]
        assert one["widths_used"] == [1242]
        assert per["variants"]["repo"]["azmae_deg"] != pytest.approx(
            one["variants"]["repo"]["azmae_deg"])
        # the committed report numbers
        assert per["variants"]["repo"]["azmae_deg"] == pytest.approx(3.785, abs=0.01)
        assert per["variants"]["truefov"]["azmae_deg"] == pytest.approx(0.908, abs=0.01)
        assert per["variants"]["calib"]["azmae_deg"] == pytest.approx(0.412, abs=0.01)

    def test_iso9613_table_is_the_20c_50rh_reference(self):
        from vid2spatial_pkg.foa_render import (
            ISO9613_ALPHA_DB_PER_KM, ISO9613_FREQ_HZ)
        want = {1000.0: 4.66, 2000.0: 9.89, 4000.0: 29.67, 8000.0: 105.29}
        for f, a in want.items():
            i = int(np.argmin(np.abs(ISO9613_FREQ_HZ - f)))
            assert ISO9613_ALPHA_DB_PER_KM[i] == pytest.approx(a)
        assert np.all(np.diff(ISO9613_ALPHA_DB_PER_KM) > 0)

    def test_vectorised_onepole_matches_the_scalar_loop(self):
        from vid2spatial_pkg.foa_render import _timevarying_onepole
        sr, T = 16000, 16000
        rng = np.random.default_rng(5)
        x = (rng.standard_normal(T) * 0.2).astype(np.float32)
        # the production contract: fc arrives box-smoothed over 0.5 s, so the
        # block approximation is tested against a control of that bandwidth
        from scipy.ndimage import uniform_filter1d
        fc = uniform_filter1d(np.linspace(8000.0, 2000.0, T),
                              size=int(sr * 0.5), mode="nearest").astype(np.float32)
        ref = np.zeros(T, dtype=np.float32)
        prev = 0.0
        for i in range(T):
            a = (2 * np.pi * fc[i]) / (2 * np.pi * fc[i] + sr)
            prev = prev + a * (x[i] - prev)
            ref[i] = prev
        got = _timevarying_onepole(x, fc, sr)
        rms = float(np.sqrt(np.mean(ref ** 2)))
        assert float(np.max(np.abs(ref - got))) < 0.02 * rms

    def test_doppler_tail_fades_instead_of_holding_dc(self):
        from vid2spatial_pkg.foa_render import apply_doppler
        sr = 16000
        t = np.arange(sr, dtype=np.float64) / sr
        x = (np.sin(2 * np.pi * 300 * t) + 1.0).astype(np.float32)  # DC offset
        # steady APPROACH at the clamp: rate > 1, so the read pointer advances
        # faster than the output is written and runs off the end of the input
        dist = (200.0 - 120.0 * t).astype(np.float32)
        y = apply_doppler(x, sr, dist, max_ratio=0.25)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
        # the run-off region must be silent, not a held DC level
        assert abs(float(y[-1])) < 1e-6
        tail = y[-int(0.05 * sr):]
        assert float(np.max(np.abs(tail))) < 0.05

    def test_build_lost_curves_tolerates_a_missing_frame_key(self):
        from vid2spatial_pkg.foa_render import build_lost_curves
        frames = [{"az": 0.0, "el": 0.0, "confidence": 0.2 if 5 <= i < 9 else 1.0}
                  for i in range(20)]
        duck, diffuse = build_lost_curves(frames, 8000, 8000, 25.0)
        assert duck.shape == (8000,)
        assert duck.min() < 0.5 and diffuse.max() > 0.2

    def test_match_length_warns_on_stride_mismatch(self, capsys):
        from vid2spatial_pkg.camera_motion import _match_length
        out = _match_length(np.zeros(50), 100, what="camera yaw")
        assert out.shape == (100,)
        assert "stride mismatch" in capsys.readouterr().out

    def test_camera_cli_args_reach_world_frame(self):
        import argparse
        from vid2spatial_pkg.config import (
            add_camera_cli_args, add_render_cli_args)
        ap = argparse.ArgumentParser()
        add_camera_cli_args(ap)
        add_render_cli_args(ap)
        default = ap.parse_args([])
        assert default.motion_mode == "camera_frame"
        assert default.fov_from_metadata is False
        assert default.focal_35mm is None
        assert default.hrir_interp == "nearest"
        assert default.confidence_gate is False
        assert default.doppler is False
        opted = ap.parse_args(["--motion-mode", "world_frame", "--focal-35mm", "27",
                               "--fov-from-metadata", "--confidence-gate",
                               "--hrir-interp", "barycentric", "--doppler"])
        assert opted.motion_mode == "world_frame"
        assert opted.focal_35mm == pytest.approx(27.0)
        assert opted.fov_from_metadata is True
        assert opted.hrir_interp == "barycentric"
        assert opted.confidence_gate is True and opted.doppler is True

    def test_from_args_carries_the_camera_flags(self):
        import argparse
        from vid2spatial_pkg.config import PipelineConfig, add_camera_cli_args
        ap = argparse.ArgumentParser()
        add_camera_cli_args(ap)
        for name, default in [("video", "v.mp4"), ("audio", "a.wav"),
                              ("traj_json", None), ("air_foa", None),
                              ("brir_L", None), ("brir_R", None),
                              ("room", "6,5,3"), ("mic", "3,2.5,1.5"),
                              ("init_bbox", None), ("fov_deg", 60.0),
                              ("stride", 1), ("method", "yolo"), ("cls", "person"),
                              ("select_track_id", None),
                              ("fallback_center_box", False), ("smooth_alpha", 0.2),
                              ("depth_backend", "auto"), ("use_depth_adapter", False),
                              ("refine_center", False),
                              ("refine_center_method", "grabcut"),
                              ("sam_ckpt", None), ("sam2_model_id", "x"),
                              ("sam2_cfg", None), ("sam2_ckpt", None),
                              ("rt60", 0.5), ("ir_backend", "auto"), ("no_ir", True),
                              ("ang_smooth_ms", 50.0), ("max_deg_per_s", None),
                              ("dist_gain_k", 1.0), ("dist_lpf_min_hz", 800.0),
                              ("dist_lpf_max_hz", 8000.0), ("occ_json", None),
                              ("estimate_occ", False), ("reverb_on", False),
                              ("rev_rt60", 0.6), ("rev_wet_min", 0.05),
                              ("rev_wet_max", 0.35), ("rev_wet_occ_boost", 0.1),
                              ("out_foa", "o.wav"), ("out_st", None),
                              ("out_bin", None), ("binaural_mode", "crossfeed"),
                              ("sofa", None), ("save_traj", None)]:
            ap.add_argument(f"--{name.replace('_', '-')}", dest=name, default=default)
        args = ap.parse_args(["--motion-mode", "world_frame", "--focal-35mm", "27"])
        cfg = PipelineConfig.from_args(args)
        assert cfg.vision.camera.motion_mode == "world_frame"
        assert cfg.vision.camera.focal_35mm == pytest.approx(27.0)
        assert cfg.vision.camera.fov_explicit is False

    def test_all_trajectory_paths_stamp_provenance(self):
        """Legacy and precomputed paths must stamp like the V2 path."""
        import inspect
        from vid2spatial_pkg import pipeline as pl
        src = inspect.getsource(pl.SpatialAudioPipeline._compute_trajectory)
        assert src.count("_apply_camera_motion") == 3, \
            "precomputed, V2 and legacy paths must all apply camera motion"
        assert src.count("_stamp_intrinsics") == 2, \
            "V2 and legacy paths must both stamp intrinsics"
        assert 'intr.setdefault("fov_detail"' in src
        assert 'intr["motion_mode"]' in src

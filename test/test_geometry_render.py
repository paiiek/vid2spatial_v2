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
        assert cfg.fov_from_metadata is True
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
            assert p["hrir_interp"].default == "barycentric"

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

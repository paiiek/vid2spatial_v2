"""
Tests for the geometry + render lane added 2026-09-04 (gap items A1, A2, A3,
A7, A8, A14, A16).

Everything here is CPU-only, model-free and deterministic.  Anything that needs
a real SOFA file or the KITTI labels skips cleanly when they are absent.
"""
from __future__ import annotations

import importlib.util
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

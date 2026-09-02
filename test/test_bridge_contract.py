"""vid2spatial <-> spatial_engine OSC plugin-boundary conformance (no bridge process).

``vid2spatial_pkg/bridge_contract.yaml`` is the contract: the ``emitted:`` half
is what vid2spatial sends, the ``bridge:`` half is extracted from
``spatial_engine/bridge/vid2spatial_osc.py`` by ``tools/extract_bridge_contract.py``.

These tests
  (i)   capture every datagram ``OSCSpatialSender`` and the demo's ``vid2spatial``
        format emit on a real local UDP socket and check address, arg count,
        arg types, ranges and port against the contract;
  (ii)  pin the semantic conventions the bridge relies on -- az sign
        (RIGHT = +), distance normalisation (1 = near), object numbering (1) --
        so flipping any of them in osc_sender / demo / trajectory_export fails;
  (iii) when the bridge tree is present, drive the real bridge handlers
        in-process with the captured datagrams and check the forwarded
        ``/adm/obj/1/aed`` (az negated, dist inverted), then re-extract the
        bridge section and fail on drift.

Env: V2S_BRIDGE_PATH overrides the bridge file (also used for mutation runs).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "test" / "demo"))

from vid2spatial_pkg import trajectory_export as texp  # noqa: E402
from vid2spatial_pkg.osc_sender import DEFAULT_OSC_PORT, OSCSpatialSender  # noqa: E402

pythonosc = pytest.importorskip("pythonosc")
from pythonosc.osc_message import OscMessage  # noqa: E402

CONTRACT_PATH = _REPO / "vid2spatial_pkg" / "bridge_contract.yaml"
EXTRACTOR = _REPO / "tools" / "extract_bridge_contract.py"
BRIDGE_PATH = Path(os.environ.get("V2S_BRIDGE_PATH",
                                  "/home/seung/mmhoa/spatial_engine/bridge/vid2spatial_osc.py"))

# One representative frame: right-front, slightly up, 2.5 m, moving.
FRAME = dict(az_deg=45.0, el_deg=10.0, dist_m=2.5, velocity_deg_s=12.0, timecode_s=1.5, frame_idx=45)


# ── capture helpers ──────────────────────────────────────────────────────────

class _Capture:
    """Local UDP listener that parses OSC datagrams."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.05)
        self.port = self.sock.getsockname()[1]
        self.msgs: list[tuple[str, list]] = []
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._run:
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            m = OscMessage(data)
            self.msgs.append((m.address, list(m.params)))

    def stop(self):
        self._run = False
        self._t.join(1.0)
        self.sock.close()


def _drain(cap: _Capture, n_expected: int, timeout: float = 2.0) -> list[tuple[str, list]]:
    t0 = time.time()
    while len(cap.msgs) < n_expected and time.time() - t0 < timeout:
        time.sleep(0.01)
    time.sleep(0.05)
    return list(cap.msgs)


def _check_msg(contract: dict, addr: str, params: list) -> None:
    spec = contract["emitted"].get(addr)
    assert spec is not None, f"{addr} not in contract emitted:"
    assert len(params) == len(spec["args"]), f"{addr}: {len(params)} args, contract {len(spec['args'])}"
    for p, a in zip(params, spec["args"]):
        want = {"float": float, "int": int}[a["type"]]
        assert type(p) is want, f"{addr}: arg {a['name']} is {type(p).__name__}, contract {a['type']}"
        rng = a.get("range")
        if rng:
            lo, hi = rng
            assert lo is None or p >= lo, f"{addr}: {a['name']}={p} < {lo}"
            assert hi is None or p <= hi, f"{addr}: {a['name']}={p} > {hi}"


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text())


@pytest.fixture(scope="module")
def sender_msgs() -> list[tuple[str, list]]:
    cap = _Capture()
    s = OSCSpatialSender(host="127.0.0.1", port=cap.port)
    assert s.connect()
    s.send_frame(**FRAME)
    s.send_xyz(1.0, 2.0, 3.0, timecode_s=1.5)
    msgs = _drain(cap, 9)
    cap.stop()
    return msgs


@pytest.fixture(scope="module")
def sender_meters_msgs() -> list[tuple[str, list]]:
    cap = _Capture()
    s = OSCSpatialSender(host="127.0.0.1", port=cap.port, distance_mode="meters")
    assert s.connect()
    s.send_frame(**FRAME)
    msgs = _drain(cap, 7)
    cap.stop()
    return msgs


@pytest.fixture(scope="module")
def demo_msgs() -> list[tuple[str, list]]:
    import server as demo  # test/demo/server.py
    cap = _Capture()
    demo._osc_client = None
    demo._osc_config.update({"enabled": True, "host": "127.0.0.1", "port": cap.port, "format": "vid2spatial"})
    traj = {"frames": [{"az": math.radians(FRAME["az_deg"]), "el": math.radians(FRAME["el_deg"]),
                        "dist_m": FRAME["dist_m"], "d_rel": 0.3}]}
    demo.send_osc_trajectory(traj, fps=1000.0)
    msgs = _drain(cap, 3)
    cap.stop()
    demo._osc_config["enabled"] = False
    demo._osc_client = None
    return msgs


# ── (i) every emitted datagram matches the contract ──────────────────────────

def test_contract_shape(contract):
    assert contract["schema"] == 1
    assert set(contract) >= {"transport", "conventions", "emitted", "export", "bridge"}
    assert contract["transport"]["bridge_listen_port"] == contract["bridge"]["listen_port"]
    assert contract["transport"]["bridge_forward_port"] == contract["bridge"]["target_port"]


def test_sender_default_port_is_bridge_listen_port(contract):
    assert DEFAULT_OSC_PORT == contract["transport"]["sender_default_port"]
    assert OSCSpatialSender().config.port == contract["bridge"]["listen_port"]


def test_sender_frame_every_message_in_contract(contract, sender_msgs):
    addrs = [a for a, _ in sender_msgs]
    assert addrs == ["/vid2spatial/azimuth", "/vid2spatial/elevation", "/vid2spatial/distance",
                     "/vid2spatial/velocity", "/vid2spatial/timecode", "/vid2spatial/frame",
                     "/vid2spatial/spatial", "/vid2spatial/xyz", "/vid2spatial/timecode"]
    for addr, params in sender_msgs:
        _check_msg(contract, addr, params)


def test_sender_meters_mode_in_contract(contract, sender_meters_msgs):
    addrs = [a for a, _ in sender_meters_msgs]
    assert "/vid2spatial/distance_m" in addrs and "/vid2spatial/distance" not in addrs
    for addr, params in sender_meters_msgs:
        _check_msg(contract, addr, params)
    dm = dict(sender_meters_msgs)["/vid2spatial/distance_m"]
    assert dm == [FRAME["dist_m"]]


def test_demo_vid2spatial_format_in_contract(contract, demo_msgs):
    addrs = [a for a, _ in demo_msgs]
    assert addrs == ["/vid2spatial/azimuth", "/vid2spatial/elevation", "/vid2spatial/distance"]
    for addr, params in demo_msgs:
        _check_msg(contract, addr, params)
        assert "demo.vid2spatial" in contract["emitted"][addr]["emitters"]


def test_every_handled_address_is_emitted_by_sender(contract, sender_msgs):
    """Every address the bridge maps a handler for is something send_frame emits."""
    sent = {a for a, _ in sender_msgs}
    for addr, spec in contract["emitted"].items():
        if spec["bridge"] == "handled":
            assert addr in contract["bridge"]["handlers"], f"{addr} marked handled but bridge has no handler"
            assert addr in sent
        else:
            assert addr not in contract["bridge"]["handlers"], f"{addr} marked ignored but bridge handles it"
    for addr in contract["bridge"]["handlers"]:
        assert addr in contract["emitted"], f"bridge handles {addr} which vid2spatial never emits"


def test_bridge_arity_satisfied_by_sender(contract, sender_msgs):
    """Each handled datagram carries at least the floats the bridge requires."""
    by_addr = dict(sender_msgs)
    for addr, h in contract["bridge"]["handlers"].items():
        params = by_addr[addr]
        n_float = sum(isinstance(p, float) for p in params)
        assert n_float >= h["required"]["float"], f"{addr}: {n_float} floats < required {h['required']}"
        assert n_float >= h["reads"]["float"], f"{addr}: bridge reads {h['reads']} floats, sender gives {n_float}"


# ── (ii) semantic conventions the bridge depends on ──────────────────────────

def test_az_sign_convention_right_positive(contract, sender_msgs, demo_msgs):
    """Pipeline az is RIGHT = +. The bridge negates it; if vid2spatial ever
    pre-negated, the engine would pan the wrong way."""
    assert contract["conventions"]["az_deg"]["positive"] == "RIGHT"
    assert contract["conventions"]["az_deg"]["bridge_transform"] == "negate"
    assert dict(sender_msgs)["/vid2spatial/azimuth"] == [FRAME["az_deg"]]
    assert dict(sender_msgs)["/vid2spatial/spatial"][0] == FRAME["az_deg"]
    assert dict(demo_msgs)["/vid2spatial/azimuth"] == pytest.approx([FRAME["az_deg"]])
    # bridge formula recorded in the contract must be a plain negation
    assert contract["bridge"]["translate"]["az"]["expr"].replace(" ", "") == "-float(az_pipeline)"


def test_distance_normalisation_near_is_one(contract, sender_msgs, demo_msgs):
    c = contract["conventions"]["dist_norm"]
    assert (c["near"], c["far"]) == (1.0, 0.0)
    want_sender = 1.0 - min(FRAME["dist_m"] / c["sender_distance_max_m"], 1.0)
    want_demo = 1.0 - min(FRAME["dist_m"] / c["demo_distance_max_m"], 1.0)
    assert dict(sender_msgs)["/vid2spatial/distance"] == pytest.approx([want_sender])
    assert dict(demo_msgs)["/vid2spatial/distance"] == pytest.approx([want_demo])
    # monotone: closer => larger wire value
    s = OSCSpatialSender()
    assert s._normalize_distance(0.5) > s._normalize_distance(5.0) > s._normalize_distance(50.0)
    assert s._normalize_distance(0.0) == 1.0 and s._normalize_distance(1e9) == 0.0
    assert texp.distance_to_norm(FRAME["dist_m"], c["sender_distance_max_m"]) == pytest.approx(want_sender)
    assert contract["bridge"]["translate"]["dist"]["expr"].replace(" ", "") == "1.0-float(dist_v2s)"


def test_spatial_bundle_carries_metres_not_normalised(contract, sender_msgs):
    """/vid2spatial/spatial[2] is METRES; the bridge normalises it itself (20 m)."""
    sp = dict(sender_msgs)["/vid2spatial/spatial"]
    assert sp[2] == FRAME["dist_m"]
    assert contract["bridge"]["handlers"]["/vid2spatial/spatial"]["dist_max_m"] == 20.0


def test_elevation_identity(contract, sender_msgs):
    assert contract["conventions"]["el_deg"]["bridge_transform"] == "identity"
    assert dict(sender_msgs)["/vid2spatial/elevation"] == [FRAME["el_deg"]]


def test_object_numbering_agrees(contract, sender_msgs, tmp_path):
    """vid2spatial sends no object id; bridge auto-assigns 1 (1-based, ADR 0006);
    trajectory_export defaults to /adm/obj/1/aed. All three must agree."""
    on = contract["conventions"]["object_numbering"]
    assert on["base"] == "wire_1_based"
    assert on["bridge_first_id"] == contract["bridge"]["object_numbering"]["first_auto_id"] == 1
    assert on["export_default_object_id"] == contract["export"]["default_object_id"] == 1
    assert all("/obj/" not in a for a, _ in sender_msgs)
    p = texp.export_trajectory_json([{"az": 0.0, "el": 0.0, "dist_m": 1.0}], tmp_path / "t.json")
    doc = json.loads(p.read_text())
    assert doc["osc_address"] == "/adm/obj/1/aed"
    assert doc["object_id"] == 1
    assert contract["bridge"]["output"]["address"] == "/adm/obj/{N}/aed"


def test_export_adm_columns_apply_bridge_transform(contract):
    rows = texp.trajectory_to_rows([{"az": math.radians(45.0), "el": math.radians(10.0), "dist_m": 2.5}],
                                   30.0, 1, contract["conventions"]["dist_norm"]["sender_distance_max_m"])
    r = rows[0]
    assert r["az_adm_deg"] == pytest.approx(-r["az_deg"])
    assert r["el_adm_deg"] == pytest.approx(r["el_deg"])
    assert r["dist_adm"] == pytest.approx(1.0 - r["dist_norm"])
    assert r["az_deg"] == pytest.approx(45.0)


# ── (iii) real bridge in-process + drift alarm (skips if bridge tree absent) ──

def _load_bridge():
    if not BRIDGE_PATH.exists():
        pytest.skip(f"bridge source not present: {BRIDGE_PATH}")
    spec = importlib.util.spec_from_file_location("v2s_bridge_under_test", BRIDGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeClient:
    def __init__(self):
        self.sent: list[tuple[str, list]] = []

    def send_message(self, addr, args):
        self.sent.append((addr, list(args)))


def _run_bridge(mod, msgs):
    srv = mod.BridgeServer(iir_alpha=1.0, rate_limit_hz=1e9)
    srv._client = _FakeClient()
    disp = {"/vid2spatial/azimuth": srv._handle_azimuth, "/vid2spatial/elevation": srv._handle_elevation,
            "/vid2spatial/distance": srv._handle_distance, "/vid2spatial/spatial": srv._handle_spatial}
    for addr, params in msgs:
        disp.get(addr, srv._handle_fallback)(addr, *params)
    return srv._client.sent


def test_real_bridge_forwards_sender_frame(contract, sender_msgs):
    mod = _load_bridge()
    out = _run_bridge(mod, sender_msgs)
    assert out, "bridge forwarded nothing"
    assert {a for a, _ in out} == {"/adm/obj/1/aed"}, "bridge must forward obj 1 (1-based) only"
    for _, args in out:
        assert len(args) == 3 and all(type(a) is float for a in args)
    az, el, dist = out[-1][1]
    assert az == pytest.approx(-FRAME["az_deg"])
    assert el == pytest.approx(FRAME["el_deg"])
    # last flush comes from /spatial (metres, 20 m normalisation): dist_adm = dist_m / 20
    assert dist == pytest.approx(FRAME["dist_m"] / 20.0)
    # and the /distance-driven flush (before /spatial) inverted the sender's 10 m normalisation
    idx = [a for a, _ in sender_msgs].index("/vid2spatial/distance")
    assert out[idx][1][2] == pytest.approx(FRAME["dist_m"] / 10.0)


def test_real_bridge_forwards_demo_frame(contract, demo_msgs):
    mod = _load_bridge()
    out = _run_bridge(mod, demo_msgs)
    az, el, dist = out[-1][1]
    assert out[-1][0] == "/adm/obj/1/aed"
    assert az == pytest.approx(-FRAME["az_deg"])
    assert dist == pytest.approx(FRAME["dist_m"] / contract["conventions"]["dist_norm"]["demo_distance_max_m"])


def test_bridge_listen_and_forward_ports(contract):
    mod = _load_bridge()
    assert mod._DEFAULTS["listen_port"] == contract["transport"]["bridge_listen_port"] == DEFAULT_OSC_PORT
    assert mod._DEFAULTS["target_port"] == contract["transport"]["bridge_forward_port"]


def test_bridge_contract_no_drift():
    """Re-extract from the bridge tree and fail on semantic drift (the alarm)."""
    if not BRIDGE_PATH.exists():
        pytest.skip(f"bridge source not present: {BRIDGE_PATH}")
    r = subprocess.run([sys.executable, str(EXTRACTOR), "--check", "--bridge", str(BRIDGE_PATH)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"bridge contract drift:\n{r.stdout}\n{r.stderr}"

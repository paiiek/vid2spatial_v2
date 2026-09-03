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
    msgs = _drain(cap, 8)
    cap.stop()
    return msgs


@pytest.fixture(scope="module")
def sender_legacy_msgs() -> list[tuple[str, list]]:
    """Opt-in legacy bundle path (--legacy-spatial)."""
    cap = _Capture()
    s = OSCSpatialSender(host="127.0.0.1", port=cap.port, legacy_spatial=True)
    assert s.connect()
    s.send_frame(**FRAME)
    msgs = _drain(cap, 7)
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
                     "/vid2spatial/xyz", "/vid2spatial/timecode"]
    assert "/vid2spatial/spatial" not in addrs, (
        "the legacy metric bundle must not be emitted by default: it arrives after "
        "/distance and the bridge re-normalises it with its own constant")
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
            if "legacy_spatial" in str(spec.get("condition", "")):
                assert addr not in sent, f"{addr} is legacy-gated but was emitted by default"
            else:
                assert addr in sent
        else:
            assert addr not in contract["bridge"]["handlers"], f"{addr} marked ignored but bridge handles it"
    for addr in contract["bridge"]["handlers"]:
        assert addr in contract["emitted"], f"bridge handles {addr} which vid2spatial never emits"


def test_bridge_arity_satisfied_by_sender(contract, sender_msgs):
    """Each handled datagram carries at least the floats the bridge requires."""
    by_addr = dict(sender_msgs)
    for addr, h in contract["bridge"]["handlers"].items():
        if addr not in by_addr:  # condition-gated (legacy bundle)
            continue
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
    assert "/vid2spatial/spatial" not in dict(sender_msgs)
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


def test_spatial_bundle_carries_metres_not_normalised(contract, sender_legacy_msgs):
    """/vid2spatial/spatial[2] is METRES; the bridge normalises it itself.

    Only emitted under --legacy-spatial (see test_legacy_bundle_is_opt_in)."""
    sp = dict(sender_legacy_msgs)["/vid2spatial/spatial"]
    assert sp[2] == FRAME["dist_m"]
    assert sp[0] == FRAME["az_deg"]
    assert "dist_max_m" in contract["bridge"]["handlers"]["/vid2spatial/spatial"]


def test_sender_and_bridge_distance_laws_agree(contract):
    """LOUD GUARD on the attach boundary.

    send_frame emits /vid2spatial/distance normalised over the sender's
    distance_max_m AND, last, /vid2spatial/spatial carrying raw metres that the
    bridge normalises over its own DISTANCE_MAX_M.  The bundle arrives last and
    wins, so if the two constants disagree the engine silently receives a
    rescaled distance -- exactly the 10 m vs 20 m halving that shipped before
    fix/lane-bridge-handoff.  Never let them drift apart again.
    """
    from vid2spatial_pkg.osc_sender import OSCConfig
    bridge_max = contract["bridge"]["handlers"]["/vid2spatial/spatial"]["dist_max_m"]
    sender_max = OSCConfig().distance_max_m
    contract_max = contract["conventions"]["dist_norm"]["sender_distance_max_m"]
    assert sender_max == bridge_max == contract_max, (
        f"distance normalisation disagrees: sender={sender_max} m, "
        f"bridge={bridge_max} m, contract={contract_max} m -- the engine would "
        f"receive distances scaled by {bridge_max / sender_max:.2f}x")
    # and the two paths must therefore produce the same normalised value
    for dist_m in (0.0, 2.5, 5.0, 9.9, 10.0, 25.0):
        via_distance = 1.0 - min(dist_m / sender_max, 1.0)
        via_spatial = max(0.0, min(1.0, 1.0 - dist_m / bridge_max))
        assert abs(via_distance - via_spatial) < 1e-9, dist_m


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
    # With the legacy bundle suppressed, the sender's own 10 m law is the last
    # (and only) word on distance, whatever the bridge's DISTANCE_MAX_M is.
    assert dist == pytest.approx(FRAME["dist_m"] / 10.0)


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


# ── attach-readiness: tools/attach_engine.py ────────────────────────────────

def _attach():
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "tools" / "attach_engine.py"
    spec = importlib.util.spec_from_file_location("attach_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _defaults(mod):
    return {a.dest: a.default for a in mod.build_parser()._actions}


def test_attach_defaults_match_the_contract(contract):
    """The attach command's defaults are the engine's values, not guesses."""
    d = _defaults(_attach())
    assert d["object_id"] == contract["export"]["default_object_id"] == 1
    assert d["distance_max_m"] == \
        contract["bridge"]["handlers"]["/vid2spatial/spatial"]["dist_max_m"] == 10.0
    assert d["az_sign"] == "right-positive"
    assert d["port"] == contract["bridge"]["listen_port"] == 9000
    assert d["host"] == "127.0.0.1"


def test_attach_rejects_zero_object_id():
    """ADM object numbers are 1-based; 0 must be refused, not silently sent."""
    mod = _attach()
    with pytest.raises(SystemExit):
        mod.main(["--object-id", "0", "--check-engine"])


def test_attach_preflight_catches_low_latency_mode_file(tmp_path, monkeypatch):
    """A stale /tmp/.spe_bridge_mode silences the bridge with no error at all."""
    mod = _attach()
    f = tmp_path / "mode"
    monkeypatch.setattr(mod, "BRIDGE_MODE_FILE", f)
    f.write_text("low_latency\n")
    with pytest.raises(mod.PreflightError, match="low_latency"):
        mod._check_bridge_mode()
    f.write_text("ai\n")
    assert "forwarding enabled" in mod._check_bridge_mode()
    f.unlink()
    assert "no override" in mod._check_bridge_mode() or "no /" in mod._check_bridge_mode()


def test_attach_preflight_catches_distance_law_mismatch(monkeypatch):
    """The 10 m vs 20 m halving must fail the preflight, not ship silently."""
    mod = _attach()
    assert "distance_max_m=10.0" in mod._check_constants()
    from vid2spatial_pkg import osc_sender

    class Bad:
        distance_max_m = 20.0

    monkeypatch.setattr(osc_sender, "OSCConfig", lambda: Bad())
    with pytest.raises(mod.PreflightError, match="disagrees"):
        mod._check_constants()


def test_attach_preflight_fails_when_nothing_is_listening():
    """Unreachable engine must fail loudly rather than stream into a void."""
    mod = _attach()
    with pytest.raises(mod.PreflightError):
        mod._check_roundtrip("127.0.0.1", 9, timeout=0.4)


def test_attach_dry_run_emits_engine_side_values(tmp_path, capsys):
    """Dry run must show the ADM values the engine will actually receive."""
    mod = _attach()
    traj = tmp_path / "t.json"
    traj.write_text(json.dumps({"fps": 30.0, "frames": [
        {"frame": 0, "az": math.radians(-45.0), "el": 0.0, "dist_m": 2.5}]}))
    assert mod.main([str(traj), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "/adm/obj/1/aed" in out
    # az -45 (left of image) must reach the engine as +45 (ADM is left-positive)
    assert "45.0000" in out
    # 2.5 m over a 10 m range -> v2s 0.75 near -> ADM dist 0.25
    assert "0.2500" in out


def test_engine_audio_diagnostic_is_runnable_and_documents_the_layout_cause():
    """The engine-silence report was wrong: a missing speaker layout, not the
    engine, was the cause. Keep the corrected conclusion and the matrix that
    proves it attached to the script, and keep the script executable."""
    script = Path(__file__).resolve().parent.parent / "tools" / "repro_engine_silence.sh"
    assert script.exists()
    assert os.access(script, os.X_OK), "diagnostic must stay executable"
    body = script.read_text()
    # the corrected verdict, not the old "engine renders silence" claim
    assert "VALID SPEAKER LAYOUT" in body
    assert "engine HEALTHY" in body
    # it must exercise the object path AND the object-bypassing noise path,
    # both with and without a layout, or it cannot isolate the cause
    for probe in ("/noise/", "/adm/obj/1/aed", "--layout", "A_nolayout_noise",
                  "C_layout_noise", "D_layout_object"):
        assert probe in body, probe
    # the two red herrings must stay recorded so nobody re-chases them
    assert "INTERNAL" in body and "0-BASED" in body
    assert "--object-source sine" in body


def test_readme_documents_the_layout_requirement():
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert "speaker layout, or it renders silence" in readme
    assert "--layout" in readme
    # and the id-base fact the engine's own wire contract pins
    assert "internal 0-based" in readme


# ── A10: the live 2x distance error ─────────────────────────────────────────

def test_legacy_bundle_is_opt_in_and_overrides_distance(contract, sender_legacy_msgs):
    """The bundle exists, is off by default, and demonstrably wins when on.

    This is the failure that shipped: /spatial arrives after /distance and the
    bridge normalises its metres with its OWN constant, so a bridge whose
    DISTANCE_MAX_M differs from the sender's places every object at the wrong
    distance -- 2x too near against the installed 20 m bridge.
    """
    from vid2spatial_pkg.osc_sender import OSCConfig
    assert OSCConfig().legacy_spatial is False
    addrs = [a for a, _ in sender_legacy_msgs]
    assert addrs[-1] == "/vid2spatial/spatial", "the bundle is emitted last -- that is why it wins"
    for addr, params in sender_legacy_msgs:
        _check_msg(contract, addr, params)

    mod = _load_bridge()
    # the bridge's own metric constant, read out of its source (it is inline)
    import re
    m = re.search(r"1\.0\s*-\s*dist_m\s*/\s*([0-9.]+)", BRIDGE_PATH.read_text())
    assert m, "could not locate the bridge's /spatial normalisation constant"
    bridge_max = float(m.group(1))

    out = _run_bridge(mod, sender_legacy_msgs)
    dist_legacy = out[-1][1][2]
    idx = addrs.index("/vid2spatial/distance")
    dist_correct = out[idx][1][2]
    assert dist_correct == pytest.approx(FRAME["dist_m"] / 10.0)
    assert dist_legacy == pytest.approx(min(1.0, FRAME["dist_m"] / bridge_max))
    if bridge_max != 10.0:
        assert dist_legacy != pytest.approx(dist_correct), (
            f"bridge normalises /spatial over {bridge_max} m, sender over 10 m, "
            "yet the forwarded distance did not change -- test is not exercising the bug")


@pytest.mark.parametrize("dist_m,want_adm", [(0.0, 0.0), (2.5, 0.25), (5.0, 0.5), (10.0, 1.0)])
def test_ten_metre_trajectory_arrives_as_adm_dist_one(dist_m, want_adm):
    """A10 regression, end to end over real UDP into the real bridge handlers.

    A source at the 10 m horizon must reach the engine as ADM dist = 1.0 (far),
    and every intermediate distance must land on the 10 m law. Before the fix
    the trailing /spatial bundle halved all of these against the 20 m bridge.
    """
    mod = _load_bridge()
    cap = _Capture()
    s = OSCSpatialSender(host="127.0.0.1", port=cap.port)
    assert s.connect()
    s.send_frame(az_deg=0.0, el_deg=0.0, dist_m=dist_m)
    msgs = _drain(cap, 6)
    cap.stop()
    assert "/vid2spatial/spatial" not in [a for a, _ in msgs]
    out = _run_bridge(mod, msgs)
    assert out[-1][0] == "/adm/obj/1/aed"
    assert out[-1][1][2] == pytest.approx(want_adm, abs=1e-6)


def test_legacy_spatial_reaches_the_cli():
    """--legacy-spatial must exist, default off, and feed OSCConfig."""
    import inspect
    from vid2spatial_pkg import osc_sender as m
    src = inspect.getsource(m.main)
    assert "--legacy-spatial" in src and "legacy_spatial=args.legacy_spatial" in src
    assert "legacy_spatial" in inspect.signature(m.OSCSpatialSender.__init__).parameters
    assert m.OSCSpatialSender(legacy_spatial=True).config.legacy_spatial is True

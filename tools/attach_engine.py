#!/usr/bin/env python3
"""Stream a vid2spatial trajectory to a running spatial_engine bridge.

This is the attach point. vid2spatial is developed detached from the engine, so
everything that could silently disagree across the boundary is an explicit flag
here, defaulted to the values the engine's bridge actually uses:

  --distance-max-m 10.0   metres mapped to normalised 1.0 (near) .. 0.0 (far).
                          MUST equal the bridge's DISTANCE_MAX_M or the engine
                          receives a rescaled distance and nothing errors.
  --object-id 1           ADM object numbers are 1-BASED (/adm/obj/1/aed).
  --az-sign right-positive
                          vid2spatial azimuth is RIGHT-positive; ADM-OSC is
                          LEFT-positive, so the bridge negates. Pass
                          left-positive only if you are feeding pre-negated
                          angles and want that suppressed.

Preflight (--check-engine) refuses to stream when the wire contract has drifted
or the engine is unreachable, instead of streaming into a void.

Usage
-----
  # preflight only: contract + reachability + one round-trip
  python3 tools/attach_engine.py --check-engine --host 127.0.0.1 --port 9000

  # stream a trajectory file
  python3 tools/attach_engine.py traj.json --host 127.0.0.1 --port 9000

  # dry run: print every packet instead of sending
  python3 tools/attach_engine.py traj.json --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

CONTRACT = _REPO / "vid2spatial_pkg" / "bridge_contract.yaml"
EXTRACTOR = _REPO / "tools" / "extract_bridge_contract.py"


# ── preflight ────────────────────────────────────────────────────────────────

class PreflightError(RuntimeError):
    pass


def _check_contract() -> str:
    """Wire contract must not have drifted from the engine bridge source."""
    if not CONTRACT.exists():
        raise PreflightError(f"bridge contract missing: {CONTRACT}")
    r = subprocess.run([sys.executable, str(EXTRACTOR), "--check"],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        raise PreflightError(
            "wire contract has DRIFTED from the engine bridge — do NOT attach.\n"
            + out + "\n  Fix the engine or run: python3 tools/extract_bridge_contract.py")
    return out.splitlines()[-1] if out else "contract OK"


def _check_constants() -> str:
    """Sender and bridge distance laws must agree, or distance is silently rescaled."""
    import yaml
    from vid2spatial_pkg.osc_sender import OSCConfig
    doc = yaml.safe_load(CONTRACT.read_text())
    bridge_max = doc["bridge"]["handlers"]["/vid2spatial/spatial"]["dist_max_m"]
    sender_max = OSCConfig().distance_max_m
    if sender_max != bridge_max:
        raise PreflightError(
            f"distance normalisation disagrees: sender={sender_max} m vs "
            f"bridge={bridge_max} m — the engine would receive distances scaled "
            f"by {bridge_max / sender_max:.2f}x, with no error anywhere.")
    obj_base = doc["export"]["default_object_id"]
    if obj_base != 1:
        raise PreflightError(f"ADM object ids must be 1-based, contract says {obj_base}")
    return f"distance_max_m={sender_max} m, object ids 1-based, az right-positive -> bridge negates"


BRIDGE_MODE_FILE = Path("/tmp/.spe_bridge_mode")


def _check_bridge_mode() -> str:
    """The bridge polls a GLOBAL file and silently stops forwarding in
    low_latency mode.

    /tmp/.spe_bridge_mode is a world-writable path shared by every bridge and
    WebGUI on the box. A stale "low_latency" in it overrides both config.yaml
    and --mode, and the bridge then accepts every packet and forwards NOTHING,
    logging one line at startup and nothing after. Observed live: 40 frames in,
    0 out. Catch it here rather than in a silent listening session.
    """
    if not BRIDGE_MODE_FILE.exists():
        return "no /tmp/.spe_bridge_mode override (bridge uses its own config)"
    mode = BRIDGE_MODE_FILE.read_text().strip()
    if mode == "low_latency":
        raise PreflightError(
            f"{BRIDGE_MODE_FILE} says 'low_latency' — the bridge will forward "
            f"NOTHING to the engine and will not say so. Remove the file or "
            f"write 'ai' into it: echo ai > {BRIDGE_MODE_FILE}")
    if mode not in ("ai", ""):
        raise PreflightError(f"{BRIDGE_MODE_FILE} holds unknown mode {mode!r}")
    return f"{BRIDGE_MODE_FILE} = {mode!r} (forwarding enabled)"


def _check_reachable(host: str, port: int, timeout: float = 2.0) -> str:
    """UDP has no handshake, so probe the port and catch the ICMP refusal."""
    try:
        addr = socket.getaddrinfo(host, port, proto=socket.IPPROTO_UDP)[0][4]
    except OSError as e:
        raise PreflightError(f"cannot resolve {host}:{port}: {e}") from e
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        # An OSC no-op the bridge ignores; a closed port answers ICMP
        # port-unreachable, which the next send surfaces as ECONNREFUSED.
        for _ in range(2):
            s.sendto(_osc_packet("/vid2spatial/timecode", [0.0]), addr)
            time.sleep(0.15)
    except OSError as e:
        raise PreflightError(
            f"engine bridge unreachable at {host}:{port} ({e}). "
            f"Start it with: python3 bridge/vid2spatial_osc.py "
            f"--listen-port {port} --target-port 9100") from e
    finally:
        s.close()
    return f"{host}:{port} accepted UDP (no ICMP refusal)"


def _check_roundtrip(host: str, port: int, timeout: float = 2.0) -> str:
    """Send one frame through the bridge and catch what it forwards on 9100."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        return "round-trip SKIPPED (remote host; cannot bind its forward port)"
    try:
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        rx.bind(("127.0.0.1", 9100))
    except OSError:
        return "round-trip SKIPPED (port 9100 already bound — the engine has it)"
    rx.settimeout(timeout)
    try:
        from vid2spatial_pkg.osc_sender import OSCSpatialSender
        s = OSCSpatialSender(host=host, port=port)
        if not s.connect():
            raise PreflightError("OSCSpatialSender.connect() failed")
        s.send_frame(az_deg=45.0, el_deg=10.0, dist_m=2.5)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data, _ = rx.recvfrom(4096)
            except socket.timeout:
                break
            addr = data.split(b"\0", 1)[0].decode("ascii", "replace")
            if addr.startswith("/adm/obj/"):
                return f"round-trip OK: bridge forwarded {addr}"
        raise PreflightError(
            "no /adm/obj/N/aed seen on 9100 within "
            f"{timeout}s — the bridge is not forwarding. Is it running?")
    finally:
        rx.close()


def preflight(host: str, port: int, *, roundtrip: bool = True) -> int:
    steps = [("wire contract", lambda: _check_contract()),
             ("boundary constants", lambda: _check_constants()),
             ("bridge mode", lambda: _check_bridge_mode()),
             ("reachability", lambda: _check_reachable(host, port))]
    if roundtrip:
        steps.append(("round-trip", lambda: _check_roundtrip(host, port)))
    failed = 0
    for name, fn in steps:
        try:
            print(f"  [ OK ] {name:20s} {fn()}")
        except PreflightError as e:
            print(f"  [FAIL] {name:20s} {e}", file=sys.stderr)
            failed += 1
            break  # later steps are meaningless once one fails
    if failed:
        print("\nPREFLIGHT FAILED — not attaching.", file=sys.stderr)
        return 1
    print("\nPREFLIGHT OK — safe to attach.")
    return 0


# ── minimal OSC encoder (probe only; real streaming uses OSCSpatialSender) ────

def _osc_packet(address: str, args: list) -> bytes:
    def pad(b: bytes) -> bytes:
        return b + b"\0" * (4 - len(b) % 4)
    out = pad(address.encode("ascii") + b"\0")
    tags = "," + "".join("i" if isinstance(a, int) else "f" for a in args)
    out += pad(tags.encode("ascii") + b"\0")
    import struct
    for a in args:
        out += struct.pack(">i" if isinstance(a, int) else ">f", a)
    return out


# ── streaming ────────────────────────────────────────────────────────────────

def load_trajectory(path: Path) -> tuple[list, float]:
    doc = json.loads(path.read_text())
    frames = doc["frames"] if isinstance(doc, dict) else doc
    fps = float(doc.get("fps", 30.0)) if isinstance(doc, dict) else 30.0
    if not frames:
        raise SystemExit(f"{path}: no frames")
    return frames, fps


def stream(frames, fps, host, port, *, object_id, distance_max_m, az_sign,
           realtime, dry_run, limit=None):
    from vid2spatial_pkg.osc_sender import OSCSpatialSender, OSCConfig
    cfg = OSCConfig(host=host, port=port, distance_max_m=distance_max_m)
    sender = OSCSpatialSender(config=cfg) if _accepts_config() else \
        OSCSpatialSender(host=host, port=port)
    sender.config.distance_max_m = distance_max_m
    flip = -1.0 if az_sign == "left-positive" else 1.0

    if not dry_run and not sender.connect():
        raise SystemExit(f"could not connect to {host}:{port}")

    shown = 0
    for i, f in enumerate(frames):
        if limit is not None and i >= limit:
            break
        az_deg = math.degrees(float(f.get("az", 0.0))) * flip
        el_deg = math.degrees(float(f.get("el", 0.0)))
        dist_m = float(f.get("dist_m", f.get("dist", 1.0)))
        t_s = i / fps
        if dry_run or shown < 3:
            norm = max(0.0, min(1.0, 1.0 - min(dist_m / distance_max_m, 1.0)))
            print(f"  frame {i:4d} t={t_s:6.3f}s -> "
                  f"/vid2spatial/azimuth [{az_deg:.4f}]  "
                  f"/vid2spatial/elevation [{el_deg:.4f}]  "
                  f"/vid2spatial/distance [{norm:.4f}]  "
                  f"/vid2spatial/spatial [{az_deg:.4f}, {el_deg:.4f}, {dist_m:.4f}, 0.0, {t_s:.4f}]"
                  f"   => engine /adm/obj/{object_id}/aed "
                  f"[{-az_deg:.4f}, {el_deg:.4f}, {1.0 - norm:.4f}]")
            shown += 1
        if not dry_run:
            sender.send_frame(az_deg=az_deg, el_deg=el_deg, dist_m=dist_m,
                              timecode_s=t_s, frame_idx=i)
            if realtime:
                time.sleep(1.0 / fps)
    n = min(len(frames), limit) if limit is not None else len(frames)
    print(f"{'would send' if dry_run else 'sent'} {n} frames "
          f"to {host}:{port} as object {object_id}")


def _accepts_config() -> bool:
    import inspect
    from vid2spatial_pkg.osc_sender import OSCSpatialSender
    return "config" in inspect.signature(OSCSpatialSender.__init__).parameters


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("trajectory", nargs="?", type=Path,
                    help="trajectory .json (omit with --check-engine)")
    ap.add_argument("--host", default="127.0.0.1", help="engine bridge host")
    ap.add_argument("--port", type=int, default=9000, help="bridge listen port")
    ap.add_argument("--object-id", type=int, default=1,
                    help="ADM object number, 1-BASED (default 1)")
    ap.add_argument("--distance-max-m", type=float, default=10.0,
                    help="metres -> normalised 1..0; must equal the bridge's "
                         "DISTANCE_MAX_M (default 10.0)")
    ap.add_argument("--az-sign", choices=("right-positive", "left-positive"),
                    default="right-positive",
                    help="sign of the azimuth you are feeding (default "
                         "right-positive, the vid2spatial convention)")
    ap.add_argument("--check-engine", action="store_true",
                    help="preflight only: contract, constants, reachability, round-trip")
    ap.add_argument("--no-roundtrip", action="store_true",
                    help="skip the round-trip step of the preflight")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="stream without checking (not recommended)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every packet instead of sending")
    ap.add_argument("--limit", type=int, help="only the first N frames")
    ap.add_argument("--no-realtime", action="store_true",
                    help="send as fast as possible instead of at fps")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.object_id < 1:
        ap.error("--object-id is 1-based; 0 is not a valid ADM object")

    if args.check_engine:
        print(f"Preflight against {args.host}:{args.port}")
        return preflight(args.host, args.port, roundtrip=not args.no_roundtrip)

    if args.trajectory is None:
        ap.error("a trajectory file is required unless --check-engine is given")

    if not args.skip_preflight and not args.dry_run:
        print(f"Preflight against {args.host}:{args.port}")
        if preflight(args.host, args.port, roundtrip=not args.no_roundtrip):
            return 1
        print()

    frames, fps = load_trajectory(args.trajectory)
    print(f"{args.trajectory}: {len(frames)} frames @ {fps} fps")
    stream(frames, fps, args.host, args.port,
           object_id=args.object_id, distance_max_m=args.distance_max_m,
           az_sign=args.az_sign, realtime=not args.no_realtime,
           dry_run=args.dry_run, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

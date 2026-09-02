#!/usr/bin/env python3
"""Extract the bridge-side half of the vid2spatial <-> spatial_engine OSC contract.

Why this exists
---------------
vid2spatial is developed detached from spatial_engine (a "plugin boundary").
The only consumer of its OSC stream is ``spatial_engine/bridge/vid2spatial_osc.py``,
which listens on 9000, negates azimuth, inverts distance, and forwards
``/adm/obj/N/aed`` on 9100. Any of those can drift silently: the bridge is
Python in another repo, and nothing on the vid2spatial side runs it.

This tool reads the bridge SOURCE with regexes (no OSC socket, no engine
process) and writes the ``bridge:`` section of
``vid2spatial_pkg/bridge_contract.yaml``. The committed YAML is what
``test/test_bridge_contract.py`` checks every emitted datagram against;
``--check`` re-extracts from the bridge tree and exits non-zero when the
SEMANTIC content differs (ports, handled addresses, arg counts, defaults,
translation formulas, output address, object base). Provenance (line
numbers, commits) is refreshed but does not fail ``--check``.

The vid2spatial-side ``emitted:`` / ``conventions:`` sections are authored by
hand and are NOT touched by this tool; the test suite proves them against
live captures of ``OSCSpatialSender`` and the demo's ``vid2spatial`` format.

Usage
-----
  python3 tools/extract_bridge_contract.py            # refresh bridge: section
  python3 tools/extract_bridge_contract.py --check    # drift alarm (exit 1)
  python3 tools/extract_bridge_contract.py --bridge PATH/vid2spatial_osc.py
  (env V2S_BRIDGE_PATH also overrides the bridge file)
If the bridge file is absent, --check prints a SKIP line and exits 0.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_BRIDGE = Path(os.environ.get(
    "V2S_BRIDGE_PATH", "/home/seung/mmhoa/spatial_engine/bridge/vid2spatial_osc.py"))
DEFAULT_ADR = Path("/home/seung/mmhoa/spatial_engine-proto/docs/adr/vid2spatial_osc_contract.md")
DEFAULT_OUT = _REPO / "vid2spatial_pkg" / "bridge_contract.yaml"

PROVENANCE_KEYS = ("source",)  # per-entry line numbers / commits: reported, never a failure
_HEADER = """\
# vid2spatial <-> spatial_engine OSC plugin boundary.
# `emitted:` / `conventions:` / `export:` are AUTHORED and proven by
# test/test_bridge_contract.py against live UDP captures of every sender.
# `bridge:` is GENERATED from the bridge source by tools/extract_bridge_contract.py
# (`--check` fails on semantic drift). Do not hand-edit `bridge:`.
"""

_RE_DEFAULT = re.compile(r'^\s*(listen_port|target_port|target_host|mode|iir_alpha|'
                         r'rate_limit_hz|max_objects)\s*=\s*([^,\n]+),', re.M)
_RE_MAP = re.compile(r'disp\.map\(\s*"([^"]+)"\s*,\s*self\.(_\w+)\s*\)')
_RE_DEF = re.compile(r'^\s*def\s+(\w+)\s*\(')
_RE_ARGS0 = re.compile(r'float\(args\[0\]\)\s+if\s+args\s+else\s+(-?[\d.]+)')
_RE_UPDATE = re.compile(r'_update_state\(\s*"(\w+)"\s*,\s*([^)]*)\)')
_RE_LEN = re.compile(r'len\(args\)\s*>=\s*(\d+)')
_RE_ARGI = re.compile(r'float\(args\[(\d+)\]\)')
_RE_SPATIAL_NORM = re.compile(r'dist_norm\s*=\s*(max\(.*?\)\))')
_RE_SEND = re.compile(r'send_message\(\s*f"([^"]+)"\s*,\s*\[([^\]]*)\]')
_RE_RETURN = re.compile(r'return\s+(.+)')
_RE_NEXT = re.compile(r'self\._next\s*=\s*(.+)')
_RE_STATE_DEFAULT = re.compile(r's\.get\(\s*"(\w+)"\s*,\s*(-?[\d.]+)\)')


def _git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _lineno(src: str, idx: int) -> int:
    return src.count("\n", 0, idx) + 1


def _func(src: str, name: str, cls: str | None = None) -> tuple[str, int]:
    """Body text + line number of `def name(` (first occurrence, or within class cls)."""
    lo = 0
    if cls:
        m = re.search(rf'^class\s+{cls}\b', src, re.M)
        if not m:
            raise ValueError(f"class {cls} not found")
        lo = m.start()
    m = re.search(rf'^([ \t]*)def\s+{name}\s*\(', src[lo:], re.M)
    if not m:
        raise ValueError(f"def {name} not found")
    start = lo + m.start()
    indent = len(m.group(1))
    line_end = src.index("\n", start) + 1
    end = len(src)
    for mm in re.finditer(r'^([ \t]*)(?:def|class)\s', src[line_end:], re.M):
        if len(mm.group(1)) <= indent:
            end = line_end + mm.start()
            break
    return src[start:end], _lineno(src, start)


def _literal(text: str):
    text = text.strip()
    if text.startswith(("'", '"')):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text in ("{}", "[]"):
        return {} if text == "{}" else []
    return text


def extract(bridge: Path) -> dict:
    src = bridge.read_text()
    rel = f"bridge/{bridge.name}"

    # -- defaults ------------------------------------------------------------
    m = re.search(r'_DEFAULTS\s*=\s*dict\(', src)
    if not m:
        raise ValueError("_DEFAULTS not found")
    defaults = {k: _literal(v) for k, v in _RE_DEFAULT.findall(src[m.end():m.end() + 600])}
    for req in ("listen_port", "target_port", "target_host", "iir_alpha", "rate_limit_hz", "max_objects"):
        if req not in defaults:
            raise ValueError(f"_DEFAULTS.{req} not found")

    # -- translation formulas -------------------------------------------------
    translate = {}
    for name, key in (("az_pipeline_to_adm", "az"), ("dist_v2s_to_adm", "dist"), ("elev_to_adm", "el")):
        body, ln = _func(src, name, cls="OscTranslator")
        r = _RE_RETURN.search(body)
        if not r:
            raise ValueError(f"{name}: no return")
        translate[key] = {"expr": r.group(1).strip(), "source": f"{rel}:{ln}"}

    # -- object numbering ------------------------------------------------------
    body, ln = _func(src, "__init__", cls="ObjectMapper")
    nx = _RE_NEXT.search(body)
    if not nx:
        raise ValueError("ObjectMapper._next not found")
    next_expr = nx.group(1).strip()
    first = eval(next_expr.replace("self._map.values()", "[]"))  # noqa: S307 - literal arithmetic on bridge source
    obj = {"first_auto_id": int(first), "next_expr": next_expr,
           "max_objects_default": defaults["max_objects"],
           "source": f"{rel}:{ln}"}

    # -- handlers ---------------------------------------------------------------
    start_body, _ = _func(src, "start", cls="BridgeServer")
    handlers = {}
    for addr, hname in _RE_MAP.findall(start_body):
        hbody, hln = _func(src, hname, cls="BridgeServer")
        h = {"handler": hname, "source": f"{rel}:{hln}"}
        a0 = _RE_ARGS0.search(hbody)
        if a0:
            h["reads"] = {"float": 1}
            h["required"] = {"float": 0}
            h["default_if_absent"] = float(a0.group(1))
        ln_m = _RE_LEN.search(hbody)
        if ln_m:
            idxs = [int(i) for i in _RE_ARGI.findall(hbody)]
            h["reads"] = {"float": max(idxs) + 1 if idxs else 0}
            h["required"] = {"float": int(ln_m.group(1))}
        up = _RE_UPDATE.search(hbody)
        if up:
            h["track"] = up.group(1)
            h["state_fields"] = [kv.split("=")[0].strip() for kv in up.group(2).split(",")]
        sn = _RE_SPATIAL_NORM.search(hbody)
        if sn:
            h["dist_norm_expr"] = sn.group(1)
            mm = re.search(r'/\s*([\d.]+)', sn.group(1))
            h["dist_max_m"] = float(mm.group(1)) if mm else None
        handlers[addr] = h
    if not handlers:
        raise ValueError("no disp.map handlers found")
    fallback = "set_default_handler" in start_body

    # -- flush / output ------------------------------------------------------------
    fbody, fln = _func(src, "_flush", cls="BridgeServer")
    sd = _RE_SEND.search(fbody)
    if not sd:
        raise ValueError("_flush: send_message not found")
    out_addr = re.sub(r'\{[^}]*\}', "{N}", sd.group(1))
    n_args = len([a for a in sd.group(2).split(",") if a.strip()])
    state_defaults = {k: float(v) for k, v in _RE_STATE_DEFAULT.findall(fbody)}
    output = {"address": out_addr, "args": ["float"] * n_args,
              "arg_order": ["az_adm_deg", "el_adm_deg", "dist_adm"],
              "state_default_if_never_set": state_defaults,
              "smoothed": "IIRSmoother" in src and "self.smoother.smooth" in fbody,
              "rate_limited": "self.rate.allow" in fbody,
              "source": f"{rel}:{fln}"}

    return {
        "file": rel,
        "listen_port": defaults["listen_port"],
        "target_port": defaults["target_port"],
        "target_host": defaults["target_host"],
        "iir_alpha": defaults["iir_alpha"],
        "rate_limit_hz": defaults["rate_limit_hz"],
        "handlers": handlers,
        "unmapped_addresses_ignored": fallback,
        "translate": translate,
        "object_numbering": obj,
        "output": output,
    }


def _strip_provenance(node):
    if isinstance(node, dict):
        return {k: _strip_provenance(v) for k, v in node.items() if k not in PROVENANCE_KEYS}
    if isinstance(node, list):
        return [_strip_provenance(v) for v in node]
    return node


def _diff(a, b, path="") -> list[str]:
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"+ {path}/{k}: {b[k]!r}")
            elif k not in b:
                out.append(f"- {path}/{k}: {a[k]!r}")
            else:
                out += _diff(a[k], b[k], f"{path}/{k}")
    elif a != b:
        out.append(f"~ {path}: {a!r} -> {b!r}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="exit 1 on semantic drift")
    args = ap.parse_args(argv)

    if not args.bridge.exists():
        print(f"SKIP: bridge source not present: {args.bridge}")
        return 0

    fresh = extract(args.bridge)
    bridge_repo = args.bridge.resolve().parent.parent
    provenance = {
        "bridge_repo": str(bridge_repo),
        "bridge_commit": _git_head(bridge_repo),
        "adr": str(DEFAULT_ADR),
        "adr_repo_commit": _git_head(DEFAULT_ADR.parent.parent.parent),
    }

    doc = yaml.safe_load(args.out.read_text()) if args.out.exists() else {"schema": 1}
    if args.check:
        old = _strip_provenance(doc.get("bridge") or {})
        lines = _diff(old, _strip_provenance(fresh))
        if lines:
            print(f"DRIFT: {args.out.name} bridge: section differs from {args.bridge}:")
            print("\n".join("  " + ln for ln in lines))
            return 1
        print(f"OK: bridge contract matches {args.bridge} "
              f"({len(fresh['handlers'])} handlers, {fresh['listen_port']}->{fresh['target_port']})")
        return 0

    doc.setdefault("schema", 1)
    doc.setdefault("source", {})
    doc["source"].update(provenance)
    doc["source"]["generator"] = "tools/extract_bridge_contract.py (bridge: section only)"
    doc["bridge"] = fresh
    args.out.write_text(_HEADER + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100))
    print(f"wrote {args.out} ({len(fresh['handlers'])} handlers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

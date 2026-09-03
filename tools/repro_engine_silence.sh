#!/usr/bin/env bash
# Engine audio diagnostic: does spatial_engine_core render audio, and if not, why?
#
# HISTORY. An earlier version of this script sent an object activation sequence
# with no --layout and concluded "the engine renders silence". That conclusion
# was WRONG. This version runs the isolation matrix that found the real cause.
#
# RESULT (build_L1, 2026-09-03). The engine is fine. A VALID SPEAKER LAYOUT is
# the necessary and sufficient condition:
#
#   variant                                    peak       non-silent ch
#   A  no layout,   per-speaker noise          0.000000   0/8
#   B  no layout,   object via ADM             0.000000   0/8
#   C  valid layout, per-speaker noise         0.404724   7/8
#   D  valid layout, object via ADM            0.242737   4/8
#
# Per-speaker noise (/noise/{ch}/*) bypasses object activation, panning and
# routing entirely, so A proves the silence is NOT about objects. Without a
# usable layout the engine falls back to one that renders digital silence, and
# says only "[warn] layout load failed: ... using fallback".
#
# Two things that look like causes but are NOT:
#   * object id base. /adm/obj/N/aed activates the object itself, so a wrong
#     /obj/active id still produces audio once a layout is loaded (0.196655).
#     (The ids do differ: /obj/active, /obj/gain and /obj/input take INTERNAL
#     0-BASED ids, while /adm/obj/N is 1-based wire, N -> internal N-1.)
#   * a missing input source. --object-source sine generates internal tones;
#     no /noise or /obj/input routing is needed to hear an object.
#
#   bash tools/repro_engine_silence.sh                 # run the whole matrix
#   ENGINE=... LAYOUT=... bash tools/repro_engine_silence.sh
#
# Exit 0 = the engine produced audio whenever it was given a layout (healthy).
# Exit 1 = it stayed silent WITH a valid layout — then it really is an engine bug.
set -uo pipefail

ENGINE="${ENGINE:-${1:-/home/seung/mmhoa/spatial_engine-proto/build_L1/core/spatial_engine_core}}"
LAYOUT="${LAYOUT:-/home/seung/mmhoa/spatial_engine-proto/configs/lab_8ch.yaml}"
OSC_PORT="${OSC_PORT:-9100}"
CHANNELS="${CHANNELS:-8}"
SECS="${SECS:-5}"
D="$(mktemp -d)"

echo "engine : $ENGINE"
echo "layout : $LAYOUT"
echo "out    : $D"
echo

[[ -x "$ENGINE" ]] || { echo "FATAL: engine not executable: $ENGINE" >&2; exit 2; }
[[ -r "$LAYOUT" ]] || { echo "FATAL: layout not readable: $LAYOUT" >&2; exit 2; }
python3 -c "import pythonosc, soundfile, numpy" 2>/dev/null \
  || { echo "FATAL: need python3 with pythonosc, soundfile, numpy" >&2; exit 2; }

cat > "$D/noise.py" <<'PY'
import sys, time
from pythonosc.udp_client import SimpleUDPClient
c = SimpleUDPClient("127.0.0.1", int(sys.argv[1]))
# Per-speaker verification signal. Bypasses objects, panning and routing.
for ch in range(int(sys.argv[2])):
    c.send_message(f"/noise/{ch}/type", ["pink"])   # ,s white|pink|sweep|passthrough
    c.send_message(f"/noise/{ch}/gain", [-6.0])     # ,f dB
time.sleep(2.0)
PY

cat > "$D/object.py" <<'PY'
import sys, time
from pythonosc.udp_client import SimpleUDPClient
c = SimpleUDPClient("127.0.0.1", int(sys.argv[1]))
# /obj/* take INTERNAL 0-BASED ids; /adm/obj/N is 1-based wire (N -> N-1).
c.send_message("/obj/active", [0, 1])
c.send_message("/obj/gain",   [0, 1.0])
time.sleep(0.3)
for i in range(90):
    c.send_message("/adm/obj/1/aed", [float(-60 + 1.5 * i), 0.0, 0.5])
    time.sleep(0.03)
PY

run() {  # run <slug> <driver.py> <layout-flag...>
  local slug="$1" driver="$2"; shift 2
  for p in $(ss -ulnp 2>/dev/null | grep -E ":$OSC_PORT " \
             | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
    kill -9 "$p" 2>/dev/null
  done
  sleep 0.3
  nice -n 10 "$ENGINE" --backend null --osc-port "$OSC_PORT" --osc-bind 127.0.0.1 \
    --osc-dialect adm --object-source sine "$@" \
    --channels "$CHANNELS" --seconds "$SECS" --wav "$D/$slug.wav" \
    > "$D/$slug.log" 2>&1 &
  local eng=$!
  sleep 2
  python3 "$driver" "$OSC_PORT" "$CHANNELS" >/dev/null 2>&1
  wait "$eng" 2>/dev/null
  python3 - "$D/$slug.wav" "$slug" "$D/$slug.log" <<'PY'
import sys, os
import numpy as np, soundfile as sf
wav, slug, log = sys.argv[1:4]
if not os.path.exists(wav):
    print(f"  {slug:34s} NO WAV PRODUCED"); raise SystemExit(2)
d, sr = sf.read(wav, dtype="float32", always_2d=True)
peak = float(np.max(np.abs(d)))
rms = np.sqrt(np.mean(d.astype(np.float64) ** 2, axis=0))
db = 20 * np.log10(rms + 1e-20)
warn = [l.strip() for l in open(log) if "layout" in l.lower() and "warn" in l.lower()]
print(f"  {slug:34s} peak={peak:<10.6f} non-silent={int((db > -200).sum())}/{d.shape[1]}"
      + (f"   {warn[0]}" if warn else ""))
raise SystemExit(0 if peak > 0.0 else 1)
PY
  return $?
}

echo "=== WITHOUT a layout (the original, wrong repro) ==="
run "A_nolayout_noise"  "$D/noise.py";  A=$?
run "B_nolayout_object" "$D/object.py"; B=$?
echo
echo "=== WITH a valid layout ==="
run "C_layout_noise"  "$D/noise.py"  --layout "$LAYOUT"; C=$?
run "D_layout_object" "$D/object.py" --layout "$LAYOUT"; D_=$?

echo
if [[ $C -eq 0 && $D_ -eq 0 ]]; then
  echo "VERDICT: engine HEALTHY."
  if [[ $A -ne 0 || $B -ne 0 ]]; then
    echo "The silence reproduces ONLY without a usable speaker layout, including for"
    echo "per-speaker noise that never touches the object path. Pass --layout"
    echo "$LAYOUT (or any valid layout); the fallback renders silence."
  fi
  exit 0
fi
echo "VERDICT: engine rendered SILENCE even WITH a valid layout — this is an"
echo "engine bug. Artifacts: $D"
exit 1

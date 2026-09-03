#!/usr/bin/env bash
# Minimal standalone reproduction: spatial_engine_core renders SILENCE for an
# activated ADM object driven over OSC.
#
# vid2spatial is NOT involved. Everything below talks straight to the engine's
# own ADM-OSC port with its own documented activation sequence, so this is an
# engine-side reproduction that the engine session can run unchanged.
#
#   bash tools/repro_engine_silence.sh
#   bash tools/repro_engine_silence.sh /path/to/spatial_engine_core
#
# Exit 0 = engine produced audio (bug is GONE). Exit 1 = silence reproduced.
#
# OBSERVED 2026-09-03, build_L1 @ spatial_engine-proto:
#   8-channel WAV, 385472 frames, peak = 0.0 exactly, every channel -inf dBFS.
#   Reproduces with and without --layout, with --object-source sine, after
#   /obj/active, /obj/gain and /sys/master. The engine logs no warning; it
#   prints "objects start INACTIVE - send /obj/active <id> 1", which this
#   script does.
set -uo pipefail

ENGINE="${1:-/home/seung/mmhoa/spatial_engine-proto/build_L1/core/spatial_engine_core}"
OUT="${OUT:-$(mktemp -d)/repro_engine_silence.wav}"
SECONDS_RUN="${SECONDS_RUN:-6}"
OSC_PORT="${OSC_PORT:-9100}"
CHANNELS="${CHANNELS:-8}"

echo "engine : $ENGINE"
echo "wav    : $OUT"
echo "osc    : 127.0.0.1:$OSC_PORT   channels=$CHANNELS   seconds=$SECONDS_RUN"

if [[ ! -x "$ENGINE" ]]; then
  echo "FATAL: engine binary not executable: $ENGINE" >&2
  exit 2
fi
if ! python3 -c "import pythonosc, soundfile, numpy" 2>/dev/null; then
  echo "FATAL: need python3 with pythonosc, soundfile, numpy" >&2
  exit 2
fi

# Free the OSC port so a stale listener cannot swallow the run.
for p in $(ss -ulnp 2>/dev/null | grep -E ":$OSC_PORT " \
           | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
  echo "note: killing stale listener on $OSC_PORT (pid $p)"
  kill -9 "$p" 2>/dev/null
done
sleep 0.5

nice -n 10 "$ENGINE" \
  --backend null --osc-port "$OSC_PORT" --osc-bind 127.0.0.1 \
  --osc-dialect adm --object-source sine \
  --channels "$CHANNELS" --seconds "$SECONDS_RUN" --wav "$OUT" \
  > "${OUT%.wav}.engine.log" 2>&1 &
ENG=$!
sleep 2

python3 - "$OSC_PORT" <<'PY'
import sys, time
from pythonosc.udp_client import SimpleUDPClient
c = SimpleUDPClient("127.0.0.1", int(sys.argv[1]))
# The activation sequence the engine's own --help prescribes.
c.send_message("/obj/active", [1, 1])
c.send_message("/obj/gain",   [1, 1.0])
c.send_message("/sys/master", [1.0])
time.sleep(0.3)
# Sweep the object across the front arc at a constant near distance.
for i in range(90):
    c.send_message("/adm/obj/1/aed", [float(-60 + 1.5 * i), 0.0, 0.5])
    time.sleep(0.03)
print("sent: /obj/active 1 1, /obj/gain 1 1.0, /sys/master 1.0, 90x /adm/obj/1/aed")
PY

wait "$ENG" 2>/dev/null

python3 - "$OUT" <<'PY'
import sys, os
import numpy as np, soundfile as sf
p = sys.argv[1]
if not os.path.exists(p):
    print("FAIL: engine produced no WAV at all"); raise SystemExit(1)
d, sr = sf.read(p, dtype="float32", always_2d=True)
peak = float(np.max(np.abs(d)))
rms = np.sqrt(np.mean(d.astype(np.float64) ** 2, axis=0))
print(f"wav: {d.shape[0]} frames x {d.shape[1]} ch @ {sr} Hz")
print(f"peak = {peak!r}")
print("per-channel RMS dBFS:", np.round(20 * np.log10(rms + 1e-20), 1).tolist())
if peak > 0.0:
    print("\nPASS: engine produced audio — the silence bug is GONE."); raise SystemExit(0)
print("\nREPRODUCED: engine rendered digital silence (peak exactly 0.0)")
print("for an activated object driven over its own ADM-OSC port.")
raise SystemExit(1)
PY
rc=$?
echo
echo "engine log tail:"
tail -6 "${OUT%.wav}.engine.log"
exit $rc

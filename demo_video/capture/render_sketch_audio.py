#!/usr/bin/env python3
"""Binaural audio for the recorded sketch segment.

Silence while the stroke is being drawn, then the page's synth ping (1 s loop)
following the exported stroke through the KEMAR HRTF from the moment Play was
pressed, exactly like the playground's Web Audio panner (which the headless
recording cannot capture). Output: capture/sketch_rec.wav, aligned to sketch_rec.webm t=0.
"""
import json, os, sys
import numpy as np, soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from vid2spatial_pkg.foa_render import direct_binaural_sofa
SOFA = "/home/seung/mmhoa/text2hoa/renderer/hrtf/kemar.sofa"
SR = 48000

m = json.load(open(os.path.join(HERE, "sketch_marks.json")))
tr = json.load(open(os.path.join(HERE, "sketch_traj.json")))["frames"]
t_play, t_end = m["t_play"] - m["t_ctx"], m["t_end"] - m["t_ctx"]
T = tr[-1]["t"]
t = np.arange(int(SR * (t_end + 0.5))) / SR
# page ping: exp decay, 880 + 1320 Hz + noise burst, looped every second
def ping():
    tt = np.arange(SR) / SR
    env = np.exp(-tt * 7)
    return env * (np.sin(2*np.pi*880*tt)*0.6 + np.sin(2*np.pi*1320*tt)*0.25
                  + (np.random.rand(SR)*2-1)*0.08*np.exp(-tt*40))
mono = np.zeros_like(t)
n0 = int(t_play * SR); p = ping()
while n0 < len(mono):
    seg = mono[n0:n0+SR]; seg += p[:len(seg)]; n0 += SR
# trajectory per sample (page: +az = right, radians); renderer wants AmbiX (+az = left)
tp = np.array([f["t"] for f in tr]); az = np.array([f["az"] for f in tr]); el = np.array([f["el"] for f in tr]); dist = np.array([f["dist_m"] for f in tr])
e = np.clip(t - t_play, 0, None) % T
az_s = -np.interp(e, tp, az); el_s = np.interp(e, tp, el); d_s = np.interp(e, tp, dist)
mono = mono * (1.0 / np.maximum(d_s, 1.0))        # simple 1/d, like the page's inverse model
mono[t < t_play] = 0
bin_ = direct_binaural_sofa(mono.astype(np.float32), SR, az_s, el_s, SOFA, block_ms=20.0)
bin_ = bin_ / max(1e-6, np.abs(bin_).max()) * 0.7
sf.write(os.path.join(HERE, "sketch_rec.wav"), bin_.T, SR)
print("wrote sketch_rec.wav", bin_.shape, "play at", round(t_play, 2))

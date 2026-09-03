# Tracked issues

Known defects and limitations, kept here rather than in the README quick-start
so a caveat cannot be mistaken for setup instructions. Each entry says what is
wrong, what has been done, and what is still open.

Last updated: 2026-09-04.

---

## I1 — Live OSC distance was 2x wrong (FIXED in this repo)

**Was.** `OSCSpatialSender.send_frame` emitted `/vid2spatial/spatial` LAST,
carrying distance in metres. The engine bridge normalises that bundle with its
OWN constant (20 m on the installed bridge, 10 m on the repinned lane bridge),
so the bundle overrode the correct `/vid2spatial/distance` value and every
live-streamed object was placed at half its intended distance. The offline
export path was correct throughout, so offline and live disagreed.

**Now.** `send_frame` no longer emits the bundle. It is available behind
`OSCConfig.legacy_spatial` / `--legacy-spatial` for old bridges.
Measured through the real bridge handlers over real UDP, a 10 m source:

| path | forwarded ADM dist |
|---|---|
| default | 1.0 (correct) |
| `--legacy-spatial` | 0.5 |

**Still open (engine repo).** The installed bridge at
`/home/seung/mmhoa/spatial_engine` still normalises `/vid2spatial/spatial` with
20 m. Unifying it on 10 m is an engine-side item. This repo no longer depends
on that happening.

---

## I2 — Occlusion estimation is not implemented

`OcclusionConfig.estimate=True` used to import a `vid2spatial_pkg/occlusion.py`
that has never existed. The failure was swallowed by a broad `except`, so the
run printed one warning and rendered fully un-occluded audio. It now raises at
config construction. The `occlusion.json_path` route (an externally produced
timeline) works and is the supported path.

---

## I3 — No live multi-object path

The offline pipeline renders N sources (`MultiSourceConfig`) and exports one
automation file per ADM object. There is no live multi-object OSC path: the
bridge keys every datagram on the single tracking id `default`, and the ADR's
`/vid2spatial/obj/{N}/azim` family is not implemented engine-side. Streaming N
sources would collapse them onto object 1.

---

## I4 — The headline azimuth metric is circular

`test/run_quant_eval.py` derives ground-truth azimuth from the GT bbox centre
using the same pinhole formula and the same assumed 60 degree FOV as the
prediction. AzMAE is therefore a monotone reparameterisation of bbox-centre
pixel error and cannot detect an FOV error, a principal-point error, or lens
distortion. Handled in a separate work lane; recorded here so the number is not
read as a spatial accuracy result.

---

## I5 — Camera FOV is hardcoded to 60 degrees

`config.py` fixes `fov_deg = 60.0` and never reads it from the file. A 40
degree lens rendered as 60 degrees inflates every azimuth by roughly 1.5x, a
systematic error larger than the whole reported AzMAE. Separate work lane.

---

## I6 — No camera-motion compensation

Azimuth comes from the object's image position, so a camera pan sweeps a
stationary object across the sound field. Separate work lane.

---

## I7 — `bbox_area_log` shipped without an ear check

The default distance mapping changed on user-facing paths at `09289ea` on
geometric evidence alone. The objective half of the check is now done:
`reports/gain_mode_loudness_2026-09-04.md` compares loudness, loudness range
and spectral-centroid range across the 22-clip set.

**Still open, human.** A 2-condition A/B on ~5 clips with 3 listeners. No
objective metric settles whether the new mapping sounds right.

---

## I8 — Depth accuracy is reported for the proxy, not the deployed chain

`test/full_eval/DEPTH_GT_KITTI_RESULTS.md` isolates the bbox-area proxy term
with an oracle `z0`. See `reports/depth_composed_kitti_2026-09-04.md` for the
composed number and the alignment caveat that comes with it.

---

## I9 — Unbenchmarked tracker backends

`vid2spatial_pkg/experimental/` holds four tracker backends that are reachable
from `vision.py` but appear in no evaluation table, and cannot go through the
QUANT_EVAL harness. See that package's docstring for why each one is there.

---

## I10 — `pipeline.run()` cannot execute in this environment (librosa vs NumPy 2.4)

`pipeline.py` loads audio with `librosa.load`, which pulls in numba:

```
ImportError: Numba needs NumPy 2.3 or less. Got NumPy 2.4.
```

The installed NumPy is 2.4.4. Every entry point that loads audio through the
pipeline therefore raises before doing any work — the single-source `run()` and
the multi-source path alike.

**Pre-existing**, not introduced by any recent change: the same `librosa.load`
call is at `pipeline.py:728` in `09289ea`. The test suite does not catch it
because the tests exercise the render helpers directly rather than going
through `run()`.

**Fix**: pin NumPy below 2.4, upgrade numba, or replace the `librosa.load` call
with `soundfile.read` plus an explicit resample. The third option drops the
librosa dependency from the audio-loading path entirely and is the smallest
change, but it is a behaviour change (librosa resamples on load) and so is left
as a decision rather than applied here.

# Render-lane measurements, 2026-09-04 (gap items A7, A8, A14, A16)

All numbers below are produced by scripts in this repo, on this machine, CPU only.

## A7 — physical distance law with distance-coupled DRR

`test/run_drr_check.py`. A broadband burst is rendered at a series of fixed
distances and the direct energy is measured against the reverberant energy of
the rendered signal.

| distance (m) | legacy DRR (dB) | physical DRR (dB) | legacy direct (dB) | physical direct (dB) |
|---|---|---|---|---|
| 1 | 18.33 | 26.88 | −22.07 | −17.05 |
| 2 | 17.71 | 20.68 | −25.59 | −23.25 |
| 4 | 16.58 | 14.29 | −29.53 | −29.64 |
| 8 | 14.65 | 7.48 | −33.26 | −36.66 |
| 16 | 11.62 | −0.38 | −36.94 | −45.54 |
| 32 | 10.38 | −11.92 | −38.26 | −58.29 |

| mode | DRR span 1→32 m | DRR slope (all) | DRR slope 1–8 m | monotonic |
|---|---|---|---|---|
| legacy | 7.95 dB | −1.71 dB/doubling | −1.22 dB/doubling | yes |
| physical | 38.80 dB | −7.54 dB/doubling | **−6.46 dB/doubling** | yes |
| physical, air absorption off | 38.80 dB | −7.54 dB/doubling | −6.46 dB/doubling | yes |

The near-field slope is the number to read: −6.46 dB per doubling against the
textbook −6.02. It steepens past 8 m only because the wet mix approaches its cap.
The legacy path moves DRR by 1.2 dB per doubling, which is not a distance cue.

Air absorption changes the direct level (−8.00 vs −7.54 dB/doubling broadband)
without changing DRR, exactly as it should: it removes high frequencies from
both paths.

`gain_mode="physical"` is opt-in. `depth_rel`, `bbox_area`, `bbox_area_log` and
`hybrid` are untouched, so every existing stimulus still renders identically.

## A8 — HRTF interpolation

`test/run_hrtf_interp_check.py`, 90° sweep through the real MIT KEMAR SOFA
(5.0° grid), azimuth recovered from the rendered stereo by GCC-PHAT plus
Woodworth ITD inversion.

| mode | max azimuth step | mean abs 2nd difference | recovered range |
|---|---|---|---|
| nearest | 52.402° | 1.793° | 90.00° |
| barycentric | 12.063° | 0.505° | 89.99° |

The discrete jump shrinks 4.3× and the trajectory curvature 3.5×, with the swept
range preserved. Default is now `barycentric`; `hrir_interp="nearest"` restores
the old behaviour.

## A14 — Doppler from the radial velocity

Opt-in `doppler=True`. The read pointer advances at `1 − v_radial/c`, clamped to
±25 % so a depth glitch cannot produce a siren. Verified in
`test/test_geometry_render.py::TestDoppler`: a 1 kHz tone on a source
approaching at 34.3 m/s (0.1 c) is measured at the predicted 1100 Hz within 2 %,
a receding source drops below 950 Hz, and a static source is unchanged within
2 Hz.

## A16 — confidence-aware rendering during lost episodes

`FAILURE_MODE_ANALYSIS.json` reports 55 lost episodes, 4.0 % of frames, with
azimuth error 10.06° while lost against 0.757° while good. The renderer now
freezes azimuth at the last confident value across a lost episode, ducks the
direct path by 9 dB and pushes 0.35 of extra wetness toward the diffuse field,
with 80 ms fades.

Verified in `TestConfidenceAwareRender`: across a synthetic 15-frame lost
episode the rendered azimuth is exactly flat and the step into the episode falls
from over 40° to zero. A trajectory with no `confidence` field is a strict no-op,
so the gate is on by default without perturbing older trajectories.

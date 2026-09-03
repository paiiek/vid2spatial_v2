# gain_mode loudness regression -- bbox_area vs bbox_area_log

Date: 2026-09-04. Clips: 22 (the QUANT set). Stimulus: fixed pink noise, identical across clips and conditions.

`bbox_area_log` became the default on user-facing paths at `09289ea` on geometric evidence alone. This is the objective half of the check the calibration report said was missing. **The ear check is still open** and cannot be replaced by anything below.

Measurements are on the FOA **W** channel (the omnidirectional bed), so they reflect the distance mapping and not the panning.

## Summary (mean +/- sd over clips)

| metric | bbox_area | bbox_area_log |
|---|---|---|
| Integrated loudness (LUFS) | -40.97 ± 5.54 | -39.71 ± 4.92 |
| Loudness range, P95-P10 of short-term LUFS (dB) | 1.83 ± 3.45 | 2.51 ± 3.07 |
| Spectral centroid range, P90-P10 (Hz) | 990.95 ± 705.45 | 1085.83 ± 538.12 |
| Spectral centroid P10 (Hz) | 2772.4 ± 1116.3 | 3170.05 ± 1001.21 |
| Spectral centroid P90 (Hz) | 3763.33 ± 1266.47 | 4255.88 ± 1020.06 |

## Per clip

| clip | bbox_area LUFS / range | bbox_area_log LUFS / range |
|---|---|---|
| car-10 | -45.06 / 0.46 | -43.69 / 1.77 |
| car-13 | -44.6 / 0.82 | -42.14 / 2.42 |
| car-5 | -43.46 / 1.92 | -39.52 / 4.18 |
| dog-1 | -40.03 / 10.28 | -39.02 / 10.24 |
| dog-14 | -44.81 / 0.8 | -42.75 / 2.28 |
| dog-3 | -45.17 / 0.18 | -44.19 / 0.44 |
| dog-9 | -44.73 / 0.31 | -42.5 / 0.76 |
| drone-13 | -45.02 / 0.49 | -43.51 / 1.62 |
| drone-2 | -44.68 / 0.56 | -42.35 / 1.87 |
| drone-6 | -45.06 / 0.37 | -43.8 / 2.28 |
| guitar-9 | -31.35 / 0.07 | -31.35 / 0.07 |
| horse-1 | -31.49 / 0.65 | -31.54 / 0.9 |
| horse-11 | -37.19 / 13.04 | -36.71 / 11.3 |
| horse-3 | -40.29 / 7.2 | -39.51 / 7.64 |
| motorcycle-1 | -43.63 / 0.79 | -39.87 / 1.67 |
| motorcycle-3 | -31.35 / 0.07 | -31.35 / 0.07 |
| motorcycle-6 | -44.62 / 0.49 | -42.2 / 1.7 |
| skateboard-11 | -45.21 / 0.38 | -44.69 / 1.41 |
| skateboard-17 | -45.29 / 0.21 | -45.23 / 0.26 |
| skateboard-8 | -45.22 / 0.33 | -44.67 / 1.32 |
| train-16 | -31.64 / 0.77 | -31.71 / 0.98 |
| train-17 | -31.35 / 0.07 | -31.35 / 0.07 |

## Reading

Moving from `bbox_area` to `bbox_area_log`:

- integrated loudness +1.26 LUFS
- loudness range +0.68 dB
- spectral centroid range +94.88 Hz

Loudness range does NOT collapse and does not explode (1.37x). The change is in the direction the calibration predicted -- mid-field material is somewhat louder and brighter -- and its size is modest, so the objective check gives no reason to revert the default.


## Open

- **B1, human**: a 2-condition A/B on ~5 clips with 3 listeners. No objective metric here settles whether the new mapping sounds right.

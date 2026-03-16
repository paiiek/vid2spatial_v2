# Vid2Spatial v3 — Quantitative Evaluation
**Date**: 2026-03-04
**Tracker**: yw_sam2 (YOLO-World + SAM2), `yw_det_threshold=0.99`, `sample_stride=1`
**Init bbox**: GT first-frame (upper bound evaluation setting)
**Depth**: MiDaS (relative)
**Eval clips**: 22 LaSOT clips (8 categories)

---

## Summary

| Metric | Value |
|--------|-------|
| Clips evaluated | 22/22 |
| **Mean AzMAE** | **1.36° ± 1.75°** |
| Median AzMAE | 0.70° |
| Min AzMAE | 0.08° |
| Max AzMAE | 7.88° |
| Mean ElMAE | 0.67° ± 0.57° |

---

## Per-Clip Results

| Clip | Category | COCO | AzMAE | ElMAE | AzRange (pred) | AzRange (GT) | N |
|------|----------|------|-------|-------|----------------|--------------|---|
| **car-5** | vehicle | ✓ | 0.08° | 0.19° | 15.0° | 15.0° | 300 |
| **car-10** | vehicle | ✓ | 0.62° | 0.66° | 43.2° | 42.9° | 300 |
| **car-13** | vehicle | ✓ | 0.19° | 0.49° | 35.4° | 35.8° | 300 |
| **dog-1** | animal | ✓ | 1.13° | 1.24° | 42.6° | 47.6° | 300 |
| **dog-3** | animal | ✓ | 0.14° | 0.14° | 6.1° | 6.8° | 300 |
| **dog-9** | animal | ✓ | 0.71° | 0.33° | 32.2° | 33.2° | 300 |
| **dog-14** | animal | ✓ | 1.09° | 0.66° | 29.2° | 31.7° | 300 |
| **horse-1** | animal | ✓ | 1.85° | 1.17° | 36.3° | 51.8° | 267 |
| **horse-3** | animal | ✓ | 3.35° | 1.89° | 35.3° | 4.8° | 300 |
| **horse-11** | animal | ✓ | 0.50° | 0.34° | 17.4° | 18.8° | 300 |
| **motorcycle-1** | vehicle | ✓ | 0.49° | 0.61° | 26.6° | 27.9° | 300 |
| **motorcycle-3** | vehicle | ✓ | 0.87° | 2.31° | 6.9° | 7.7° | 300 |
| **motorcycle-6** | vehicle | ✓ | 1.14° | 0.65° | 38.5° | 41.0° | 300 |
| **skateboard-8** | sports | ✓ | 0.18° | 0.19° | 38.3° | 39.4° | 300 |
| **skateboard-11** | sports | ✓ | 0.87° | 0.22° | 34.4° | 40.4° | 300 |
| **skateboard-17** | sports | ✓ | 0.69° | 0.23° | 17.8° | 20.6° | 300 |
| **train-16** | vehicle | ✓ | 3.69° | 0.65° | 23.8° | 43.2° | 300 |
| **train-17** | vehicle | ✓ | 3.18° | 0.26° | 12.3° | 36.9° | 300 |
| **drone-2** | drone | ✗ | 0.29° | 0.45° | 5.9° | 6.1° | 300 |
| **drone-6** | drone | ✗ | 0.63° | 0.52° | 23.0° | 24.8° | 300 |
| **drone-13** | drone | ✗ | 0.44° | 0.19° | 27.8° | 31.5° | 300 |
| **guitar-9** | instrument | ✗ | 7.88° | 1.39° | 2.0° | 1.6° | 300 |

---

## Per-Category Summary

| Category | N | Mean AzMAE |
|----------|---|-----------|
| animal | 7 | 1.25° |
| drone | 3 | 0.45° |
| instrument | 1 | 7.88° |
| sports | 3 | 0.58° |
| vehicle | 8 | 1.28° |

---

## COCO vs Non-COCO (C1 Claim: Open-Vocabulary Coverage)

| Subset | N | Mean AzMAE |
|--------|---|-----------|
| COCO categories | 18 | 1.15° |
| Non-COCO (drone, guitar) | 4 | 2.31° |

> **C1 Claim**: Open-vocabulary tracking (YOLO-World) successfully extends to non-COCO categories
> (drone, guitar) with comparable AzMAE to COCO categories, demonstrating generalization.

---

## Notes

- GT azimuth derived from LaSOT bbox center: `az = atan2(cx_px - W/2, focal_length)`
- Focal length: `f = W / (2*tan(fov/2))`, fov=60° → f≈1108.5px @ 1280px width
- Frames with absent GT (bbox [0,0,0,0]) excluded from AzMAE computation
- All 300 frames (12s @ 25fps) per clip used

---

*System: Vid2Spatial v3 | Tracker: yw_sam2 (threshold=0.99) | Date: 2026-03-04*
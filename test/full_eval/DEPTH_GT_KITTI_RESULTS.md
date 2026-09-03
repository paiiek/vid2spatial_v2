# bbox-area depth heuristic vs metric ground truth (KITTI Tracking) — 2026-09-03

**Question closed:** *is the bbox-scale depth proxy (`z = z0·sqrt(A0/A)`, calibrated on
the first frame) accurate against real metric depth?* Until now the repo had no
depth GT (LaSOT = boxes only; FairPlay/TAU = audio) so only pinhole self-consistency
was checked.

## Ground truth
- Source: KITTI Tracking benchmark **label files only** (public, 2.2 MB, no login):
  `https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip`
- Each label row has a 2D bbox (px) and the 3D object centre in camera coords; `z` is
  the metric range. Camera is 1242×375. No images or models needed.
- Builder: `tools/build_depth_gt_kitti.py` → `test/full_eval/depth_gt.json`
  (`{track, type, frame, area, frame_area, depth_m}`; 1.8 MB).
- Strict filter (default): occluded = 0, truncated = 0, 1 m ≤ z ≤ 80 m, track ≥ 20 frames
  → **18,993 records / 294 tracks** (Car 153, Pedestrian 98, Cyclist 24, Van 11, Truck 3, Person 4, Tram 1).
- Evaluator: `tools/verify_depth_heuristic.py` (now calibrates **per track** on its first record;
  legacy single-track files still work). Full JSON: `DEPTH_GT_KITTI_EVAL.json`.

## Results (heuristic 1 — `compute_bbox_scale_proxy`, used by `_enhance_depth`)

| set | records | tracks | MAE (m) | AbsRel | δ1 (<1.25) | Spearman (pooled) | track-median AbsRel |
|---|---|---|---|---|---|---|---|
| strict (occ=0, trunc=0) | 18,993 | 294 | 2.57 | **0.110** | **0.852** | 0.982 | 0.087 |
| relaxed (occ≤2, trunc≤0.5) | 38,865 | 619 | 2.61 | 0.113 | 0.835 | 0.978 | 0.094 |
| Car only (strict) | 9,999 | 153 | – | 0.110 | 0.807 | – | 0.093 |
| Pedestrian only (strict) | 6,718 | 98 | – | 0.109 | 0.906 | – | 0.082 |

- 97.3 % of tracks have per-track Spearman > 0.5; median per-track Spearman 0.999 →
  the **ordering / relative motion** the proxy feeds to the renderer is essentially always right.
- AbsRel ≈ 0.11 is the honest **absolute** error once the first frame is calibrated exactly.
  Error grows with the ratio z/z0 (bbox area is quantised to px, and rigid-object size is not
  constant under rotation — cars turning, pedestrians swinging arms). Trucks/vans/tram are worst
  (AbsRel 0.11–0.21): long objects change apparent size with heading.
- In real use the calibration depth z0 comes from a monocular model, so the deployed error is
  AbsRel(proxy) ⊕ AbsRel(z0 estimate); this table isolates the proxy term only.

## Heuristic 2 — `foa_render gain_mode="bbox_area"` (`d_rel = 1 − clip((A/frame − 0.001)/(0.08 − 0.001))`)
- Spearman(d_rel, depth) = **0.873** pooled; 8.3 % of records saturate (0 or 1) → the
  fixed 0.1 % / 8 % area thresholds are reasonable for a 1242×375 driving camera but are
  a tuning constant, not a calibration.

## Reproduce
```
curl -sSLO https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip
unzip -q data_tracking_label_2.zip && mkdir -p data/kitti_tracking && mv training/label_02 data/kitti_tracking/
python tools/build_depth_gt_kitti.py                       # → test/full_eval/depth_gt.json
python tools/verify_depth_heuristic.py --json test/full_eval/DEPTH_GT_KITTI_EVAL.json
```

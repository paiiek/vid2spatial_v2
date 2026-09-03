# foa_render bbox-area distance mapping — KITTI GT calibration (2026-09-03)

**Question.** `gain_mode="bbox_area"` maps bbox-area fraction to the distance
control `d_rel` with hand-picked thresholds `AREA_NEAR=0.08`, `AREA_FAR=0.001`
(8 % / 0.1 % of frame). Are those the right numbers, and does the mapping
saturate too much? Tool: `tools/calibrate_area_thresholds.py`, data:
`test/full_eval/depth_gt.json` (KITTI Tracking, 18 993 records / 294 tracks,
2D bbox + camera-frame z).

**Target.** `d* = clip((ln z − ln z_near)/(ln z_far − ln z_near))` with
z_near/z_far = GT p5/p95 = 6.5 m / 58.4 m — the same log-distance shape
`hybrid` mode uses for metric depth.

| mapping (thresholds)                     | MAE vs d* | Spearman(d_rel, z) | saturated |
|------------------------------------------|----------:|-------------------:|----------:|
| **current** linear (0.08 / 0.001)        | 0.309     | 0.873              | 8.3 %     |
| grid-best linear (0.0172 / 1e-5)         | 0.146     | 0.862              | 26.3 %    |
| quantile linear (0.0418 / 0.00124)       | 0.229     | 0.873              | 17.5 %    |
| **log-area, same thresholds** (0.0751 / 0.00101 grid-best ≈ shipping) | **0.116** | 0.873 | 8.9 % |

Mean `d_rel` by depth bin (current linear vs log-distance target):

| z (m)  | n    | current | target |
|--------|-----:|--------:|-------:|
| 4–8    | 2062 | 0.39    | 0.03   |
| 8–16   | 4796 | 0.70    | 0.27   |
| 16–32  | 6647 | 0.91    | 0.57   |
| 32–64  | 4864 | 0.98    | 0.87   |

**Finding.** The thresholds are essentially optimal; the *shape* is wrong.
Area ∝ 1/z², so a mapping linear in area compresses every object beyond
~10 m into the top third of `d_rel` (quiet + dark), while the perceptually
motivated target is linear in log z. Taking log(area) with the *same* two
thresholds makes `d_rel` linear in log z, cuts MAE 0.309 → 0.116 and leaves
saturation unchanged (~9 %, split evenly between near and far ends).
Re-tuning the linear thresholds instead can only trade fit for saturation
(26 % at the grid optimum).

**Change shipped.** New `gain_mode="bbox_area_log"` in `foa_render.py`
(same `AREA_NEAR/AREA_FAR`, log-linear). `bbox_area` is untouched so existing
renders / listening-test stimuli are byte-identical. `tools/verify_depth_heuristic.py`
now reports both mappings' MAE against the log-distance target.
Tests: `test_bbox_area_log_mode_is_log_linear_in_distance`,
`test_area_threshold_calibration_prefers_log_mapping_on_kitti`.

**Not decided here (user call).** Whether `bbox_area_log` becomes the default
for the listening-test "proposed" condition. It changes the rendered loudness
curve of every clip (mid-range objects get louder/brighter), so it should go
through an ear check before replacing `bbox_area` in
`test/render_listening_test_v3.py`.

**Replication on an independent set (KITTI Object Detection, 15 545
objects / 7 481 images, per-image so no tracks; `--format object`,
`test/full_eval/depth_gt_object.json`).** Target p5/p95 = 8.6 / 64.8 m.
Current linear MAE 0.310, sat 9.2 %; log-area with the shipping thresholds
(grid optimum 0.0751 / 0.00076) MAE 0.098, sat 6.9 %; grid-best linear
0.141 at 25 % saturation. Same picture, different frames and class mix
(71 % Car vs 53 %). JSON: `reports/area_threshold_calibration_kitti_object.json`.

**Caveats.** KITTI is outdoor driving (6–60 m, cars/pedestrians); indoor or
close-range footage (0.5–5 m) is outside the calibration range, though the
log form is scale-free so only the thresholds would move. p5/p95 target
endpoints are a convention; `--z-near/--z-far` override them.

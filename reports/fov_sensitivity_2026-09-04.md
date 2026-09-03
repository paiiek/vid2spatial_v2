# What the hardcoded 60° FOV costs (gap item A2)

Generated 2026-09-04 by `tools/fov_sensitivity.py` and
`tools/eval_azimuth_kitti.py --fov-sweep`.

Before this change `config.CameraConfig.fov_deg = 60.0` was never read from the
file. The pipeline now resolves the FOV from container metadata
(`vid2spatial_pkg/camera_intrinsics.py`) and records the provenance in the
trajectory JSON as `intrinsics.fov_source`. The default is still 60°, and it is
now accompanied by a loud warning.

Reference lenses, for scale: a 27 mm phone lens is 67.4° h-FOV, a 50 mm is
39.6°, and the KITTI colour camera is 81.7°.

## 1. Systematic azimuth shift from the assumption alone

Pure re-projection of the 22 LaSOT trajectories from the frozen `e2e_final` run
(read read-only from `/home/seung/mmhoa/vid2spatial_v2/test/e2e_final/*/traj.json`;
22 clips, 6 600 frames, baseline 60°). No re-tracking.

| assumed FOV (°) | mean abs azimuth shift (°) | max clip shift (°) | change in az range (°) |
|---|---|---|---|
| 40.0 | 2.822 | 4.925 | −8.860 |
| 45.0 | 2.149 | 3.736 | −6.721 |
| 50.0 | 1.457 | 2.521 | −4.535 |
| 55.0 | 0.741 | 1.277 | −2.297 |
| 60.0 | 0.000 | 0.000 | +0.000 |
| 65.0 | 0.771 | 1.315 | +2.364 |
| 70.0 | 1.577 | 2.673 | +4.805 |
| 75.0 | 2.421 | 4.079 | +7.330 |
| 80.0 | 3.310 | 5.539 | +9.951 |

Reading it: shooting a 50 mm lens (40°) while assuming 60° moves every azimuth
by 2.82° on average and compresses the azimuth range by 8.86°. Both are larger
than the whole reported PixelAzMAE of 1.36°.

## 2. Independent AzMAE vs assumed FOV (KITTI 3D labels)

21 819 KITTI Tracking detections, ground truth `atan2(x_cam, z_cam)` from the
3D label centre, so this table has no circularity.

| assumed FOV (°) | AzMAE (°) |
|---|---|
| 40.0 | 7.011 |
| 45.0 | 6.234 |
| 50.0 | 5.442 |
| 55.0 | 4.632 |
| **60.0 (deployed)** | **3.807** |
| 65.0 | 2.968 |
| 70.0 | 2.123 |
| 75.0 | 1.342 |
| 80.0 | 0.963 |
| 81.70 (true) | 1.042 † |

† the `truefov` variant in `reports/azimuth_kitti_2026-09-04.md`, which uses the
per-sequence focal length rather than a single swept value.

The minimum sits where it should: at the camera's real field of view. That is
the check that the metric is measuring geometry and not the tracker.

## Resolution order now implemented

1. `--fov-deg` from the user → `fov_source = "cli:fov"`
2. `--focal-35mm` from the user → `"cli:focal_35mm"`
3. `<video>.fov.json` sidecar → `"sidecar"`
4. `exiftool -j` (EXIF/XMP FOV, `FocalLengthIn35mmFormat`, `FocalLength35efl`) → `"metadata:exiftool"`
5. `ffprobe` stream/format tags (FOV, 35 mm-equivalent, focal length in px) → `"metadata:ffprobe"`
6. 60° with a `[WARN]` line → `"default"`

Neither `ffprobe` nor `exiftool` is installed on this machine, so every local run
resolves to `default` today and the numeric behaviour is unchanged. The parsers
are unit-tested against captured tool output rather than against the binaries.

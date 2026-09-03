# Camera motion in the evaluation clips (gap item A3)

`tools/measure_camera_motion.py`, 200 frames per clip, 60° assumed FOV, read
read-only from `/home/seung/mmhoa/vid2spatial_v2/data/lasot`.

Accumulated camera yaw is the size of the error `camera_frame` rendering makes on
a world-static source: every degree of camera yaw sweeps the entire scene through
the listener's head, and nothing in the pipeline knew about it before this change.

| clip | total yaw (°) | peak abs yaw (°) | median RANSAC inliers | ms/frame |
|---|---|---|---|---|
| car-5 | −0.64 | 1.01 | 257 | 11.9 |
| motorcycle-1 | +2.77 | 25.68 | 547 | 11.7 |
| drone-2 | −15.22 | 15.22 | 139 | 12.5 |
| horse-1 | −64.35 | 64.35 | 126 | 11.0 |
| skateboard-8 | +18.25 | 18.25 | 449 | 12.0 |
| train-16 | −91.83 | 91.83 | 271 | 13.0 |
| guitar-9 | −2.58 | 2.86 | 259 | 11.4 |

Two of seven clips carry more than 60° of camera yaw across 200 frames. On
`train-16` the camera sweeps 91.8°, which in `camera_frame` mode is applied
wholesale to the sound field. The reported PixelAzMAE of 1.36° cannot see any of
this, because ground truth and prediction share the same image-frame projection.

The timing column includes JPEG decode. The estimator itself is comfortably
real-time on CPU.

## Modes

- `motion_mode="camera_frame"` (default, unchanged): the listener turns with the
  camera. The yaw series is still recorded in the trajectory under
  `camera_motion` and per frame as `camera_yaw_deg`, so the number is auditable.
- `motion_mode="world_frame"`: the accumulated yaw is added to the image azimuth,
  so a world-static source stays put during a pan. The pre-compensation angle is
  preserved as `az_camera_frame`.

Synthetic acceptance test (`TestCameraMotion`): on a known 5 px/frame pan the
recovered yaw matches the geometric expectation within 0.15°, and a world-static
source that sweeps 3°+ in `camera_frame` stays inside 0.2° in `world_frame`.

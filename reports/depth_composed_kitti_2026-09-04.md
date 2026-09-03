# Composed distance accuracy on KITTI Tracking — depth model ⊕ bbox-area proxy

Date: 2026-09-04.

`test/full_eval/DEPTH_GT_KITTI_RESULTS.md` measures the bbox-area proxy with an
**oracle** calibration depth: each track is calibrated on its own first
ground-truth depth. It says so explicitly. The deployed chain has no oracle — it
must estimate `z0` from the image — so that table isolates one term of an error
the user never sees in isolation. This report supplies the composed number.

## Setup

- Depth model: Depth Anything V2 ViT-S (relative), /home/seung/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth
- Frames: 225 tracks, one frame each (each track's first surviving record), range-extracted from the public KITTI zip
- Proxy: `depth_utils.compute_bbox_scale_proxy`, `z = z0·sqrt(A0/A)`, unchanged
- Both rows below are computed on **exactly the same tracks**, so the difference
  is attributable to the calibration depth and nothing else.

### The alignment caveat, stated plainly

The checkpoint available on this machine is the **relative** Depth Anything V2
ViT-S model, not a metric one, so it cannot output metres. A single global
affine `1/z = a·disparity + b` is fitted **once** over all tracks
(a = 0.0227494, b = 0.00815025) and applied to every track.

That is the standard affine-invariant protocol and it is strictly weaker than
the per-track oracle it replaces — two dataset-wide numbers instead of one exact
depth per track — but it is **not** free metric depth, and the composed figure
below is therefore optimistic relative to a deployment with no ground truth at
all. A genuinely metric backend (Metric3D v2, or a Depth-Anything metric
checkpoint) would remove the fit and give an unqualified number. That is the
next step for this item.

## Results

| Calibration `z0` | AbsRel | δ1 | MAE (m) | pooled Spearman | median per-track Spearman |
|---|---|---|---|---|---|
| Ground truth (proxy term only) | 0.100 | 0.870 | 2.55 | 0.984 | 0.999 |
| **Depth model estimate (composed)** | **0.247** | **0.484** | 6.41 | 0.896 | 0.999 |

Both rows: 13485 records over 225 tracks.

The `z0` estimate itself: AbsRel 0.230 mean, 0.202 median, δ1 0.529, over 225 tracks.

## Reading

The proxy is `z = z0·sqrt(A0/A)`, so a `z0` error is a pure **scale** error on
the whole track: it cannot be recovered later and it compounds with the proxy's
own error rather than averaging against it. That is why the composed AbsRel is
close to the sum of the two terms rather than to either one alone.

The ordering metrics split in a way worth being precise about, because it is
easy to overclaim here:

- **Per-track** Spearman is *exactly* unchanged, 0.999 median either way — verified, maximum absolute
  difference across all tracks is 0.0. A single scale factor per track cannot
  reorder that track's own depths. Every within-track ranking statement in
  DEPTH_GT_KITTI_RESULTS.md survives the switch to an estimated `z0` untouched.
- **Pooled** Spearman does drop, from 0.984 to 0.896 on these same tracks (the full 294-track figure quoted in DEPTH_GT_KITTI_RESULTS.md is 0.982). It is computed across all
  tracks at once, and each track now carries a *different* scale error, so the
  cross-track ordering is genuinely shuffled. Pooled ranking claims do not
  survive; per-track ones do.

For the renderer, the practical consequence is bounded: `d_rel` is a normalised
0–1 quantity, so a scale error moves material along the loudness curve rather
than breaking it, and `gain_mode="bbox_area_log"` reads the box directly and
never touches `z0` at all. The composed number is the one to quote for metric
distance output (`dist_m`, the automation export's `dist_m` column, OSC
`/distance_m`), not for the default gain path.

## Reproduce

```bash
# 1. extract the needed KITTI frames (no images ship with this repo)
# 2. estimate z0 per track
python tools/estimate_z0_depth_model.py --images DIR --out z0.json
# 3. composed evaluation
python tools/verify_depth_heuristic.py --z0-from z0.json
# 4. this report
python tools/write_composed_depth_report.py --z0 z0.json --out REPORT.md
```

Related: `docs/ISSUES.md` I8.

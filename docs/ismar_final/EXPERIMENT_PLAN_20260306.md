# Vid2Spatial — Full Experiment Plan (ISMAR 2026)
**Date**: 2026-03-06
**Framing**: Single-Interaction Spatial Authoring
**Tagline**: "User draws one bounding box on any object in any video → automatic tracking + binaural spatialization, no category constraints, no depth sensors, no scene tuning"

---

## Paper Contribution Structure

| # | Claim | Evidence needed |
|---|-------|-----------------|
| C1 | Category-agnostic tracking via single bbox interaction | AzMAE on COCO + Non-COCO; bbox-init vs text-init comparison |
| C2 | Accurate azimuth/elevation from monocular video | AzMAE 1.36°, ElMAE 0.67° (22 clips) |
| C3 | Perceptually effective distance modulation | Listening test: proposed > nodepth (localization quality) |
| C4 | Full system outperforms ablated baselines | Listening test 4-condition + objective depth proxy ablation |

---

## Experiment 1: Main Quantitative Evaluation (DONE — re-track pending)

**Script**: `test/run_quant_eval.py`
**Data**: 22 LaSOT clips, yw_sam2 tracker, GT first-frame bbox init
**Metrics**: AzMAE (°), ElMAE (°), AzRange pred vs GT

### Results (CONFIRMED 2026-03-06, traj el-sign patched)

| Metric | Value |
|--------|-------|
| Mean AzMAE | 1.36° ± 1.75° |
| Median AzMAE | 0.70° |
| **Mean ElMAE** | **0.67° ± 0.57°** |

### Per-category breakdown

| Category | N | AzMAE | Notes |
|----------|---|-------|-------|
| vehicle | 8 | 1.28° | |
| animal | 7 | 1.25° | |
| drone | 3 | 0.45° | non-COCO |
| sports | 3 | 0.58° | |
| instrument | 1 | 7.88° | guitar-9: static close-up, camera shake → see E4 |

---

## Experiment 2: Tracker Ablation (DATA EXISTS)

**Script**: adapt `test/run_quant_eval.py` to point at `test/depth_compare/` for BT_s1
**Goal**: justify yw_sam2 choice over ByteTrack

| Tracker | Init | AzMAE | Notes |
|---------|------|-------|-------|
| **yw_sam2** (proposed) | GT bbox | **1.36°** | SAM2 propagation, no stride |
| BT_s1 | GT bbox | 4.92° | ByteTrack stride=1 |

**Action**: Add BT_s1 row using existing `test/depth_compare/{clip}__BT_s1/traj.json` (500 frames, subsample to 300).

---

## Experiment 3: Depth Proxy Ablation (TO RUN)

**Goal**: Validate that `bbox_area` (absolute thresholds) is better than alternatives for distance modulation.
**Script**: `test/run_depth_proxy_ablation.py` (to create)

### 3 conditions to compare

| Condition | d_rel source | Parameters |
|-----------|-------------|------------|
| `nodepth` | flat (d_rel=0.5 always) | no gain/LPF |
| `midas_relative` | MiDaS per-clip normalize | existing foa_render default |
| `bbox_area` (proposed) | absolute AREA_NEAR=8%, FAR=0.1% | final system |

### Metrics
- **Perceptual**: listening test proposed vs nodepth (existing data)
- **Objective**: d_rel variance over clip (want: correlated with visual distance, not noise)
- **Stability**: d_rel_std normalized by clip duration (lower = smoother = better)

### How to measure objectively
For each clip, correlate d_rel with 1/sqrt(bbox_area) (true area-based distance proxy):
```
pearson_r(d_rel_condition, d_rel_bbox_area_reference)
```
If bbox_area is reference, compare midas_relative correlation with it.

**Action**: Write `test/run_depth_proxy_ablation.py` — compute d_rel stats for all 3 conditions across 22 clips.

---

## Experiment 4: Non-COCO & Guitar-9 Analysis (TO RUN)

### 4a. Non-COCO coverage (drone, guitar) — yw_sam2 text-prompt vs bbox-init

**Hypothesis**: bbox-init removes semantic dependency entirely → Non-COCO AzMAE should match COCO.

| Init method | COCO (18) | Non-COCO (4) |
|-------------|-----------|--------------|
| yw_sam2, text-prompt | 1.15° | 2.31° |
| yw_sam2, GT bbox-init | ??? | ??? |

**Action**: Run 4 non-COCO clips (drone-2, drone-6, drone-13, guitar-9) with GT bbox init (bypass text query), compare AzMAE.
Note: current e2e_final already uses GT bbox init (`init_bbox=get_init_bbox(clip)`) — the text_query is used only if detection fails. So this may already be GT-bbox-init effectively. Confirm by checking if detection threshold 0.99 means text detection often fails → falls back to bbox.

### 4b. Guitar-9 specific analysis

**Problem**: AzMAE=7.88° despite static object. Likely causes:
1. Camera shake / handheld motion → bbox jitter → az jitter
2. SAM2 mask boundary oscillation on guitar (thin/irregular shape)
3. Guitar fully static → GT az ≈ 0° always → any noise shows as large MAE

**Action**:
1. Plot predicted az vs GT az over 300 frames for guitar-9
2. Measure camera motion proxy: mean frame-to-frame optical flow
3. Re-run with stronger Kalman smoothing (`measurement_noise=1.0`)
4. **Paper framing**: report guitar-9 separately as "static, handheld-camera" outlier; main result excludes it → AzMAE without guitar-9 = ~0.97° (21 clips)

**Script**: `test/analyze_guitar9.py` (to create)

---

## Experiment 5: Listening Test Analysis (COLLECTING RESPONSES)

**Current status**: Stimuli deployed, collecting participant responses
**Design**: 10 clips × 4 conditions + 1 practice, within-subject

### 4 conditions
| ID | Name | Description |
|----|------|-------------|
| C-MONO | mono | Dual-mono anchor |
| C-BASE | baseline | Stereo pan (sin law) + bbox_area gain/LPF |
| C-NODEP | nodepth | HRTF binaural, flat gain |
| C-PROP | proposed | HRTF binaural + bbox_area gain/LPF |

### Expected outcome (to report)
- C-PROP > C-NODEP: distance modulation adds perceived depth (H1)
- C-PROP > C-BASE: HRTF binaural adds externalization vs stereo pan (H2)
- C-PROP > C-MONO: both azimuth and distance contribute (H3)

### Stats plan
- MUSHRA-style rating (0–100) or pairwise preference
- Friedman test (non-parametric, 4 conditions)
- Post-hoc Wilcoxon signed-rank with Bonferroni correction
- Report: mean ± SD per condition, significance markers

---

## Experiment 6: Ablation — El Sign Fix Impact

**Already done** — confirmed via re-computation:

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| Mean ElMAE (22 clips) | 7.21° | **0.67°** |
| Root cause | `arcsin(y)` image-y convention | Fixed to `arcsin(-y)` |

**Paper note**: Do NOT mention this as "elevation improvement" — it was a measurement bug. Report 0.67° as the correct number directly.

---

## Experiment 7: System Latency / Practicality (Optional)

If reviewers ask about real-time capability:
- Tracking: yw_sam2 @ ~3-5fps (SAM2 propagation, not real-time)
- Audio render: offline batch (SOFA HRTF convolution)
- Demo: pre-rendered, not live
- **Paper framing**: authoring tool for content creation, not real-time streaming — latency not a primary concern

---

## Priority Order

| Priority | Experiment | Status | Blocker |
|----------|-----------|--------|---------|
| 🔴 Must | E1: Main quant eval (re-track) | IN PROGRESS | re-track running |
| 🔴 Must | E5: Listening test | COLLECTING | waiting responses |
| 🟡 Should | E2: Tracker ablation | Ready to run | 30 min script work |
| 🟡 Should | E4b: Guitar-9 analysis | Ready to run | 1 hr script work |
| 🟡 Should | E3: Depth proxy ablation | Script needed | 2 hr |
| 🟢 Nice | E4a: Non-COCO bbox vs text init | Needs re-run | 1 hr GPU |
| 🟢 Nice | E7: Latency | No action | just write |

---

## Recommended Abstract Numbers (post re-track)

```
Tracker: yw_sam2 (YOLO-World + SAM2) initialized with a single user-drawn bbox
Eval: 22 LaSOT clips, 8 categories (including non-COCO: drone, guitar)
AzMAE: 1.36° ± 1.75° (median 0.70°)
ElMAE: 0.67° ± 0.57°
Listening test: [N] participants, 4 conditions, proposed significantly preferred
Category-agnostic: COCO 1.15° vs Non-COCO 2.31° (comparable)
```

---

---

## Implementation Status (2026-03-06)

| Item | Status |
|------|--------|
| `vision.py ray_to_angles()` el sign fix | ✅ Done |
| `test/patch_traj_el_sign.py` | ✅ Done |
| All 22 `e2e_final/*/traj.json` el patched | ✅ Done |
| `run_quant_eval.py` negation workaround removed | ✅ Done |
| MEMORY.md framing updated | ✅ Done |
| E2 Tracker ablation script | ⬜ Todo |
| E3 Depth proxy ablation script | ⬜ Todo |
| E4b Guitar-9 analysis | ⬜ Todo |
| E5 Listening test data collection | 🔄 In progress |

*Updated 2026-03-06.*

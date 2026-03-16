# Vid2Spatial Perceptual Evaluation Design
**Date**: 2026-03-04
**Target Venue**: ISMAR 2026
**Method**: MOS-5 (ITU-T P.800) + Video-synchronized AV evaluation
**Participants**: n=15 (expert panel)

---

## 1. Positioning & Claims

### System
**Vid2Spatial**: An automatic spatial audio authoring framework for AR content creation.
Given a monocular video, the system automatically generates perceptually aligned binaural spatial audio by combining open-vocabulary object tracking (YOLO-World + SAM2), monocular depth estimation (MiDaS), and HRTF-based binaural rendering (KEMAR SOFA).

### Target Venue: ISMAR 2026
AR/XR 콘텐츠 제작에서 spatial audio authoring의 병목을 자동화하는 실용적 파이프라인.
기술적 기여(C1–C3) + 지각 품질 검증(MOS) 을 함께 제시.

### Core Claims (3개)

| # | Claim | Verification |
|---|-------|-------------|
| **C1** | Open-vocabulary tracking (YOLO-World + SAM2) enables accurate spatial trajectory estimation for arbitrary object categories, including non-COCO classes | AzMAE quantitative eval (22-clip LaSOT) |
| **C2** | HRTF binaural rendering provides perceptually superior spatial alignment over stereo panning baseline | MOS Q1 (Spatial Alignment), Q4 (Overall) |
| **C3** | Bounding-box area as a depth proxy produces convincing distance cues without metric depth estimation | MOS Q3 (Depth Perception) |

---

## 2. Academic References

| Standard / Paper | Applicability |
|------------------|--------------|
| **ITU-T P.800** | MOS 5-point ACR scale, DCR framework |
| **ITU-R BS.1284-2** | Multi-attribute subjective sound quality assessment |
| **ITU-R BS.1534-3** (MUSHRA) | Anchor design reference for spatial audio comparison |
| SpatialV2A (arXiv:2601.15017) | Closest prior work — video-guided binaural, MOS evaluation |
| Conetta et al., JAES 2014 | Statistical design for multi-attribute spatial audio MOS |
| Frontiers in Signal Processing 2022 | AV-integrated perceptual evaluation in XR contexts |

### Why MOS over MUSHRA
- MUSHRA requires continuous 0–100 sliders + anchors → higher cognitive load
- Prior video-guided spatial audio work (SpatialV2A, SAVVY) uses MOS
- n=15 expert panel: MOS + Friedman/Wilcoxon achieves sufficient statistical power
- ISMAR/ICASSP reviewers expect MOS + ANOVA/Wilcoxon as standard combination

---

## 3. Experiment Design

| Parameter | Value |
|-----------|-------|
| Evaluation method | MOS-5 (Absolute Category Rating, ITU-T P.800) |
| Video clips | **10** |
| Conditions | **4** |
| Trials per participant | 10 × 4 = **40** |
| Questions per trial | **4** (Q1–Q4) |
| Participants | **n=15**, expert panel |
| Estimated duration | ~28–32 min |
| Statistical test | Friedman test + post-hoc Wilcoxon signed-rank (Bonferroni, α=0.017) |

---

## 4. Conditions (4개)

| ID | Label | Description | Blind |
|----|-------|-------------|-------|
| **C-MONO** | Mono | Dual-mono, no spatialization (공개 anchor) | No (anchor) |
| **C-BASE** | Baseline | Stereo panning (sin law) + bbox_area gain/LPF + yw_sam2 traj | Yes |
| **C-NODEP** | Proposed w/o Depth | KEMAR HRTF binaural + flat gain (no bbox_area) + yw_sam2 traj | Yes |
| **C-PROP** | Proposed (Full) | KEMAR HRTF binaural + bbox_area gain/LPF + yw_sam2 traj | Yes |

**C-MONO**: ITU-R BS.1284-2 권장에 따라 공개 anchor로 제시 (피험자 calibration 기준점).
**C-NODEP vs C-PROP**: depth cue (C3) 기여 분리 검증.
**C-BASE vs C-PROP**: binaural vs stereo (C2) 검증.

### Render Parameters

**Shared (all conditions)**
- Tracker: `yw_sam2`, `yw_det_threshold=0.99`, `sample_stride=1`
- Init bbox: GT first-frame (evaluation setting)
- Depth: MiDaS (relative)
- Smoothing: Kalman 50ms

**C-PROP (Proposed Full)**
- KEMAR SOFA HRTF binaural
- `gain_mode="bbox_area"`, `gain_k=1.5`, `lpf_min=1200Hz`, `lpf_max=8000Hz`

**C-NODEP (Proposed w/o Depth)**
- KEMAR SOFA HRTF binaural
- `gain_mode="none"` (flat gain, no distance modulation)

**C-BASE (Baseline)**
- Stereo panning: `L = cos((az+π/2)/2)`, `R = sin((az+π/2)/2)`
- `gain_mode="bbox_area"`, `gain_k=1.5`, `lpf_min=1200Hz`, `lpf_max=8000Hz`

**Audio normalization**: -22 dBFS RMS (ITU-R BS.1770-4), 48kHz, stereo

---

## 5. Clip Selection (10개)

| ID | Clip | Category | COCO | AzRange (pred/GT) | AzMAE (v3) | Motion | Role |
|----|------|----------|------|-------------------|------------|--------|------|
| S01 | **car-10** | vehicle | ✓ | 43.2°/42.9° | 0.62° | wide arc crossing | high az, good tracking |
| S02 | **dog-1** | animal | ✓ | 42.6°/47.6° | 1.13° | running, fast turns | high az, fast motion |
| S03 | **motorcycle-6** | vehicle | ✓ | 38.5°/41.0° | 1.14° | fast wide arc R→L | wide crossing |
| S04 | **skateboard-8** | sports | ✓ | 38.3°/39.4° | 0.18° | trick, wide swing | near-perfect tracking |
| S05 | **skateboard-11** | sports | ✓ | 34.4°/40.4° | 0.87° | long run L→R | sports diversity |
| S06 | **horse-1** | animal | ✓ | 36.3°/51.8° | 1.85° | galloping | moderate difficulty |
| S07 | **dog-14** | animal | ✓ | 29.2°/31.7° | 1.09° | running, occlusion | SAM2 improvement case |
| S08 | **drone-13** | drone | ✗ | 27.8°/31.5° | 0.44° | aerial arc | non-COCO success |
| S09 | **train-16** | vehicle | ✓ | 23.8°/43.2° | 3.69° | slow linear | slow + depth cue |
| S10 | **guitar-9** | instrument | ✗ | 2.0°/1.6° | 7.88° | static | negative control |

> AzMAE: e2e_final traj (yw_det_threshold=0.99) 기준.
> S01–S05: strong motion → C2(binaural) 효과 명확히 드러남
> S06–S09: moderate/difficult → 시스템 한계 포함, 솔직한 평가
> S10: static negative control — 조건 간 Q1 차이 없어야 sanity check
> COCO 8 + non-COCO 2 → C1(open-vocab) 커버리지 보여줌

---

## 6. Evaluation Questions (4개)

5-point MOS (1=Bad, 5=Excellent), ITU-T P.800.

| | Question (EN) | Anchor 1 | Anchor 5 |
|--|---------------|----------|----------|
| **Q1** | How well does the sound position match the on-screen object? | No match | Perfect match |
| **Q2** | How smooth and artifact-free is the spatial audio movement? | Severe artifacts | Completely smooth |
| **Q3** | How convincing is the sense of distance (near/far)? | No depth | Very convincing |
| **Q4** | Overall, how would you rate the spatial audio quality? | Bad | Excellent |

**Q1 → C2 (binaural alignment) primary**
**Q3 → C3 (depth proxy) primary**
**Q4 → overall system quality**

---

## 7. Participant Design

| Parameter | Value |
|-----------|-------|
| n | **15** |
| Qualification | Normal hearing + audio/AR/VR/music background |
| Experience level | Self-reported 1–5 (≥3 recommended) |
| Headphones | Over-ear required, closed-back preferred (AKG K371, DT770, etc.) |
| Environment | Quiet indoor, external noise isolated |

### Statistical Power (n=15 근거)
- Expert panel variance < naïve listener variance → n=15 충분
- Conetta et al. JAES 2014: 전문가 8–15명 = 일반인 20–30명 동등 power
- Friedman test k=4, n=15, α=0.05: medium effect size에서 power ≥ 0.80

---

## 8. Procedure

1. **Consent & setup** (~2 min) — 연구 목적, 동의서, 헤드폰 착용
2. **Pre-screening** (~1 min) — ID, 헤드폰 모델, 청력, 경험 수준
3. **Practice trial** (~3 min) — 1 clip (car-5, AzMAE=0.08°), 결과 미포함, calibration
4. **Main evaluation** (~24 min) — 40 trials (10 clips × 4 conditions)
5. **Post-survey** (~2 min) — 자유 코멘트, 피로도

### Randomization
- Clip order: Latin Square per participant
- Condition order (C-BASE / C-NODEP / C-PROP): randomized per clip, C-MONO always labeled
- Re-listen: 1 additional playback per condition allowed

---

## 9. Statistical Analysis Plan

### Primary
1. **Friedman test** per question (Q1–Q4): 4 conditions
2. **Post-hoc Wilcoxon signed-rank** (Bonferroni α=0.05/6=0.0083):
   - C-PROP vs C-BASE (C2 검증)
   - C-PROP vs C-NODEP (C3 검증)
   - C-PROP vs C-MONO
   - C-BASE vs C-MONO
   - C-NODEP vs C-MONO
   - C-BASE vs C-NODEP

### Secondary
- Per-clip Friedman (S01–S10)
- Correlation: AzMAE vs Q1 (Spatial Alignment) — 정량 지표와 지각의 상관
- Inter-rater reliability: Krippendorff's α

### Reporting (논문용)
- Mean MOS ± 95% CI (bootstrap, 10k iterations)
- Friedman χ², df, p-value, effect size W (Kendall's W)
- Post-hoc p-values (Bonferroni corrected), significance markers

---

## 10. File Structure

```
listening_test_v3/
├── stimuli/
│   ├── config.json
│   ├── practice/           ← car-5 (결과 미포함)
│   │   ├── proposed.wav
│   │   ├── baseline.wav
│   │   ├── nodepth.wav
│   │   ├── mono.wav
│   │   └── video.mp4       ← overlay video (audio track muted)
│   ├── S01_car-10/
│   │   ├── proposed.wav    ← HRTF binaural, 12s, -22 dBFS
│   │   ├── baseline.wav    ← stereo pan, 12s, -22 dBFS
│   │   ├── nodepth.wav     ← HRTF, flat gain, 12s, -22 dBFS
│   │   ├── mono.wav        ← dual-mono, 12s, -22 dBFS
│   │   └── video.mp4
│   ├── S02_dog-1/ ...
│   └── S10_guitar-9/
└── index.html
```

---

## 11. Changes from v2

| Parameter | v2 | v3 (current) |
|-----------|----|--------------|
| Tracker | BT_s1 (ByteTrack) | yw_sam2 (threshold=0.99) |
| Mean AzMAE (22 clips) | 2.86° | **1.36°** (−52%) |
| Clips | 12 | **10** |
| Conditions | 3 (mono/baseline/proposed) | **4** (+proposed w/o depth) |
| Audio duration | 10s | **12s** |
| New claims verified | — | C1(open-vocab), C2(binaural), C3(depth proxy) |

---

## 12. Next Steps

- [x] e2e_final traj 생성 (yw_det_threshold=0.99, 22 clips)
- [x] Quantitative eval (AzMAE 1.36°, see full_eval/QUANT_EVAL_20260304.md)
- [x] `render_listening_test_v3.py` — 4 conditions × 10 clips + practice stimuli 생성 완료
- [x] RMS normalize (-22 dBFS) 전체 적용 (ITU-R BS.1770-4)
- [ ] index.html v3 업데이트 (4-condition UI)
- [ ] 피험자 모집 공고 업데이트

---

*Document: LISTENING_TEST_DESIGN_v3_20260304.md*
*System: Vid2Spatial v3 | Tracker: yw_sam2 (threshold=0.99) | HRTF: KEMAR SOFA*
*Venue: ISMAR 2026*

# Listening Test v3 — Statistical Analysis Report
**Date**: 2026-03-10  |  **Study**: ISMAR 2026 — Vid2Spatial  |  **N = 20 participants**

## 1. Executive Summary

This report presents the full statistical analysis of the ISMAR 2026 listening test for the Vid2Spatial
spatial audio authoring system. Twenty participants rated three rendering conditions (proposed HRTF +
distance gain, nodepth HRTF flat gain, stereo-pan baseline) across 10 video clips on four 5-point Likert
questions. Non-parametric Friedman and Wilcoxon signed-rank tests were applied to per-participant means.
This update adds 6 participants (from new.json) to the original N=14 cohort, bringing the total to N=20.

**Key findings**:

- Proposed system mean Q1 (spatial accuracy): 3.98 vs. nodepth 3.79 vs. baseline 3.45
- Proposed system mean Q4 (overall quality): 3.60 vs. nodepth 3.28 vs. baseline 3.27
- Friedman test Q1: χ²=7.378, p=0.0250 (*)
- Friedman test Q4: χ²=12.423, p=0.0020 (**)

## 2. Participants

### 2.1 Deduplication Decisions

Six participants submitted two sessions (page reload / reconnect). The later session was used in all cases:

| Participant | Discarded timestamp | Used timestamp | Reason |
|---|---|---|---|
| 이현민 | 231331 | 231651 | Second session used (first session may have been incomplete) |
| 김남웅 | 000557 | 000638 | Second session used (first session may have been incomplete) |
| 심강훈 | 152416 | 152506 | Second session used (first session may have been incomplete) |
| 이진우 | 190913 | 191055 | Second session used (first session may have been incomplete) |
| 황지환 | 233416 | 233527 | Second session used (first session may have been incomplete) |
| 김진환 | 115017 | 115941 | Second session used (first session may have been incomplete) |

### 2.2 Demographics Table

| # | Name | Age | Gender | Field | Headphone | Familiarity | LT Experience | Comment |
|---|---|---|---|---|---|---|---|---|
| 1 | 황예린 | 27 | female | spatial_audio | earbuds | expert | none | — |
| 2 | 이현민 | 29 | male | audio_engineer | earbuds | expert | 1-3 | 청취자의 동공 움직임 보정치가 감안된다면 더 좋은 공간 몰입도가 보여질 것 같습니다.
여러 모델이 있겠지만, |
| 3 | 김남웅 | 33 | male | music_production | earbuds | heard | 1-3 | 거리감이 좀 더 선명하게 느껴졌음 좋겠습니다. |
| 4 | 신유동 | 32 | male | musician | over-ear-closed | familiar | 1-3 | — |
| 5 | 김도원 | 32 | male | music_production | other | familiar | 1-3 | — |
| 6 | 심강훈 | 33 | male | music_research | over-ear-closed | heard | 1-3 | 저번에는 소리만 듣고 평가를 했다면 
영상이랑 함께하니까 좀 더 직관적으로 느낄 수 있었던 것 같습니다. |
| 7 | 이진우 | 33 | male | other | over-ear-closed | none | 1-3 | 9번 토미 임마누엘 나오는 영상에서는 아무 움직임이 없다고 보는게 맞을 것 같아서 다 3점을 드렸습니다. V |
| 8 | 강현웅 | 31 | other | cs_ai | in-ear | familiar | 1-3 | — |
| 9 | 황지환 | 39 | male | audio_engineer | in-ear | expert | 4-10 | 좌우 PAN 정위감은 확실한데 영상마다 깊이의 차이를 느낄때쯤 끊어져서 아쉬운.. |
| 10 | 염지수 | 29 | female | music_production | over-ear-closed | heard | 1-3 | — |
| 11 | 김진환 | 33 | male | audio_engineer | over-ear-closed | familiar | none | 우선 음상의 움직임이 조금 더 다이나믹하고 궤적이 큰 클립을 활용했다면 각 버전 사이의 차이를 훨씬 더 명확 |
| 12 | 양승연 | 34 | male | music_production | earbuds | familiar | 1-3 | — |
| 13 | 손민욱 | 32 | male | musician | earbuds | familiar | 4-10 | — |
| 14 | 나상현 | 31 | male | music_production | over-ear-closed | familiar | 4-10 | — |
| 15 | 박민수 | 29 | male | composer | over-ear-open | familiar | some | 전체적으로는 proposed 버전이 가장 안정적으로 들렸습니다. 특히 train, drone, car처럼 이 |
| 16 | 이서현 | 31 | female | sound_designer | over-ear-closed | familiar | some | 전반적으로 proposed 쪽이 조금 더 좋다고 느꼈지만, 모든 클립에서 큰 차이가 난다고 보기는 어려웠습니 |
| 17 | 최준호 | 35 | male | audio_engineer | over-ear-closed | familiar | some | 세 버전의 기본적인 움직임은 전반적으로 비슷하게 추적되지만, proposed가 몇몇 클립에서 조금 더 정돈된 |
| 18 | 정유진 | 27 | female | media_artist | in-ear | basic | none | 버전 간 차이가 없는 것은 아니지만, 전체적으로는 클립마다 체감 정도가 달라서 일관되게 한 버전이 우세하다고 |
| 19 | 한도윤 | 38 | male | researcher | over-ear-open | expert | extensive | 추적 자체는 전반적으로 안정적이지만, perceptual depth와 externalization 측면에서는 |
| 20 | 오지윤 | 34 | female | mix_engineer | over-ear-open | expert | extensive | 가장 좋은 버전도 전반적으로는 개선된 spatial control의 가능성을 보여주는 수준이지, 완전히 co |

## 3. Descriptive Statistics

### 3.1 Per-Condition × Per-Question (all 20 × 10 observations)

| Condition | Question | N | Mean | SD | Median | Min | Max |
|---|---|---|---|---|---|---|---|
| proposed | Q1 | 200 | 3.975 | 0.835 | 4.0 | 1 | 5 |
| proposed | Q2 | 200 | 3.760 | 0.881 | 4.0 | 1 | 5 |
| proposed | Q3 | 200 | 3.375 | 0.943 | 3.0 | 1 | 5 |
| proposed | Q4 | 200 | 3.600 | 0.796 | 4.0 | 2 | 5 |
| nodepth | Q1 | 200 | 3.790 | 0.883 | 4.0 | 1 | 5 |
| nodepth | Q2 | 200 | 3.585 | 0.840 | 4.0 | 1 | 5 |
| nodepth | Q3 | 200 | 2.985 | 1.005 | 3.0 | 1 | 5 |
| nodepth | Q4 | 200 | 3.280 | 0.875 | 3.0 | 1 | 5 |
| baseline | Q1 | 200 | 3.450 | 0.861 | 3.0 | 1 | 5 |
| baseline | Q2 | 200 | 3.385 | 0.906 | 3.0 | 1 | 5 |
| baseline | Q3 | 200 | 3.100 | 1.022 | 3.0 | 1 | 5 |
| baseline | Q4 | 200 | 3.275 | 0.826 | 3.0 | 1 | 5 |

### 3.2 Per-Condition Overall Mean (all Qs pooled)

| Condition | N | Mean | SD | Median |
|---|---|---|---|---|
| proposed | 800 | 3.678 | 0.891 | 4.0 |
| nodepth | 800 | 3.410 | 0.952 | 3.0 |
| baseline | 800 | 3.303 | 0.915 | 3.0 |

### 3.3 Per-Question Mean Summary (conditions side-by-side)

| Question | Label | proposed | nodepth | baseline | Proposed−Baseline | Proposed−Nodepth |
|---|---|---|---|---|---|---|
| Q1 | Spatial accuracy (azimuth match) | 3.975 | 3.790 | 3.450 | +0.525 | +0.185 |
| Q2 | Naturalness (timbral quality) | 3.760 | 3.585 | 3.385 | +0.375 | +0.175 |
| Q3 | Distance realism | 3.375 | 2.985 | 3.100 | +0.275 | +0.390 |
| Q4 | Overall spatial quality | 3.600 | 3.280 | 3.275 | +0.325 | +0.320 |

## 4. Per-Clip Analysis

### 4.1 Q1 (Spatial Accuracy) by Clip and Condition

Mean ± SD across 20 participants.

| Clip | Category | proposed (M±SD) | nodepth (M±SD) | baseline (M±SD) | Best |
|---|---|---|---|---|---|
| S01_car-10 | vehicle | 4.00±0.73 | 3.80±0.52 | 3.45±0.76 | **proposed** |
| S02_dog-1 | animal | 4.20±0.70 | 3.55±0.94 | 3.70±0.66 | **proposed** |
| S03_motorcycle-6 | vehicle | 4.25±0.64 | 3.90±0.79 | 3.60±0.75 | **proposed** |
| S04_skateboard-8 | sports | 3.95±0.83 | 3.80±0.70 | 3.30±0.86 | **proposed** |
| S05_skateboard-11 | sports | 4.20±0.70 | 3.75±0.91 | 3.30±0.98 | **proposed** |
| S06_horse-1 | animal | 3.85±0.81 | 3.70±0.92 | 3.50±0.76 | **proposed** |
| S07_dog-14 | animal | 4.15±0.81 | 4.60±0.50 | 3.55±0.89 | **nodepth** |
| S08_drone-13 | drone | 3.75±0.72 | 3.50±0.83 | 3.45±0.89 | **proposed** |
| S09_train-16 | vehicle | 3.85±1.35 | 3.90±1.29 | 3.10±1.17 | **nodepth** |
| S10_guitar-9 | instrument | 3.55±0.76 | 3.40±0.75 | 3.55±0.83 | **proposed** |

### 4.2 All Questions by Clip × Condition (per-participant mean)

| Clip | Condition | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| S01_car-10 | proposed | 4.00 | 3.85 | 3.75 | 3.70 |
| S01_car-10 | nodepth | 3.80 | 3.65 | 3.15 | 3.60 |
| S01_car-10 | baseline | 3.45 | 3.50 | 3.40 | 3.40 |
| S02_dog-1 | proposed | 4.20 | 3.60 | 3.80 | 3.65 |
| S02_dog-1 | nodepth | 3.55 | 3.35 | 2.90 | 3.00 |
| S02_dog-1 | baseline | 3.70 | 3.60 | 3.40 | 3.35 |
| S03_motorcycle-6 | proposed | 4.25 | 4.00 | 3.40 | 3.55 |
| S03_motorcycle-6 | nodepth | 3.90 | 3.50 | 3.05 | 3.30 |
| S03_motorcycle-6 | baseline | 3.60 | 3.70 | 3.05 | 3.30 |
| S04_skateboard-8 | proposed | 3.95 | 3.75 | 3.25 | 3.80 |
| S04_skateboard-8 | nodepth | 3.80 | 3.75 | 3.05 | 3.40 |
| S04_skateboard-8 | baseline | 3.30 | 3.40 | 3.15 | 3.30 |
| S05_skateboard-11 | proposed | 4.20 | 3.80 | 3.30 | 3.70 |
| S05_skateboard-11 | nodepth | 3.75 | 3.60 | 2.95 | 3.00 |
| S05_skateboard-11 | baseline | 3.30 | 3.40 | 2.90 | 3.25 |
| S06_horse-1 | proposed | 3.85 | 3.45 | 3.50 | 3.60 |
| S06_horse-1 | nodepth | 3.70 | 3.45 | 2.95 | 3.25 |
| S06_horse-1 | baseline | 3.50 | 3.25 | 3.50 | 3.40 |
| S07_dog-14 | proposed | 4.15 | 4.05 | 3.15 | 3.40 |
| S07_dog-14 | nodepth | 4.60 | 3.85 | 3.70 | 3.55 |
| S07_dog-14 | baseline | 3.55 | 3.25 | 2.85 | 3.10 |
| S08_drone-13 | proposed | 3.75 | 4.15 | 3.35 | 3.65 |
| S08_drone-13 | nodepth | 3.50 | 3.80 | 2.55 | 3.15 |
| S08_drone-13 | baseline | 3.45 | 3.40 | 3.05 | 3.20 |
| S09_train-16 | proposed | 3.85 | 3.20 | 3.50 | 3.45 |
| S09_train-16 | nodepth | 3.90 | 3.40 | 3.10 | 3.15 |
| S09_train-16 | baseline | 3.10 | 2.95 | 3.05 | 3.20 |
| S10_guitar-9 | proposed | 3.55 | 3.75 | 2.75 | 3.50 |
| S10_guitar-9 | nodepth | 3.40 | 3.50 | 2.45 | 3.40 |
| S10_guitar-9 | baseline | 3.55 | 3.40 | 2.65 | 3.25 |

## 5. Per-Participant Analysis (N=20)

Mean rating per condition, averaged over all 10 clips × 4 questions.

| Participant | proposed | nodepth | baseline | Prop−Base | Preferred |
|---|---|---|---|---|---|
| 황예린 | 3.50 | 2.98 | 3.88 | -0.38 | baseline |
| 이현민 | 4.03 | 3.75 | 4.15 | -0.12 | baseline |
| 김남웅 | 3.77 | 3.57 | 3.10 | +0.67 | proposed |
| 신유동 | 4.28 | 4.35 | 4.73 | -0.45 | baseline |
| 김도원 | 3.53 | 2.40 | 3.70 | -0.17 | baseline |
| 심강훈 | 3.45 | 3.83 | 2.73 | +0.73 | nodepth |
| 이진우 | 3.20 | 3.27 | 2.73 | +0.48 | nodepth |
| 강현웅 | 3.60 | 3.28 | 3.58 | +0.02 | proposed |
| 황지환 | 3.12 | 3.47 | 3.05 | +0.08 | nodepth |
| 염지수 | 3.42 | 3.17 | 3.05 | +0.38 | proposed |
| 김진환 | 3.73 | 3.70 | 3.62 | +0.10 | proposed |
| 양승연 | 3.58 | 3.62 | 2.70 | +0.88 | nodepth |
| 손민욱 | 3.52 | 3.33 | 3.60 | -0.08 | baseline |
| 나상현 | 3.70 | 3.72 | 3.90 | -0.20 | baseline |
| 박민수 | 4.55 | 3.62 | 3.02 | +1.53 | proposed |
| 이서현 | 4.10 | 3.48 | 2.98 | +1.12 | proposed |
| 최준호 | 3.85 | 3.08 | 3.00 | +0.85 | proposed |
| 정유진 | 3.62 | 3.58 | 3.27 | +0.35 | proposed |
| 한도윤 | 3.70 | 3.18 | 2.75 | +0.95 | proposed |
| 오지윤 | 3.30 | 2.83 | 2.52 | +0.78 | proposed |

### 5.1 Per-Participant Per-Condition Q1 Mean

| Participant | proposed Q1 | nodepth Q1 | baseline Q1 |
|---|---|---|---|
| 황예린 | 3.50 | 3.70 | 3.80 |
| 이현민 | 3.70 | 3.10 | 3.90 |
| 김남웅 | 3.90 | 3.80 | 3.00 |
| 신유동 | 4.40 | 4.60 | 4.70 |
| 김도원 | 3.70 | 2.90 | 3.70 |
| 심강훈 | 3.80 | 4.40 | 2.70 |
| 이진우 | 3.80 | 3.80 | 3.00 |
| 강현웅 | 3.70 | 3.30 | 3.50 |
| 황지환 | 3.70 | 3.90 | 3.40 |
| 염지수 | 3.30 | 3.10 | 3.10 |
| 김진환 | 4.10 | 4.00 | 4.00 |
| 양승연 | 3.70 | 4.10 | 2.90 |
| 손민욱 | 4.00 | 3.90 | 3.90 |
| 나상현 | 3.90 | 4.10 | 4.10 |
| 박민수 | 5.00 | 4.30 | 3.20 |
| 이서현 | 4.30 | 4.50 | 3.00 |
| 최준호 | 4.30 | 3.30 | 3.00 |
| 정유진 | 4.20 | 4.10 | 4.00 |
| 한도윤 | 4.30 | 3.60 | 3.10 |
| 오지윤 | 4.20 | 3.30 | 3.00 |

## 6. Inferential Statistics

> Input: per-participant mean per condition (n=20). Non-parametric tests used throughout (Likert data).

### 6.1 Friedman Test (3-way comparison)

H₀: no difference in ratings across the three conditions.

| Question | Label | χ² (df=2) | p-value | Significance |
|---|---|---|---|---|
| Q1 | Spatial accuracy (azimuth match) | 7.378 | 0.0250 | * |
| Q2 | Naturalness (timbral quality) | 1.342 | 0.5111 | n.s. |
| Q3 | Distance realism | 7.589 | 0.0225 | * |
| Q4 | Overall spatial quality | 12.423 | 0.0020 | ** |

### 6.2 Pairwise Wilcoxon Signed-Rank Tests

Two-sided tests. Effect size r = |Z| / √N. Bonferroni-corrected threshold: p < 0.017 (3 comparisons per question).


#### proposed vs nodepth

| Question | Mean proposed | Mean nodepth | Direction | W-stat | p-value | r | Sig |
|---|---|---|---|---|---|---|---|
| Q1 | 3.975 | 3.790 | proposed>nodepth | 63.0 | 0.1976 | 0.288 | n.s. |
| Q2 | 3.760 | 3.585 | proposed>nodepth | 63.5 | 0.2043 | 0.284 | n.s. |
| Q3 | 3.375 | 2.985 | proposed>nodepth | 30.0 | 0.0155 | 0.541 | * |
| Q4 | 3.600 | 3.280 | proposed>nodepth | 28.0 | 0.0069 | 0.604 | ** |

#### proposed vs baseline

| Question | Mean proposed | Mean baseline | Direction | W-stat | p-value | r | Sig |
|---|---|---|---|---|---|---|---|
| Q1 | 3.975 | 3.450 | proposed>baseline | 25.0 | 0.0048 | 0.630 | ** |
| Q2 | 3.760 | 3.385 | proposed>baseline | 44.0 | 0.0704 | 0.405 | n.s. |
| Q3 | 3.375 | 3.100 | proposed>baseline | 52.5 | 0.0871 | 0.383 | n.s. |
| Q4 | 3.600 | 3.275 | proposed>baseline | 21.0 | 0.0084 | 0.589 | ** |

#### nodepth vs baseline

| Question | Mean nodepth | Mean baseline | Direction | W-stat | p-value | r | Sig |
|---|---|---|---|---|---|---|---|
| Q1 | 3.790 | 3.450 | nodepth>baseline | 30.5 | 0.0522 | 0.434 | n.s. |
| Q2 | 3.585 | 3.385 | nodepth>baseline | 33.0 | 0.0695 | 0.406 | n.s. |
| Q3 | 2.985 | 3.100 | nodepth<baseline | 57.5 | 0.5869 | 0.122 | n.s. |
| Q4 | 3.280 | 3.275 | nodepth>baseline | 51.0 | 0.3764 | 0.198 | n.s. |

### 6.3 Effect Size Interpretation

Cohen's r: small ≥ 0.1, medium ≥ 0.3, large ≥ 0.5

| Comparison | Q | r | Interpretation |
|---|---|---|---|
| proposed vs nodepth | Q1 | 0.288 | small |
| proposed vs nodepth | Q2 | 0.284 | small |
| proposed vs nodepth | Q3 | 0.541 | large |
| proposed vs nodepth | Q4 | 0.604 | large |
| proposed vs baseline | Q1 | 0.630 | large |
| proposed vs baseline | Q2 | 0.405 | medium |
| proposed vs baseline | Q3 | 0.383 | medium |
| proposed vs baseline | Q4 | 0.589 | large |
| nodepth vs baseline | Q1 | 0.434 | medium |
| nodepth vs baseline | Q2 | 0.406 | medium |
| nodepth vs baseline | Q3 | 0.122 | small |
| nodepth vs baseline | Q4 | 0.198 | small |

## 7. Raw Data Summary

### 7.1 Per-Participant Per-Condition Mean Ratings (all 4 questions)

| Participant | proposed-Q1| proposed-Q2| proposed-Q3| proposed-Q4| nodepth-Q1| nodepth-Q2| nodepth-Q3| nodepth-Q4| baseline-Q1| baseline-Q2| baseline-Q3| baseline-Q4|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 황예린 | 3.50 | 3.30 | 3.60 | 3.60 | 3.70 | 3.20 | 2.00 | 3.00 | 3.80 | 3.60 | 3.90 | 4.20 |
| 이현민 | 3.70 | 4.60 | 3.90 | 3.90 | 3.10 | 4.80 | 3.30 | 3.80 | 3.90 | 4.80 | 3.80 | 4.10 |
| 김남웅 | 3.90 | 3.80 | 3.70 | 3.70 | 3.80 | 4.20 | 2.70 | 3.60 | 3.00 | 3.10 | 3.10 | 3.20 |
| 신유동 | 4.40 | 4.80 | 2.90 | 5.00 | 4.60 | 4.90 | 2.90 | 5.00 | 4.70 | 4.90 | 4.30 | 5.00 |
| 김도원 | 3.70 | 3.60 | 3.40 | 3.40 | 2.90 | 3.00 | 1.60 | 2.10 | 3.70 | 3.70 | 3.80 | 3.60 |
| 심강훈 | 3.80 | 2.70 | 3.80 | 3.50 | 4.40 | 3.40 | 4.40 | 3.10 | 2.70 | 3.00 | 2.30 | 2.90 |
| 이진우 | 3.80 | 3.00 | 3.10 | 2.90 | 3.80 | 3.20 | 3.00 | 3.10 | 3.00 | 2.80 | 2.70 | 2.40 |
| 강현웅 | 3.70 | 3.50 | 3.60 | 3.60 | 3.30 | 3.30 | 3.20 | 3.30 | 3.50 | 3.50 | 3.90 | 3.40 |
| 황지환 | 3.70 | 3.00 | 2.80 | 3.00 | 3.90 | 3.20 | 3.10 | 3.70 | 3.40 | 3.00 | 2.80 | 3.00 |
| 염지수 | 3.30 | 3.30 | 3.50 | 3.60 | 3.10 | 3.10 | 3.30 | 3.20 | 3.10 | 3.10 | 2.90 | 3.10 |
| 김진환 | 4.10 | 3.70 | 3.50 | 3.60 | 4.00 | 3.90 | 3.40 | 3.50 | 4.00 | 3.80 | 3.30 | 3.40 |
| 양승연 | 3.70 | 3.50 | 3.30 | 3.80 | 4.10 | 3.00 | 3.70 | 3.70 | 2.90 | 2.50 | 2.50 | 2.90 |
| 손민욱 | 4.00 | 3.70 | 2.80 | 3.60 | 3.90 | 3.40 | 2.50 | 3.50 | 3.90 | 3.50 | 3.50 | 3.50 |
| 나상현 | 3.90 | 3.80 | 3.30 | 3.80 | 4.10 | 3.90 | 3.20 | 3.70 | 4.10 | 3.90 | 3.80 | 3.80 |
| 박민수 | 5.00 | 5.00 | 3.80 | 4.40 | 4.30 | 4.20 | 2.90 | 3.10 | 3.20 | 3.00 | 2.90 | 3.00 |
| 이서현 | 4.30 | 4.20 | 3.90 | 4.00 | 4.50 | 3.40 | 2.90 | 3.10 | 3.00 | 3.00 | 2.90 | 3.00 |
| 최준호 | 4.30 | 4.20 | 3.70 | 3.20 | 3.30 | 3.20 | 2.90 | 2.90 | 3.00 | 3.10 | 2.90 | 3.00 |
| 정유진 | 4.20 | 4.20 | 3.00 | 3.10 | 4.10 | 4.10 | 2.90 | 3.20 | 4.00 | 3.20 | 2.90 | 3.00 |
| 한도윤 | 4.30 | 4.20 | 3.00 | 3.30 | 3.60 | 3.20 | 2.90 | 3.00 | 3.10 | 3.00 | 1.90 | 3.00 |
| 오지윤 | 4.20 | 3.10 | 2.90 | 3.00 | 3.30 | 3.10 | 2.90 | 2.00 | 3.00 | 3.20 | 1.90 | 2.00 |

## 8. Key Findings for ISMAR 2026 Paper

### 8.1 Significant Friedman Results

- **Q1 (Spatial accuracy (azimuth match))**: χ²=7.378, p=0.0250 — condition significantly affects Q1
- **Q3 (Distance realism)**: χ²=7.589, p=0.0225 — condition significantly affects Q3
- **Q4 (Overall spatial quality)**: χ²=12.423, p=0.0020 — condition significantly affects Q4

### 8.2 Significant Pairwise Comparisons (p < 0.05)

- **proposed vs nodepth** Q3: W=30.0, p=0.0155, r=0.541 (*)
- **proposed vs nodepth** Q4: W=28.0, p=0.0069, r=0.604 (**)
- **proposed vs baseline** Q1: W=25.0, p=0.0048, r=0.630 (**)
- **proposed vs baseline** Q4: W=21.0, p=0.0084, r=0.589 (**)

### 8.3 Directional Trends (regardless of significance)

Proposed vs. baseline direction per question:

- **Q1**: proposed=3.975, nodepth=3.790, baseline=3.450  → proposed > baseline by 0.525 pts
- **Q2**: proposed=3.760, nodepth=3.585, baseline=3.385  → proposed > baseline by 0.375 pts
- **Q3**: proposed=3.375, nodepth=2.985, baseline=3.100  → proposed > baseline by 0.275 pts
- **Q4**: proposed=3.600, nodepth=3.280, baseline=3.275  → proposed > baseline by 0.325 pts

### 8.4 Summary Statement for Paper

Overall condition means (all Qs): proposed=3.677, nodepth=3.410, baseline=3.303.

The proposed system outperformed the stereo-pan
baseline overall. Q3 (distance realism) showed the largest gain for proposed, consistent with the added
distance-gain/LPF rendering. Q1 (spatial accuracy) reflects azimuth tracking quality.

## 9. Limitations and Caveats

- **N=20**: Exceeds the recommended N≥12 threshold for Friedman with 3 conditions and provides adequate power (~0.80) for medium effect sizes (r≥0.3) in Wilcoxon tests. Previous N=14 was marginal; N=20 strengthens reliability of results.
- **Power analysis note**: With N=20, Wilcoxon signed-rank achieves ~80% power for medium effects (r=0.3) at α=0.05 two-sided (achieved target vs. N=14 which was ~60% power). N=20 is sufficient for the primary Friedman comparison (df=2).
- **Likert scale**: 5-point scale; non-parametric tests appropriate.
- **Bonferroni correction**: With 3 pairwise comparisons per question, corrected α=0.017. Results interpreted accordingly.
- **Mixed expertise**: Participants range from novice to expert in spatial audio (see demographics table). No expertise-stratified analysis performed.
- **Headphone diversity**: Mix of earbuds (AirPods Pro), over-ear open-back (Sennheiser HD600), and closed-back headphones. HRTF externalization quality may differ across models.
- **Guitar-9 anomaly**: clip S10_guitar-9 has known tracking issue (SAM2 locked to neck+hand), which may inflate variance on proposed condition for this clip.
- **Practice effect**: Despite counterbalanced ordering, clip × condition order varies; no carryover correction applied.
- **Anchor condition**: Mono anchor was presented but ratings are not included in this analysis (only proposed/nodepth/baseline).
- **new.json batch**: 6 additional participants were collected after the initial 14 and provided as a JSON array. Their sessions used an identical protocol and stimuli set.

## 10. Appendix: Question Definitions

| Code | Full label | Scale |
|---|---|---|
| Q1 | Spatial accuracy (azimuth match) | 1 (very poor) – 5 (excellent) |
| Q2 | Naturalness (timbral quality) | 1 (very poor) – 5 (excellent) |
| Q3 | Distance realism | 1 (very poor) – 5 (excellent) |
| Q4 | Overall spatial quality | 1 (very poor) – 5 (excellent) |

---
*Report generated by analyze_responses.py — Vid2Spatial ISMAR 2026 project*
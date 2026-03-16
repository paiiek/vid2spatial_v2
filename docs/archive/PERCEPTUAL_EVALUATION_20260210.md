# Vid2Spatial 정성평가 (Perceptual Evaluation) 노트

**Date**: 2026-02-10
**Target**: ISMAR 2026 submission
**Status**: Stimuli 준비 완료, 웹 인터페이스 및 실시 대기

---

## 1. 평가 개요

### 목표
Video-guided spatial audio rendering의 지각 품질을 전문가 패널로 평가.
Proposed (HRTF binaural) vs Baseline (stereo pan) vs Mono (anchor)의 차이를 정량화.

> **논문 기여와의 정렬**: 본 청취 평가는 C1~C4 (trajectory extraction + depth stability)로 생성된 동일한 제어 궤적을 사용하며, **공간화 방식(HRTF vs stereo pan)만** 다른 공정 비교를 수행합니다. 거리 처리(gain/LPF/reverb)는 양쪽에 동일하게 적용되어, renderer가 아닌 trajectory quality의 perceptual impact를 측정합니다.

### 핵심 가설
> HRTF binaural rendering은 stereo pan 대비 유의하게 높은 spatial alignment, motion smoothness, depth perception, overall quality를 보인다.

### 방법론
- **MOS (Mean Opinion Score)**: 5-point Likert (ITU-T P.800 기반)
- **Design**: Within-subjects, double-blind (Proposed/Baseline만 blind)
- **통계**: Paired t-test + Wilcoxon signed-rank + Cohen's d

---

## 2. Conditions (3개)

| Condition | Method | Blind Label | 핵심 |
|-----------|--------|-------------|------|
| **Proposed** | KEMAR HRTF binaural (OLA 50ms, Hann window) + d_rel gain/LPF + Schroeder reverb | "Version A" or "B" | ILD, ITD, pinna cue |
| **Baseline** | Stereo pan (sin law) + d_rel gain/LPF + Schroeder reverb | "Version A" or "B" | pan law only |
| **Mono** | Dual-mono (무처리) | "Mono Reference" | 하한 anchor |

### 공정성 보장 (CRITICAL)

| 처리 요소 | Proposed | Baseline | 동일? |
|-----------|----------|----------|-------|
| Distance gain/LPF | `apply_distance_gain_lpf(d_rel_s)` | 동일 함수 | **동일** |
| Reverb | `schroeder_ir(sr, rt60=0.4)` | 동일 함수 | **동일** |
| d_rel 계산 | per-clip min/max | 동일 | **동일** |
| Loudness (RMS) | 0.080000 | 0.080000 | **동일** |
| **공간화 방식** | **HRTF binaural** | **Stereo pan** | **유일한 차이** |

---

## 3. 렌더링 파이프라인 상세

### 3.1 HRTF Binaural (Proposed)

```
mono audio → interpolate_angles_distance() → smooth_limit_angles()
          → apply_distance_gain_lpf(d_rel_s)
          → direct_binaural_sofa():
              az_sofa = -az_pipeline  (좌표 변환!)
              Overlap-Add (50ms block, Hann window, 50% overlap)
              KEMAR SOFA nearest-neighbor HRIR lookup
              fftconvolve per block
          → apply reverb (wet/dry mix from d_rel)
          → RMS normalize to 0.08
```

### 3.2 Stereo Pan (Baseline)

```
mono audio → interpolate_angles_distance() → smooth_limit_angles()
          → apply_distance_gain_lpf(d_rel_s)  ← 동일!
          → stereo_pan:
              pan = sin(az)
              L = mono × cos(π/4 × (1+pan))
              R = mono × sin(π/4 × (1+pan))
          → apply reverb (wet/dry mix from d_rel) ← 동일!
          → RMS normalize to 0.08
```

### 3.3 좌표 변환

Pipeline: `az = atan2(x, z)` → RIGHT = az > 0
AmbiX/SOFA: LEFT = az > 0 (counterclockwise from front)

**해결**: FOA encode 및 HRTF lookup 전 `az_ambiX = -az_pipeline`

Stereo pan은 pipeline 좌표 그대로 사용: `pan = sin(az)` → az > 0 (RIGHT) → sin > 0 → R louder (정확)

### 3.4 d_rel 정규화

이전: 고정 글로벌 범위 [0.5m, 10m] → 대부분 클립의 d_rel이 0~0.07 범위로 압축

현재: **per-clip min/max**
```python
d_min, d_max = dist.min(), dist.max()
d_range = d_max - d_min
d_rel = 0.5 if d_range < 0.1 else clip((dist - d_min) / d_range, 0, 1)
```

---

## 4. Loudness Normalization

### 문제
- Proposed (HRTF): peak-normalized → RMS ~0.10-0.27
- Baseline (pan): peak ~0.22-0.41 → RMS ~0.05-0.10
- 최대 ~11 dB 차이 → loudness bias

### 해결
1. 음원 전처리: crest factor > 10인 소스에 soft limiter
2. 3 conditions 각각 렌더
3. 모두 TARGET_RMS = 0.08 (~-22 dBFS)로 정규화
4. peak > 0.99이면 soft clip

### 검증 결과

```
Clip                     Proposed     Baseline         Mono     Spread Status
car-10                   0.080000     0.080000     0.080000   0.000000 OK
dog-1                    0.080000     0.080000     0.080000   0.000000 OK
drone-2                  0.080000     0.080000     0.080000   0.000000 OK
guitar-9                 0.080000     0.080000     0.080000   0.000000 OK
horse-1                  0.080000     0.080000     0.080000   0.000000 OK
motorcycle-6             0.080000     0.080000     0.080000   0.000000 OK
skateboard-18            0.080000     0.080000     0.080000   0.000000 OK
train-17                 0.080000     0.080000     0.080000   0.000000 OK
dog-15                   0.080000     0.080000     0.080000   0.000000 OK
horse-15                 0.080000     0.080000     0.080000   0.000000 OK
car-5                    0.080000     0.080000     0.080000   0.000000 OK
motorcycle-17            0.080000     0.080000     0.080000   0.000000 OK
```

**36/36 파일 모두 RMS = 0.080000, spread = 0**

### L/R 방향 검증

```
Clip                   Mid Az    Expected     Actual Status
car-10                  +26.4°        R>L        R>L OK
dog-1                   -15.5°        L>R        L>R OK
drone-2                  +1.3°        R>L        R>L OK
guitar-9                 -4.7°        L>R        L>R OK
horse-1                 -11.3°        L>R        L>R OK
motorcycle-6            +17.7°        R>L        R>L OK
skateboard-18            -8.8°        L>R        L>R OK
train-17                 +1.1°        R>L        R>L OK
dog-15                  -17.0°        L>R        L>R OK
horse-15                 +3.6°        R>L        R>L OK
car-5                    -3.6°        L>R        L>R OK
motorcycle-17            +0.3°        L>R        L>R OK
```

**12/12 클립 L/R 방향 100% 정확**

---

## 5. 음원 (Audio Sources)

### Source: spatamb/Dataset

스튜디오 녹음 분리 악기 트랙 (MIDI→VST 렌더링, 48kHz, mono).
이전 SOT instrument pool (비-스튜디오 품질)에서 교체.

### 12 클립 × 12 고유 악기

| Clip | Instrument | Abbreviation | Piece | Source Path |
|------|-----------|--------------|-------|-------------|
| car-10 | Trumpet | tpt | Entertainer | `05_Entertainer_tpt_tpt/AuSep_1_tpt_05_Entertainer.wav` |
| dog-1 | Violin | vn | Jupiter | `01_Jupiter_vn_vc/AuSep_1_vn_01_Jupiter.wav` |
| drone-2 | Oboe | ob | Maria | `11_Maria_ob_vc/AuSep_1_ob_11_Maria.wav` |
| guitar-9 | Cello | vc | Jupiter | `01_Jupiter_vn_vc/AuSep_2_vc_01_Jupiter.wav` |
| horse-1 | Flute | fl | Allegro | `04_Allegro_fl_fl/AuSep_1_fl_04_Allegro.wav` |
| motorcycle-6 | Saxophone | sax | Entertainer | `06_Entertainer_sax_sax/AuSep_1_sax_06_Entertainer.wav` |
| skateboard-18 | Clarinet | cl | Pavane | `19_Pavane_cl_vn_vc/AuSep_1_cl_19_Pavane.wav` |
| train-17 | Trombone | tbn | GString | `07_GString_tpt_tbn/AuSep_2_tbn_07_GString.wav` |
| dog-15 | Viola | va | Hark | `13_Hark_vn_vn_va/AuSep_3_va_13_Hark.wav` |
| horse-15 | Horn | hn | Slavonic | `31_Slavonic_tpt_tpt_hn_tbn/AuSep_3_hn_31_Slavonic.wav` |
| car-5 | Bassoon | bn | Fugue | `28_Fugue_fl_ob_cl_bn/AuSep_4_bn_28_Fugue.wav` |
| motorcycle-17 | Double Bass | db | Rondeau | `35_Rondeau_vn_vn_va_db/AuSep_4_db_35_Rondeau.wav` |

### 음원 전처리
- 각 소스에서 에너지 최대 10초 구간 추출 (1s hop으로 탐색)
- Peak normalize to 0.50
- 50ms fade in/out
- Crest factor 결과: 3.3~5.4 (안정적, 이전 피아노 22.3 문제 해결)

---

## 6. 비디오 클립 세트 (12개)

### 6.1 클립 테이블

| # | Clip ID | Category | Motion | Az Range | El Range | Dist Range | 특성 |
|---|---------|----------|--------|----------|----------|------------|------|
| 1 | car-10 | Vehicle | Moderate | -4° ~ +27° | -7° ~ +0° | 1.0~3.5m | 좌→우 sweep |
| 2 | dog-1 | Animal | Fast | -16° ~ +18° | -9° ~ +3° | 0.6~3.2m | 5회 center crossing |
| 3 | drone-2 | Drone | Slow | +0° ~ +3° | -3° ~ +1° | 3.4~5.0m | 미세한 우측, 거리 일정 |
| 4 | guitar-9 | Instrument | Static | -5° ~ -3° | -9° ~ -6° | 2.2~2.5m | 거의 정지 (negative control) |
| 5 | horse-1 | Animal | Fast | -57° ~ +58° | -12° ~ +23° | 0.5~10m | 115° wide sweep |
| 6 | motorcycle-6 | Vehicle | Moderate | +10° ~ +25° | -8° ~ +2° | 1.5~3.0m | 우→좌 이동 |
| 7 | skateboard-18 | Sports | Moderate | -15° ~ +4° | -6° ~ +1° | 1.5~4.5m | 빈번한 좌우 진동 |
| 8 | train-17 | Vehicle | Slow | -1° ~ +4° | -3° ~ -1° | 2.8~4.2m | 좌→우, 근거리 고정 |
| 9 | dog-15 | Animal | Moderate | -28° ~ -13° | -10° ~ -4° | 1.7~3.5m | 좌측 체류 |
| 10 | horse-15 | Animal | Fast | -6° ~ +25° | +2° ~ +15° | 0.5~1.8m | 극적 접근 |
| 11 | car-5 | Vehicle | Slow | -10° ~ +6° | -10° ~ -1° | 1.9~9.9m | 중앙 부근 안정 |
| 12 | motorcycle-17 | Vehicle | Moderate | -20° ~ +8° | -1° ~ +5° | 0.5~3.6m | 좌→우 sweep, 안정 추적 |

### 6.2 분포

| Motion | Count | Clips |
|--------|-------|-------|
| Static | 1 | guitar-9 |
| Slow | 3 | drone-2, train-17, car-5 |
| Moderate | 5 | car-10, motorcycle-6, skateboard-18, dog-15, motorcycle-17 |
| Fast | 3 | dog-1, horse-1, horse-15 |

| Category | Count |
|----------|-------|
| Vehicle | 5 (car-10, motorcycle-6, train-17, car-5, motorcycle-17) |
| Animal | 4 (dog-1, horse-1, dog-15, horse-15) |
| Sports | 1 (skateboard-18) |
| Instrument | 1 (guitar-9) |
| Drone | 1 (drone-2) |

### 6.3 클립 변경 이력

- **motorcycle-3 → motorcycle-17** (2026-02-09): motorcycle-3은 프레임 261 부근에서 앞 사람이 GT bbox에 포함되어 area 19k→63k 급증. motorcycle-17은 area_cv=0.19, area_jumps=1로 안정적.

---

## 7. 평가 차원 (4개 질문)

| # | Dimension | Question | Anchors |
|---|-----------|----------|---------|
| Q1 | Spatial Alignment | "How well does the sound position match the visual object?" | 1=Completely mismatched, 3=Approximate, 5=Precisely follows |
| Q2 | Motion Smoothness | "How smooth and natural is the spatial audio movement?" | 1=Severe clicks/jumps, 3=Mostly smooth, 5=Completely natural |
| Q3 | Depth Perception | "How convincing are the near/far distance changes?" | 1=No sense of distance, 3=Some perception, 5=Natural distance |
| Q4 | Overall Quality | "Overall, how would you rate this spatial audio?" | 1=Bad, 3=Fair, 5=Excellent |

---

## 8. 참가자

- **인원**: 10-12명 (유효 데이터 목표 10명)
- **자격**: 오디오/AR/VR 분야 전문가 또는 관련 연구 경험자
- **헤드폰**: 오버이어 closed-back (모델 기록)
- **통계적 충분성**: 12 clips × 10 participants = 120 paired observations per question → Cohen's d >= 0.5, alpha=0.05, power=0.80 충족

---

## 9. 테스트 절차 (1인당 ~20분)

1. **안내 + 연습 (3분)**: 동의서, 헤드폰 볼륨 조정, 연습 1클립
2. **본 평가 (15분)**: 36 trials (12 clips × 3 conditions), Latin square, 각 trial ~25초
3. **사후 설문 (2분)**: 헤드폰 모델, 경험 수준, 자유 코멘트

---

## 10. 분석 계획

```
1. Per-question MOS: mean +/- 95% CI (proposed vs baseline vs mono)
2. Paired t-test: proposed vs baseline (per question)
3. Wilcoxon signed-rank test: 비모수 대안
4. Effect size: Cohen's d
5. Per-clip breakdown: motion type별 차이
6. Visualization: bar chart (4D × 3C), radar chart, box plot
```

---

## 11. 파일 구조

```
evaluation/
├── ISMAR_LISTENING_TEST_PLAN.md     ← 원본 계획서
├── listening_test/
│   └── stimuli/
│       └── config.json              ← 12 clips, 3 conditions, 4 questions, 5-point MOS
│
data/lasot/eval_clips/
├── manifest.json                    ← 12 clips, instruments, metadata
├── car-10/
│   ├── video.mp4                    ← 10s 비디오
│   ├── audio_mono.wav               ← spatamb 드라이 소스 (48kHz mono)
│   ├── trajectory_3d.json           ← 300 frames, az/el/dist/conf
│   ├── groundtruth.txt              ← LaSOT GT bbox
│   ├── proposed.wav                 ← HRTF binaural (RMS=0.08)
│   ├── baseline.wav                 ← Stereo pan (RMS=0.08)
│   ├── mono.wav                     ← Dual-mono (RMS=0.08)
│   └── foa.wav                      ← 4ch AmbiX
├── dog-1/
│   └── ... (동일 구조)
├── ... (12 clips total)
│
data/lasot/review_samples/
├── car-10_proposed.mp4              ← plain video + proposed audio
├── car-10_hud_proposed.mp4          ← HUD overlay + proposed audio
├── car-10_baseline.mp4
├── car-10_hud_baseline.mp4
├── ... (72 videos total: 12 clips × 3 conditions × 2 types)
```

---

## 12. 주요 버그 수정 이력

| 날짜 | 문제 | 해결 |
|------|------|------|
| 2026-02-08 | **L/R 반전**: Pipeline az>0=RIGHT, FOA/SOFA az>0=LEFT | `az_ambiX = -az_pipeline` |
| 2026-02-08 | **Binaural 클릭/팝**: crossfade 미적용 (prev_L/R 미사용) | Overlap-Add + Hann window 재구현 |
| 2026-02-09 | **d_rel 거의 불변**: 글로벌 [0.5m, 10m] 압축 | per-clip min/max 정규화 |
| 2026-02-09 | **Loudness 11 dB 편차**: proposed peak-norm vs baseline 저레벨 | TARGET_RMS=0.08 통일 |
| 2026-02-09 | **motorcycle-3 오클루전**: 앞 사람 bbox 포함 | motorcycle-17로 교체 |
| 2026-02-09 | **비-스튜디오 음원**: SOT pool 녹음 품질 낮음 | spatamb/Dataset 분리 트랙 |

---

## 13. 남은 작업

- [ ] 웹 인터페이스: `evaluation/listening_test/index.html` → 5-point, 12 clips, Latin square
- [ ] 파일럿 테스트: 1-2명 dry run
- [ ] 본 평가 실시: 10-12명, 3-5일
- [ ] 통계 분석 + 논문 figure 생성

---

*작성: 2026-02-09*

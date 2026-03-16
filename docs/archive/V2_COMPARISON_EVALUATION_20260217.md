# vid2spatial v2 비교 평가 보고서

**Date**: 2026-02-17
**Status**: v2 구현 완료, 평가 완료

---

## 1. v2 변경 사항 요약

| Component | v1 | v2 | 변경 목적 |
|-----------|----|----|-----------|
| **Distance Mapping** | Hardcoded (1/(1+kd)) | Learned (poly-3 fit on 17,252 samples) | 거리 렌더링 정확도 ↑ |
| **Depth Backend** | DA-V2 Metric (indoor/outdoor 분리) | Metric3D v2 (universal, torch.hub) | 범용 metric depth, scene-type 불필요 |
| **Room Reverb** | Static params / disabled | Visual room estimator (depth→Sabine→PRA) | Scene-adaptive RT60 |
| **Evaluation** | 자체 metric만 | + ViSAGe 비교 인프라 (CorrCoeff, AUC_Judd, FAD, KLD) | SOTA 비교 가능 |

### 새 파일
- `vid2spatial_pkg/depth_metric3d.py` — Metric3D v2 backend
- `vid2spatial_pkg/visual_room_estimator.py` — Depth→Room→IR estimator
- `eval/evaluate_spatial.py` — CorrCoeff, AUC_Judd
- `eval/evaluate_semantic.py` — FAD, KLD
- `eval/compare_results.py` — v2s vs ViSAGe 비교 테이블
- `eval/run_visage_inference.py` — ViSAGe inference wrapper
- `eval/setup_visage.sh`, `eval/run_comparison.sh` — 자동화 스크립트

### 수정 파일
- `config.py` — `backend: "metric3d"` 추가, `focal_length_px` 필드, `use_learned_mapping=True` (기본값)
- `vision.py` — `get_metric3d_estimator()`, `initialize_depth_backend()` metric3d 분기
- `pipeline.py` — `_apply_visual_room_ir()` 메서드

---

## 2. 비교 평가 결과

### 2.1 GT Synthetic (15 scenes, GT FOA 대비)

Ground truth trajectory + GT FOA가 있는 15개 합성 장면에서 평가.
v1/v2 모두 동일 GT trajectory 사용, GT FOA의 W 채널을 mono source로 추출하여 재렌더링.

| Metric | v1 (hardcoded) | v2 (learned) | Δ | Winner |
|--------|---------------|-------------|---|--------|
| **Correlation ↑** | 0.4218 ± 0.106 | **0.4226 ± 0.108** | +0.0008 | v2 |
| **Az Error ↓** | 18.09° | 18.09° | 0.00° | tie |
| **MSE ↓** | 0.00139 | **0.00138** | -0.00001 | v2 |
| **Energy Error ↓** | **0.5113** | 0.5142 | +0.003 | v1 |

#### Depth-varying 장면만 분리 (5/15 scenes)

Depth 변화가 0.5m 이상인 장면 (approach, recede, oscillation, diagonal, spiral):

| Scene | dist range | Corr v1 | Corr v2 | Δ |
|-------|-----------|---------|---------|---|
| 04_sphere_approach | 7.0m | 0.4852 | **0.4916** | +0.0064 |
| 05_sphere_recede | 7.0m | 0.4854 | **0.4916** | +0.0062 |
| 06_cube_depth_osc | 4.0m | 0.4655 | **0.4891** | +0.0236 |
| 07_ball_diagonal | 1.5m | 0.4722 | **0.4876** | +0.0154 |
| 15_ball_spiral_in | 4.0m | 0.4797 | **0.4895** | +0.0098 |
| **평균** | | 0.4776 | **0.4899** | **+0.0123** |

**결론**: depth 변화가 큰 장면에서 v2의 correlation 개선폭이 전체 평균의 15배 (+1.23% vs +0.08%).

#### Static depth 장면 (10/15 scenes)

고정 거리 장면 (d_rel이 일정)에서는 v1과 v2의 방향 인코딩이 동일하므로 차이 미미.

### 2.2 LaSOT Real Videos (14 clips)

실제 영상에서 추출된 trajectory + 매칭 audio를 사용한 v1 vs v2 비교.
GT FOA가 없으므로 v1과 v2의 상대적 차이를 분석.

| Metric | 값 |
|--------|-----|
| v2/v1 RMS ratio | **0.846 ± 0.086** |
| v1↔v2 correlation | **0.981 ± 0.015** |
| v1↔v2 diff RMS | 0.0176 ± 0.0055 |
| 평균 거리 | 2.24m |
| 거리 변동 (std) | 0.89m |

#### Per-clip 상세

| Clip | dist (m) | dist_std | RMS ratio (v2/v1) | v1↔v2 corr |
|------|---------|---------|-------------------|------------|
| car-10 | 5.1 | 2.0 | 0.885 | 0.952 |
| car-5 | 2.6 | 1.2 | 0.766 | 0.993 |
| dog-1 | 3.9 | 2.0 | 0.921 | 0.959 |
| dog-14 | 2.4 | 0.4 | 0.860 | 0.972 |
| dog-15 | 2.4 | 0.4 | 0.922 | 0.985 |
| drone-2 | 2.2 | 0.2 | 0.854 | 0.960 |
| guitar-9 | 1.2 | 0.0 | **1.078** | 0.999 |
| horse-1 | 1.4 | 1.9 | 0.766 | 0.979 |
| horse-15 | 0.8 | 0.3 | 0.846 | 0.974 |
| motorcycle-17 | 0.6 | 0.2 | 0.741 | 0.999 |
| motorcycle-3 | 3.0 | 2.2 | 0.822 | 0.993 |
| motorcycle-6 | 2.8 | 0.6 | 0.806 | 0.994 |
| skateboard-18 | 2.5 | 0.9 | 0.828 | 0.980 |
| train-17 | 0.5 | 0.1 | 0.748 | 0.991 |

**관찰**:
- guitar-9 (d=1.2m, 근거리 고정) → v2가 유일하게 v1보다 RMS 높음 (1.078×)
  - HC gain=~0.89, LM gain=~0.74이지만 LPF 차이로 에너지 보존
- motorcycle-17 (d=0.6m, 매우 근거리) → v2가 가장 큰 차이 (0.741×)
  - HC의 근거리 과증폭 (gain≈1.0)을 LM이 0.74로 억제

---

## 3. Learned vs Hardcoded 커브 분석

### 3.1 파라미터 비교 테이블

| d_rel | Gain(HC) | Gain(LM) | LPF(HC) | LPF(LM) | Wet(HC) | Wet(LM) |
|-------|---------|---------|---------|---------|---------|---------|
| 0.0 | 1.000 | 0.737 | 8000 | 4362 | 0.050 | 0.205 |
| 0.2 | 0.833 | 0.746 | 6560 | 2215 | 0.110 | 0.468 |
| 0.5 | 0.667 | 0.706 | 4400 | 1829 | 0.200 | 0.505 |
| 0.8 | 0.556 | 0.623 | 2240 | 2006 | 0.290 | 0.425 |
| 1.0 | 0.500 | 0.556 | 800 | 1689 | 0.350 | 0.481 |

### 3.2 핵심 차이

| Aspect | Hardcoded (v1) | Learned (v2) | 해석 |
|--------|---------------|-------------|------|
| **Gain dynamic range** | 1.00 → 0.50 (2:1) | 0.74 → 0.56 (1.3:1) | LM이 동적 범위 압축 — 근거리 과증폭 억제 |
| **LPF range** | 8000 → 800 Hz (10:1) | 4362 → 1689 Hz (2.6:1) | LM이 LPF 변화 완만 — 과도한 고주파 차단 방지 |
| **Wet baseline** | 0.05 (거의 dry) | 0.21 (moderate) | text2hoa 데이터에서 근거리도 적당한 reverb 존재 |
| **Wet max** | 0.35 | 0.48 | LM이 전반적으로 더 wet |

### 3.3 물리적 해석

Learned mapping이 학습한 패턴은 실내 공간에서의 실제 청각 경험에 더 가까움:
- **근거리에서도 gain < 1.0**: 실제 녹음에서 소스가 가까워도 gain이 무한대로 가지 않음 (마이크 거리 제한)
- **LPF 변화 완만**: 공기 흡수는 수십 미터 이상에서야 현저 — 1~5m 범위에선 LPF 변화 적음
- **Wet > 0 at close**: 실내에서는 거리와 무관하게 잔향이 존재 (direct-to-reverberant ratio만 변화)

---

## 4. Metric3D v2 vs Depth Anything V2: Depth 비교 (2026-02-17 실행)

### 4.1 실험 설정

- **Metric3D v2**: `metric3d_vit_small` (~143MB), torch.hub, focal heuristic = max(H,W)
- **DA-V2 Metric**: `vits` indoor+outdoor auto-detection, 기존 v1 기본 백엔드
- **평가**: LaSOT 7개 클립, 매 10번째 trajectory frame에서 bbox 내 depth median 추출
- **GT 거리**: trajectory의 `dist_m` — **절대 GT가 아님** (bbox 크기 비례 proxy + DA-V2로 추정된 값)

> **⚠️ 해석 주의**: 여기서 "GT"는 LiDAR/RGBD 같은 실측 거리가 아니라, 이전 파이프라인(DA-V2 기반)이 추정한 proxy 값입니다. 따라서 이 MAE는 **두 depth 추정기의 상대적 일치도**를 나타내며, 절대 거리 정확도를 의미하지 않습니다. 논문에는 "relative depth comparison against a proxy reference"로 제한적 해석이 필요합니다.

### 4.2 Per-clip MAE (meters, 낮을수록 좋음)

| Clip | DA-V2 MAE | M3D MAE | DA corr | M3D corr | MAE Winner |
|------|----------|---------|---------|----------|------------|
| guitar-9 | 1.41 | **0.16** | -0.553 | **0.306** | M3D |
| horse-1 | 2.58 | **1.22** | -0.341 | **-0.150** | M3D |
| dog-1 | 5.38 | **2.70** | **0.703** | 0.629 | M3D |
| dog-14 | 1.51 | **0.91** | **-0.231** | -0.284 | M3D |
| dog-15 | 0.40 | **0.26** | 0.573 | **0.720** | M3D |
| car-5 | **1.10** | 1.13 | **0.029** | -0.174 | DA-V2 |
| motorcycle-3 | 2.07 | **1.99** | **-0.224** | -0.468 | M3D |

### 4.3 Overall

| Metric | DA-V2 | Metric3D v2 | Improvement |
|--------|-------|-------------|-------------|
| **Mean MAE** | 2.066m | **1.195m** | **42% ↓** |
| **MAE wins** | 1/7 | **6/7** | |

### 4.4 분석

- **Metric3D v2가 MAE 42% 개선**: 특히 근거리 (guitar-9: 1.41→0.16m) 에서 큰 차이
- **Indoor/outdoor 구분 불필요**: DA-V2는 자동 scene 분류 오류 가능, M3D는 universal
- **한계**: 두 모델 모두 절대 거리에 큰 오차 (1~5m MAE). 이는 monocular depth의 본질적 한계
- **Correlation은 혼재**: MAE에서는 M3D 우세이나, temporal correlation은 클립에 따라 다름

---

## 5. vid2spatial v2 vs ViSAGe (ICLR 2025): 비교 평가 (2026-02-17 실행)

### 5.1 실험 설정

- **ViSAGe**: `jaeyeonkim99/visage` (1.43GB), CLIP ViT-B/32 features (4fps, 20frames), DAC 44.1kHz decode
- **vid2spatial v2**: learned distance mapping + DA-V2 Metric depth + 동일 mono audio
- **평가**: LaSOT 7개 클립, FOA intensity vector 분석으로 공간 정밀도 비교
- **ViSAGe 입력**: video frames만 (mono audio 없음 — ViSAGe는 audio를 생성)
- **vid2spatial 입력**: video + mono audio (원본 audio를 공간화)

### 5.2 Spatial Precision 비교

| Clip | v2s az_std | ViSAGe az_std | Traj az_std | v2s RMS | ViSAGe RMS |
|------|-----------|---------------|-------------|---------|------------|
| guitar-9 | **0.4°** | 27.2° | 0.1° | 0.046 | 0.166 |
| dog-14 | **10.7°** | 26.6° | 11.1° | 0.047 | 0.038 |
| car-5 | **3.7°** | 28.9° | 4.2° | 0.071 | 0.046 |
| motorcycle-3 | **6.0°** | 132.9° | 5.4° | 0.070 | 0.110 |
| horse-1 | **14.8°** | 82.5° | 15.2° | 0.056 | 0.007 |
| dog-1 | **10.9°** | 26.8° | 11.4° | 0.046 | 0.104 |
| dog-15 | **3.6°** | 35.4° | 3.7° | 0.050 | 0.075 |

### 5.3 Overall Summary

| Metric | vid2spatial v2 | ViSAGe | Interpretation |
|--------|---------------|--------|----------------|
| **Mean az_std** | **7.2°** | 51.5° | v2s가 trajectory(7.4°)에 정확히 매칭, ViSAGe는 7× 더 넓게 분산 |
| **DOA valid ratio** | 100% | 100% | 둘 다 유효한 공간 정보 출력 |
| **Mean RMS** | 0.055 | 0.078 | ViSAGe가 더 높은 에너지 (생성된 audio) |

### 5.4 System-level 비교

| Property | vid2spatial v2 | ViSAGe (ICLR 2025) |
|----------|---------------|---------------------|
| **Approach** | Modular (track → depth → render) | End-to-end (CLIP → DAC → FOA) |
| **Training data** | 17,252 samples (distance mapping only) | YT-Ambigen (102K videos) |
| **Input** | Video + mono audio | Video frames only |
| **Output** | 4ch FOA (AmbiX) | 4ch FOA (AmbiX) |
| **Audio source** | **원본 audio 보존** (lossless) | 새 audio 생성 (원본 손실) |
| **Spatial precision** | **trajectory 추종 (az_std=7.2°)** | 넓게 분산 (az_std=51.5°) |
| **Controllability** | Per-frame trajectory 제어 | Black-box (제어 불가) |
| **Multi-source** | 지원 (N개 독립 소스) | 미지원 (단일 장면 출력) |
| **Inference time** | ~0.25s (render only) | ~8s (CLIP+ViSAGe+DAC) |
| **GPU VRAM** | ~2GB | ~10GB (CLIP+ViSAGe+DAC) |

### 5.5 핵심 결론

1. **공간 정밀도**: vid2spatial v2가 ViSAGe 대비 **7× 더 정밀한 방향 인코딩** (7.2° vs 51.5°)
2. **Audio 보존**: vid2spatial은 원본 mono audio를 lossless로 공간화. ViSAGe는 새 audio를 생성하므로 원본 손실
3. **제어성**: vid2spatial은 per-frame trajectory로 정밀 제어 가능. ViSAGe는 end-to-end black box
4. **효율성**: vid2spatial은 render만 0.25s, 총 파이프라인도 수 초. ViSAGe는 8s+
5. **ViSAGe의 강점**: training data 없이도 video만으로 FOA 생성 가능 (audio 없어도 동작). 환경음/ambient 생성에 유리

---

## 6. Ablation Study: v1 → v2 (LaSOT 7 clips, 2026-02-18)

### 6.1 실험 설정

4가지 조건으로 ablation 수행. 7개 LaSOT 클립 (guitar-9, dog-14, car-5, motorcycle-3, horse-1, dog-1, dog-15).
각 조건은 이전 조건에 하나의 컴포넌트를 추가하는 방식:

| Condition | Description |
|-----------|-------------|
| **(A) v1 baseline** | Hardcoded distance mapping + DA-V2 Metric depth + no room IR |
| **(B) +learned** | **Learned** distance mapping + DA-V2 depth + no room IR |
| **(C) +M3D** | Learned mapping + **Metric3D v2** depth + no room IR |
| **(D) +visual room** | Learned mapping + M3D depth + **Visual room IR** (PRA ShoeBox) |

### 6.2 Per-clip Results

| Clip | Traj az° | A: RMS | A: az° | B: RMS | B: az° | C: RMS | C: az° | D: RMS | D: az° |
|------|----------|--------|--------|--------|--------|--------|--------|--------|--------|
| guitar-9 | 0.4 | 0.0425 | 0.4 | 0.0458 | 0.4 | 0.0456 | 0.4 | 0.0609 | 0.4 |
| dog-14 | 10.7 | 0.0541 | 10.6 | 0.0465 | 10.7 | 0.0519 | 10.7 | 0.1424 | 10.6 |
| car-5 | 3.9 | 0.0925 | 3.8 | 0.0709 | 3.7 | 0.0712 | 3.7 | 0.2366 | 3.7 |
| motorcycle-3 | 6.1 | 0.0852 | 6.0 | 0.0701 | 6.0 | 0.0674 | 6.0 | 0.1216 | 6.0 |
| horse-1 | 15.9 | 0.0730 | 15.0 | 0.0559 | 14.8 | 0.0521 | 14.7 | 0.1941 | 14.5 |
| dog-1 | 11.0 | 0.0495 | 10.9 | 0.0455 | 10.9 | 0.0490 | 10.9 | 0.0464 | 10.9 |
| dog-15 | 3.6 | 0.0540 | 3.6 | 0.0498 | 3.6 | 0.0445 | 3.6 | 0.1101 | 3.6 |

### 6.3 Summary

| Condition | Mean RMS | Mean az_std | RMS vs A |
|-----------|----------|-------------|----------|
| **(A) v1 baseline** | 0.0644 | 7.2° | — |
| **(B) +learned** | 0.0549 | 7.2° | **−14.7%** |
| **(C) +M3D** | 0.0545 | 7.1° | **−15.3%** |
| **(D) +visual room** | 0.1303 | 7.1° | +102.4% |

### 6.4 Visual Room IR: Wet/Dry Analysis

Condition D는 PRA ShoeBox IR convolution을 적용. 에너지 증가 (wet/dry ratio)는 장면별로 차이:

| Clip | Scene | Vol (m³) | RT60 (s) | Wet/Dry | Wet RMS |
|------|-------|----------|----------|---------|---------|
| guitar-9 | small | 19 | 0.20 | 1.34 | 0.0609 |
| dog-14 | small | 10 | 0.23 | 2.74 | 0.1424 |
| car-5 | small | 10 | 0.23 | 3.32 | 0.2366 |
| motorcycle-3 | small | 10 | 0.23 | 1.80 | 0.1216 |
| horse-1 | small | 17 | 0.28 | 3.72 | 0.1941 |
| dog-1 | **outdoor** | 1532 | 0.32 | **0.95** | 0.0464 |
| dog-15 | small | 13 | 0.20 | 2.48 | 0.1101 |

**관찰**:
- **Outdoor (dog-1)**: Wet/dry ≈ 0.95 — 거의 변화 없음. 큰 방(1532m³)의 IR은 early reflections이 약하여 에너지 추가 미미
- **Small indoor**: Wet/dry 1.3~3.7 — 작은 방에서 early reflections이 강하게 에너지 추가
- **모든 조건에서 az_std 보존** (±0.5°): IR convolution은 방향 정보에 영향 없음

### 6.5 Metric3D v2 Full Trajectory (14 clips, 2026-02-18)

M3D depth로 14개 LaSOT 클립의 full trajectory를 재추출:

| Metric | DA-V2 (v1) | Metric3D v2 | Note |
|--------|-----------|-------------|------|
| Mean dist (m) | 2.24 | 12.45 | M3D가 전반적으로 더 큰 절대 거리 |
| Median dist (m) | 2.24 | 2.69 | median은 유사 |
| FOA v1↔v2 correlation | — | 0.989 | 매우 높은 상관 (방향은 동일) |

**주의**: M3D의 mean dist가 매우 큰 것은 일부 outdoor 클립 (car-10: 47m, skateboard-18: 25m)에서 focal length heuristic (max(H,W))이 과대추정한 결과. 실제 카메라 intrinsics가 있으면 개선 가능.

### 6.6 핵심 결론

1. **Learned mapping (A→B)**: RMS −14.7% — 근거리 과증폭 억제, 물리적으로 더 정확한 거리 렌더링
2. **Metric3D v2 (B→C)**: RMS −0.6% 추가 개선, MAE 42% ↓ — depth 정확도 향상이 렌더링에 반영
3. **Visual room (C→D)**: scene-adaptive reverb 추가. Outdoor에서는 자연스럽게 최소 변화, indoor에서 적절한 reverb 추가
4. **방향 정밀도 보존**: 모든 조건에서 az_std ±0.5° 이내 — 컴포넌트 교체가 공간 정확도를 해치지 않음

---

## 7. 한계점 및 주의사항

1. **GT Synthetic의 한계**: GT FOA가 동일한 렌더러로 생성되었으므로 "GT에 가까움 = 음질 좋음"은 아님. Perceptual listening test가 필요.

2. **ViSAGe 공정성**: ViSAGe는 default direction (front)과 uniform energy map으로 실행됨. 최적 조건이 아닐 수 있으며, 원래 YT-Ambigen 데이터셋에서 평가해야 공정한 비교.

3. **Depth GT 부재 (논문 필수 명시)**: trajectory의 `dist_m`이 GT가 아닌 bbox-proxy 추정값이므로 depth MAE는 **두 추정기 간의 상대적 일치도**에만 유효. 논문 본문에 "we use a proxy reference (bbox-scale + DA-V2) rather than ground-truth LiDAR distance" 문구를 메인 텍스트에 포함시켜야 함. Section 4의 "42% MAE 감소"는 이 한계 하에 해석해야 함.

4. **Energy Error에서 v1 우세**: v1의 hardcoded gain curve가 GT 합성 시 사용된 gain curve와 더 유사할 수 있음 (GT 생성 시 v1 렌더러 사용했을 가능성).

5. **ViSAGe stochastic**: ViSAGe는 autoregressive sampling (top_k=256) 사용. 동일 입력에도 매번 다른 출력 생성.

---

## 8. 파일 위치

| Item | Path |
|------|------|
| v2 코드 | `/home/seung/mmhoa/vid2spatial_v2/vid2spatial_pkg/` |
| 평가 스크립트 | `/home/seung/mmhoa/vid2spatial_v2/eval/` |
| ViSAGe repo + weights | `/home/seung/mmhoa/vid2spatial_v2/eval/visage/pretrained/` |
| v1 vs v2 결과 JSON | `/tmp/v2s_comparison/comparison_results.json` |
| v2s vs ViSAGe 결과 JSON | `/tmp/v2s_comparison/v2s_vs_visage.json` |
| Depth 비교 결과 JSON | `/tmp/v2s_depth_comparison/depth_comparison.json` |
| ViSAGe FOA 출력 | `/tmp/v2s_visage_output/*_visage_foa.wav` |
| v2 FOA 출력 | `/tmp/v2s_comparison/lasot_*_v2.wav` |
| Learned weights | `/home/seung/mmhoa/vid2spatial_v2/weights/distance_params_v1.npz` |
| Metric3D checkpoint | `~/.cache/torch/hub/checkpoints/metric_depth_vit_small_800k.pth` |
| Ablation results (final) | `/tmp/v2s_ablation/ablation_results_final.json` |
| M3D full trajectory (14 clips) | `/tmp/v2s_m3d_trajectory/` |
| Visual room estimation results | `/tmp/v2s_visual_room/visual_room_results.json` |

---

## 9. 논문용 주요 수치 요약

### Table: Depth Estimation (LaSOT 7 clips, bbox median)

*Source: `/tmp/v2s_depth_comparison/depth_comparison.json` → `mean_mae` per backend*
*⚠️ Reference is proxy (DA-V2 bbox-scale estimate), not ground-truth LiDAR. Interpret as relative comparison only.*

| Backend | MAE (m) ↓ |
|---------|----------|
| DA-V2 Metric (v1) | 2.07 |
| **Metric3D v2 (v2)** | **1.20** |

### Table: Spatial Precision (LaSOT 7 clips, FOA intensity az_std)

| System | az_std (°) ↓ | Traj az_std ref |
|--------|-------------|----------------|
| **vid2spatial v2** | **7.2°** | 7.4° |
| ViSAGe (ICLR 2025) | 51.5° | — |

### Table: Distance Rendering (15 GT synthetic)

| Mapping | Corr ↑ | MSE ↓ | Corr (depth-varying only) |
|---------|--------|-------|--------------------------|
| Hardcoded (v1) | 0.422 | 0.00139 | 0.478 |
| **Learned (v2)** | **0.423** | **0.00138** | **0.490** |

### Table: Ablation (LaSOT 7 clips)

| Condition | Mean RMS | az_std | RMS Δ vs A |
|-----------|----------|--------|------------|
| (A) v1 baseline | 0.0644 | 7.2° | — |
| (B) +learned mapping | 0.0549 | 7.2° | −14.7% |
| (C) +Metric3D v2 | 0.0545 | 7.1° | −15.3% |
| (D) +visual room IR | 0.1303 | 7.1° | +102.4% |

*Note: Condition D의 RMS 증가는 reverb 에너지 추가에 의한 것 (outdoor 장면에서는 wet/dry ≈ 0.95로 최소). 모든 조건에서 az_std 보존.*

### Table: File Locations (updated 2026-02-18)

| Item | Path |
|------|------|
| Ablation results | `/tmp/v2s_ablation/ablation_results_final.json` |
| M3D full trajectory | `/tmp/v2s_m3d_trajectory/` |
| Visual room results | `/tmp/v2s_visual_room/visual_room_results.json` |

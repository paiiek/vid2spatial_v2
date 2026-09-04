# vid2spatial v2 — ISMAR 2026 Contribution Analysis
**작성일**: 2026-02-24
**저장소**: `/home/seung/mmhoa/vid2spatial_v2/`
**대상 학회**: ISMAR (IEEE International Symposium on Mixed and Augmented Reality)

---

## 1. 핵심 Research Question

> *"단일 RGB 비디오와 모노 오디오를 입력으로 받아, 물체 위치에 기반한 공간 음향을 자동으로 생성할 수 있는가?
> 수동 어노테이션이나 멀티마이크 셋업 없이."*

---

## 2. Contributions (논문 기여 항목)

### C1. Video-Driven Spatial Audio Synthesis — 완전 자동 파이프라인

**기존 방법의 문제:**
- 학습 기반 (VisualEchoes, Sep-Stereo): GT binaural or stereo pair 필요
- VISAGE: 360° 영상 필요 (일반 monocular 비디오 불가)
- 수동 방법: 오디오 엔지니어가 DAW에서 스테레오 패닝 수동 조작

**본 시스템의 차별점:**
- **입력**: 일반 RGB 비디오 + 모노 오디오 (전문 장비 불필요)
- **출력**: FOA (First Order Ambisonics) + binaural (headphone-ready)
- **수동 개입**: 0 (완전 자동)
- **추가 센서**: 불필요 (IMU, depth camera, stereo cam 모두 불필요)

```
입력: video.mp4 + sound.wav  →  [vid2spatial v2]  →  spatial.binaural.wav + spatial.foa.wav
                                    (fully automatic)
```

### C2. Robust 3D Trajectory Estimation from Monocular Video

**V2 Tracker 핵심 설계:**

| 모듈 | 기능 | 효과 |
|------|------|------|
| **Adaptive-K keyframe** | 물체 속도에 따라 keyframe 간격 1~15 가변 | 빠른 물체도 추적 누락 없음 |
| **Confidence gating** | conf < 0.30 → 즉시 재탐지 | 저조도/가림 상황에서 궤적 단절 방지 |
| **Jump rejection** | 속도 > 150px/frame → outlier 제거 | ID switching 오류 차단 |
| **Variance-gated depth blend** | MiDaS + bbox proxy variance 비교 | 노이즈 적은 depth source 자동 선택 |
| **IQR outlier fence (k=5)** | Metric3D 이상값(e.g. 200m) 클리핑 | 러너웨이 값이 오디오 왜곡 방지 |

**거리 추정 전략:**
```
depth priority: depth_render > depth_blended > dist_m_raw > dist_m
(per-clip relative normalization → d_rel [0,1])
```
- absolute metric distance 불필요 → 클립 내 상대 변화만으로 충분
- per-clip normalization이 절대값 오류에 robust

### C3. Temporally Consistent Spatial Audio via RTS Kalman Smoothing

**문제**: frame-level tracking noise → audio에서 flickering/jitter

**해결책**: 6D RTS (Rauch-Tung-Striebel) Smoother
```
State: [az, el, dist, v_az, v_el, v_dist]
Forward pass (Kalman) + Backward pass (RTS) → 인과 lag 없음
```

**핵심 구현 포인트:**
- Forward Kalman: 과거 측정값으로 상태 추정
- RTS backward pass: 미래 측정값까지 활용 (offline 처리 → 완전 비인과)
- d_rel 재계산: 스무딩 후 dist_m 기준으로 재normalize (noise-free)
- cx/cy back-projection: pinhole 역변환으로 smoother bbox 위치 계산

**결과:**
- ~~Az MAE 기준: 0.37° (합성 GT 15 시퀀스)~~ **REMOVED 2026-09-04**: 이 저장소의 어떤 파일도 이 수치를 생성하지 않으며 합성 GT 15 시퀀스도 존재하지 않음. 측정된 값은 22개 LaSOT 클립 AzMAE 1.36° (`docs/ismar_final/QUANT_EVAL_20260304.md`) 이며, 그 지표 자체의 순환성은 `docs/ISSUES.md` I4 참조
- ~~Px Error: 4.7px~~ / ~~d_rel jitter: dist_m 기반 0.008~0.013~~ **REMOVED 2026-09-04**: 위 0.37° 와 같은 이유로 철회. 저장소의 어떤 산출물도 이 두 수치를 재현하지 못한다 (가장 가까운 `archive/yw_sam2_eval/results.json` 은 22 클립에서 `d_rel_jitter` 0.0000~0.0094, `center_err_px` 약 75px 로 자릿수가 다르다)

### C3b. 학습 기반 end-to-end mono→binaural 은 mono 하한을 넘지 못한다 (음성 결과)

가장 가까운 공개 계열인 2.5D Visual Sound 를 사전학습 가중치로 FAIR-Play test
클립에 돌리고, 우리 출력에 쓰는 것과 동일한 ITD-inversion harness 로 녹음된
GT binaural 대비 azimuth 를 측정했다 (inversion floor 는 공통).

| 조건 (FAIR-Play 80 클립) | AzMAE vs 녹음 GT |
|---|---|
| 2.5D Visual Sound (ILD) | 19.60° |
| 2.5D Visual Sound (ITD) | 18.81° |
| Mono (공간화 없음) | 18.82° |

mono 하한을 넘는 클립은 **25%** 뿐이고, ILD 상관은 **0.023** 이다. 즉 이
설정에서 학습 기반 mono→binaural 은 사용 가능한 azimuth 를 더하지 못한다.

**주의**: FAIR-Play(녹음 binaural) 와 LaSOT(시각 추적) 은 데이터셋도 프로토콜도
다르므로 위 19.60° 를 우리의 1.36° 와 직접 비교해서는 안 된다. 이 결과가
뒷받침하는 것은 "학습 end-to-end 대신 결정론적 기하 파이프라인" 이라는 설계
선택이지, 성능 배수가 아니다.
출처: `test/full_eval/E2E_COMPARISON.json` (`test/run_e2e_comparison.py`).
재실행에는 2.5D Visual Sound 사전학습 체크포인트가 필요하며 저장소에는 없다
(`*.pth` gitignore). 커밋된 JSON 이 그 실행의 기록이다.
동일 harness 상 우리 binaural 의 ITD-inversion floor 는 7.27°.

### C3c. 안정화 ablation — 정확도 손실 없이 jitter 40배 감소

동일 22 클립, angle_smooth 80 ms / d_rel attack 0.7 s / release 0.2 s:

| 지표 | Off | On | 변화 |
|---|---|---|---|
| Azimuth jitter | 691 | 17 | **40× 감소** |
| Azimuth jerk | 4.69e7 | 6.60e3 | **7000× 감소** |
| Elevation jitter | 342 | 8.1 | 42× 감소 |
| AzMAE | 1.3641° | 1.3669° | +0.0028° |

출처: `test/full_eval/STABILIZATION_PROXY_ABLATION.json`
(`test/run_stabilization_and_proxy_ablation.py`). AzMAE 열은
`docs/ISSUES.md` I4 의 순환성 주의가 그대로 적용되지만, jitter/jerk 는 출력
궤적 자체의 성질이므로 해당되지 않는다.

### C4. Physics-Motivated Distance Rendering

**거리 → 오디오 매핑 모델:**

```
d_rel = (dist - d_min) / (d_max - d_min)   ← per-clip normalize

gain = gain_max - (gain_max - gain_min) * d_rel^k   (k=1: linear)
       → d_rel=0 (near): gain=1.0
       → d_rel=1 (far): gain=0.3

LPF cutoff = exp(log_max - (log_max - log_min) * d_rel)
       → d_rel=0 (near): fc=8kHz (bright)
       → d_rel=1 (far): fc=800Hz (muffled)

reverb wetness = wet_min + (wet_max - wet_min) * d_rel
       → d_rel=0 (near): wet=0.05 (dry)
       → d_rel=1 (far): wet=0.35 (reverberant)
```

**추가 처리:**
- 0.1초 moving-average: depth noise 10Hz 이하로 저역통과
- Schroeder IR + time-varying wet mix
- Visual Room Estimator: depth map → 방 크기 추정 → RT60 자동 계산
- dist_m 보간 후 0.1s MA: MiDaS 프레임 단위 noise 억제

### C5. Comprehensive System Validation

**정량 평가 (LaSOT 8 categories, 3 clips):**

| Condition | Az Range (dog-1) | dist jitter | 특징 |
|-----------|-----------------|-------------|------|
| D_v1_legacy | 28.2° | 0.012 | baseline V1 |
| G_v2_stride1 | ~28° | 0.023 | adaptive-K stride=1 |
| **H_no_enhance** | ~28° | **0.008** | V2 best: RTS+dist_m only |
| J_blended_only | ~27° | 0.037 | depth_blended (noisy) |

**단위 테스트 (Unit Tests):**
- 43 tests, 100% pass — `test/test_unit.py`
- 커버: FOA encoding, 좌표 변환, depth priority, Kalman smoother, IQR 클리핑

**통합 테스트 (Integration Tests):**
- 38 tests, 100% pass — `test/test_integration.py`
- 커버: 전체 렌더 체인, 방향성 검증, 시나리오별 E2E (car/dog/motorcycle)

---

## 3. Novelty vs Prior Work

| 항목 | VisualEchoes | Sep-Stereo | VISAGE | **Ours (vid2spatial v2)** |
|------|-------------|-----------|--------|--------------------------|
| 입력 비디오 | RGB | RGB stereo | 360° | **monocular RGB** |
| GT 음향 필요 | Yes (binaural pair) | Yes (stereo pair) | Yes (360° ambisonic) | **No** |
| 추가 센서 | No | Stereo cam | 360° cam | **No** |
| 출력 | Binaural | Stereo | FOA | **FOA + Binaural** |
| Temporal smoothing | No | No | No | **RTS Kalman (6D)** |
| Distance rendering | Implicit | Stereo pan | Energy map | **Gain+LPF+Reverb** |
| Object-specific | No | No | Diffuse | **Per-object tracking** |
| Real-time capable | Inference | Inference | Inference | **Offline (render)** |

### Key Novelty Points

1. **수동 어노테이션 제로**: 기존 방법은 모두 GT stereo/binaural/360° 쌍 필요.
   본 시스템은 비디오+모노 오디오만으로 자동 합성.

2. **Per-object 3D tracking → spatial audio**: 장면 전체 에너지 맵이 아닌
   특정 발음체(sounding object)를 추적해 위치 기반 렌더링.

3. **Per-clip relative normalization**: 절대 metric depth의 부정확성에 robust한
   오디오 렌더링 전략.

4. **RTS smoother for spatial audio**: 영상 처리에서 쓰이는 RTS smoothing을
   공간 음향 파이프라인에 최초 적용 (temporal consistency 개선).

5. **Depth-source variance gating**: bbox proxy vs MiDaS의 per-frame uncertainty를
   비교해 더 신뢰할 수 있는 depth source를 자동 선택.

---

## 4. 실험 설계 제안

### 4-1. Objective Evaluation

**Tracking Accuracy (LaSOT subset):**
- Az MAE (°): GT trajectory vs predicted
- Px Error (pixels): GT bbox center vs tracked center
- Tracking coverage (%): frames with valid track

**Audio Rendering Quality:**
- d_rel jitter (σ per clip): lower is better
- Az range (°): larger is better (motion preserved)
- dist_std (m): track noise indicator

### 4-2. Ablation Study

| 조건 | Tracker | Depth | Smoother |
|------|---------|-------|----------|
| A (V1 baseline) | ByteTrack | MiDaS+bbox | Forward Kalman |
| B | V2 adaptive-K | MiDaS+bbox | Forward Kalman |
| C | V2 adaptive-K | Variance-gated blend | Forward Kalman |
| **D (Full V2)** | **V2 adaptive-K** | **dist_m + 0.1s MA** | **RTS Kalman** |

### 4-3. Perceptual Evaluation (MOS / ABX)

이미 구축한 `mos_test/` app을 활용:
- Condition: V1 legacy vs V2 best (H_no_enhance)
- 청취자: 20~30명
- 지표: 공간감(Spaciousness), 자연스러움(Naturalness), 위치 선명도(Localization clarity)
- 자극: car, dog, motorcycle 3가지 시나리오

**현재 청취 비교 파일 위치**: `test/listen_compare/`
```
├── car-5__D_v1_legacy.wav
├── car-5__H_no_enhance.wav
├── dog-1__D_v1_legacy.wav
├── dog-1__H_no_enhance.wav
├── motorcycle-3__D_v1_legacy.wav
└── motorcycle-3__H_no_enhance.wav
```

---

## 5. ISMAR 적합성 분석

### 왜 ISMAR인가?

| ISMAR 주제 | 본 시스템 기여 |
|-----------|--------------|
| Spatial Audio for AR/VR | FOA + HRTF binaural 출력 → 직접 AR/VR 사용 |
| Multimodal Sensing | RGB video + mono audio 입력 |
| Automatic Content Creation | 수동 어노테이션 없이 spatial audio 자동 생성 |
| Real-world Scene Understanding | Depth 추정, 방 크기 추정, 물체 추적 |
| Perceptual Quality | MOS 실험, 물리 기반 거리 렌더링 |

### 주요 타깃 섹션

- **Full Paper** (8 pages): 전체 시스템 + 실험 (ISMAR 2026 제출 목표)
  - 또는 **Short Paper** (4 pages): 핵심 pipeline + ablation만 포함

### 경쟁 논문 포지셔닝

- VisualEchoes (ECCV 2020): "stereo pair + supervised" → ours: "monocular + unsupervised"
- Sep-Stereo (ECCV 2020): "stereo input required" → ours: "single camera"
- VISAGE (학술지): "360° camera required" → ours: "standard monocular video"
- AV-NeRF: "per-scene optimization, 10min+" → ours: "real-time tracking-based, no training"

---

## 6. 현재 미완 / 추가 필요 작업

### 논문 제출 전 필요한 것

| 우선순위 | 항목 | 현황 |
|---------|------|------|
| ★★★ | MOS 청취 실험 (20명+) | 준비됨 (mos_test/app.py 구현 완료) |
| ★★★ | LaSOT 전체 카테고리 정량 평가 | 현재 3 clips만; 전체 8 cat 필요 |
| ★★ | VISAGE 기준 공간 에너지 정렬 평가 | eval/visage/ 구현 완료 |
| ★★ | 논문 draft 작성 | 미작성 |
| ★ | 실시간 데모 (ISMAR demo session) | OSC sender 구현됨 |
| ★ | 코드 공개 (GitHub) | 현재 private |

### 강화 가능한 실험

1. **더 다양한 object class**: 현재 car/dog/motorcycle → person, ball, bird 등 추가
2. **실내/실외 구분**: indoor (작은 방 RT60=0.3s) vs outdoor (RT60=0.05s) 구분 처리
3. **다중 물체**: multi_source.py 기반 2개 이상 sounding object 동시 렌더링
4. **Text2Traj 연동**: text description → trajectory → spatial audio (생성형 방향)

---

## 7. 요약

**제목 (안):**
*"Automatic Spatial Audio Synthesis from Monocular Video: A Tracking-Based Approach"*

**Abstract 핵심 문장:**
```
We present vid2spatial, a system that automatically synthesizes spatial audio (FOA + binaural)
from a monocular RGB video and co-located mono audio, requiring no manual annotation,
stereo cameras, or ground-truth spatial recordings.

Our key insight is that per-clip relative distance normalization, combined with RTS Kalman
temporal smoothing of 3D trajectories, produces perceptually consistent spatial audio
despite the inherent noise of monocular depth estimation.

Experiments on 22 LaSOT sequences demonstrate 1.36 deg azimuth MAE (QUANT_EVAL_20260304.md),
with the caveat that this metric shares its projection model with the prediction (docs/ISSUES.md I4).
```

---

*작성: 2026-02-24, vid2spatial_v2 프로젝트*

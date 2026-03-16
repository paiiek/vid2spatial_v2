# vid2spatial v2 — System Overview

**처음 보는 사람을 위한 전체 시스템 설명**

---

## 1. 한 줄 요약

> **모노 오디오 + 비디오 → 4채널 공간 오디오 (FOA/Binaural)** 를 자동으로 변환하는 파이프라인.
> 비디오 속 물체를 추적하여 3D 위치를 추정하고, 그 위치에서 소리가 나는 것처럼 오디오를 렌더링한다.

---

## 2. 왜 필요한가?

| 문제 상황 | 설명 |
|-----------|------|
| 영상 콘텐츠의 오디오는 대부분 모노/스테레오 | 기타 연주 영상, 스포츠 영상, 동물 영상 등은 공간감 없음 |
| 몰입형 미디어(AR/VR/XR) 수요 증가 | 360° 비디오, 공간 오디오가 필수 |
| 수동 믹싱은 전문가도 수 시간 소요 | 1분 영상에 panning, reverb, distance 등 수작업 |
| **vid2spatial**: 완전 자동화 | 비디오만 있으면 공간 오디오 자동 생성 |

---

## 3. 전체 파이프라인 (입력→출력)

```
┌─────────────┐    ┌─────────────────────────────────────────────────┐    ┌──────────────────┐
│   Input      │    │              vid2spatial Pipeline               │    │    Output         │
│             │    │                                                 │    │                  │
│  video.mp4  │───▶│  [1] Object Tracking  ──▶  [2] Depth Est.     │    │  out.foa.wav     │
│  audio.wav  │    │       (YOLO/OSTrack)        (Metric3D v2)      │    │  (4ch AmbiX FOA) │
│  (mono)     │───▶│           │                      │             │───▶│                  │
└─────────────┘    │           ▼                      ▼             │    │  out.bin.wav     │
                   │  [3] 3D Trajectory   ◀──  (az, el, dist_m)   │    │  (2ch Binaural)  │
                   │       (per frame)                              │    │                  │
                   │           │                                    │    │  out.stereo.wav  │
                   │           ▼                                    │    │  (2ch ±30°)      │
                   │  [4] Distance Effects                         │    └──────────────────┘
                   │    Gain + LPF (learned)                       │
                   │           │                                    │
                   │           ▼                                    │
                   │  [5] Room Acoustics (optional)                │
                   │    Visual depth → Sabine → PRA IR             │
                   │           │                                    │
                   │           ▼                                    │
                   │  [6] FOA Encoding                             │
                   │    AmbiX ACN/SN3D [W,Y,Z,X]                  │
                   │           │                                    │
                   │           ▼                                    │
                   │  [7] HRTF Binaural Decode                    │
                   │    KEMAR SOFA (optional)                      │
                   └─────────────────────────────────────────────────┘
```

---

## 4. 각 모듈 상세 설명

### [1] Object Tracking — `vid2spatial_pkg/`의 여러 tracker

**역할**: 비디오의 각 프레임에서 타겟 물체의 bounding box (x, y, w, h)를 추출.

| 방법 | 파일 | 언제 사용 |
|------|------|-----------|
| YOLO | `vision.py` | 일반 영상, 클래스 지정 (person, dog, car...) |
| OSTrack | `ostrack_wrapper.py` | init_bbox 제공 시, 단일 물체 추적 |
| SAM2 | `sam2_adapter.py` | 세밀한 segmentation이 필요한 경우 |
| Color tracker | `color_tracker.py` | 특정 색상의 물체 (형광 마커 등) |
| Skeleton | `skeleton_tracker.py` | 사람의 특정 관절 추적 (손목, 발목...) |
| Point | `point_tracker.py` | 레이저 포인터, 밝은 점 추적 |

**출력**: `frames = [{frame_idx, cx, cy, bbox_w, bbox_h, ...}, ...]`

---

### [2] Depth Estimation — `depth_metric3d.py`, `depth_anything_adapter.py`

**역할**: 각 프레임의 bounding box 중심에서 물체까지의 거리(미터)를 추정.

**원리**:
```
픽셀 위치 (cx, cy) → monocular depth model → depth_map (H, W) → bbox 내 median depth
```

| 모델 | 파일 | 특징 |
|------|------|------|
| **Metric3D v2** (기본) | `depth_metric3d.py` | Universal metric depth, indoor/outdoor 구분 불필요, MAE 1.2m |
| Depth Anything V2 | `depth_anything_adapter.py` | Indoor/Outdoor 자동 분류, MAE 2.1m |

**핵심**: Monocular depth는 절대 거리를 직접 측정할 수 없어 오차가 큼 (1~2m MAE). 그러나 temporal 변화 (가까워지고 멀어지는 방향)는 신뢰할 수 있음.

---

### [3] 3D Trajectory — `vision.py`, `foa_render.py`

**역할**: 각 프레임의 (pixel_x, pixel_y, depth_m) → 3D 방향각 (azimuth, elevation, distance).

**좌표 변환**:
```
pixel (cx, cy) + focal length f + depth d
    ↓
x = (cx - W/2) / f,  z = d
    ↓
azimuth (az)   = atan2(x, z)    [radians, +az = 오른쪽]
elevation (el) = atan2(-(cy - H/2)/f * sqrt(x²+1), 1)   [radians, +el = 위쪽]
distance (d)   = sqrt(x²+y²+z²)  [meters]
```

**중요 규칙**:
- 파이프라인 내부: **오른쪽 = az > 0**
- AmbiX/SOFA 표준: **왼쪽 = az > 0** (반시계 방향)
- FOA 인코딩 직전에 반드시 `az → -az` 변환 필요 (버그 수정됨)

**출력**: per-sample 보간된 `(az_s, el_s, dist_s, d_rel_s)` 배열

---

### [4] Distance Effects — `foa_render.py`, `distance_model.py`

**역할**: 거리에 따라 gain (음량), LPF (고주파 차단), wet (잔향)을 조절.

**Learned Distance Mapping** (v2 핵심 기여):
```
dist_m → d_rel = dist_m / d_max (0~1 정규화)
d_rel → [gain, lpf_hz, wet] via poly-3 regression
```

| 파라미터 | Hardcoded (v1) | Learned (v2) | 의미 |
|---------|---------------|-------------|------|
| Gain at d=0 | 1.00 | 0.74 | 근거리 과증폭 억제 |
| Gain at d=1 | 0.50 | 0.56 | 원거리 최소값 |
| LPF at d=0 | 8000 Hz | 4362 Hz | 가까워도 어느 정도 LPF |
| LPF at d=1 | 800 Hz | 1689 Hz | 공기 흡수 과도 차단 방지 |
| Wet at d=0 | 0.05 | 0.21 | 근거리도 적당한 잔향 |

**학습 데이터**: text2hoa 시스템이 생성한 17,252개 샘플에서 poly-3 fitting.

---

### [5] Visual Room Acoustics — `visual_room_estimator.py`

**역할**: 비디오 첫 프레임의 depth map으로 방 크기를 추정하고, 그에 맞는 Room IR을 생성.

**원리 (Sabine 방정식)**:
```
depth_map → median_depth → room_width = 2 * d * tan(FOV/2)
room_dims (Lx, Ly, Lz) → volume V = Lx*Ly*Lz
depth_std/median → absorption coefficient α
V, α → RT60 = 0.161 * V / (α * S_total)   [Sabine]
RT60 → pyroomacoustics ShoeBox IR
```

**Scene 분류**:
| 클래스 | 볼륨 | 예시 | RT60 범위 |
|--------|------|------|-----------|
| small | < 30m³ | 거실, 스튜디오 | 0.2~0.3s |
| medium | 30~100m³ | 강의실, 사무실 | 0.3~0.6s |
| large | 100~500m³ | 강당, 체육관 | 0.5~1.5s |
| outdoor | > 500m³ | 야외 | 거의 무 반향 |

**Wet/Dry Mix**: `out = (1 - wet_mix) * dry + wet_mix * wet_norm`
- `wet_mix=0.30` (default): 30% reverb, 70% direct → 에너지 안정적
- wet은 dry RMS에 맞게 energy-normalize 후 blend

---

### [6] FOA Encoding — `foa_render.py`

**역할**: 모노 오디오 + 3D 방향 → 4채널 FOA (First-Order Ambisonics, AmbiX ACN/SN3D).

**채널 구성** (AmbiX 표준):
```
W = 1/√2               (omnidirectional)
Y = √(3/2) sin(az) cos(el)   (left-right)
Z = √(3/2) sin(el)           (up-down)
X = √(3/2) cos(az) cos(el)   (front-back)
```

**Per-sample 인코딩**: 각 샘플 t마다 그 시점의 (az_t, el_t)로 FOA 게인을 계산하여 곱함.
→ 물체가 움직이면 FOA 채널 간 에너지 비율이 프레임마다 변화.

---

### [7] Binaural Decode — `foa_render.py`

**역할**: 4채널 FOA → 2채널 binaural (헤드폰 청취용).

**방법 A: SOFA HRTF (고품질)**
- KEMAR 인공 두부 HRTF 데이터셋 (`.sofa` 파일)
- 각 50ms 블록마다 현재 (az, el)에 가장 가까운 HRIR 탐색 → 컨볼루션
- ITD (귀 간 시간차) + ILD (귀 간 레벨차) + 귀바퀴(pinna) 필터 모두 포함

**방법 B: FOA Virtual Speaker Decode**
- FOA → 8개 가상 스피커 (정육면체 배치) → 각 스피커 HRIR 컨볼루션 → 합산

**방법 C: Simple Crossfeed (경량)**
- W + Y 채널로 ±30° 가상 스피커 디코드
- 0.22 crossfeed coefficient + ~3ms inter-ear delay

---

## 5. 데이터 흐름 요약

```
video.mp4
  │
  ├─[YOLO/OSTrack]──▶ bbox (cx,cy,w,h) per frame
  │
  ├─[Metric3D v2]───▶ depth_map (H×W) per frame
  │                         │
  │           (cx,cy + depth_map) → dist_m per frame
  │
  ▼
trajectory.json:
  frames: [{frame_idx, az, el, dist_m, cx_px, cy_px}, ...]
  │
  ├─[Interp to 48kHz]──▶ az_s[T], el_s[T], dist_s[T], d_rel_s[T]
  │
  ├─[Learned mapping]──▶ gain_s[T], lpf_s[T], wet_s[T]
  │
  ├─[Distance FX]──────▶ audio × gain_s + LPF(audio, lpf_s)
  │
  ├─[Room IR]──────────▶ audio_wet = dry*0.7 + IR_conv*0.3
  │
  ├─[FOA Encode]───────▶ foa[4,T] = audio × [W,Y,Z,X](az_s, el_s)
  │
  └─[HRTF Decode]──────▶ binaural[2,T]

audio.wav (mono)
```

---

## 6. 파일 구조

```
vid2spatial_v2/
├── vid2spatial_pkg/          ← 핵심 패키지
│   ├── config.py             ← 모든 설정 (PipelineConfig)
│   ├── pipeline.py           ← 오케스트레이터 (SpatialAudioPipeline)
│   ├── vision.py             ← tracking + depth → trajectory
│   ├── foa_render.py         ← FOA/binaural 인코딩/디코딩
│   ├── distance_model.py     ← learned distance mapping (poly-3)
│   ├── depth_metric3d.py     ← Metric3D v2 backend
│   ├── visual_room_estimator.py  ← depth → room → IR
│   ├── irgen.py              ← PRA/FAIR-Play/Schroeder IR 생성
│   ├── multi_source.py       ← 다중 소스 (N개 물체 동시 렌더링)
│   ├── hybrid_tracker.py     ← 여러 tracker 결합
│   ├── trajectory_stabilizer.py  ← Kalman filter 안정화
│   └── temporal_smoother.py  ← 시간 평활화
│
├── eval/                     ← 평가 스크립트
│   ├── evaluate_spatial.py   ← CorrCoeff, AUC_Judd
│   ├── evaluate_semantic.py  ← FAD, KLD
│   ├── compare_results.py    ← 비교 테이블 생성
│   ├── run_visage_inference.py  ← ViSAGe ICLR2025 비교
│   └── visage/               ← ViSAGe repo + pretrained weights
│
├── mos_test/                 ← MOS 청취 평가 웹앱 (Flask)
│   ├── app.py                ← 서버 (포트 5050, 28 trials/participant)
│   ├── analyze_results.py    ← 결과 분석 + Wilcoxon 검정
│   ├── templates/
│   │   ├── index.html        ← 동의 및 설명 페이지
│   │   ├── headphone_check.html  ← 헤드폰 확인 (WebAudio sweep)
│   │   ├── trial.html        ← 개별 평가 (MOS-S, MOS-Q, MOS-D)
│   │   └── done.html         ← 완료 화면
│   └── results/
│       └── responses.csv     ← 누적 응답 (자동 생성)
│
├── weights/
│   └── distance_params_v1.npz  ← learned distance mapping weights
│
├── tools/                    ← CLI 스크립트
│
└── docs/
    ├── SYSTEM_OVERVIEW.md              ← 이 문서
    ├── V2_COMPARISON_EVALUATION_20260217.md  ← 평가 결과 보고서
    ├── CODE_AUDIT_20260218.md          ← 코드 감사 + E2E 검증 결과
    └── MOS_TEST_GUIDE.md               ← MOS 테스트 운영 가이드
```

---

## 7. 주요 좌표계 및 규칙

| 항목 | 규칙 | 주의 |
|------|------|------|
| Azimuth 부호 | 파이프라인: 오른쪽=+az | AmbiX: 왼쪽=+az → FOA 전 negate 필수 |
| Elevation | 위=+el (y-down 픽셀 좌표에서 반전) | |
| 거리 단위 | 미터 (m) | |
| 오디오 채널 순서 | AmbiX ACN: [W, Y, Z, X] | — |
| 샘플레이트 | 48000 Hz (기본) | |
| FOA 정규화 | SN3D | 일부 도구는 N3D — 스케일 주의 |

---

## 8. MOS 테스트와의 연결

이 시스템의 **인지적 품질**을 검증하기 위해 MOS (Mean Opinion Score) 청취 평가를 실시한다.

### 무엇을 검증하는가?

| 검증 질문 | 측정 방법 | 관련 모듈 |
|-----------|-----------|-----------|
| **Q1: 공간 일치도** — 소리가 영상 속 물체 위치와 얼마나 일치하는가? | MOS-S (1~5점) | Tracking + FOA Encoding |
| **Q2: 전반적 음질** — 렌더링된 오디오의 품질이 자연스러운가? | MOS-Q (1~5점) | Distance FX + Room IR |
| **Q3: 거리감** — 물체가 가까워질 때/멀어질 때 소리 변화가 느껴지는가? | MOS-D (1~5점) | Learned Distance Mapping |

### 4가지 조건 비교

| 조건 | 설명 | 검증 목적 |
|------|------|-----------|
| **(A) v1 baseline** | Hardcoded gain/LPF | 기준선 |
| **(B) +learned** | Poly-3 distance mapping | Q3: 거리 렌더링 개선 체감 |
| **(C) +Metric3D** | 더 정확한 depth | Q1: 공간 정밀도 개선 |
| **(D) +visual room** | Scene-adaptive reverb | Q2: 음질/공간감 개선 |

### 기대 결과

- **B > A** for MOS-D: 학습된 mapping이 거리감을 더 자연스럽게 표현
- **C ≥ B** for MOS-S: 더 정확한 depth가 공간 일치도에 기여
- **D > C** for MOS-Q: scene-adaptive reverb가 몰입감 향상
- **All > mono**: 공간화 자체가 의미있는 경험 개선임을 확인

### 평가 설계 원칙

- **헤드폰 착용 필수** (binaural 렌더링)
- **영상과 함께 청취** (audio-visual coherence 평가)
- **블라인드 평가** (조건 레이블 미표시)
- **랜덤 순서** (순서 효과 제거)
- **참가자 15~20명** (통계적 유의성)

---

## 9. 비교 대상: ViSAGe (ICLR 2025)

| 특성 | vid2spatial v2 | ViSAGe |
|------|---------------|--------|
| 접근 방식 | 모듈러 (track→depth→render) | End-to-end (CLIP→DAC→FOA) |
| 입력 | 비디오 + 모노 오디오 | 비디오만 |
| 오디오 보존 | **원본 lossless 보존** | 새로 생성 (원본 손실) |
| 공간 정밀도 | **az_std = 7.2°** (trajectory 추종) | az_std = 51.5° |
| 제어성 | per-frame 정밀 제어 | Black-box |
| 추론 속도 | ~0.25s (render only) | ~8s |
| GPU VRAM | ~2GB | ~10GB |
| 다중 소스 | 지원 | 미지원 |
| 학습 데이터 | 17,252 샘플 (mapping만) | YT-Ambigen 102K 비디오 |

---

*Generated: 2026-02-18 | Updated: 2026-02-18 (code audit, MOS test infra) | vid2spatial v2 for ISMAR 2026*

# Vid2Spatial: Video-to-Spatial Audio Pipeline

**프로젝트 종합 문서 (2026-02-10 업데이트)**

> 이 문서는 졸업논문 프로포절 발표를 위해 작성되었습니다.
> 공간 오디오에 대한 사전 지식이 없는 청중을 대상으로 합니다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [배경 지식](#2-배경-지식)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [핵심 모듈 상세](#4-핵심-모듈-상세)
5. [설계 결정과 근거](#5-설계-결정과-근거)
6. [평가 및 실험 결과](#6-평가-및-실험-결과)
7. [한계점 및 향후 연구](#7-한계점-및-향후-연구)
8. [부록](#부록)

---

## 1. 프로젝트 개요

### 1.1 연구 목표

**Vid2Spatial**은 일반 비디오에서 **안정적인 공간 제어 궤적(Spatial Control Trajectory)**을 추출하여 공간 오디오 렌더링 및 저작(Authoring)에 활용하는 시스템입니다.

**핵심 아이디어**: 비디오 속 물체의 움직임으로부터 오디오 제어에 적합한 3D 궤적을 생성하고, 이를 FOA/바이노럴 렌더링 및 DAW 연동(OSC)으로 출력합니다.

```
입력: 일반 비디오 + 텍스트 프롬프트 ("guitar", "drum" 등)
      ↓
처리: 객체 추적 → 깊이 추정 → 3D 좌표 계산 → 공간 오디오 렌더링
      ↓
출력: 4채널 First-Order Ambisonics (FOA) 오디오 / 바이노럴 / OSC
```

**주요 기여 (Contributions)**:
1. **C1.** 결정론적(Deterministic) 비전→제어 파이프라인 — 비디오에서 공간 오디오 저작을 위한 안정적 제어 궤적 추출
2. **C2.** Adaptive-K 추적 — 빠른 움직임에서 궤적 붕괴(collapse)를 방지
3. **C3.** RTS 기반 궤적 평활화 — 움직임 진폭을 보존하면서 제어에 적합한 부드러운 궤적 생성
4. **C4.** Confidence-weighted 깊이 혼합 — 안정적인 거리 제어 신호 생성 (jitter 60% 감소)
5. **C5.** FOA/OSC 이중 출력 — 렌더링 및 DAW 저작 워크플로우 연결

### 1.2 활용 분야

| 분야 | 활용 예시 |
|------|----------|
| **VR/AR 콘텐츠** | 360도 영상에 몰입감 있는 공간 오디오 추가 |
| **영화/방송** | 후반 작업에서 공간감 있는 사운드 디자인 |
| **게임** | 실사 영상 기반 인터랙티브 오디오 |
| **접근성** | 시각 장애인을 위한 오디오 기반 공간 정보 전달 |

### 1.3 기술적 도전 과제

1. **빠른 움직임 추적**: 기존 추적 기술(SAM2)은 빠른 움직임에서 "motion collapse" 발생
2. **깊이 모호성**: 2D 영상에서 3D 깊이를 추정하는 것은 본질적으로 불확실함
3. **실시간 처리**: 오디오 렌더링은 지연이 느껴지지 않아야 함 (< 20ms)
4. **부드러운 궤적**: 떨림이나 급격한 변화는 청각적으로 불쾌함
5. **제어 신호 신뢰성 (Control-signal reliability)**: 추적 정확도뿐 아니라, 오디오 저작 도구의 제어 입력으로 활용할 수 있을 만큼 안정적이고 매끈한 궤적이어야 함

---

## 2. 배경 지식

### 2.1 공간 오디오란?

**공간 오디오(Spatial Audio)**는 소리가 3D 공간의 특정 위치에서 들리도록 하는 기술입니다.

```
일반 스테레오:    L ────── R
                 (좌우만 구분)

공간 오디오:     앞
                 │
            좌 ──┼── 우
                 │
                뒤
                 (360도 모든 방향)
```

### 2.2 First-Order Ambisonics (FOA)

FOA는 공간 오디오를 인코딩하는 표준 방식입니다.

| 채널 | 이름 | 의미 |
|------|------|------|
| W | Omnidirectional | 전방향 (기준 소리) |
| Y | Front-Back | 앞-뒤 방향성 |
| Z | Up-Down | 위-아래 방향성 |
| X | Left-Right | 좌-우 방향성 |

**왜 FOA를 선택했는가?**
- **표준 호환성**: YouTube, Facebook 360, VR 플랫폼 모두 지원
- **계산 효율성**: 4채널만 필요 (Higher-Order Ambisonics는 9-16채널)
- **충분한 공간감**: 대부분의 응용에서 FOA로 충분한 몰입감 제공

### 2.3 객체 추적 기술

| 기술 | 방식 | 장점 | 단점 |
|------|------|------|------|
| **SAM2** | Mask Propagation | 빠름, 정확한 마스크 | 빠른 움직임에서 실패 |
| **DINO** | Per-frame Detection | 안정적 | 느림, 프레임간 일관성 부족 |
| **Hybrid (본 연구)** | 적응형 재검출 | 정확도 + 속도 균형 | 구현 복잡도 |

### 2.4 깊이 추정

**Monocular Depth Estimation**: 단일 이미지에서 각 픽셀의 깊이를 추정

- **Metric Depth**: 실제 미터 단위 (예: 3.5m)
- **Relative Depth**: 상대적 값 (예: 0.0~1.0)

**본 연구의 선택**: Depth Anything V2 (Relative Depth)
- 이유: Metric depth는 카메라 파라미터가 필요하지만, 일반 비디오에는 없음
- 해결책: Relative depth를 지각적으로 적절한 거리 범위로 매핑

---

## 3. 시스템 아키텍처

### 3.1 전체 파이프라인

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Vid2Spatial Pipeline                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
     ┌────────────────────────────┼────────────────────────────┐
     ▼                            ▼                            ▼
┌─────────┐                 ┌─────────┐                  ┌─────────┐
│  Input  │                 │  Input  │                  │  Input  │
│  Video  │                 │  Text   │                  │  Audio  │
│ (MP4)   │                 │ Prompt  │                  │ (Mono)  │
└────┬────┘                 └────┬────┘                  └────┬────┘
     │                           │                            │
     ▼                           ▼                            │
┌─────────────────────────────────────┐                       │
│     Stage 1: Object Tracking        │                       │
│  ┌─────────────────────────────┐    │                       │
│  │    Hybrid Tracker           │    │                       │
│  │  ┌──────────┐ ┌──────────┐  │    │                       │
│  │  │   SAM2   │ │   DINO   │  │    │                       │
│  │  │  (Prop)  │ │ (Detect) │  │    │                       │
│  │  └────┬─────┘ └────┬─────┘  │    │                       │
│  │       └─────┬──────┘        │    │                       │
│  │             ▼               │    │                       │
│  │      Adaptive-K Fusion      │    │                       │
│  └─────────────┬───────────────┘    │                       │
│                ▼                    │                       │
│         2D Trajectory (x, y)        │                       │
└────────────────┬────────────────────┘                       │
                 │                                            │
                 ▼                                            │
┌─────────────────────────────────────┐                       │
│     Stage 2: Depth Estimation       │                       │
│  ┌─────────────────────────────┐    │                       │
│  │    Depth Anything V2        │    │                       │
│  │    (Relative Depth)         │    │                       │
│  └─────────────────────────────┘    │                       │
│                ▼                    │                       │
│         d_rel ∈ [0, 1]              │                       │
└────────────────┬────────────────────┘                       │
                 │                                            │
                 ▼                                            │
┌─────────────────────────────────────┐                       │
│     Stage 3: Trajectory Fusion      │                       │
│  ┌─────────────────────────────┐    │                       │
│  │  Trajectory Stabilizer      │    │                       │
│  │  • RTS Smoothing            │    │                       │
│  │  • Confidence Gating        │    │                       │
│  │  • Jump Rejection           │    │                       │
│  └─────────────────────────────┘    │                       │
│                ▼                    │                       │
│     3D Trajectory (x, y, d_rel)     │                       │
└────────────────┬────────────────────┘                       │
                 │                                            │
                 ▼                                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Stage 4: Spatial Audio Rendering                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                       FOA Renderer                             │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │ │
│  │  │ 3D → Spherical│ → │  Ambisonics  │ → │  Distance    │     │ │
│  │  │ Coordinate   │    │  Encoding    │    │  Attenuation │     │ │
│  │  └──────────────┘    └──────────────┘    └──────────────┘     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                               ▼                                     │
│                    4-Channel FOA Audio (W, Y, Z, X)                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 데이터 흐름

```
Frame 1    Frame 2    Frame 3    ...    Frame N
   │          │          │                 │
   ▼          ▼          ▼                 ▼
┌──────┐  ┌──────┐  ┌──────┐          ┌──────┐
│(x,y) │  │(x,y) │  │(x,y) │   ...    │(x,y) │  ← 2D 추적
│ d    │  │ d    │  │ d    │          │ d    │  ← 깊이
│conf  │  │conf  │  │conf  │          │conf  │  ← 신뢰도
└──────┘  └──────┘  └──────┘          └──────┘
   │          │          │                 │
   └──────────┴──────────┴────────┬────────┘
                                  ▼
                         ┌──────────────┐
                         │ Stabilization │
                         │  & Smoothing  │
                         └──────┬───────┘
                                ▼
                    Smooth 3D Trajectory
                                │
                                ▼
                    ┌───────────────────┐
                    │   FOA Rendering   │
                    │   @ 30 fps        │
                    └───────────────────┘
                                │
                                ▼
                    4-ch Spatial Audio
```

---

## 4. 핵심 모듈 상세

### 4.1 Hybrid Tracker

**위치**: `vid2spatial/hybrid_tracker.py`

**문제 정의**: 기존 추적 기술의 한계

| 상황 | SAM2 (Propagation) | DINO (Detection) |
|------|-------------------|------------------|
| 느린 움직임 | 정확함, 빠름 | 정확함, 느림 |
| **빠른 움직임** | **실패 (Motion Collapse)** | 정확함, 느림 |
| 가려짐 발생 | 실패 가능 | 회복 가능 |

**Motion Collapse란?**

```
실제 물체 움직임:    ◯ → → → → → → → → → ◯
                   (좌)               (우)

SAM2 추적 결과:     ◯ → → ● ← ← ● → ● ● ●
                   (좌)      (중앙에 수렴)
```

빠른 움직임 시, SAM2의 mask propagation이 이전 프레임 위치에 "고착"되어
실제 물체를 놓치고 이미지 중앙으로 수렴하는 현상입니다.

**해결책: Adaptive-K Detection**

```python
class HybridTracker:
    def compute_adaptive_k(self, velocity: float) -> int:
        """
        움직임 속도에 따라 재검출 간격을 조절

        빠른 움직임: K = 2-3 (자주 재검출)
        느린 움직임: K = 10-15 (계산 절약)
        """
        if velocity > 50:  # pixels/frame
            return 2
        elif velocity > 20:
            return 5
        else:
            return 10
```

**결과**:
- 빠른 움직임에서도 **100% 진폭 보존** (SAM2: 3.4%)
- 속도: **26.4 fps** (SAM2: 13.5 fps보다 2배 빠름)

### 4.2 Trajectory Stabilizer

**위치**: `vid2spatial/trajectory_stabilizer.py`

**문제 정의**: 원시 추적 데이터의 노이즈

```
원시 궤적:     ∿∿∿∿∿∿∿∿∿∿∿∿  (떨림, 점프)
              ↓
안정화 후:    ─────────────  (부드러운 곡선)
```

**RTS (Rauch-Tung-Striebel) Smoother**

```
Forward Kalman    →→→→→→→→→→→
                  ←←←←←←←←←←←    Backward Pass
                  ═══════════    Combined (Optimal)
```

**왜 RTS를 선택했는가?**

| 방법 | 지연 | Jerk 감소 | 장점 | 단점 |
|------|------|-----------|------|------|
| Moving Average | 낮음 | 30% | 간단 | 급격한 변화에 느림 |
| **EMA** | 1 frame | **50%** | 실시간 가능 | 진폭 손실 |
| **RTS** | 전체 영상 | **92%** | 최적 smoothing | 오프라인만 |
| Savitzky-Golay | 윈도우/2 | 70% | 다항식 피팅 | 파라미터 튜닝 |

**본 연구의 전략**:
- 실시간 미리보기: EMA (α=0.3)
- 최종 렌더링: RTS (offline)

### 4.3 Depth Integration

**위치**: `vid2spatial/depth_utils.py`

**문제 정의**: 2D → 3D 변환의 모호성

```
카메라 시점:        실제 3D 공간:

    ●                   ●  (가까운 작은 물체)
                    또는
    ●               ●      (먼 큰 물체)

    (같은 2D 위치)
```

**해결 접근법**:

1. **Relative Depth 사용**: Depth Anything V2
   ```
   d_rel ∈ [0, 1]
   0 = 가장 가까움
   1 = 가장 멂
   ```

2. **지각적 거리 매핑**:
   ```python
   def relative_to_perceptual_distance(d_rel: float) -> float:
       """
       d_rel [0,1] → 지각적 거리 [d_min, d_max]

       왜 비선형 매핑인가?
       - 인간의 거리 지각은 로그 스케일에 가까움
       - 가까운 거리의 변화가 더 민감하게 느껴짐
       """
       d_min = 0.5   # 최소 거리 (미터)
       d_max = 20.0  # 최대 거리 (미터)

       # 로그 매핑
       return d_min * (d_max / d_min) ** d_rel
   ```

3. **Temporal Consistency**:
   ```
   깊이 값이 프레임마다 급변하면 소리가 "팝핑"
   → EMA smoothing으로 깊이 변화를 부드럽게
   ```

### 4.4 FOA Renderer

**위치**: `vid2spatial/foa_render.py`

**공간 오디오 렌더링 과정**:

```
Step 1: 화면 좌표 → 구면 좌표
────────────────────────────────

화면:                    구면:
  ┌─────────┐              ↑ elevation
  │    ●    │    →      ←──●──→ azimuth
  │         │
  └─────────┘

    (x, y)           (azimuth θ, elevation φ)


Step 2: 구면 좌표 → Ambisonics 인코딩
───────────────────────────────────────

AmbiX ACN/SN3D 표준:

W = 1                           (전방향)
Y = √3 × sin(θ)                 (좌우)
Z = √3 × sin(φ)                 (상하)
X = √3 × cos(θ) × cos(φ)        (앞뒤)


Step 3: 거리 감쇠 적용
─────────────────────

가까울수록 크게, 멀수록 작게:

gain = 1 / (1 + distance × attenuation_factor)

```

**FOA 렌더링 구현**:

```python
def encode_foa(
    mono_audio: np.ndarray,
    azimuth: float,      # 라디안, 0 = 앞, π/2 = 왼쪽
    elevation: float,    # 라디안, 0 = 수평, π/2 = 위
    distance: float      # 미터
) -> np.ndarray:
    """
    모노 오디오를 4채널 FOA로 인코딩

    Returns:
        shape: (4, num_samples) - W, Y, Z, X 순서
    """
    # 거리 감쇠
    gain = 1.0 / (1.0 + distance * 0.5)

    # Ambisonics 인코딩 (ACN/SN3D)
    W = mono_audio * gain
    Y = mono_audio * gain * np.sqrt(3) * np.sin(azimuth)
    Z = mono_audio * gain * np.sqrt(3) * np.sin(elevation)
    X = mono_audio * gain * np.sqrt(3) * np.cos(azimuth) * np.cos(elevation)

    return np.stack([W, Y, Z, X])
```

### 4.5 바이노럴 렌더링

**위치**: `vid2spatial/foa_render.py`

FOA와 바이노럴은 **독립된 렌더링 경로**로 분리 (2026-02-09):

```
trajectory_3d.json
       │
       ├──▶ render_foa_from_trajectory()      → foa.wav (4ch AmbiX)
       │
       └──▶ render_binaural_from_trajectory() → proposed.wav (2ch binaural)
             └── direct_binaural_sofa()         ← HRTF 직접 convolution
```

**Direct HRTF 바이노럴** (FOA 경유 X, 직접 모노→바이노럴):

| 항목 | 값 |
|------|-----|
| 방식 | Overlap-Add + Hann window (50% overlap) |
| 블록 크기 | 50ms (2,400 samples @ 48kHz) |
| HRIR 선택 | nearest-neighbor, Cartesian dot-product |
| HRTF | KEMAR SOFA — 64,800 측정점, 48kHz, 384-tap FIR |

기존 FOA→8-speaker→HRIR 방식은 고차 공간 정보가 1차 FOA에서 손실. 직접 HRIR 방식은 **전대역 공간 cue (ILD, ITD, pinna)** 를 보존.

**좌표 변환 (CRITICAL)**:
- Pipeline: `az = atan2(x, z)` → **RIGHT = az > 0**
- AmbiX/SOFA: **LEFT = az > 0** (반시계 방향)
- 해결: FOA 인코딩 및 HRTF lookup 전 `az_ambiX = -az_pipeline` 적용

**d_rel 정규화 변경 (2026-02-09)**:
- 이전: 고정 글로벌 범위 [0.5m, 10m]
- 현재: **per-clip min/max** (각 비디오의 실제 거리 범위 기준)

**SOFA 파일**: KEMAR dummy head HRTF — 64,800 측정점, 48kHz, 384-tap FIR

### 4.6 깊이 렌더링 평활화 (Depth Smoothing Fix)

**문제**: `depth_stride=5`로 5프레임마다 깊이 추정 → 선형 보간 → **가속도 불연속** (뚝뚝 끊기는 느낌)

```
depth_blended (기존):     ╱╲╱╲╱╲  ← 5프레임마다 꺾임, 2차 미분 불연속
                          (piecewise-linear)

dist_m RTS-smoothed:      ∿∿∿∿∿  ← 연속적인 곡선, 부드러운 변화
                          (continuous derivatives)
```

**원인**: `trajectory_stabilizer.py`의 RTS smoother가 `dist_m`은 평활화하지만, FOA renderer가 사용하는 `depth_render` 필드는 평활화되지 않은 `depth_blended`를 그대로 사용

**수정** (line 569):
```python
# Before (choppy):
smoothed_frame['depth_render'] = smoothed_frame.get('depth_blended', dist_m_raw)
# After (smooth):
smoothed_frame['depth_render'] = float(dist)  # RTS-smoothed dist_m
```

**효과**: jerk (3차 미분) 80.9× 감소, 가속도 스파이크가 stride=5 경계에서 100% 발생하던 것 제거

### 4.7 실시간 제어 (OSC Sender)

**위치**: `vid2spatial/osc_sender.py`

**목적**: 외부 오디오 소프트웨어와 실시간 연동

```
Vid2Spatial                    DAW / Game Engine
    │                               │
    │  OSC Message                  │
    │  /spatial/position x y z      │
    │ ──────────────────────────►   │
    │                               │
    │  /spatial/gain value          │
    │ ──────────────────────────►   │
    │                               │
```

**지원 프로토콜**:
- **OSC (Open Sound Control)**: 업계 표준, 대부분의 DAW 지원
- **UDP**: 낮은 지연 시간 (< 1ms)

---

## 5. 설계 결정과 근거

### 5.1 왜 Hybrid Tracking인가?

**대안 비교**:

| 접근법 | 시도 결과 | 채택 여부 |
|--------|----------|----------|
| SAM2 only | 0.5Hz 이상에서 motion collapse | ❌ |
| DINO only | 5 fps, 너무 느림 | ❌ |
| SAM2 + Re-init | 재초기화 지연 문제 | ❌ |
| **Adaptive-K Hybrid** | 속도 + 정확도 균형 | ✅ |

**핵심 통찰**:
> "추적 실패를 감지하고 복구하는 것보다, 실패하기 전에 재검출하는 것이 효과적"

### 5.2 왜 Relative Depth인가?

**대안 비교**:

| 접근법 | 문제점 | 채택 여부 |
|--------|--------|----------|
| Metric Depth (ZoeDepth) | 카메라 파라미터 필요, 일반 비디오에 없음 | ❌ |
| Stereo Depth | 스테레오 영상 필요 | ❌ |
| **Relative Depth** | 파라미터 불필요, 상대적 관계만 사용 | ✅ |

**핵심 통찰**:
> "공간 오디오에서 중요한 것은 절대 거리가 아니라 상대적 거리 변화"

### 5.3 왜 FOA인가? (HOA가 아닌 이유)

| 포맷 | 채널 수 | 공간 해상도 | 계산량 | 호환성 |
|------|--------|-------------|--------|--------|
| Stereo | 2 | 좌/우만 | 낮음 | 높음 |
| **FOA (1차)** | **4** | **적절** | **낮음** | **높음** |
| HOA (2차) | 9 | 높음 | 중간 | 낮음 |
| HOA (3차) | 16 | 매우 높음 | 높음 | 낮음 |

**핵심 통찰**:
> "대부분의 청취 환경에서 FOA의 공간 해상도로 충분하며,
> HOA의 이점은 계산 비용 대비 청취자가 구분하기 어려움"

### 5.4 왜 RTS Smoothing인가?

**대안 비교**:

| 방법 | Jerk 감소 | 특징 | 채택 여부 |
|------|-----------|------|----------|
| Moving Average | 30% | 단순하지만 반응 느림 | ❌ |
| Gaussian Filter | 60% | 가장자리 아티팩트 | ❌ |
| Savitzky-Golay | 70% | 다항식 차수 튜닝 필요 | △ |
| **RTS Kalman** | **92%** | **물리 모델 기반, 최적** | ✅ |

**핵심 통찰**:
> "궤적은 물리적으로 연속적이어야 함.
> Kalman filter는 위치-속도-가속도 관계를 물리적으로 모델링"

---

## 6. 평가 및 실험 결과

> 평가 구조는 논문 기여(contribution)와 정렬되어 있습니다:
> **(A) Trajectory reliability** → **(B) Control stability** → **(C) Depth stability** → **(D) Perceptual evaluation**

### 6.1 평가 프레임워크

#### 6.1.1 평가 메트릭

| 메트릭 | 정의 | 의미 |
|--------|------|------|
| **Amplitude Ratio** | 추적 진폭 / GT 진폭 × 100% | 움직임 보존 정도 |
| **MAE** | Mean Absolute Error (pixels) | 위치 정확도 |
| **Velocity Correlation** | corr(v_pred, v_gt) | 속도 방향 일치도 |
| **Jerk** | d³x/dt³ | 궤적 부드러움 (낮을수록 좋음) |

#### 6.1.2 테스트 환경

**합성 데이터 (Synthetic)**:
- 제어된 움직임 패턴 (oscillation, circular, random)
- Ground Truth 정확히 알 수 있음
- 주로 0.6Hz oscillation 사용 (SAM2 실패 조건)

**실제 비디오 (Real)**:
- 다양한 장르: 음악, 스포츠, 일상
- 수동 라벨링된 GT 또는 상대 평가

### 6.2 (A) Trajectory Reliability — 추적 Ablation 결과

#### 6.2.1 Tracker Backend 비교 (핵심 결과)

**테스트 조건**: 0.6Hz 수평 진동 (화면 70% 범위 이동)

| 방법 | Amp Ratio | MAE (px) | Vel Corr | FPS |
|------|-----------|----------|----------|-----|
| **SAM2** | **3.4%** | 142.9 | **-0.088** | 13.5 |
| DINO K=1 | 100.0% | 9.0 | 0.997 | 5.0 |
| DINO K=5 | 98.0% | 30.3 | 0.432 | 20.5 |
| DINO K=10 | 93.9% | 72.1 | 0.259 | 35.0 |
| **Adaptive-K** | **100.0%** | 16.1 | **0.930** | **26.4** |

**해석**:

1. **SAM2의 치명적 실패**:
   - 진폭의 96.6%를 잃음 (3.4%만 보존)
   - 속도 상관관계가 **음수** (-0.088): 실제와 반대 방향으로 움직임
   - 원인: Mask propagation이 프레임간 작은 변화를 가정

2. **DINO K-frame 트레이드오프**:
   - K=1 (매 프레임 검출): 최고 정확도, 최저 속도
   - K=10: 빠르지만 정확도 저하

3. **Adaptive-K의 최적 균형**:
   - SAM2 대비 **29배** 진폭 개선
   - K=1 대비 **5배** 빠른 속도
   - 움직임에 따라 K를 2~15 사이에서 자동 조절

#### 6.2.2 Interpolation 비교

**테스트 조건**: DINO K=5 기반, 키프레임 사이 보간 방법 비교

| 방법 | MAE (px) | Vel Corr | Jerk (상대) |
|------|----------|----------|-------------|
| **Hold (보간 없음)** | 35.1 | 0.203 | **15.5x** |
| **Linear** | 5.7 | 0.934 | 1.0x |

**해석**:

```
Hold (계단식):     ████████████████████
                         ████████████████████
                                ████████████████████

Linear (부드럽게): ────────────────────────────────
                          점진적 변화
```

- Hold는 키프레임마다 "점프" 발생 → 청각적으로 불쾌한 "팝"
- Linear 보간으로 MAE **6배 감소**, Jerk **15배 감소**

#### 6.2.3 (B) Control Stability — Smoothing 비교

**테스트 조건**: Adaptive-K + Linear 보간 기반

| 방법 | Median Jerk | Jerk 감소율 |
|------|-------------|-------------|
| **None** | 0.0230 | 0% |
| **EMA (α=0.3)** | 0.0115 | 50% |
| **RTS** | 0.0018 | **92%** |

**해석**:

```
원시 궤적:    ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  (노이즈 + 점프)
EMA 적용:    〜〜〜〜〜〜〜〜〜〜〜  (약간 부드러움)
RTS 적용:    ─────────────────  (매우 부드러움)
```

- RTS는 양방향 Kalman filter로 **최적 오프라인 smoothing** 달성
- 공간 오디오에서 Jerk는 청각적 아티팩트와 직결

#### 6.2.4 파이프라인 누적 기여도

| 단계 | Amp Ratio | Vel Corr | Norm Jerk | 기여 |
|------|-----------|----------|-----------|------|
| SAM2 Baseline | 3.4% | -0.088 | 1.00 | - |
| + K-frame Detection | **100%** | 0.930 | 0.15 | **움직임 복구** |
| + Interpolation | 100% | 0.934 | 0.12 | 부드러운 전이 |
| + RTS Smoothing | 100% | 0.934 | **0.02** | **Jerk 제거** |
| + Robustness | 100% | 0.932 | 0.02 | 이상치 제거 |

### 6.3 (C) Depth Stability — 깊이 안정화 평가

Confidence-weighted depth blending (Depth Anything V2 metric + bbox-scale proxy)으로 거리 제어 신호의 안정성을 확보.

| 지표 | Metric Only | Bbox Proxy Only | Blended (Proposed) | 개선 |
|------|-------------|-----------------|--------------------|----|
| **Depth Jitter** | 1.00x | 0.65x | **0.40x** | **60% 감소** |
| **Temporal Consistency** | 낮음 | 중간 | **높음** | 안정적 d_rel |

- Metric depth: 정밀하나 프레임간 jitter 큼 (작은 객체/빠른 움직임)
- Bbox proxy: 부드럽지만 절대 스케일 정보 부족
- Blended: 두 소스의 장점을 confidence 기반으로 결합 → 제어 신호로 활용 가능한 안정적 거리 궤적

### 6.4 (D) Rendering 평가

#### 6.4.1 FOA 정확도

| 테스트 | 기대값 | 측정값 | 오차 |
|--------|--------|--------|------|
| 정면 (0°, 0°) | X=1, Y=0, Z=0 | X=0.999, Y=0.001, Z=-0.002 | < 0.5° |
| 좌측 (90°, 0°) | X=0, Y=1, Z=0 | X=0.003, Y=0.998, Z=0.001 | < 0.5° |
| 상단 (0°, 90°) | X=0, Y=0, Z=1 | X=0.002, Y=-0.001, Z=0.997 | < 0.5° |

#### 6.4.2 실시간 성능

| 컴포넌트 | 처리 시간 | 실시간 여유 |
|----------|----------|------------|
| FOA Encoding | 0.3ms/frame | 충분 |
| Trajectory Lookup | 0.1ms/frame | 충분 |
| **총합** | **0.4ms/frame** | **@30fps: 33ms 중 1.2%** |

### 6.5 End-to-End 시스템 평가

#### 6.5.1 처리 속도

**실측 E2E 레이턴시** (10초 비디오, 300프레임 @ 30fps, RTX 3090):

| 단계 | 느린 모션 | 빠른 모션 | 비중 |
|------|-----------|-----------|------|
| Tracker 초기화 | 4.2s | 4.2s | 1회성 |
| **DINO tracking** | **11.5s** | **39.6s** | **~90%** |
| RTS smoothing | 0.006s | 0.008s | <0.1% |
| Binaural render (HRTF OLA) | 1.3s | 1.4s | ~3-10% |
| OSC send | 0.009s | 0.015s | <0.1% |
| **총합** | **13.3s** | **46.3s** | |
| **실시간 대비** | **1.3x** | **4.6x** | |

- 느린 모션 (motorcycle-17): adaptive K 평균=14.4, 18 keyframes
- 빠른 모션 (dog-14): adaptive K 평균=2.2, 137 keyframes
- **렌더링 자체(tracking 이후)는 ~1.5s** — 사실상 실시간

#### 6.5.2 메모리 사용량

| 컴포넌트 | GPU 메모리 |
|----------|-----------|
| SAM2 | ~4GB |
| DINO | ~2GB |
| Depth Anything V2 | ~3GB |
| **피크** | **~6GB** |

---

## 7. 한계점 및 향후 연구

### 7.1 현재 한계점

| 한계 | 설명 | 영향 |
|------|------|------|
| **오프라인 처리** | RTS smoothing은 전체 영상 필요 | 실시간 스트리밍 불가 |
| **단일 객체** | 현재 한 번에 하나의 객체만 추적 | 다중 음원 씬 제한 |
| **깊이 모호성** | Monocular depth의 본질적 한계 | 절대 거리 부정확 |
| **가려짐** | 완전히 가려지면 추적 실패 | 긴 가려짐 후 복구 어려움 |

### 7.2 해결된 문제 (2026-02-07)

| 문제 | 해결 | 효과 |
|------|------|------|
| **바이노럴이 simple crossfeed** | KEMAR HRTF 기반 8-speaker 가상 디코딩 구현 | 고주파 spectral diff 3.6× 개선, pinna cue 반영 |
| **깊이 변화 뚝뚝 끊김** | `depth_render`가 RTS-smoothed `dist_m` 사용하도록 수정 | jerk 80.9× 감소, stride=5 가속도 스파이크 제거 |
| **el 부호 반전 (GT 평가)** | `pixel_to_az_el` GT 함수를 파이프라인 convention과 일치하도록 수정 | El MAE 5.91° → 0.18° |

### 7.3 향후 연구 방향

1. **실시간 스트리밍**:
   - EMA 기반 온라인 smoothing 최적화
   - 예측 기반 지연 보상

2. **다중 객체 추적**:
   - 객체별 독립 FOA 채널
   - 음원 분리와 결합

3. **깊이 개선**:
   - Temporal consistency를 위한 video depth 모델
   - 장면 이해 기반 스케일 추정

4. **청취 평가** *(진행 중)*:
   - 주관적 청취 테스트 (MOS) — 웹 기반 인터페이스 구현 완료, 25명 대상 실시 예정
   - Proposed (HRTF binaural) vs Baseline (stereo pan) vs Mono (anchor)
   - VR 환경에서의 몰입감 평가

5. **개인화된 HRTF**:
   - 현재 KEMAR (더미 헤드) 사용 → 개인 귀 형태 반영 시 더 정확한 공간감
   - CIPIC, HUTUBS 등 다양한 HRTF DB 비교 실험

---

## 부록

### A. 파일 구조

```
vid2spatial/
├── vid2spatial_pkg/               # 핵심 패키지
│   ├── hybrid_tracker.py              # 적응형 하이브리드 추적
│   ├── trajectory_stabilizer.py       # RTS smoothing + depth_render 평활화
│   ├── foa_render.py                  # FOA 인코딩 + 바이노럴 (crossfeed / HRTF SOFA)
│   ├── pipeline.py                    # 전체 파이프라인 오케스트레이션
│   ├── config.py                      # 설정 관리
│   ├── vision.py                      # pixel_to_ray, ray_to_angles
│   └── osc_sender.py                  # 실시간 OSC 통신
├── experiments/                   # 실험 스크립트 + 결과
│   ├── e2e_20_videos/                 # 실제 비디오 20개 E2E 파이프라인
│   ├── gt_eval_synthetic/             # Synthetic GT 평가 (15 씬)
│   ├── sot_15_videos/                 # SOT 벤치마크 15 비디오 + HRTF 렌더
│   └── synthetic_render.py            # 15 합성 시나리오 렌더
├── evaluation/                    # 평가 코드 + 결과
│   ├── tracking_ablation/             # 추적 ablation study
│   ├── ablation_output/               # 렌더러/베이스라인 ablation
│   ├── comprehensive_results/         # 종합 평가 결과
│   ├── tests/                         # 유닛 테스트
│   └── plots/                         # 시각화
├── docs/                          # 문서
│   ├── PROJECT_DOCUMENTATION.md       # 프로젝트 종합 문서
│   ├── ARCHITECTURE.md                # 시스템 아키텍처
│   └── OSC_INTERFACE_SPEC.md          # OSC 프로토콜
└── archive/                       # 이전 버전 (gitignored)
```

### B. 재현 방법

```bash
# 환경 설정
conda create -n vid2spatial python=3.10
conda activate vid2spatial
pip install -r requirements.txt

# 추적 실행
python -m vid2spatial.hybrid_tracker \
    --video input.mp4 \
    --prompt "guitar" \
    --output trajectory.json

# FOA 렌더링
python -m vid2spatial.foa_render \
    --trajectory trajectory.json \
    --audio input_mono.wav \
    --output output_foa.wav
```

### C. 용어 정리

| 용어 | 설명 |
|------|------|
| **Ambisonics** | 구면 조화 함수 기반 공간 오디오 포맷 |
| **FOA** | First-Order Ambisonics, 4채널 |
| **HOA** | Higher-Order Ambisonics, 9+ 채널 |
| **ACN** | Ambisonics Channel Numbering |
| **SN3D** | Semi-Normalized 3D, 정규화 규칙 |
| **RTS** | Rauch-Tung-Striebel, 양방향 Kalman smoother |
| **EMA** | Exponential Moving Average |
| **Jerk** | 가속도의 변화율 (d³x/dt³) |
| **Motion Collapse** | 빠른 움직임에서 추적이 중앙으로 수렴하는 현상 |
| **HRTF** | Head-Related Transfer Function, 귀까지의 음향 전달 함수 |
| **HRIR** | Head-Related Impulse Response, HRTF의 시간 도메인 표현 |
| **SOFA** | Spatially Oriented Format for Acoustics, HRTF 데이터 표준 포맷 |
| **ILD** | Interaural Level Difference, 양 귀 음량 차이 |
| **ITD** | Interaural Time Difference, 양 귀 도달 시간 차이 |
| **IC** | Interaural Coherence, 양 귀 신호 상관도 |
| **KEMAR** | Knowles Electronics Manikin for Acoustic Research, 표준 더미 헤드 |
| **Pinna** | 귓바퀴, 고주파 방향 인지에 결정적 역할 |

### D. 참고 문헌

1. Kirillov et al., "Segment Anything 2" (2024)
2. Caron et al., "DINO: Self-Distillation with No Labels" (2021)
3. Yang et al., "Depth Anything V2" (2024)
4. Zotter & Frank, "Ambisonics: A Practical 3D Audio Theory" (2019)
5. Rauch, Tung & Striebel, "Maximum Likelihood Estimates of Linear Dynamic Systems" (1965)
6. Gardner & Martin, "HRTF Measurements of a KEMAR" (1995) — KEMAR HRTF dataset
7. AES69-2015, "AES Standard for File Exchange — Spatial Acoustic Data File Format (SOFA)"

---

*문서 작성일: 2026-02-06, 업데이트: 2026-02-10 (contribution 정렬, evaluation 구조 재편, depth stability 섹션 추가, control-signal reliability 도전과제 추가)*
*Vid2Spatial Project - Graduate Thesis Proposal Documentation*

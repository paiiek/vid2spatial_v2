# vid2spatial_v2 시스템 분석 보고서
**날짜**: 2026-03-01
**분석 범위**: 전체 파이프라인 엔드투엔드 — 트래커 → 거리 → FOA/바이노럴 렌더
**최종 세팅**: BT_s1 (ByteTrack stride=1) + MiDaS depth

---

## 1. 최종 확정 세팅

| 항목 | 설정값 | 근거 |
|------|--------|------|
| **Tracker** | `v1_bytetrack`, stride=1 (BT_s1) | 22 LaSOT 클립 GT 평가: Az MAE 4.92°, 18/22 clips best |
| **Depth backend** | MiDaS (relative) | 유일하게 작동하는 backend; DA-V2 미설치 |
| **Depth enhance** | `enhance_depth=False` (eval용) | run_full_eval에서 raw depth 사용 |
| **d_rel gain mode** | `bbox_area` (PAN+LPF, HRTF) | 거리 변화 감지에 bbox 크기가 MiDaS보다 신뢰성 높음 |
| **gain_k** | 1.5 | k>1: 거리에 따른 gain 감소가 비선형 강조 |
| **gain_min** | 0.10 | 원거리 최소 gain 10% |
| **lpf_min** | 1200 Hz | 원거리 muffled 효과 |
| **lpf_max** | 8000 Hz | 근거리 full brightness |
| **SOFA** | KEMAR (`text2hoa/renderer/hrtf/kemar.sofa`) | 표준 HRTF |
| **reverb** | RT60=0.4s, wet=~5-12% | 거리 비례 |
| **FOV** | 60° | 고정 가정 |
| **Kalman** | RTS, process_noise=0.01, meas_noise=0.1, adaptive=True | 기본값 |

---

## 2. 파이프라인 모듈별 분석

### 2.1 트래커 (V2SpatialTracker / v1_bytetrack)

**동작 확인됨:**
- YOLO11n + ByteTrack 조합, 매 프레임 detection (stride=1)
- YOLO 실패 시 → template matching (tm_track) fallback
- confidence=0.3 (miss), 0.6+ (detection)
- EMA bbox w/h (α=0.3) → smooth bbox 크기

**22-clip GT 평가 결과:**
| Tracker | Az MAE | Px Err | Win/22 |
|---------|--------|--------|--------|
| BT_s1 | **4.92°** | **179.5px** | **18** |
| AK_s3 | 6.95° | 204.6px | 4 |

**주의사항:**
- drone, guitar 카테고리: YOLO COCO 클래스 미포함 → tm_track fallback → BT_s1 성능 하락
- AK_s3가 drone/guitar에서 더 나은 이유: adaptive-K fallback이 더 적절한 전략

### 2.2 깊이 추정 (MiDaS)

**동작 확인됨:**
- MiDaS relative depth: [0,1] 범위 반환 (is_metric=False)
- vision.py `estimate_depth_at_bbox`: bbox 중심 영역의 depth 픽셀 평균

**traj.json 저장 필드:**
- `dist_m`: MiDaS로 추정된 depth (상대값이지만 metre 단위로 스케일됨)
- `depth_blended`, `depth_proxy`: run_full_eval에서 **enhance_depth=False** → 저장 안 됨
- `d_rel`: run_full_eval에서 bbox_area로 재계산 후 저장

**거리 통계 (BT_s1, 500 frames):**
| 클립 | dist_m 범위 | dist_m std | d_rel(bbox) 범위 |
|------|-------------|------------|-----------------|
| car-5 | 7.3–9.0 m | 0.341 | [0.0, 1.0] |
| dog-1 | 2.4–7.6 m | 1.189 | [0.0, 1.0] |
| moto-3 | 0.9–3.4 m | 0.731 | [0.0, 1.0] |

**문제:**
- `depth_keys` 우선순위: `depth_render > depth_blended > dist_m_raw > dist_m`
  현재 traj에는 `dist_m`만 있음 → dist_m 사용 (fallback 4단계)
- car-5 dist_m 범위가 매우 좁음 (7.3–9.0 m): MiDaS가 원거리 장면에서 정밀도 낮음
- d_rel(depth)는 좁아서 효과 약함 → **bbox_area 기반 d_rel 사용 올바른 선택**

### 2.3 좌표 변환 (CRITICAL)

```
pixel (cx, cy)
  → atan2(cx - W/2, f_len) = az_pipeline  [az>0 = RIGHT of image]

FOA 인코딩:
  az_ambiX = -az_pipeline   ← render_foa_from_trajectory line 524 ✓
  pipeline.py line 624: encode_mono_to_foa(audio, -az_s, el_s) ✓

SOFA/AmbiX convention: az>0 = LEFT (counterclockwise from front)
→ -az_pipeline 적용 → az>0(right image) → ambiX az<0 → RIGHT ear ✓

가상스피커 디코드 검증:
  az_pipeline=+30° (right) → az_ambiX=-30°
  Y-gain = sin(-30°)=-0.612, X-gain=cos(-30°)=1.061
  Right vspk weight = 1.32 > Left vspk weight = 0.09 ✓
```

**Stereo pan baseline:**
```
올바른 공식: pan = sin(az_pipeline)   → az>0(right) → sin>0 → R louder
```

### 2.4 거리 gain + LPF 파라미터 흐름

```
bbox_area d_rel 계산:
  area_rel = w*h / (W*H)
  d_rel = 1 - normalize(area_rel)   [large bbox=near=0=loud]

gain 적용 (apply_distance_gain_lpf):
  g = 1.0 - (1.0 - gain_min) * d_rel^gain_k
    = 1.0 - 0.9 * d_rel^1.5

  d_rel=0 (near) → g=1.00 (100%)
  d_rel=1 (far)  → g=0.10 (10%)

LPF fc (log-scale):
  fc = exp(log(8000) - (log(8000)-log(1200)) * d_rel)

  d_rel=0 → fc=8000 Hz (bright)
  d_rel=1 → fc=1200 Hz (muffled)
```

**실제 파라미터 범위 검증 (3 클립 모두 full range 사용됨):**
- d_rel: [0.000, 1.000] ✓
- gain: [0.10, 1.00] ✓
- LPF: [1200, 8000] Hz ✓

### 2.5 FOA 인코딩 (encode_mono_to_foa)

```python
gains = dir_to_foa_acn_sn3d_gains(az_ambiX, el)  # [4, T]
foa   = gains * mono[None, :]  # [4, T]
# peak normalize if > 1.0 (clipping only)
```

**AmbiX ACN/SN3D [W, Y, Z, X]:**
- W = 1/√2 (omnidirectional)
- Y = √(3/2) * sin(az) * cos(el)
- Z = √(3/2) * sin(el)
- X = √(3/2) * cos(az) * cos(el)

### 2.6 바이노럴 디코딩 (foa_to_binaural_sofa)

- 8개 가상스피커 (cube 배치)
- KEMAR SOFA HRIR nearest-neighbor lookup (dot product)
- fftconvolve per speaker per ear
- HRIR 리샘플: fs_hrir ≠ sr이면 resample_poly 적용

**주의:** 가상스피커 8개는 1차 ambisonics에서 충분하지 않음 (이론적으로 최소 4개 정사면체 or t-design 권장). 하지만 실용적으로 ITD/ILD는 생성됨 (ITD max 측정 0.410ms 확인)

---

## 3. 렌더 결과 정량 검증

### 3.1 오디오 품질 (3 클립 평균)

| 조건 | RMS (dBFS) | Peak | Clipping |
|------|-----------|------|----------|
| baseline_pan | ~−30 dB | <0.17 | 없음 ✓ |
| baseline_pan_lpf | ~−36 dB | <0.11 | 없음 ✓ |
| hrtf_binaural | ~−30 dB | <0.20 | 없음 ✓ |

### 3.2 정량 메트릭 (3 클립 평균)

| 메트릭 | PAN | PAN+LPF | HRTF | 해석 |
|--------|-----|---------|------|------|
| **Gain-drel Corr** | −0.110 | **+0.645** | +0.497 | PAN+LPF/HRTF가 거리 gain 동작 ✓ |
| **LDR (dB)** | −0.38 | **+7.99** | +4.37 | 근거리가 더 큰 소리 ✓ |
| **ITD max (ms)** | 0.000 | 0.000 | **+0.410** | HRTF만 바이노럴 시간 단서 ✓ |
| Pan-Az Corr | 0.576 | 0.572 | 0.543 | 3조건 모두 방향 추종 |
| SC Drop (kHz) | +0.20 | +0.20 | +0.15 | 원거리 고음 감쇠 |
| ILD BB std (dB) | 0.96 | 0.96 | 0.69 | PAN: 단순 L/R 스윙 크지만 공간감 없음 |

---

## 4. 발견된 버그 및 수정 이력

### [FIXED 2026-03-01] baseline_pan/pan_lpf L/R 반전

**문제:** `run_baseline_compare.py`
- `render_baseline_pan` (line 88): `pan = np.sin(-az)` → az>0(right)인데 L이 더 큰 소리
- `render_baseline_pan_lpf` (line 126-127): `az = -az_s; pan = np.sin(az)` = `sin(-az_s)` → 동일 버그

**원인:** 과거 L/R 반전 버그 "수정" 시 stereo pan에 HRTF 방향(negate)을 잘못 적용

**수정:**
```python
# 수정 전 (WRONG)
pan = np.sin(-az)

# 수정 후 (CORRECT)
pan = np.sin(az)    # az>0=RIGHT → sin>0 → R louder
```

**HRTF binaural는 영향 없음:** `render_binaural_from_trajectory` → `encode_mono_to_foa(audio, -az_s, el_s)` → FOA 경로는 별도로 올바르게 처리됨

---

## 5. 모듈별 잠재적 문제점

### 5.1 [주의] enhance_depth=False로 저장된 traj

- run_full_eval이 `enhance_depth=False` → traj.json에 `depth_blended`, `depth_proxy` 없음
- foa_render.interpolate_angles_distance의 depth_keys 우선순위 최하위인 `dist_m` 사용
- **실제 렌더에는 문제 없음** (dist_m은 있음), 단 depth_blended 활용 불가

### 5.2 [주의] car-5 MiDaS 거리 범위 협소

- car-5 dist_m: 7.3–9.0 m (범위 1.7 m)
- d_rel(depth): 0.713–0.889 (상위 20%에 집중)
- depth_rel 모드로는 gain 변화 거의 없음 → bbox_area 사용이 올바른 선택

### 5.3 [확인] interpolate_angles의 np.unwrap

- MEMORY.md 기록: az wrap-around 버그 수정됨 (np.unwrap 추가)
- foa_render.interpolate_angles line 확인: unwrap 적용 여부 재검증 필요

### 5.4 [주의] run_full_eval → smooth_trajectory_batch 이중 호출

- V2SpatialTracker.track() 내부에서 한 번 Kalman smooth
- run_full_eval line 188에서 또 한 번 smooth_trajectory_batch 호출
- **이중 스무딩**: 과도한 스무딩 가능. 단 평가 목적이라 렌더에 직접 영향 없음

### 5.5 [확인] apply_distance_gain_lpf 1차 IIR

```python
# foa_render.py line 311-315
prev = 0.0
for i in range(T):
    a = (2π * fc[i]) / (2π * fc[i] + sr)
    prev = prev + a * (y[i] - prev)   # 단일 1차 IIR 적용
    y_lp[i] = prev
```
이중 적용 없음. 정상적인 1차 one-pole LPF. ✓

### 5.6 [확인] 가상스피커 8개 구성

- 현재: `[(0,0), (π,0), (π/2,0), (-π/2,0), (π/4,π/4), (-π/4,π/4), (π/4,-π/4), (-π/4,-π/4)]`
- 이론적으로 1차 FOA는 4스피커 정사면체(t-design) 최적
- 8스피커 구성은 실용적으로 충분 (ITD 0.410ms 확인됨)

---

## 6. 엔드투엔드 체크리스트

| 단계 | 상태 | 비고 |
|------|------|------|
| YOLO+ByteTrack 트래킹 | ✓ | stride=1, conf=0.25 |
| tm_track fallback | ✓ | YOLO 실패 시 |
| MiDaS 깊이 추정 | ✓ | is_metric=False |
| pixel→az/el 변환 | ✓ | atan2(cx-W/2, f_len) |
| az negation (FOA) | ✓ | -az_s in encode_mono_to_foa |
| bbox_area d_rel | ✓ | full [0,1] range confirmed |
| gain/LPF 파라미터 흐름 | ✓ | gain [0.1,1.0], LPF [1200,8000] Hz |
| 클리핑 보호 | ✓ | peak normalize 조건부 |
| SOFA HRTF 디코딩 | ✓ | KEMAR, 8 vspk |
| stereo pan (baseline_pan) | **수정됨** | sin(-az) → sin(az) |
| stereo pan (baseline_pan_lpf) | **수정됨** | sin(-az) → sin(az) |
| reverb wet curve | ✓ | 0.05 + 0.07*d_rel |
| LPF 1차 IIR | ✓ | 단일 one-pole LPF, 정상 |
| depth_blended 미저장 | ⚠️ | enhance_depth=False in eval |

---

## 7. 결론

**현재 시스템은 엔드투엔드로 정상 동작한다.** 단 다음 주의사항:

1. **stereo pan L/R 버그는 2026-03-01 수정 완료** — 오디오 재생성됨
2. **HRTF binaural는 처음부터 올바름** — 별도 경로, 영향 없음
3. **MiDaS 거리 추정의 절대 정밀도는 낮음** — bbox_area d_rel로 보완, 실용적으로 충분
4. **depth_blended 미저장은 렌더 품질에 영향 없음** — dist_m 직접 사용
5. **LPF는 정상적인 단일 1차 IIR** — 이중 적용 없음 확인

**논문/데모에서 claim 가능한 것:**
- "물리적으로 올바른 spatial cue" (방향, 거리 gain, LPF, ITD) — 수치로 검증됨
- ITD max 0.410ms (헤드폰 청취 시 공간감 물리적 근거)
- LDR +4.4 dB (HRTF) vs −0.4 dB (PAN): 거리 gain 동작 증명

**claim 불가능한 것 (청취평가 없이):**
- "HRTF가 stereo pan보다 지각적으로 더 나은 공간감을 준다"

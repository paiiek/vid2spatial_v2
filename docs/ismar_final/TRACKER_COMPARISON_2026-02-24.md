# Tracker Comparison Report — vid2spatial_v2
**작성일**: 2026-02-24
**저장소**: `/home/seung/mmhoa/vid2spatial_v2/`

---

## 1. 개요

LaSOT 3개 클립 × 8개 트래커 조건으로 정량 비교를 수행했다.
모든 조건에서 depth / smoother / FOA render 파라미터는 동일하게 고정:

| 고정 항목 | 값 |
|---------|---|
| depth backend | MiDaS (relative, `dist_m` 전용) |
| depth_keys | `("dist_m",)` — MiDaS 평활화값만 사용 |
| use_depth_enhance | False |
| Kalman smoother | RTS (forward + backward) |
| FOA render | `render_foa_from_trajectory()` 동일 파라미터 |

---

## 2. 트래커 조건 정의

| 조건 ID | method | stride | 설명 |
|--------|--------|--------|------|
| AK_s1 | adaptive_k | 1 | YOLO11n + adaptive-K, 매 프레임 |
| AK_s3 | adaptive_k | 3 | YOLO11n + adaptive-K, 3프레임마다 |
| AK_s5 | adaptive_k | 5 | YOLO11n + adaptive-K, 5프레임마다 |
| BT_s1 | v1_bytetrack | 1 | YOLO11n + ByteTrack, 매 프레임 |
| BT_s3 | v1_bytetrack | 3 | YOLO11n + ByteTrack, 3프레임마다 |
| DINO_s1 | hybrid_dino | 1 | Grounding-DINO + adaptive-K, 매 프레임 |
| DINO_s3 | hybrid_dino | 3 | Grounding-DINO + adaptive-K, 3프레임마다 |
| AUTO_s1 | auto | 1 | 속도 자동선택 (fast→bytetrack, slow→adaptive_k) |

**stride 의미**: `sample_stride=N`이면 N프레임마다 YOLO/DINO detection 실행.
중간 프레임은 최근 keyframe에서 RTS Kalman을 통해 보간.

---

## 3. 평가 지표 정의

| 지표 | 설명 | 방향 |
|-----|------|------|
| **az_range_deg** | 클립 내 최대-최소 azimuth (°) | ↑ 클수록 좋음 (실제 이동 반영) |
| **d_rel_jitter** | per-sample d_rel 차분의 RMS | ↓ 작을수록 좋음 (거리 gain 안정) |
| conf_mean | detection confidence 평균 | ↑ 클수록 좋음 |
| conf_low_pct | conf < 0.35 비율 (%) | ↓ 작을수록 좋음 |
| dist_std_m | dist_m 표준편차 (m) | 참고 지표 |

`d_rel_jitter = sqrt(mean(diff(d_rel)^2))` — 거리 기반 gain/LPF 변화의 떨림 정도.
`az_range_deg` — 오디오에서 좌우 이동이 얼마나 넓게 들리는지의 직접 지표.

---

## 4. 전체 결과

### 4-1. car-5 (직선 고속 이동)

| 조건 | az_range° | d_jitter | conf | <35% | dist_std |
|-----|----------:|--------:|-----:|-----:|--------:|
| **BT_s1** | **52.0** | 0.00876 | 0.596 | 1.3% | 1.279 |
| DINO_s1 | 50.7 | 0.00882 | 0.434 | **50.8%** | 1.449 |
| BT_s3 | 50.1 | **0.00837** | 0.573 | 2.6% | 1.287 |
| DINO_s3 | 43.1 | 0.00846 | 0.581 | 3.6% | 1.289 |
| **AK_s5** | 43.3 | **0.00744** | **0.604** | **0.0%** | 1.154 |
| AK_s1 | 41.0 | 0.00878 | **0.693** | **0.0%** | 1.239 |
| AUTO_s1 | 41.0 | 0.00878 | **0.693** | **0.0%** | 1.239 |
| AK_s3 | 36.9 | 0.00984 | 0.599 | **0.0%** | 1.437 |

**분석**:
- BT_s1 az_range 52.0° 최대 → 직선 고속 차량은 ByteTrack이 우수
- DINO_s1: az range는 BT_s1에 근접하나 conf<35% 50.8% → 탐지 불안정
- AK_s5: jitter 최소(0.00744), conf 안정(0.0% low) → 오디오 품질 균형 최적
- AUTO_s1 = AK_s1 (car는 속도가 adaptive_k 선택 기준 아래로 판정)

### 4-2. dog-1 (빠르고 방향 전환 많음)

| 조건 | az_range° | d_jitter | conf | <35% | dist_std | 비고 |
|-----|----------:|--------:|-----:|-----:|--------:|------|
| **AK_s3** | **44.9** | 0.01271 | 0.486 | 0.8% | 0.994 | ★ jump-reject fix 후 |
| BT_s1 | 44.5 | 0.01219 | 0.340 | 86.6% | 1.132 | conf 불안정 |
| DINO_s1 | 44.5 | 0.01219 | 0.340 | 86.6% | 1.132 | BT와 동일 궤적 |
| AUTO_s1 | 44.5 | 0.01219 | 0.340 | 86.6% | 1.132 | auto→bytetrack 선택 |
| AK_s1 | 36.7 | 0.01595 | 0.505 | 0.7% | 1.284 | |
| BT_s3 | 26.2 | 0.01109 | 0.297 | 97.7% | 1.025 | stride 3 + BT 조합 불안 |
| AK_s5 | 11.3 | 0.01371 | 0.481 | 1.4% | 1.121 | stride 너무 커서 실패 |

**분석**:
- AK_s3: jump-reject fix 후 12.7° → 44.9°로 대폭 개선 (BT_s1 수준)
- BT 계열: conf<35% 86.6~97.7% → ByteTrack 저신뢰 association 다수, 그러나 az range는 유지
- DINO_s1 = BT_s1 (AUTO가 bytetrack 선택 → DINO도 내부적으로 동일 path)
- AK_s5: stride=5 + 빠른 개 → unlock이 느려서 최악

### 4-3. motorcycle-3 (중속 직선+곡선)

| 조건 | az_range° | d_jitter | conf | <35% | dist_std |
|-----|----------:|--------:|-----:|-----:|--------:|
| **AK_s5** | **48.2** | **0.01219** | 0.498 | **1.6%** | 2.462 |
| BT_s1 | 46.6 | 0.01363 | 0.474 | 42.0% | 2.352 |
| DINO_s1 | 46.6 | 0.01363 | 0.474 | 42.0% | 2.352 |
| AK_s1 | 46.4 | 0.01377 | **0.518** | **0.2%** | 2.213 |
| AUTO_s1 | 46.4 | 0.01377 | **0.518** | **0.2%** | 2.213 |
| BT_s3 | 46.0 | 0.01250 | 0.406 | 60.0% | 2.311 |
| DINO_s3 | 41.8 | 0.01232 | 0.484 | 26.1% | 2.351 |
| AK_s3 | 40.6 | 0.01327 | 0.494 | 0.7% | 2.552 |

**분석**:
- AK_s5: az range 최대 + jitter 최소 + conf 안정 → 삼박자
- BT 계열: az range는 준수하나 conf<35% 42~60% → detection 불안정
- DINO = BT (middleware 동일)
- AUTO_s1 = AK_s1 (중속이므로 adaptive_k 선택됨)

---

## 5. 클립별 종합 랭킹 (composite = az_norm↑ + d_jitter_norm↓)

| clip | #1 | #2 | #3 |
|------|----|----|-----|
| car-5 | BT_s3 (1.49) | BT_s1 (1.45) | AK_s5 (1.42) |
| dog-1 | AK_s3 (1.76) | BT_s1 (1.76) | AUTO_s1 (1.76) |
| motorcycle-3 | AK_s5 (2.00) | BT_s3 (1.51) | DINO_s3 (1.07) |

---

## 6. 권장 트래커 설정

| 물체 타입 | 속도 | 권장 | 이유 |
|---------|-----|------|------|
| 차량, 빠른 직선 이동 | fast | **BT_s3** | az range 최대, jitter 낮음, conf 안정 |
| 개, 사람 (방향전환 많음) | fast | **AK_s3** | fix 후 az range BT급, conf 압도적 안정 |
| 오토바이, 말 (중속) | medium | **AK_s5** | az range+jitter 균형 최고, conf 안정 |
| 클래스 불명 / 범용 | - | **AUTO_s1** | 속도 자동 판정 (car→BT, dog→BT, moto→AK) |

> ⚠️ **DINO**: car에서 az=50.7°로 우수하나 conf<35%=50.8% → 실용 불가
> ⚠️ **BT_s3 dog**: conf<35%=97.7% → 거의 모든 프레임 저신뢰

---

## 7. 핵심 버그 수정: Jump Reject 영구 고착

### 7-1. 버그 내용

**파일**: `vid2spatial_pkg/v2_spatial_tracker.py` — `_track_adaptive_k()`

**증상**: dog-1에서 AK tracker가 frame 3에서 az=-9°로 고착, 전체 클립 동안 이동 없음
→ az_range 13° (GT: ~44°)

**근본 원인** (코드 분석):

```python
# jump reject 발생 시 (line 373-380, 수정 전)
if vel > self.max_velocity_px:
    jump_reject_count += 1
    current_k = 1           # 다음 프레임 즉시 재탐지
    keyframes.append(..., last_kf 위치 반복)
    frames_since_detect = 0
    continue
    # ← last_cx, last_cy 갱신 없음!
```

- jump reject 후 `current_k=1` → 다음 프레임 즉시 재탐지
- 그러나 `last_cx/last_cy`는 여전히 jump reject 직전 값 유지
- 빠른 개: YOLO가 실제 위치에서 탐지하면 또 vel > max_velocity_px → 또 reject
- **결과**: 영원히 frame 3 위치에서 벗어나지 못함

### 7-2. 수정 내용

```python
# 추가 변수 (line 335-336)
consecutive_jump_rejects = 0
MAX_CONSECUTIVE_JUMP_REJECTS = 5   # 5번 연속 reject → unlock

# jump reject 블록 수정 (line 376-391)
if vel > self.max_velocity_px:
    jump_reject_count += 1
    consecutive_jump_rejects += 1
    if consecutive_jump_rejects >= MAX_CONSECUTIVE_JUMP_REJECTS:
        # Unlock: 물체가 일관되게 멀리 이동 → 위치 초기화
        last_cx, last_cy = None, None
        consecutive_jump_rejects = 0
        # Fall through: 이번 detection을 신뢰 가능한 새 기준점으로 수용
    else:
        current_k = 1
        keyframes.append(..., last_kf 위치 반복)
        frames_since_detect = 0
        continue

# 성공적 accept 시 카운터 리셋 (line 412)
consecutive_jump_rejects = 0
```

### 7-3. 효과

| 조건 | 수정 전 az_range | 수정 후 az_range | 개선폭 |
|-----|---------------:|---------------:|------:|
| AK_s1 (dog) | 13.2° | **36.7°** | +23.5° |
| AK_s3 (dog) | 12.7° | **44.9°** | +32.2° |
| AK_s5 (dog) | 5.0° | 11.3° | +6.3° |

AK_s3이 이제 BT_s1(44.5°)과 동급 az range를 내면서 conf 안정성은 훨씬 우위.

### 7-4. AK_s5 dog 여전히 낮은 이유

stride=5이면 YOLO detection이 5프레임마다만 실행.
5번 consecutive reject가 쌓이려면 최소 25프레임이 필요하고,
그 동안 stuck position이 오디오에 반영됨 → unlock이 느림.
속도가 매우 빠른 물체에는 stride≤3이 필수.

---

## 8. 파일 위치

```
test/tracker_compare/
├── car-5__AK_s1/     traj.json + output.binaural.wav + overlay.mp4
├── car-5__AK_s3/     ...
├── car-5__AK_s5/     ...
├── car-5__BT_s1/     ...
├── car-5__BT_s3/     ...
├── car-5__DINO_s1/   ...
├── car-5__DINO_s3/   ...
├── car-5__AUTO_s1/   ...
├── dog-1__*/         (AK 계열: jump-reject fix 적용 후 결과)
├── motorcycle-3__*/  ...
└── tracker_compare_raw.json   (dog-1 AK 재실행 결과)

test/tracker_listen/
├── car-5__BT_s1.binaural.wav     ← car 청취용 베스트
├── car-5__BT_s3.binaural.wav
├── car-5__AK_s5.binaural.wav
├── dog-1__AK_s3.binaural.wav     ← dog 청취용 베스트 (fix 후)
├── dog-1__AK_s1.binaural.wav
├── dog-1__BT_s1.binaural.wav
├── motorcycle-3__AK_s5.binaural.wav  ← moto 청취용 베스트
├── motorcycle-3__AK_s1.binaural.wav
└── ...
```

---

## 9. 실행 방법

```bash
# 특정 clips × conditions 재실행
CUDA_VISIBLE_DEVICES=1 python test/run_tracker_compare.py \
    --clips dog-1 car-5 motorcycle-3 \
    --conds AK_s3 BT_s1 AK_s5 \
    --skip_done

# 결과 확인 (traj.json에서 직접 계산)
python -c "
import json, math, numpy as np
data = json.load(open('test/tracker_compare/dog-1__AK_s3/traj.json'))
azs = [math.degrees(f['az']) for f in data['frames']]
print(f'az_range={max(azs)-min(azs):.1f}°')
"
```

---

*작성: 2026-02-24, vid2spatial_v2 tracker comparison*

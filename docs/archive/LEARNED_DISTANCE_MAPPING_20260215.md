# Learned Distance-to-Audio Parameter Mapping

**Date**: 2026-02-15
**Status**: Implemented (default disabled, `use_learned_mapping=False`)

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Baseline: Hardcoded Formulas](#2-baseline-hardcoded-formulas)
3. [Approach: Data-Driven Curve Fitting](#3-approach-data-driven-curve-fitting)
4. [Implementation](#4-implementation)
5. [Training](#5-training)
6. [Results](#6-results)
7. [Bug Fix: dist_s to d_rel_s](#7-bug-fix-dist_s-to-d_rel_s)
8. [Limitations and Honest Assessment](#8-limitations-and-honest-assessment)
9. [Usage](#9-usage)
10. [File Manifest](#10-file-manifest)

---

## 1. Motivation

vid2spatial의 거리 기반 오디오 렌더링은 3가지 파라미터를 제어한다:

| Parameter | Role |
|-----------|------|
| **Gain** (linear) | 거리에 따른 음량 감쇠 |
| **LPF cutoff** (Hz) | 거리에 따른 고주파 감쇠 (공기 흡수 모사) |
| **Wet mix** | 거리에 따른 잔향 비율 |

기존 구현은 수동으로 설계한 수식(선형/로그)을 사용하며, 실제 공간 오디오 데이터와의
정합성이 검증되지 않았다. text2hoa 프로젝트에서 축적된 17,252개의 레이블 데이터를
활용하여 데이터 기반 커브로 교체할 수 있는지 실험하였다.

---

## 2. Baseline: Hardcoded Formulas

`foa_render.py`의 기존 매핑 (d_rel = normalized distance [0, 1]):

```
gain_lin = 1.0 - dist_gain_k * 0.7 * d_rel
         = 1.0 at d_rel=0, 0.3 at d_rel=1

lpf_hz   = exp(log(8000) * (1-d_rel) + log(800) * d_rel)
         = 8000 Hz at d_rel=0, 800 Hz at d_rel=1

wet      = wet_min + (wet_max - wet_min) * d_rel
         = 0.05 at d_rel=0, 0.35 at d_rel=1
```

이 수식들의 문제점:
- 선형 가정: 실제 음향 감쇠는 inverse-square law에 가까움
- LPF 커브: 공기 흡수 계수는 주파수와 거리에 비선형적으로 의존
- Wet 커브: 실내 음향에서 직접음/반사음 비율은 거리의 제곱에 비례

---

## 3. Approach: Data-Driven Curve Fitting

### 3.1 Data Source

text2hoa의 학습 데이터 (`text2spatial_v3_boosted4.jsonl`, 17,252 records):

```json
{
  "dist_m": 2.5,        // 거리 (meters)
  "gain_db": -6.0,      // 감쇠 (dB)
  "wet_mix": 0.3,       // 잔향 비율
  "drr_db": 8.0         // Direct-to-Reverberant Ratio
}
```

### 3.2 Preprocessing

```
d_rel     = clip((dist_m - 0.6) / 5.4, 0, 1)    # 0.6m~6.0m -> [0,1]
gain_lin  = 10^(gain_db / 20)                      # dB -> linear
wet       = wet_mix                                 # direct use
lpf       = DRR-derived curve (see training script)
```

### 3.3 Two Model Types

**Polynomial (degree 3)** - 기본 모드:
- `numpy.polynomial.polynomial.polyfit(d_rel, target, deg=3)`
- gain, wet: 직접 fitting
- lpf: `log(lpf)`에 fitting 후 `exp()` 적용
- 추론 시간: ~1ms (전체 오디오 어레이)

**MLP (2-layer)** - 옵션:
- `Linear(1→32) + ReLU + Linear(32→3)`
- Per-column target normalization으로 학습 후 denorm을 output layer에 fold
- PyTorch로 학습, numpy로 추론 (torch 불필요)

---

## 4. Implementation

### 4.1 New Files

| File | Description |
|------|-------------|
| `vid2spatial_pkg/distance_model.py` | DistanceParamModel 클래스 (poly + MLP inference) |
| `tools/train_distance_model.py` | 학습 스크립트 (standalone) |
| `weights/distance_params_v1.npz` | 학습된 계수 (poly + MLP) |
| `weights/distance_model_diagnostic.png` | 진단 플롯 |

### 4.2 Modified Files

**`vid2spatial_pkg/config.py`** — SpatialConfig에 2개 필드 추가:
```python
use_learned_mapping: bool = False       # 학습된 매핑 사용 여부
learned_model_path: Optional[str] = None  # None = default weights
```

**`vid2spatial_pkg/foa_render.py`** — 3개 함수에 분기 추가:
- `apply_distance_gain_lpf()`: gain/LPF에 `model.predict_gain()`, `model.predict_lpf()` 분기
- `build_wet_curve_from_dist_occ()`: wet에 `model.predict_wet()` 분기
- `render_foa_from_trajectory()`, `render_binaural_from_trajectory()`: 파라미터 전달

**`vid2spatial_pkg/pipeline.py`** — config 전달 + 버그 수정 (Section 7 참고)

### 4.3 DistanceParamModel API

```python
from vid2spatial_pkg.distance_model import DistanceParamModel, get_distance_model

# Singleton cache (returns None if weights missing)
model = get_distance_model()

# Or explicit load
model = DistanceParamModel.load("weights/distance_params_v1.npz")

# Predict all 3 params
gain_lin, lpf_hz, wet = model.predict(d_rel_array)

# Individual predictions
gain = model.predict_gain(d_rel)   # [0.3, 1.1]
lpf  = model.predict_lpf(d_rel)   # [200, 20000] Hz
wet  = model.predict_wet(d_rel)    # [0.0, 1.0]
```

---

## 5. Training

### 5.1 Commands

```bash
cd /home/seung/mmhoa/vid2spatial
/home/seung/miniforge3/bin/python3 tools/train_distance_model.py
```

### 5.2 Training Process

1. **Stage A**: Polynomial fit (numpy polyfit, degree 3)
2. **Stage B**: MLP training (PyTorch, 50 epochs, lr=1e-3, MSE loss)
   - Per-column normalization: targets scaled to [0,1] for balanced loss
   - Denorm folding: `W2_out = W2_norm * t_range`, `b2_out = b2_norm * t_range + t_min`
3. **Stage C**: Evaluation (80/20 train/val split)
4. **Stage D**: Save weights + diagnostic plot

### 5.3 MLP Scale Imbalance Fix

첫 학습 시 MLP이 실패한 원인: `log(lpf)` 값 (~7-8)이 gain/wet (~0-1)보다 MSE에서
과도하게 지배. Per-column normalization으로 해결:

```
t_norm = (t - t_min) / t_range   # 학습 시 [0,1]로 정규화
W2_denorm = W2_norm * t_range    # 추론 시 output layer에 fold
b2_denorm = b2_norm * t_range + t_min
```

---

## 6. Results

### 6.1 MAE Comparison (Validation Set, 80/20 Split)

| Metric | Hardcoded | Poly-3 | MLP | Poly Improvement |
|--------|-----------|--------|-----|------------------|
| **Gain MAE** | 0.1281 | **0.0810** | 0.0840 | **-36.8%** |
| **Wet MAE** | 0.3110 | **0.1990** | 0.2102 | **-36.0%** |

Poly가 MLP보다 약간 더 좋은 성능. 1차원 입력에서 3차 다항식이 충분히 표현력을 가짐.

### 6.2 Curve Comparison (Key d_rel Points)

| d_rel | Gain (HC→Poly) | LPF (HC→Poly) | Wet (HC→Poly) |
|-------|----------------|----------------|----------------|
| 0.0 | 1.00 → 0.74 | 8000 → 4362 Hz | 0.05 → 0.20 |
| 0.5 | 0.65 → 0.71 | 2529 → 1829 Hz | 0.20 → 0.51 |
| 1.0 | 0.30 → 0.56 | 800 → 1689 Hz | 0.35 → 0.48 |

주요 차이점:
- **Gain**: 근거리에서 더 낮고(0.74 vs 1.0), 원거리에서 더 높음(0.56 vs 0.3) → 더 완만한 감쇠
- **LPF**: 전체적으로 더 낮은 cutoff → 더 적극적인 고주파 감쇠
- **Wet**: 근거리부터 0.20으로 높고, 거리에 따른 변화폭이 작음 → 항상 잔향 존재

### 6.3 E2E Verification

두 모드 모두 기존 trajectory JSON과 정상적으로 FOA 출력 생성 확인.

---

## 7. Bug Fix: dist_s to d_rel_s

### 발견 위치
`pipeline.py` L338 (구현 시점 기준)

### 문제
```python
# Before (BUG):
wet = build_wet_curve_from_dist_occ(
    dist_s, occ_s, ...   # dist_s = meters (e.g., 3.0m)
)
```

`build_wet_curve_from_dist_occ()`는 입력을 `clip(0, 1)`으로 처리하므로,
1m 이상의 거리는 모두 1.0으로 클리핑 → 항상 최대 wet 값 적용.

### 수정
```python
# After (FIXED):
wet = build_wet_curve_from_dist_occ(
    d_rel_s, occ_s, ...  # d_rel_s = normalized [0, 1]
)
```

### 영향
이전까지 모든 vid2spatial 출력에서 reverb wet이 과도하게 적용되고 있었음.
거리와 무관하게 거의 최대 잔향이 일괄 적용되는 상태였기 때문에,
사실상 거리 기반 잔향 제어가 작동하지 않은 것과 같은 상태.

---

## 8. Limitations and Honest Assessment

### 8.1 Cross-Domain Transfer 문제

학습 데이터가 **text2hoa** 출신이라는 근본적 한계:

| 문제 | 설명 |
|------|------|
| **Domain mismatch** | text2hoa labels는 텍스트 설명에서 유래. vid2spatial의 시각적 거리와 1:1 대응 아님 |
| **Low R²** | 거리만으로 gain/wet 분산의 ~10%만 설명 (R² ≈ 0.1). 나머지 90%는 음원 유형, 환경 등에 의존 |
| **No perceptual validation** | 청취 테스트 미실시. 수치적 MAE 개선이 지각적 품질 개선을 보장하지 않음 |
| **No vid2spatial-specific GT** | vid2spatial 도메인에서의 정답 레이블이 존재하지 않음 |

### 8.2 논문 사용 적절성

**주요 기여(main contribution)로는 부적절**:
- Cross-domain transfer의 이론적 정당성 부족
- 청취 테스트 결과 없음
- R² = 0.1로 모델 설명력이 매우 낮음

**활용 가능한 방법**:
- Engineering improvement로 기술 (System Description 섹션)
- Future work 방향으로 제시 ("향후 vid2spatial 도메인 전용 GT를 구축하면...")
- 수동 수식 대비 데이터 기반 접근의 잠재력을 보여주는 예시

### 8.3 향후 개선 방향

1. **vid2spatial 전용 GT 구축**: 실제 녹음된 공간 오디오에서 거리-파라미터 쌍 추출
2. **Perceptual loss**: MSE 대신 MUSHRA/ABX 청취 테스트 기반 최적화
3. **Multi-input model**: 거리뿐 아니라 음원 유형, 공간 크기 등을 입력으로 확장
4. **Online adaptation**: 사용자 피드백으로 커브를 실시간 조정

---

## 9. Usage

### 9.1 기본 사용 (hardcoded, 기존 동작)

```bash
python run.py --video input.mp4 --audio input.wav --out-foa output.foa.wav
```

### 9.2 학습된 매핑 활성화

```bash
python run.py --video input.mp4 --audio input.wav --out-foa output.foa.wav \
    --use-learned-mapping
```

### 9.3 커스텀 가중치

```bash
python run.py ... --use-learned-mapping \
    --learned-model-path /path/to/custom_weights.npz
```

### 9.4 Python API

```python
from vid2spatial_pkg.config import PipelineConfig, SpatialConfig

config = PipelineConfig(
    video_path="input.mp4",
    audio_path="input.wav",
    spatial=SpatialConfig(
        use_learned_mapping=True,
        learned_model_path=None,  # None = default weights
    ),
    output=OutputConfig(foa_path="output.foa.wav"),
)
```

---

## 10. File Manifest

```
vid2spatial/
├── vid2spatial_pkg/
│   ├── distance_model.py      # [NEW] DistanceParamModel (poly + MLP)
│   ├── config.py               # [MOD] +use_learned_mapping, +learned_model_path
│   ├── foa_render.py           # [MOD] learned mapping branches in gain/LPF/wet
│   └── pipeline.py             # [MOD] config pass-through + dist_s→d_rel_s bug fix
├── tools/
│   └── train_distance_model.py # [NEW] Training script (standalone)
├── weights/
│   ├── distance_params_v1.npz  # [NEW] Trained coefficients
│   └── distance_model_diagnostic.png  # [NEW] Diagnostic plot
└── docs/
    └── LEARNED_DISTANCE_MAPPING_20260215.md  # This document
```

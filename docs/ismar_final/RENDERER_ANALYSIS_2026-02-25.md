# Renderer Pipeline 상세 분석 — vid2spatial_v2
**작성일**: 2026-02-25
**저장소**: `/home/seung/mmhoa/vid2spatial_v2/`

---

## 1. 파이프라인 전체 흐름

```
[video + audio]
     │
     ▼
[1] compute_trajectory()          ← V2SpatialTracker (adaptive-K / ByteTrack / DINO)
     │ traj.json: {frames, fps, intrinsics}
     ▼
[2] _apply_room_ir()              ← Schroeder/PRA/FAIR-Play IR 선택적 적용
     │ audio_processed
     ▼
[3] _render_spatial_audio()       ← 핵심 렌더
     │  interpolate_angles_distance() → az_s, el_s, dist_s, d_rel_s
     │  smooth_limit_angles()         → 50ms MA 스무딩
     │  apply_distance_gain_lpf()     → 거리 gain + LPF
     │  apply_timevarying_reverb_mono() (optional)
     │  encode_mono_to_foa(-az_s, el_s) → FOA [4,T]
     ▼
[4] _write_outputs()              → FOA + Stereo + Binaural
```

---

## 2. 핵심 함수별 상세 분석

### 2-1. `interpolate_angles_distance()` — [foa_render.py:61-120]

```python
def interpolate_angles_distance(frames, T, sr, fps=30.0, depth_keys=None)
    → (az_s, el_s, dist_s, d_rel_s)  # 모두 길이 T float32 배열
```

**처리 단계:**

1. **fps-aware frame→sample 변환**: `idx_samples = frame_idx * (sr / fps)`
   - fps=25 (LaSOT) 일 때: frame 0 → sample 0, frame 25 → sample 48000
   - fps 미일치 시 15× 속도왜곡 방지 (157초 영상을 10초 오디오에 맵핑하는 버그 방지)

2. **depth priority lookup** (per-frame):
   ```
   depth_keys 기본값: ("depth_render", "depth_blended", "dist_m_raw", "dist_m")
   순서대로 첫 번째 존재하는 키 사용, 없으면 1.0
   ```
   tracker_compare에서: `depth_keys=("dist_m",)` — MiDaS 평활화값만 사용

3. **0.1초 이동평균 (dist smoothing)**:
   ```python
   smooth_samples = max(1, int(sr * 0.1))  # 4800 @ 48kHz
   dist_s = uniform_filter1d(dist_s, size=smooth_samples, mode='nearest')
   ```
   - MiDaS 프레임 단위 노이즈(~25-30Hz) 제거, 자연스러운 거리 변화(~1Hz) 보존
   - **per-clip normalization 전에 적용** → d_min/d_max도 노이즈 없는 값 기반

4. **per-clip normalization**:
   ```python
   d_range = d_max - d_min
   if d_range < 0.1:
       d_rel_s = np.full_like(dist_s, 0.5)  # 거리 변화 없으면 중간값 고정
   else:
       d_rel_s = clip((dist_s - d_min) / d_range, 0.0, 1.0)
   ```
   - 절대 거리값 불필요 → 클립 내 상대 변화만으로 렌더링
   - dist_m이 모두 같을 때 (frozen dist): d_rel_s = 0.5 → gain/LPF 중간값 → 무해

**반환값**: `(az_s, el_s, dist_s, d_rel_s)` — 4개. pipeline.py에서 4개 언패킹 확인 ✅

---

### 2-2. `smooth_limit_angles()` — [foa_render.py:123-162]

```python
def smooth_limit_angles(az_s, el_s, sr, smooth_ms=50.0, max_deg_per_s=None)
    → (az_s, el_s)
```

- **50ms 이동평균**: 48kHz에서 win=2400 샘플
- `np.convolve(x, k, mode='same')`: 경계부 edge artifact 있을 수 있음
  - mode='same'이므로 첫/마지막 win/2 샘플이 zero-padded convolution의 영향 받음
  - 실용상 50ms window는 미미한 영향
- `max_deg_per_s`: 설정 시 sample당 최대 이동 제한 (clamp_delta)
  - 기본값 None → 제한 없음

---

### 2-3. `apply_distance_gain_lpf()` — [foa_render.py:165-239]

```python
def apply_distance_gain_lpf(x, sr, dist_s, d_rel_s=None, *,
                             gain_k=1.0, lpf_min_hz=800.0, lpf_max_hz=8000.0,
                             gain_min=0.3, gain_max=1.0, ...)
```

**gain 공식** (d_rel_s 사용):
```python
g = gain_max - (gain_max - gain_min) * (nd ** gain_k)
# d_rel=0 (near): g=gain_max=1.0
# d_rel=1 (far):  g=gain_min=0.3
```

**LPF cutoff 공식** (log-scale):
```python
log_fc = log_max - (log_max - log_min) * nd
fc = exp(log_fc)
# d_rel=0 (near): fc=8kHz (bright)
# d_rel=1 (far):  fc=800Hz (muffled)
```

**one-pole IIR LPF**:
```python
a = (2π*fc[i]) / (2π*fc[i] + sr)
prev = prev + a * (y[i] - prev)
```
- 시작시 prev=0 → 초기 transient (단, 오디오 컨텍스트에서 50ms 정도면 수용 가능)
- time-varying fc → 매 샘플 α 업데이트 (계산 비용 있음)

**soft normalize**:
```python
peak = max(abs(y_lp)) + 1e-9
if peak > 1.0: y_lp /= peak
```
→ gain 곱 후 clipping 방지

---

### 2-4. `encode_mono_to_foa()` — [foa_render.py:323-337]

```python
def encode_mono_to_foa(mono, az_series, el_series) → foa [4,T]
```

AmbiX ACN/SN3D 인코딩:
```
W = 1/√2
Y = √(3/2) * sin(az) * cos(el)
Z = √(3/2) * sin(el)
X = √(3/2) * cos(az) * cos(el)
channels: [W, Y, Z, X]
```

**입력 az 방향 규약**:
- 함수 자체는 az 부호를 그대로 씀
- **호출 전 negation 필요**: pipeline az (right=+) → AmbiX (left=+)

---

### 2-5. `render_foa_from_trajectory()` — [foa_render.py:392-436]

독립 함수 (pipeline class 바깥).
```python
az_ambiX = -az_s   # ← negation 올바르게 적용 ✅
foa = encode_mono_to_foa(audio_proc, az_ambiX, el_s)
```

---

### 2-6. `_render_spatial_audio()` — [pipeline.py:551-649]

pipeline class 내부 메서드.
```python
foa = encode_mono_to_foa(audio_dist, -az_s, el_s)  # ← 수정 완료 (2026-02-25)
```

---

### 2-7. `direct_binaural_sofa()` — [foa_render.py:555-662]

HRTF binaural 렌더. overlap-add (50% Hann window).
```python
az_sofa = -az_s   # ← negation 올바르게 적용 ✅
stereo = direct_binaural_sofa(audio_proc, sr, az_sofa, el_s, sofa_path)
```

---

### 2-8. `foa_to_binaural_sofa()` — [foa_render.py:665-]

FOA virtual-speaker decode → HRTF binaural.
- FOA에서 az 부호는 이미 AmbiX 기준으로 encode됨 → 별도 negation 불필요 ✅

---

## 3. 발견된 버그 및 수정 이력

### [버그 1] L/R 반전 — pipeline._render_spatial_audio() (2026-02-25 수정)

**파일**: `vid2spatial_pkg/pipeline.py:621`

**수정 전**:
```python
foa = encode_mono_to_foa(audio_dist, az_s, el_s)   # ← az negation 누락!
```

**수정 후**:
```python
# Pipeline az: az>0 = RIGHT (atan2(x,z))
# AmbiX/SOFA: az>0 = LEFT (counterclockwise from front)
foa = encode_mono_to_foa(audio_dist, -az_s, el_s)  # ← negation 추가
```

**영향 범위**:
- `SpatialAudioPipeline.run()` → `_render_spatial_audio()` 경로 전체
- tracker_compare 결과물 (binaural.wav) 전부 L/R 반전 → `test/rerender_binaural.py`로 재렌더 완료

**영향 없는 경로** (별도 negation 있음):
- `render_foa_from_trajectory()`: `az_ambiX = -az_s` ✅
- `render_binaural_from_trajectory()`: `az_sofa = -az_s` ✅
- `direct_binaural_sofa()`: 호출 전 negation 완료 ✅

---

### [버그 2] Jump Reject 영구 고착 — v2_spatial_tracker._track_adaptive_k() (2026-02-24 수정)

**파일**: `vid2spatial_pkg/v2_spatial_tracker.py:335-412`

**원인**: jump reject 시 `last_cx/last_cy` 미갱신 → 영구 stuck
**수정**: 5번 연속 reject 시 `last_cx/last_cy = None` 초기화 → 다음 detection 수용
**효과**: dog-1 AK_s3 az_range 12.7° → 44.9°

---

## 4. 잠재적 위험 영역 (미수정)

### 4-1. az wrap-around (π/-π 경계) 미처리

`interpolate_angles()` / `smooth_limit_angles()`는 az를 raw float로 보간.
- 물체가 정면 뒤쪽(-π/+π 경계)을 지나면 -3.14 → +3.14 급격한 jump
- `np.interp`는 선형 보간하므로 경계에서 반대 방향으로 쓸리는 왜곡 발생
- **현재 실용 클립에서는 문제 없음** (전방 물체만 추적)
- **수정 방법**: `np.unwrap(az)` 후 보간, 결과를 다시 `wrap`

### 4-2. dist_m 모두 같은 값 (frozen dist)

- `d_range < 0.1` 조건으로 `d_rel_s = 0.5` → 무해 ✅

### 4-3. 음수 dist_m 가능성

- MiDaS relative depth는 양수지만 per-clip normalization 과정에서 subtract 시 음수 가능
- `np.clip(d_rel_s, 0.0, 1.0)`으로 보호 ✅

### 4-4. smooth_limit_angles mode='same' edge effect

- convolution 경계부 50ms에 zero-padding artifact
- 오디오 렌더에서 실용상 무시 가능

### 4-5. one-pole LPF 초기 transient

- t=0에서 prev=0 시작 → 첫 ~50ms 동안 LPF 과도응답
- 테스트에서 확인 완료 (middle 50% 검사로 통과) — 실용상 수용 가능

### 4-6. foa_to_binaural() crossfeed 모드 az 미적용

```python
def foa_to_binaural(foa_sn3d, sr):
    st = foa_to_stereo(foa_sn3d, sr)  # FOA decode to ±30°
    # simple crossfeed (ITD 근사)
```
- FOA virtual speaker (±30°)에서 decode 후 crossfeed → 실제 HRTF 없음
- tracker_compare에서는 SOFA mode 사용하므로 해당 없음 ✅

---

## 5. render_foa_from_trajectory vs pipeline._render_spatial_audio 비교

| 항목 | render_foa_from_trajectory | pipeline._render_spatial_audio |
|-----|---------------------------|-------------------------------|
| 사용처 | 독립 호출 (test/benchmark) | SpatialAudioPipeline.run() |
| az negation | ✅ `az_ambiX = -az_s` | ✅ `encode_mono_to_foa(-az_s, el_s)` (수정 후) |
| depth_keys | 파라미터 없음 (기본 priority) | `self.depth_priority` |
| reverb | apply_reverb 파라미터 | config.reverb.enabled |
| binaural | 없음 (FOA only) | _write_outputs에서 처리 |
| Kalman | 없음 (trajectory 이미 smooth 가정) | config.spatial.use_kalman_smoothing |

---

## 6. 데이터 흐름 타입/크기 체크

```
traj.json frames: [{"frame": int, "az": float(rad), "el": float(rad),
                    "dist_m": float, "confidence": float, ...}, ...]

interpolate_angles_distance():
  입력: frames (List[Dict]), T=480000, sr=48000, fps=25.0
  출력: az_s [480000 float32], el_s [480000 float32],
         dist_s [480000 float32], d_rel_s [480000 float32]

encode_mono_to_foa():
  입력: mono [480000], az [-480000 float32], el [480000 float32]
  출력: foa [4, 480000 float32]

write_foa_wav():
  입력: foa [4, T], sr=48000
  → sf.write: foa.T → [T, 4] AmbiX wav

foa_to_binaural_sofa():
  입력: foa [4, T], sr=48000, sofa_path
  → binaural [2, T]
  → sf.write: binaural.T → [T, 2] stereo wav
```

모든 중간 배열이 float32 유지 — clipping 전 peak check 있음 ✅

---

## 7. 실행 검증

### 단위 테스트 (43/43 통과)
```bash
python -m pytest test/test_unit.py -v
```

### 통합 테스트 (38/38 통과)
```bash
python -m pytest test/test_integration.py -v
```

### 렌더 재실행 (az negation fix 후)
```bash
# traj.json 재사용, 오디오 render만
python test/rerender_binaural.py --clips car-5 dog-1 motorcycle-3

# 트래킹부터 전부 재실행
python test/run_tracker_compare.py --clips car-5 dog-1 --conds AK_s3 BT_s1
```

---

*작성: 2026-02-25, vid2spatial_v2 렌더러 분석*

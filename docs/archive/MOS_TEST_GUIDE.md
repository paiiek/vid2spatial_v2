# MOS 청취 평가 운영 가이드

**vid2spatial v2 — Perceptual Evaluation Study**
**Date**: 2026-02-18
**목적**: 각 ablation 조건의 인지적 품질 차이를 MOS (Mean Opinion Score)로 측정

---

## 1. 개요

### 1.1 평가 목적

자동화 spatial audio 시스템의 품질을 주관적으로 평가한다. 4가지 ablation 조건(A-D)이 실제 청취 시 체감 차이를 만드는지 확인하고, 각 컴포넌트(learned mapping, Metric3D, visual room IR)의 기여를 측정한다.

### 1.2 평가 설계

| 항목 | 값 |
|------|----|
| 참가자 수 | 15~20명 (권장) |
| 클립 수 | 7개 |
| 조건 수 | 4개 (A, B, C, D) |
| 총 평가 항목 | 7 × 4 = **28개** |
| 예상 소요 시간 | **20~30분** |
| 청취 방식 | 헤드폰 필수 (binaural) |
| 평가 방식 | Blind (조건 레이블 미표시) |
| 순서 | 클립 내 조건 순서 참가자별 랜덤 |

### 1.3 4가지 조건 설명

| 조건 | 내부 이름 | 설명 |
|------|-----------|------|
| **(A) v1 baseline** | `A_v1baseline` | 하드코딩 gain/LPF curve, DA-V2 depth, reverb 없음 |
| **(B) +learned** | `B_learned` | **Learned** poly-3 distance mapping (v2 핵심 기여) |
| **(C) +M3D** | `C_m3d` | B + **Metric3D v2** depth (MAE 42% ↓) |
| **(D) +visual room** | `D_full` | C + **Visual room IR** (scene-adaptive reverb) |

### 1.4 3가지 MOS 차원

| 차원 | 약어 | 질문 (한국어) | 예상 주요 조건 |
|------|------|--------------|----------------|
| **Spatial Coherence** | MOS-S | 소리가 영상 속 물체 위치(좌우, 앞뒤, 거리)와 얼마나 잘 일치합니까? | C > B (M3D depth 효과) |
| **Perceptual Quality** | MOS-Q | 공간화된 오디오의 전반적인 음질이 얼마나 자연스럽고 좋습니까? | D > C (room IR 효과) |
| **Distance Perception** | MOS-D | 물체가 가까워지거나 멀어질 때 소리의 음량/음색 변화가 자연스럽게 느껴집니까? | B > A (learned mapping 효과) |

---

## 2. 시스템 설정

### 2.1 사전 요구사항

```bash
# Flask 설치
pip install flask

# 오디오 파일 존재 확인
ls /tmp/mos_binaural/*.wav | wc -l  # 28개여야 함
```

### 2.2 오디오 파일 생성 (없는 경우)

28개 binaural 파일을 생성한다. 각 파일명 형식: `{clip_id}_{condition_internal}_bin.wav`

```bash
# 각 클립 × 4 조건 × binaural 렌더링
# 예시: dog-14 클립, 조건 A
python /path/to/vid2spatial_v2/tools/run.py \
  --video data/lasot/dog/dog-14/*.jpg \  # 또는 mp4
  --audio data/lasot/dog/dog-14/audio.wav \
  --out-bin /tmp/mos_binaural/dog-14_A_v1baseline_bin.wav \
  --no-learned-mapping \  # 조건 A용
  --depth-backend auto    # DA-V2

# 조건별 옵션 차이:
# A: --no-learned-mapping --depth-backend auto --no-ir
# B: (default) --depth-backend auto --no-ir
# C: --depth-backend metric3d --no-ir
# D: --depth-backend metric3d --ir-backend visual
```

또는 기존 binaural 파일이 있다면:
```
/tmp/mos_binaural/
├── guitar-9_A_v1baseline_bin.wav
├── guitar-9_B_learned_bin.wav
├── guitar-9_C_m3d_bin.wav
├── guitar-9_D_full_bin.wav
├── dog-14_A_v1baseline_bin.wav
... (총 28개)
```

### 2.3 서버 시작

```bash
cd /home/seung/mmhoa/vid2spatial_v2/mos_test

# 기본 포트 (5050)
python app.py

# 포트 변경 또는 오디오 디렉터리 지정
python app.py --port 5050 --audio-dir /tmp/mos_binaural
```

서버 출력:
```
[mos] Audio files available: 28/28
[mos] Starting server on http://0.0.0.0:5050
[mos] Admin results: http://localhost:5050/admin/results
```

---

## 3. 참가자 진행 절차

### 3.1 평가 URL

참가자에게 아래 URL 제공:
```
http://<서버IP>:5050/
```

로컬 테스트:
```
http://localhost:5050/
```

### 3.2 참가자 흐름

```
/ (동의 및 설명)
    ↓ (동의 체크박스 + 시작 버튼)
/headphone-check (헤드폰 확인)
    - 4가지 체크리스트 수동 확인
    - 좌→우 스윕 음원 재생 (WebAudio API)
    ↓ (모두 확인 후)
/trial (반복, 총 28회)
    - 오디오 플레이어
    - MOS-S, MOS-Q, MOS-D 1~5점 평가
    - 전체 청취 후 제출 버튼 활성화
    ↓ (28번 완료)
/done (완료 화면)
    - 참가자 ID, 응답 수 표시
```

### 3.3 참가자 안내 문구 (예시)

> 이 실험은 헤드폰 착용이 필수입니다. 각 영상 클립을 보면서 오디오를 듣고, 3가지 항목에 대해 1~5점으로 평가해 주세요. 소리가 영상 속 물체 위치와 얼마나 자연스럽게 일치하는지가 핵심 평가 기준입니다.

---

## 4. 결과 수집 및 분석

### 4.1 실시간 결과 확인

관리자 URL:
```
http://localhost:5050/admin/results
```

조건별 평균 MOS 테이블 표시.

### 4.2 CSV 다운로드

```
http://localhost:5050/admin/export
```

또는 직접 파일 접근:
```bash
cat /home/seung/mmhoa/vid2spatial_v2/mos_test/results/responses.csv
```

CSV 컬럼:
```
timestamp, participant_id, clip_id, condition_id,
mos_s, mos_q, mos_d, listen_duration_s, replays
```

### 4.3 분석 실행

```bash
python /home/seung/mmhoa/vid2spatial_v2/mos_test/analyze_results.py \
  --csv /home/seung/mmhoa/vid2spatial_v2/mos_test/results/responses.csv
```

출력:
- 조건별 mean ± std 테이블 (MOS-S, MOS-Q, MOS-D)
- Wilcoxon signed-rank test (A vs B, B vs C, C vs D)
- 막대 그래프 (선택, matplotlib)

---

## 5. 통계 분석 방법

### 5.1 주요 가설

| 비교 | MOS 차원 | 귀무가설 |
|------|---------|---------|
| A vs B | MOS-D | H₀: 학습 mapping이 거리감에 차이 없음 |
| B vs C | MOS-S | H₀: Metric3D가 공간 일치도에 차이 없음 |
| C vs D | MOS-Q | H₀: Visual room IR이 음질에 차이 없음 |

### 5.2 통계 검정

- **Wilcoxon signed-rank test** (paired, 순서 척도)
  - `scipy.stats.wilcoxon(a_scores, b_scores, alternative='less')`
  - α = 0.05 (단측 검정: B > A)

- **참가자 수 및 검정력**
  - n=15: 효과 크기 d=0.5, α=0.05에서 검정력 ≈ 0.6
  - n=20: 검정력 ≈ 0.7 (권장)
  - 실제 효과가 크다면 (d≥0.8) n=10도 충분

### 5.3 기대 결과

| 비교 | 기대 방향 | 이유 |
|------|---------|------|
| MOS-D: B > A | ★★★ 가능성 높음 | Poly-3 mapping이 거리 dynamic range 감소, 더 자연스러운 변화 |
| MOS-S: C ≥ B | ★★ 가능성 있음 | Depth MAE 42% 감소가 공간 일치도에 반영 (효과 미약할 수 있음) |
| MOS-Q: D > C | ★★ 가능성 있음 | Reverb가 몰입감 향상 (단, energy 증가로 과도할 경우 역효과) |
| MOS-Q: D < C | ★ 역효과 가능 | 소형 실내 IR이 과도한 경우 (wet/dry = 1.3~3.7 before wet_mix) |

---

## 6. 파일 구조

```
mos_test/
├── app.py                    ← Flask 서버 (메인)
├── analyze_results.py        ← 결과 분석 스크립트
├── templates/
│   ├── index.html            ← 동의 및 설명 페이지
│   ├── headphone_check.html  ← 헤드폰 확인 (WebAudio sweep)
│   ├── trial.html            ← 개별 시험 평가 페이지
│   └── done.html             ← 완료 페이지
└── results/
    └── responses.csv         ← 모든 응답 누적 (자동 생성)
```

---

## 7. 기술 세부사항

### 7.1 세션 관리

- 각 참가자: `uuid4()[:8]` participant_id 발급
- Flask session: server-side cookie, `random.seed(participant_id)`로 재현 가능한 랜덤 순서
- 세션 만료: 브라우저 닫으면 새 세션 → 새 participant_id

### 7.2 오디오 전송

```
GET /audio/{filename}
→ send_from_directory(AUDIO_DIR, filename)
→ 브라우저 <audio> 태그로 스트리밍
```

파일 형식: WAV PCM_24, 48kHz, 2ch (binaural)

### 7.3 응답 제출

```
POST /submit-trial
Content-Type: application/json
Body: {
    "mos_s": 3,
    "mos_q": 4,
    "mos_d": 3,
    "listen_duration_s": 12.4,
    "replays": 1
}
```

- `listen_duration_s`: 실제 청취 시간 (JS 측정)
- `replays`: 재생 횟수 (0 = 한 번만 들음)
- 오디오 재생 완료 전까지 제출 버튼 비활성화

### 7.4 응답 검증 기준

분석 시 아래 응답은 제외 권장:
- `listen_duration_s < 클립길이 * 0.5`: 충분히 듣지 않음
- 모든 항목에 동일 점수 (예: 항상 3점): 무성의 응답 의심
- 총 소요 시간 < 10분: 너무 빠름

---

## 8. 보안 및 운영

### 8.1 공개 배포 시 주의

```python
# app.py 보안 설정 (필요 시)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# admin 페이지 보호 (선택)
@app.route("/admin/results")
def admin_results():
    if request.args.get('key') != 'YOUR_ADMIN_KEY':
        return "Unauthorized", 401
```

### 8.2 백업

```bash
# 결과 정기 백업
cp results/responses.csv results/responses_backup_$(date +%Y%m%d).csv
```

### 8.3 서버 안정성

```bash
# nohup으로 백그라운드 실행
nohup python app.py --port 5050 > mos_server.log 2>&1 &
echo $! > mos_server.pid

# 중지
kill $(cat mos_server.pid)
```

---

## 9. 결과 해석 기준

### 9.1 MOS 점수 해석

| 점수 | 품질 수준 | 해석 |
|------|-----------|------|
| 4.5 - 5.0 | 우수 (Excellent) | 완벽하게 자연스러움 |
| 4.0 - 4.4 | 좋음 (Good) | 약간의 개선 여지 있음 |
| 3.5 - 3.9 | 보통 (Fair) | 눈에 띄는 문제 없으나 자연스럽지 않음 |
| 3.0 - 3.4 | 나쁨 (Poor) | 명확한 문제 있음 |
| < 3.0 | 매우 나쁨 (Bad) | 수용 불가 |

*ITU-T P.800 기준 해석*

### 9.2 논문 작성 시 보고 형식

```
Table X: MOS listening test results (N=XX participants, 28 trials per participant)
                MOS-S           MOS-Q           MOS-D
(A) Baseline    X.XX ± X.XX     X.XX ± X.XX     X.XX ± X.XX
(B) +Learned    X.XX ± X.XX*    X.XX ± X.XX     X.XX ± X.XX*
(C) +M3D        X.XX ± X.XX*    X.XX ± X.XX     X.XX ± X.XX
(D) +VisRoom    X.XX ± X.XX     X.XX ± X.XX*    X.XX ± X.XX

* p < 0.05, Wilcoxon signed-rank test (vs. preceding condition)
```

---

*Generated: 2026-02-18 | vid2spatial v2 MOS Test Guide*

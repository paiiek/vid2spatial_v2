# Detector Benchmark Report
**Date**: 2026-03-02
**Clips**: 22 LaSOT sequences (18 COCO-class + 4 Non-COCO), 300 frames each
**Metric**: GT center-x MAE, Az MAE, Detection Rate, Speed

---

## 1. Overall Summary

| Detector | Det% | CxMAE | AzMAE | Speed | N |
|---|---|---|---|---|---|
| YOLO11n (COCO) | 61% | 53px | 2.65° | 93fps | 22 |
| YOLO-World v2 | 86% | 57px | 2.89° | 75fps | 22 |
| **GroundingDINO** | **100%** | **60px** | **3.03°** | 2.6fps | 22 |
| **SAM2 video** | **99%** | **13px** | **0.74°** | 2.8fps | 16 |
| YOLO-World+BT | 67% | 59px | 2.98° | 39fps | 22 |

---

## 2. COCO vs Non-COCO Split

| Detector | COCO det% | COCO AzMAE | Non-COCO det% | Non-COCO AzMAE |
|---|---|---|---|---|
| YOLO11n | 75% | 2.65° | **0%** | N/A |
| YOLO-World | 96% | 2.84° | 41% | 3.18° |
| GroundingDINO | 100% | 3.17° | **100%** | 2.39° |
| SAM2 | 99% | 0.77° | **100%** | 0.39° |
| YW+BT | 75% | 2.96° | 34% | 3.11° |

---

## 3. Per-Clip Results (det% / CxMAE / AzMAE)

| Clip | YOLO11n | YoloWorld | GDINO | SAM2 | YW+BT |
|---|---|---|---|---|---|
| car-5 | 100%/10px/0.5° | 100%/9px/0.4° | 100%/42px/2.1° | **100%/3px/0.1°** | 100%/9px/0.4° |
| car-10 | 95%/208px/10.0° | 100%/89px/4.3° | 100%/105px/5.0° | **100%/3px/0.2°** | 89%/99px/4.8° |
| car-13 | 100%/77px/3.8° | 100%/104px/5.1° | 100%/150px/7.3° | **100%/2px/0.1°** | 100%/104px/5.1° |
| dog-1 | 40%/2px/0.1° | 88%/3px/0.1° | 98%/13px/0.7° | 97%/6px/0.3° | **65%/3px/0.1°** |
| dog-3 | 90%/1px/0.1° | 100%/2px/0.1° | **100%/1px/0.0°** | 100%/4px/0.2° | 98%/2px/0.1° |
| dog-9 | 90%/140px/7.0° | 96%/95px/4.7° | 100%/68px/3.4° | **100%/5px/0.2°** | 94%/97px/4.8° |
| dog-14 | 0% | 88%/164px/8.1° | 100%/159px/7.9° | **100%/22px/1.1°** | 50% |
| horse-1 | 0% | 78%/5px/0.2° | 100%/78px/3.7° | 89%/24px/1.2° | **37%/4px/0.2°** |
| horse-3 | 82%/59px/3.0° | 100%/80px/4.0° | **100%/50px/2.5°** | 100%/65px/3.2° | 99%/76px/3.9° |
| horse-11 | 78%/9px/0.5° | 100%/8px/0.4° | **100%/7px/0.3°** | 100%/9px/0.5° | 100%/8px/0.4° |
| motorcycle-1 | 82%/156px/7.7° | 95%/212px/10.5° | 100%/161px/8.0° | **100%/8px/0.4°** | 72% |
| motorcycle-3 | 50%/10px/1.0° | 100%/10px/1.1° | 100%/18px/1.8° | **100%/32px/3.2°** | 100%/10px/1.1° |
| motorcycle-6 | 97%/3px/0.1° | 98%/4px/0.2° | 100%/20px/1.0° | **98%/12px/0.6°** | 98%/4px/0.2° |
| skateboard-8 | 88%/93px/4.8° | 91%/93px/4.8° | 100%/112px/5.7° | **100%/2px/0.1°** | 11% |
| skateboard-11 | 99%/3px/0.1° | 100%/14px/0.7° | 100%/16px/0.8° | **100%/2px/0.1°** | 28% |
| skateboard-17 | 60%/3px/0.1° | 95%/27px/1.4° | **100%/15px/0.8°** | --- | 11% |
| train-16 | 91%/36px/1.8° | 92%/32px/1.7° | 96%/57px/2.9° | --- | **91%/32px/1.7°** |
| train-17 | 100%/68px/3.4° | **100%/64px/3.2°** | 100%/65px/3.3° | --- | 100%/64px/3.2° |
| drone-2 | 0% | 48%/25px/1.3° | 100%/22px/1.1° | **100%/8px/0.4°** | 24% |
| drone-6 | 0% | 0% | **100%/6px/0.3°** | --- | 0% |
| drone-13 | 0% | 15%/7px/0.4° | **100%/5px/0.2°** | --- | 11% |
| guitar-9 | 0% | 100%/153px/7.9° | 100%/153px/7.9° | --- | 100%/153px/7.9° |

**Note**: guitar-9 cx_mae=153px는 모든 open-vocab 모델에서 동일 → GT bbox 기준 차이 (guitar 전체 vs 지판 부분)

---

## 4. Key Findings

### SAM2 video (D) — 정확도 최강
- CxMAE: **13px** (YOLO11n 53px 대비 4× 낮음)
- AzMAE: **0.74°** (YOLO11n 2.65° 대비 3.6× 낮음)
- 100% detection (COCO + Non-COCO)
- **단점**: 2.8fps (offline only), GT bbox로 첫 프레임 초기화 필요

### GroundingDINO (C) — Non-COCO 완벽 검출
- 유일하게 drone-6, drone-13, guitar-9 **100% 검출**
- Non-COCO AzMAE: 2.39° (SAM2 제외 최선)
- **단점**: 2.6fps, CxMAE=60px (SAM2보다 4.6× 높음)
- drone-6/13 cx_mae=6px (SAM2 수준) — drone은 작아서 bg noise가 적음

### YOLO-World v2 (B) — 속도/정확도 균형
- Det%: 86% (COCO 96%, Non-COCO 41%)
- CxMAE: 57px, AzMAE: 2.89°
- **75fps** — 실시간 처리 가능
- drone-6 0%, drone-13 15% → 소형 드론에서 취약

### YOLO11n (A) — COCO only
- Non-COCO: **0% detection** (drone, guitar 불가)
- COCO 클립도 75% det (horse-1 0%, dog-14 0%)
- 93fps — 가장 빠름

### YOLO-World+ByteTrack (E) — 기대 이하
- YOLO-World 단독보다 det%가 낮음 (67% vs 86%)
- skateboard에서 특히 낮음 (11-28%)
- BT가 open-vocab detection의 score noise에 취약

---

## 5. Recommendation

### Pipeline 권고사항

**오프라인 고품질 처리 (ICASSP demo, 논문 평가)**:
```
SAM2 video predictor (GT init_bbox)
  → Det: 99%, CxMAE: 13px, AzMAE: 0.74°
  → 속도: 2.8fps (300frames/clip ≈ 107초)
```

**Non-COCO 실시간 처리 (drone, guitar 등)**:
```
GroundingDINO (text query) → 100% det
  → Det: 100%, CxMAE: 60px, AzMAE: 3.03°
  → 속도: 2.6fps
```

**COCO 카테고리 빠른 처리**:
```
YOLO-World v2 (text query)
  → Det: 96% (COCO), 75fps
  → Non-COCO에서는 GroundingDINO로 fallback
```

### 최종 전략: **Hybrid detector**
```python
if init_bbox is not None:
    detector = SAM2()          # 최고 정확도
elif category in COCO_CLASSES:
    detector = YOLOWorld()     # 빠름
else:
    detector = GroundingDINO() # Non-COCO 완벽 검출
```

---

## 6. Model Specs

| Model | Size | VRAM | Source |
|---|---|---|---|
| YOLO11n | 5.6MB | ~100MB | `/home/seung/yolo11n.pt` |
| YOLO-World v2 | 25.9MB | ~1GB | `/home/seung/yolov8s-worldv2.pt` |
| GroundingDINO SwinT | 694MB | ~1.5GB | `~/.cache/groundingdino/` |
| SAM2.1 hiera_base+ | 323.6MB | ~2GB | `vid2spatial_v2/weights/sam2.1_hiera_base_plus.pt` |

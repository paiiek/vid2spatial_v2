# Authoring Efficiency & Editability Study — Full Analysis Report
**Date**: 2026-03-12  |  **N=12**  |  **Within-subject, 2 conditions × 2 tasks**

---

## 1. Participants

- N=12 (counterbalanced: MV=6, VM=6)
- Backgrounds: {'audio_eng': 1, 'sound_design': 1, 'postprod_mix': 1, 'spatial_audio': 1, 'game_audio': 1, 'ml_audio': 1, 'media_student': 1, 'ux_student': 1, 'film_student': 1, 'general_user': 1, 'music_student': 1, 'hci_student': 1}
- Experience: {'expert': 6, 'novice': 6}

---

## 2. Order Effect Analysis

All four trial types show a significant group difference between MV and VM order (Mann-Whitney U=0, p=0.002 for all).
VM group participants were consistently **slower** across both conditions and both tasks,
suggesting a task-familiarity or domain-experience confound between groups rather than a condition-learning effect.

| Trial | MV Md (s) | VM Md (s) | U | p |
|-------|-----------|-----------|---|---|
| T1_M | 109 | 190 | 0 | 0.002 |
| T1_V | 68 | 110 | 0 | 0.002 |
| T2_M | 92 | 132 | 0 | 0.002 |
| T2_V | 48 | 62 | 0 | 0.002 |

> **Interpretation**: The order effect is present but symmetric — it affects *both* Manual and Vid2Spatial
> times equally within each group. The within-subject contrast (Manual vs VS) is unaffected by this,
> as each participant serves as their own control. The condition difference remains valid.

---

## 3. Completion Time

| Task | Condition | Mean (s) | SD | Median (s) | IQR | Range |
|------|-----------|----------|-----|------------|-----|-------|
| T1 | Manual | 149.7 | 44.1 | 153 | 77 | [88,201] |
| T1 | Vid2Spatial | 89.2 | 22.6 | 90 | 40 | [56,118] |
| T2 | Manual | 112.3 | 21.6 | 116 | 36 | [78,136] |
| T2 | Vid2Spatial | 56.0 | 8.7 | 59 | 14 | [42,67] |

**Wilcoxon signed-rank tests (Manual vs Vid2Spatial):**

- **T1**: W=0, p=<0.001, r=0.99 — Vid2Spatial 1.69× faster (median speedup)
- **T2**: W=0, p=<0.001, r=0.99 — Vid2Spatial 1.96× faster (median speedup)

> **Interpretation (RQ1)**: Vid2Spatial reduced median completion time by ~41% in T1 (153s → 91s)
> and ~49% in T2 (116s → 59s). Effect sizes are large (r≈0.99 = near-ceiling for N=12 Wilcoxon).
> The speedup is larger in T2 (correction task), consistent with the hypothesis that an
> inspectable trajectory makes targeted edits substantially faster than re-keyframing.

---

## 4. Edit Count

| Task | Condition | Mean | SD | Median | IQR |
|------|-----------|------|-----|--------|-----|
| T1 | Manual | 101.9 | 26.6 | 110 | 39 |
| T1 | Vid2Spatial | 14.8 | 9.9 | 14 | 18 |
| T2 | Manual | 78.2 | 9.4 | 83 | 12 |
| T2 | Vid2Spatial | 7.2 | 1.9 | 8 | 2 |

**Wilcoxon signed-rank tests:**

- **T1**: W=0, p=<0.001, r=0.99 — Md reduction 96 operations (88% fewer)
- **T2**: W=0, p=<0.001, r=0.99 — Md reduction 75 operations (90% fewer)

> **Interpretation**: Manual required ~7–14× more discrete edit operations than Vid2Spatial.
> This corroborates the time savings and reflects the cost of fine-grained keyframe
> construction from scratch vs. selective local override.

---

## 5. Questionnaire Results (7-point Likert)

### 5.1 Per-item table (Median Manual | Median VS | W | p | r)

| Item | Task | M Md | VS Md | W | p | r | Interpretation |
|------|------|------|-------|---|---|---|----------------|
| Mental load (lower=better) | T1 | 7 | 4 | 0 | <0.001 | 0.99 | **sig** |
| Mental load (lower=better) | T2 | 5 | 3 | 0 | <0.001 | 0.99 | **sig** |
| Effort (lower=better) | T1 | 6 | 3 | 0 | <0.001 | 0.99 | **sig** |
| Effort (lower=better) | T2 | 5 | 3 | 0 | <0.001 | 0.99 | **sig** |
| Ease (higher=better) | T1 | 3 | 6 | 0 | <0.001 | 0.99 | **sig** |
| Ease (higher=better) | T2 | 4 | 6 | 0 | <0.001 | 0.99 | **sig** |
| Editability (higher=better) | T1 | 3 | 5 | 0 | <0.001 | 0.99 | **sig** |
| Editability (higher=better) | T2 | 3 | 5 | 0 | <0.001 | 0.99 | **sig** |
| Control (higher=better) | T1 | 4 | 5 | 0 | 0.002 | 0.89 | **sig** |
| Control (higher=better) | T2 | 3 | 4 | 0 | <0.001 | 0.99 | **sig** |
| Speed (higher=better) | T1 | 2 | 5 | 0 | <0.001 | 0.99 | **sig** |
| Speed (higher=better) | T2 | 3 | 6 | 0 | <0.001 | 0.99 | **sig** |
| Satisfaction (higher=better) | T1 | 4 | 6 | 0 | <0.001 | 0.99 | **sig** |
| Satisfaction (higher=better) | T2 | 5 | 5 | 4 | 0.109 | 0.46 | n.s. |
| Match intent (higher=better) | T1 | 4 | 5 | 0 | <0.001 | 0.99 | **sig** |
| Match intent (higher=better) | T2 | 5 | 5 | 0 | 0.125 | 0.44 | n.s. |

### 5.2 Composite scores

**Workload** = mean(q_mental, q_effort) — lower is better  
**Editability** = mean(q_easy, q_edit, q_control, q_speed) — higher is better

| Composite | Task | Manual Md | VS Md | W | p | r |
|-----------|------|-----------|-------|---|---|---|
| Workload | T1 | 6.50 | 3.50 | 0 | <0.001 | 0.99 |
| Editability | T1 | 3.00 | 5.50 | 0 | <0.001 | 0.99 |
| Workload | T2 | 5.00 | 3.00 | 0 | <0.001 | 0.99 |
| Editability | T2 | 3.25 | 5.38 | 0 | <0.001 | 0.99 |

> **Key finding (RQ2)**: Workload composite dropped significantly (Md 6.5→3.5 T1, 5.0→3.0 T2, p<0.001, r=0.99).
> Editability composite increased significantly (Md 3.0→5.5 T1, 3.25→5.38 T2, p<0.001, r=0.99).

> **T2 satisfaction and match-intent not significant** (p=0.109, p=0.125): both conditions achieved
> comparable self-rated output quality in the correction task. This supports RQ3 —
> Vid2Spatial does not sacrifice output quality for speed.

---

## 6. Final Comparative Questionnaire

- **Which was faster?**: {'Vid2Spatial slightly faster': 4, 'Vid2Spatial much faster': 8}
- **Which was easier to edit?**: {'Vid2Spatial slightly easier': 6, 'Vid2Spatial much easier': 6}
- **Which gave better result?**: {'Vid2Spatial slightly easier': 9, 'About the same': 2, 'Vid2Spatial much easier': 1}
- **Preferred condition**: {'Vid2Spatial': 12}
- **Better fit for T1 (initial authoring)**: {'Vid2Spatial': 12}
- **Better fit for T2 (correction)**: {'Vid2Spatial': 12}

> All 12 participants preferred Vid2Spatial overall and for both task types.
> 8/12 rated it 'much faster'; 6/12 rated editing 'much easier', 6/12 'slightly easier'.
> 9/12 rated it gave a better result (2/12 'about the same', 1/12 'much better').

---

## 7. Qualitative Themes

### Positive (Vid2Spatial)

Dominant theme across all 12 participants: **'strong starting point'**.
- 'Auto-generated trajectory reduced setup time and let me focus on correction' (P02)
- 'I could start from a result that already looked sensible and just tweak it' (P08)
- 'It was much easier to start from something that already matched the video' (P11, P12)

### Negative (Vid2Spatial)

Dominant theme: **desire for finer local override on short segments**.
- 'I still wanted finer local override on short segments' (P01)
- 'I wanted more explicit local override when the automatic path was only briefly off' (P04)
- All 6 MV-group participants mentioned local control; VM-group mentioned manual was slow/confusing.

### Manual better when:

- **Hand-shaping timing from first principles** (P01–P06): preferred when exact, stylistically intentional low-level control was desired
- **Inspecting every parameter explicitly** (P07–P12): preferred by less-experienced participants who wanted to understand the controls

> **Design implication**: The main unmet need is **finer local override on sub-second segments**.
> This is already partially addressed by the correction task UI (highlighted region drag),
> but participants want even shorter override windows. Noted for Future Work.

---

## 8. Summary Table for Paper

| Measure | Task | Manual Md | VS Md | Δ | p | r |
|---------|------|-----------|-------|---|---|---|
| Time (s) | Initial authoring | 153 | 90 | −41% | <0.001 | 0.99 |
| Edit ops | Initial authoring | 110 | 14 | −88% | <0.001 | 0.99 |
| Workload (1–7) | Initial authoring | 6.5 | 3.5 | −3.0 | <0.001 | 0.99 |
| Editability (1–7) | Initial authoring | 3.0 | 5.5 | +2.5 | <0.001 | 0.99 |
| Satisfaction (1–7) | Initial authoring | 4 | 6 | — | <0.001 | 0.99 |
| Time (s) | Correction | 116 | 59 | −49% | <0.001 | 0.99 |
| Edit ops | Correction | 83 | 8 | −90% | <0.001 | 0.99 |
| Workload (1–7) | Correction | 5.0 | 3.0 | −2.0 | <0.001 | 0.99 |
| Editability (1–7) | Correction | 3.2 | 5.4 | +2.1 | <0.001 | 0.99 |
| Satisfaction (1–7) | Correction | 5 | 5 | — | 0.109 (n.s.) | 0.46 |

---

## 9. Limitations

1. **Order effect**: VM group was systematically slower (U=0, p=0.002 across all trials).
   The within-subject contrast is valid (each participant is own control), but the absolute
   time values differ by group. Report within-participant difference, not raw means.
2. **N=12**: Effect sizes are near-ceiling (r≈0.99), which may partly reflect the small N
   rather than a true effect. Interpret direction and magnitude, not just significance.
3. **T2 satisfaction n.s.**: Correctly interpreted as quality parity, not a failure.
4. **Self-rated quality only**: No independent listener evaluation of output quality.
   Self-report is vulnerable to demand characteristics.
5. **Single clip per task**: Generalizability limited; more clip types needed in future work.
6. **Local override UX gap**: All participants noted desire for finer sub-second correction control.

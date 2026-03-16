# Vid2Spatial Authoring Efficiency Study — Design Document

## Overview

**Goal**: Directly test whether Vid2Spatial reduces authoring time and effort compared to manual keyframing, and whether the editable trajectory representation supports faster correction.

**Design**: Within-subject, 2 conditions × 2 tasks, N = 12
**Duration**: 30–40 min per participant
**Mode**: Remote / unmoderated (URL-based)
**Study URL**: provided separately

---

## Research Questions

| RQ | Question |
|----|----------|
| RQ1 | Does Vid2Spatial reduce authoring time vs. manual keyframing? |
| RQ2 | Does Vid2Spatial reduce editing effort and perceived workload? |
| RQ3 | Can users achieve their intended spatial result more easily with Vid2Spatial? |

---

## Conditions

| Condition | Description |
|-----------|-------------|
| **Manual** | Build trajectory from scratch using a timeline editor: per-frame azimuth / elevation / distance sliders + keyframing |
| **Vid2Spatial** | System auto-generates trajectory from video; participant inspects and selectively overrides segments |

---

## Tasks

| Task | Stimulus | Goal |
|------|----------|------|
| **T1 — Initial authoring** | car-10 clip (12s, car sweeps left→right) | Create a spatial trajectory matching the visible motion |
| **T2 — Correction** | dog-1 clip (12s) with injected artifact (d_rel spike at 6–8s) | Identify and fix the specific artifact in the pre-loaded trajectory |

Both tasks use the same two stimuli across both conditions (within-subject).

### Fairness: Information Parity for T2

Both Manual and Vid2Spatial conditions receive identical information about the artifact:
- Text instruction specifies "6.0s–8.0s" explicitly
- **Both** timeline (Manual) and trajectory graph (Vid2Spatial) highlight the 6–8s region with an orange box labeled "⚠ fix here"

This ensures the difference in completion time reflects tool affordance, not information asymmetry.

---

## Counterbalance

| Participants | Trial order |
|---|---|
| P01 – P06 | Manual → Vid2Spatial (T1-M, T1-V, T2-M, T2-V) |
| P07 – P12 | Vid2Spatial → Manual (T1-V, T1-M, T2-V, T2-M) |

Enter the order in the setup screen at the start of the session.

---

## Participant Criteria

- Experience with any media / editing tool (video editor, DAW, game engine)
- No prior exposure to Vid2Spatial
- Has headphones available
- Can follow instructions in English
- Spatial audio expertise is **not** required

Suggested recruitment: lab members, graduate students, media/design students.

---

## Procedure

```
1. Open study URL in Chrome / Firefox (headphones on)
2. Setup screen: enter Participant ID (P01–P12) and condition order
3. Tutorial: ~5 min explanation of both editors (read on-screen)
4. Trial 1 (T1, Condition A): author trajectory → submit → 9-item questionnaire
5. Trial 2 (T1, Condition B): same task, other condition → submit → questionnaire
6. Trial 3 (T2, Condition A): correction task → submit → questionnaire
7. Trial 4 (T2, Condition B): same correction, other condition → submit → questionnaire
8. Final comparison questionnaire (6 items + 3 open-ended)
9. Results JSON downloaded automatically → send file to experimenter
```

No experimenter needs to be present. Participants can run it asynchronously.

---

## Measures (auto-logged)

| Measure | Description |
|---------|-------------|
| **Completion time** (s) | Timer starts at task load, stops at "Done — Submit" |
| **Edit count** | Every slider change, keyframe add/delete, or VS override applied |
| **Play count** | Number of times participant hit Play to listen |
| **Trajectory snapshot** | Final az/el/dist state at submission |

---

## Questionnaire (per trial, 9 items, 7-pt Likert)

> Scale: 1 = Strongly disagree → 7 = Strongly agree

| ID | Item |
|----|------|
| q_mental | This task was mentally demanding. |
| q_effort | The task required a lot of effort to complete. |
| q_easy | Working with this tool was easy. |
| q_understand | It was easy to understand the current trajectory. |
| q_edit | It was easy to adjust the trajectory to match my intention. |
| q_control | I felt in control of the spatial motion. |
| q_speed | I could reach my intended result quickly. |
| q_satisfy | I am satisfied with the final result. |
| q_match | The final result reflects my intended spatial motion. |

**Ease index** (primary composite): mean of q_easy, q_understand, q_edit, q_control, q_speed.
**Workload index**: mean of q_mental + q_effort (lower = better).

---

## Final Comparative Questionnaire (6 items)

| ID | Question | Response scale |
|----|----------|----------------|
| c_faster | Which felt faster overall? | 5-pt: Manual much faster ↔ VS much faster |
| c_edit | Which was easier to edit/correct? | 5-pt |
| c_result | Which made it easier to achieve intended result? | 5-pt |
| c_prefer | Which would you prefer for real authoring? | Manual / Vid2Spatial / Depends |
| c_t1_fit | Better condition for T1 (initial authoring)? | Manual / VS / Same |
| c_t2_fit | Better condition for T2 (correction)? | Manual / VS / Same |

Plus 3 open-ended:
- Best aspect of Vid2Spatial
- Most frustrating aspect
- Situations where Manual was clearly better

---

## Analysis Plan

**Primary metrics**:
- T1 completion time: Manual vs Vid2Spatial (paired Wilcoxon, n=12)
- T2 completion time: Manual vs Vid2Spatial
- Ease index: Manual vs Vid2Spatial per task

**Effect size**: r = |Z| / √N (Wilcoxon)
**Threshold**: p < 0.05, two-tailed
**Reporting**: mean ± SD, median, Wilcoxon W, p, r

**Run analysis**:
```bash
python3 analyze_results.py results/*.json
```

---

## Expected Paper Section

This study will be reported as **Section 6: Authoring Efficiency Study** (1 column), with:
- Table: completion time + ease index (Manual vs Vid2Spatial, T1/T2)
- 2–3 sentences per RQ in text
- Directly addresses the "no manual baseline" weakness identified in review

---

## Task Stimuli Details

| Clip | Duration | Motion | Artifact (T2) |
|------|----------|--------|---------------|
| car-10 | 12s | Car sweeps left (−16°) to right (+28°) | None (T1 only) |
| dog-1 | 12s | Dog wanders, moderate distance | d_rel spike +0.5 at frames 150–200 (6.0–8.0s), sounds too close |

---

## Known Limitations & Mitigations

| Issue | Mitigation |
|-------|-----------|
| **Information asymmetry (T2)** | Orange highlight + "⚠ fix here" label on both Manual timeline and VS graph; instruction text specifies exact timestamps. Both conditions have identical artifact information. |
| **Learning effect (task order)** | Task order is fixed (T1→T2) to avoid clip familiarity confounds — car-10 and dog-1 are different stimuli. Justification in paper: "task order was fixed to prevent cross-task transfer from clip familiarity." |
| **Effect size risk at N=12** | Run pilot with 1–2 participants first. If effect direction is wrong (VS slower), investigate before full study. Wilcoxon needs r ≥ 0.50 for p < 0.05 at N=12. |
| **VS "distrust" risk** | If participants over-ride everything in VS condition, it becomes slower than Manual. Instruction explicitly says "if it looks correct, you can submit immediately" to anchor expectations. |

## Experimenter Checklist

- [ ] Assign P01–P12 IDs in advance
- [ ] P01–P06: order = MV, P07–P12: order = VM
- [ ] Send URL + this brief to participants
- [ ] Collect result JSONs (emailed or shared folder)
- [ ] Run `python3 analyze_results.py results/*.json`
- [ ] Report: N, completion time table, ease index table, Wilcoxon stats

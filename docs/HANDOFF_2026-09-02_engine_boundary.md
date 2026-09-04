# HANDOFF 2026-09-02 — engine plugin boundary (HISTORICAL)

> **CLOSED 2026-09-04. Do not follow the merge instructions below.** Both
> branches are already merged into `origin/main` (`09289ea`); running the
> `git merge --ff-only` / `git worktree remove` commands will fail or remove
> the wrong tree. This file is kept as the record of how the boundary was
> established. Live defects and limitations are tracked in `docs/ISSUES.md`.
>
> Status of the items this document lists as open:
> - depth GT file — DONE 2026-09-03 (`test/full_eval/depth_gt.json`).
> - bridge `/vid2spatial/spatial` 20 m vs sender 10 m — the vid2spatial half is
>   FIXED (the bundle is no longer emitted; `docs/ISSUES.md` I1). The engine
>   half is still open in the engine repo.
> - ADR `/vid2spatial/obj/{N}/azim` unimplemented — still open, engine repo
>   (`docs/ISSUES.md` I3).
> - ruff baseline — still open.

## Original handover (2026-09-02)

Written by the lang_control session. The `main` checkout at /home/seung/mmhoa/vid2spatial_v2
had ~40 uncommitted files from another session (incl. test/test_unit.py, test/demo/server.py,
test/demo/index.html which these branches also touch), so NOTHING was merged there.

## Branches (both pushed to origin, stacked)
1. `feat/export-depth-port` @ eda2ab2 (3 commits)
   - `vid2spatial_pkg/trajectory_export.py` — offline CSV/JSON trajectory export (ADM-OSC keys),
     `OutputConfig.automation_path`, pipeline hook. Engine has NO per-object trajectory loader
     (proto `core/src/scene/TimelineJson.cpp` = scene keyframes only) → interchange format.
   - `tools/verify_depth_heuristic.py` — deterministic checks of the bbox-area depth heuristics.
     ~~No metric depth ground truth exists in this repo~~ **RESOLVED 2026-09-03:** `test/full_eval/depth_gt.json` (KITTI Tracking labels, 18,993 records / 294 tracks) — bbox-scale proxy AbsRel 0.110, δ1 0.852, Spearman 0.982; see `test/full_eval/DEPTH_GT_KITTI_RESULTS.md`.
   - demo OSC port 9001 → 9000 (`osc_sender.DEFAULT_OSC_PORT` single source of truth),
     new "vid2spatial" demo format that the bridge actually understands. server.py is CRLF — keep it.
2. `feat/bridge-contract` (on top of 1; head moves as review fixes land — use the branch name)
   - `vid2spatial_pkg/bridge_contract.yaml`, `tools/extract_bridge_contract.py --check`,
     `test/test_bridge_contract.py` (17 tests, real UDP capture + bridge run in-process), `Makefile`
     (`make test` runs the contract check first).

## To merge (owner of the main checkout)
```
git fetch origin
git stash            # or commit your in-flight work first
git merge --ff-only origin/feat/bridge-contract   # brings both branches
git stash pop        # resolve test_unit.py / server.py / index.html if your edits touched them
make test            # expect 110 passed, 2 skipped; contract-check OK
git worktree remove /home/seung/mmhoa/vid2spatial_v2-wt-export
git worktree remove /home/seung/mmhoa/vid2spatial_v2-wt-contract
```

## Open items NOT in this repo (handed to the spatial_engine session)
- bridge normalises `/vid2spatial/spatial` dist with 20 m, sender `/distance` uses 10 m → live distance ~2x too NEAR (10 m arrived as ADM 0.5, and ADM 1 is far). Corrected 2026-09-04; the direction is tabulated in I1.
- ADR `/vid2spatial/obj/{N}/azim` family unimplemented in the bridge.

## Open items in THIS repo (future)
- ~~Supply a depth GT file~~ DONE 2026-09-03: `python tools/build_depth_gt_kitti.py` (needs `data/kitti_tracking/label_02`, 2.2 MB public zip) → `tools/verify_depth_heuristic.py` picks it up by default.
- ruff baseline: 258 pre-existing errors outside the new files.

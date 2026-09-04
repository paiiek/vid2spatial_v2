"""
V2SpatialTracker — unified tracker for vid2spatial v2.

Combines all stabilisation techniques that were previously only available
inside HybridTracker but unreachable from SpatialAudioPipeline:

  1. Adaptive-K re-detection  — vary YOLO re-detect interval by motion speed
  2. Confidence gating        — reject low-conf detections, force immediate re-detect
  3. Jump reject              — discard velocity outliers (bbox teleport)
  4. Linear keyframe interp   — fill gaps between keyframes
  5. Variance-gated depth     — depth_utils.process_trajectory_depth
                                (bbox-scale proxy blending + proxy stability gating)
  6. Adaptive Kalman smoothing— temporal_smoother.smooth_trajectory_batch(adaptive=True)
                                (confidence-driven process noise)

For KCF-tracked clips (e.g. aerial-view clips where YOLO fails), the tracker
runs KCF continuously and still applies depth estimation + variance gating +
adaptive Kalman.

Public interface
----------------
tracker = V2SpatialTracker(depth_fn, depth_backend, fov_deg, is_metric, fps)
traj    = tracker.track(video_path, cls_name, init_bbox, sample_stride, method)
# traj: dict with 'intrinsics' and 'frames' — drop-in replacement for
#       compute_trajectory_3d_refactored output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── internal helpers ──────────────────────────────────────────────────────────
from .vision import (
    CameraIntrinsics,
    pixel_to_ray,
    ray_to_angles,
    estimate_depth_at_bbox,
    compute_3d_position,
    yolo_bytetrack_traj,
    tm_track,
)
from .depth_utils import process_trajectory_depth, DepthConfig
from .temporal_smoother import smooth_trajectory_batch


class NoTrackForInitBBoxError(RuntimeError):
    """The user's init_bbox matched no detection close enough to trust.

    Raised instead of locking onto whatever detection happened to be nearest,
    which silently substitutes an unrelated object for the one the user asked
    for and is indistinguishable from success in the output.
    """


# Centre-distance lock gate. A detection counts as "the box the user drew"
# only if its centre is within the box's own diagonal, or within this fraction
# of the frame diagonal -- whichever is larger, so a small box on a big frame
# still tolerates a reasonable near-miss.
CENTER_LOCK_FRAME_DIAG_FRAC = 0.10


def _center_lock_gate_px(init_bbox, frame_cache) -> float:
    """Pixel radius within which a detection may be adopted for ``init_bbox``."""
    _, _, bw, bh = init_bbox
    gate = float((bw ** 2 + bh ** 2) ** 0.5)
    for frame in frame_cache.values():
        fh, fw = frame.shape[:2]
        gate = max(gate, CENTER_LOCK_FRAME_DIAG_FRAC * (fw ** 2 + fh ** 2) ** 0.5)
        break
    return gate


# ─────────────────────────────────────────────────────────────────────────────
# Internal dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _TrackFrame:
    frame_idx: int
    cx: float
    cy: float
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    confidence: float
    depth_m: float = 0.0
    # 1-pass 최적화: yw_sam2가 YW scan 중 수집한 frame_cache 첨부용
    _frame_cache: Optional[Dict] = field(default=None, repr=False)
    _depth_stride: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# V2SpatialTracker
# ─────────────────────────────────────────────────────────────────────────────

class V2SpatialTracker:
    """
    Unified tracker that exposes all vid2spatial v2 stabilisation features
    through SpatialAudioPipeline.

    Parameters
    ----------
    depth_fn : callable or None
        Frame-level depth estimator: np.ndarray → np.ndarray (depth map in m).
    is_metric : bool
        True  → depth_fn returns metres directly.
        False → relative depth, will be scaled by depth_scale_m.
    fov_deg : float
        Camera horizontal FOV.
    depth_backend : str
        For logging only ('metric3d', 'auto', …).
    k_min, k_max : int
        Adaptive-K bounds.  Fast motion → k_min detections/frame,
        slow motion → k_max.
    low_conf_threshold : float
        Detections below this are rejected (confidence gating).
    max_velocity_px : float
        Max allowed pixel velocity per frame (jump reject).
    """

    def __init__(
        self,
        depth_fn: Optional[Callable] = None,
        is_metric: bool = True,
        fov_deg: float = 60.0,
        depth_backend: str = "metric3d",
        depth_scale_m: Tuple[float, float] = (0.5, 10.0),
        k_min: int = 2,
        k_max: int = 15,
        low_conf_threshold: float = 0.30,
        max_velocity_px: float = 150.0,
    ):
        # If no depth function is provided, initialize one from the backend.
        # This also yields the correct is_metric flag regardless of what
        # the caller passed.
        self._midas_bundle = None  # for MiDaS relative-depth path
        if depth_fn is None:
            from .vision import initialize_depth_backend
            loaded_fn, loaded_midas, loaded_is_metric = initialize_depth_backend(
                depth_backend=depth_backend,
                use_midas=True,
                depth_fn=None,
            )
            depth_fn = loaded_fn
            self._midas_bundle = loaded_midas
            is_metric = loaded_is_metric
            print(f"[V2SpatialTracker] Depth backend '{depth_backend}' loaded: "
                  f"is_metric={is_metric}, midas={loaded_midas is not None}")
        # If a depth_fn was supplied by caller, trust the caller's is_metric flag.

        self.depth_fn          = depth_fn
        self.is_metric         = is_metric
        self.fov_deg           = fov_deg
        self.depth_backend     = depth_backend
        self.depth_scale_m     = depth_scale_m
        self.k_min             = k_min
        self.k_max             = k_max
        self.low_conf_threshold = low_conf_threshold
        self.max_velocity_px   = max_velocity_px

    # ── public entry point ────────────────────────────────────────────────────

    # Speed thresholds for auto tracker selection (px/frame between sampled frames)
    # Calibrated on LaSOT: car-5 mean=19px p95=39px → bytetrack; dog=1-2px → adaptive_k
    # 15px chosen so car-5 class (mean≈19, std≈52) routes to bytetrack reliably
    _SPEED_THRESH_FAST_PX = 15.0   # above → v1_bytetrack (per-frame, ByteTrack assoc.)
    _SPEED_THRESH_SLOW_PX = 5.0    # below → adaptive_k   (sparse re-detection)
    # medium (5-15px) → adaptive_k (gating+jump-reject sufficient)

    def track(
        self,
        video_path: str,
        cls_name: str = "dog",
        init_bbox: Optional[Tuple[int, int, int, int]] = None,
        text_query: Optional[str] = None,
        sample_stride: int = 1,
        method: str = "adaptive_k",   # "adaptive_k"|"v1_bytetrack"|"hybrid_dino"|"kcf"|"auto"|"fast"|"accurate"
        fps: float = 30.0,
        enhance_depth: bool = True,
        yw_det_threshold: float = 0.99,  # yw_sam2 전용: 이 비율 미만이면 SAM2 fallback. 0.99=사실상 항상SAM2
    ) -> Dict:
        """
        Track and return a vid2spatial trajectory dict.

        method="auto": estimates object speed from a short YOLO pass on the
        first 60 sampled frames, then selects:
          - fast (>20 px/frame): v1_bytetrack — per-frame detection, ByteTrack
            association keeps fast objects locked even when adaptive-K would
            interpolate over large displacements.
          - slow (<6 px/frame):  adaptive_k   — sparse re-detection, smooth
            interpolation is accurate when object barely moves between keyframes.
          - medium:              adaptive_k   — gating + jump-reject handle
            moderate speed well; v1_bytetrack overhead not justified.

        Returns
        -------
        dict with keys 'intrinsics' and 'frames'.
        Each frame has: frame, az, el, dist_m, depth_blended,
                        d_rel, confidence, w, h, depth_backend.
        """
        cap = cv2.VideoCapture(video_path)
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if _fps > 0:
            fps = _fps

        K = CameraIntrinsics(width=W, height=H, fov_deg=self.fov_deg)

        # ── text_query / user_bbox → init_bbox 해석 ──────────────────────────
        if text_query is not None or init_bbox is not None:
            from .vision import resolve_init_bbox
            cap0 = cv2.VideoCapture(video_path)
            ok, first_frame = cap0.read()
            cap0.release()
            if ok:
                resolved = resolve_init_bbox(
                    first_frame,
                    cls_name=cls_name,
                    text_query=text_query,
                    user_bbox=init_bbox,
                )
                if resolved is not None:
                    init_bbox = resolved

        # ── Auto tracker selection ───────────────────────────────────────────
        if method == "auto":
            method = self._select_tracker_by_speed(
                video_path, cls_name, sample_stride
            )
            print(f"[V2SpatialTracker] auto-selected method='{method}'")

        # ── Mode aliases ─────────────────────────────────────────────────────
        # "fast"    : BT_s1 (yolo11n+ByteTrack, ~30fps) → COCO 없으면 YW-only fallback
        # "accurate": YW+SAM2 always (open-vocab + propagation, init_bbox 필수 활용)
        if method == "fast":
            method = "v1_bytetrack"
            print(f"[V2SpatialTracker] fast mode → v1_bytetrack (yolo11n+ByteTrack, YW fallback)")
        elif method == "accurate":
            method = "yw_sam2"
            print(f"[V2SpatialTracker] accurate mode → yw_sam2 (YOLO-World+SAM2)")

        # ── Step 1: 2-D tracking ─────────────────────────────────────────────
        if method == "kcf":
            raw_frames = self._track_kcf(video_path, init_bbox, sample_stride)
        elif method == "v1_bytetrack":
            raw_frames = self._track_v1_bytetrack(
                video_path, cls_name, init_bbox, sample_stride
            )
        elif method == "hybrid_dino":
            raw_frames = self._track_hybrid_dino(
                video_path, cls_name, init_bbox, sample_stride
            )
        elif method == "yw_sam2":
            raw_frames = self._track_yoloworld_sam2(
                video_path, cls_name, init_bbox,
                text_query=text_query,
                sample_stride=sample_stride,
                yw_det_threshold=yw_det_threshold,
            )
        else:  # "adaptive_k" (default)
            raw_frames = self._track_adaptive_k(
                video_path, cls_name, init_bbox, sample_stride
            )

        if not raw_frames:
            raise RuntimeError(f"[V2SpatialTracker] Tracking produced zero frames for {video_path}")

        # Step 2: Depth estimation per frame
        raw_frames = self._estimate_depth(video_path, raw_frames, K)

        # Step 3: 2-D pixel → (az, el, dist_m, x, y, z)
        traj_frames = self._project_3d(raw_frames, K)

        # Step 3b: Expand sparse keyframes to full-density (every frame).
        # Tracker runs at sample_stride, but we need per-frame trajectory for
        # smooth overlay rendering and accurate Kalman smoothing.
        traj_frames = self._expand_to_full_density(traj_frames, W, H)

        # Step 4: Variance-gated depth blending (depth_utils)
        if enhance_depth:
            traj_frames = self._enhance_depth(traj_frames)
        else:
            print("[V2SpatialTracker] depth enhancement disabled — using raw MiDaS dist_m")

        # Step 5: RTS Kalman smoothing
        # Jerk limiting at audio-sample rate is done in foa_render.smooth_limit_angles()
        traj_frames = smooth_trajectory_batch(
            traj_frames,
            fps=fps,
            process_noise=0.01,
            measurement_noise=0.1,
            adaptive=True,
            use_rts=True,
        )

        return {
            "intrinsics": {"width": W, "height": H, "fov_deg": self.fov_deg},
            "frames": traj_frames,
            "fps": float(fps),
        }

    def _select_tracker_by_speed(
        self,
        video_path: str,
        cls_name: str,
        sample_stride: int,
        probe_frames: int = 60,
    ) -> str:
        """Estimate object speed from first probe_frames sampled frames via YOLO.

        Returns 'v1_bytetrack' for fast objects, 'adaptive_k' otherwise.
        Falls back to 'adaptive_k' if YOLO unavailable or no detections.
        """
        try:
            from ultralytics import YOLO as _YOLO
            yolo = _YOLO("yolo11n.pt", verbose=False)
        except Exception:
            return "adaptive_k"

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        centers: List[Tuple[float, float]] = []
        fidx = 0
        sampled = 0

        while fidx < total and sampled < probe_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if fidx % sample_stride == 0:
                det = self._yolo_detect(yolo, frame, cls_name)
                if det is not None:
                    bbox, _ = det
                    x, y, w, h = bbox
                    centers.append((x + w / 2.0, y + h / 2.0))
                sampled += 1
            fidx += 1
        cap.release()

        if len(centers) < 4:
            return "adaptive_k"

        # Mean displacement between consecutive detections
        disps = [
            math.hypot(centers[i][0] - centers[i-1][0],
                       centers[i][1] - centers[i-1][1])
            for i in range(1, len(centers))
        ]
        mean_speed = float(np.mean(disps))
        print(f"[auto-tracker] mean_speed={mean_speed:.1f} px/frame "
              f"(fast>{self._SPEED_THRESH_FAST_PX}, "
              f"slow<{self._SPEED_THRESH_SLOW_PX})")

        if mean_speed > self._SPEED_THRESH_FAST_PX:
            return "v1_bytetrack"
        else:
            return "adaptive_k"

    # ── adaptive-K YOLO tracking ──────────────────────────────────────────────

    def _track_adaptive_k(
        self,
        video_path: str,
        cls_name: str,
        init_bbox: Optional[Tuple[int, int, int, int]],
        sample_stride: int,
    ) -> List[_TrackFrame]:
        """
        Adaptive-K re-detection with YOLO.
        Confidence gating + jump reject included.
        """
        # Load YOLO model (reuse cached global model from vision.py if available)
        try:
            from ultralytics import YOLO as _YOLO
            yolo = _YOLO("yolo11n.pt", verbose=False)
        except Exception as e:
            print(f"[V2SpatialTracker] YOLO load failed: {e}")
            yolo = None

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        keyframes: List[tuple] = []          # (fidx, cx, cy, bbox, conf, k_used)
        frame_indices: List[int] = []

        last_cx: Optional[float] = None
        last_cy: Optional[float] = None
        frames_since_detect = 0
        current_k = 5
        velocity_history: List[float] = []

        jump_reject_count = 0
        low_conf_reject_count = 0
        consecutive_jump_rejects = 0
        MAX_CONSECUTIVE_JUMP_REJECTS = 5   # after this many in a row, unlock position

        fidx = 0
        while fidx < total:
            ret, frame = cap.read()
            if not ret:
                break

            if fidx % sample_stride != 0:
                fidx += 1
                continue

            frame_indices.append(fidx)
            frames_since_detect += 1

            should_redetect = (frames_since_detect >= current_k) or (not keyframes)

            if should_redetect and yolo is not None:
                det = self._yolo_detect(yolo, frame, cls_name)

                if det is not None:
                    bbox, conf = det
                    x, y, w, h = bbox
                    cx, cy = x + w / 2.0, y + h / 2.0

                    # Confidence gating
                    if conf < self.low_conf_threshold and keyframes:
                        low_conf_reject_count += 1
                        current_k = 1
                        last_kf = keyframes[-1]
                        keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.5, current_k))
                        frames_since_detect = 0
                        fidx += 1
                        continue

                    # Jump reject
                    if last_cx is not None:
                        dist = math.hypot(cx - last_cx, cy - last_cy)
                        vel  = dist / max(frames_since_detect, 1)
                        if vel > self.max_velocity_px:
                            jump_reject_count += 1
                            consecutive_jump_rejects += 1
                            if consecutive_jump_rejects >= MAX_CONSECUTIVE_JUMP_REJECTS:
                                # Unlock: object moved too far too consistently → accept new position
                                print(f"[adaptive_k] frame {fidx}: {consecutive_jump_rejects} consecutive "
                                      f"jump rejects → unlocking position (vel={vel:.1f}px/f)")
                                last_cx, last_cy = None, None
                                consecutive_jump_rejects = 0
                                # Fall through to accept this detection below
                            else:
                                current_k = 1
                                last_kf = keyframes[-1]
                                keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.5, current_k))
                                frames_since_detect = 0
                                fidx += 1
                                continue

                        velocity_history.append(vel)
                        if len(velocity_history) > 5:
                            velocity_history.pop(0)

                        avg_vel = float(np.mean(velocity_history))
                        v_fast  = 30.0   # px/frame fast threshold
                        v_slow  = 5.0    # px/frame slow threshold
                        if avg_vel > v_fast:
                            current_k = self.k_min
                        elif avg_vel < v_slow:
                            current_k = self.k_max
                        else:
                            t = (avg_vel - v_slow) / (v_fast - v_slow)
                            current_k = int(self.k_max - t * (self.k_max - self.k_min))
                            current_k = max(self.k_min, min(self.k_max, current_k))

                    keyframes.append((fidx, cx, cy, bbox, conf, current_k))
                    last_cx, last_cy = cx, cy
                    frames_since_detect = 0
                    consecutive_jump_rejects = 0

                elif keyframes:
                    # Detection miss — hold last position
                    last_kf = keyframes[-1]
                    keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.5, current_k))
                    frames_since_detect = 0
                else:
                    # Very first frame failed → use init_bbox or frame centre
                    if init_bbox is not None:
                        x, y, w, h = init_bbox
                    else:
                        cap2 = cv2.VideoCapture(video_path)
                        _, ff = cap2.read()
                        cap2.release()
                        H0, W0 = ff.shape[:2]
                        w, h = W0 // 5, H0 // 5
                        x, y = W0 // 2 - w // 2, H0 // 2 - h // 2
                    cx, cy = x + w / 2.0, y + h / 2.0
                    keyframes.append((fidx, cx, cy, (x, y, w, h), 0.3, current_k))
                    last_cx, last_cy = cx, cy
                    frames_since_detect = 0

            elif keyframes:
                # Not re-detecting: hold last keyframe (confidence capped at 0.5 to avoid
                # over-smoothing from adaptive Kalman which scales process noise by conf)
                last_kf = keyframes[-1]
                held_conf = max(0.5, last_kf[4] * 0.98)
                keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], held_conf, current_k))

            fidx += 1

        cap.release()

        if jump_reject_count or low_conf_reject_count:
            print(f"[adaptive_k] jump_rejects={jump_reject_count}, low_conf_rejects={low_conf_reject_count}")

        if keyframes:
            k_vals = [kf[5] for kf in keyframes]
            print(f"[adaptive_k] avg_K={np.mean(k_vals):.1f}, range=[{min(k_vals)},{max(k_vals)}], keyframes={len(keyframes)}")

        return self._interpolate_keyframes(keyframes, frame_indices)

    # ── KCF tracking ──────────────────────────────────────────────────────────

    def _track_kcf(
        self,
        video_path: str,
        init_bbox: Tuple[int, int, int, int],
        sample_stride: int,
    ) -> List[_TrackFrame]:
        """KCF tracker with confidence proxy (PSR-based)."""
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tracker = cv2.TrackerKCF_create()
        ret, first = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("[V2SpatialTracker] Cannot read first frame for KCF")

        tracker.init(first, init_bbox)

        raw: List[_TrackFrame] = []
        fidx = 0

        while fidx < total:
            if fidx == 0:
                x, y, w, h = init_bbox
                cx, cy = x + w / 2.0, y + h / 2.0
                if fidx % sample_stride == 0:
                    raw.append(_TrackFrame(fidx, cx, cy, (x, y, w, h), 0.9))
                fidx += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            ok, bbox_out = tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox_out]
                cx, cy = x + w / 2.0, y + h / 2.0
                conf = 0.8  # KCF has no intrinsic confidence; use fixed value
            else:
                # Tracker lost — hold last position
                if raw:
                    prev = raw[-1]
                    x, y, w, h = prev.bbox
                    cx, cy = prev.cx, prev.cy
                    conf = 0.3
                else:
                    x, y, w, h = init_bbox
                    cx, cy = x + w / 2.0, y + h / 2.0
                    conf = 0.3

            if fidx % sample_stride == 0:
                raw.append(_TrackFrame(fidx, cx, cy, (x, y, w, h), conf))

            fidx += 1

        cap.release()
        return raw

    # ── depth estimation ──────────────────────────────────────────────────────

    def _estimate_depth(
        self,
        video_path: str,
        raw_frames: List[_TrackFrame],
        K: CameraIntrinsics,
    ) -> List[_TrackFrame]:
        """Run depth estimator on each tracked frame.

        1-pass 최적화: _track_yoloworld_sam2가 YW scan 중 수집한 frame_cache가
        _TrackFrame에 첨부돼 있으면 VideoCapture 재오픈 없이 바로 사용.
        depth_stride가 첨부돼 있으면 해당 stride마다만 MiDaS 추론,
        중간 프레임은 마지막 계산값 재사용 (depth 변화가 느려 오디오 품질에 무영향).
        """
        has_depth = (self.depth_fn is not None) or (self._midas_bundle is not None)
        if not has_depth:
            # depth_m must be in the units _project_3d will read it in.
            # A flat 2.0 was metres, but with no backend is_metric is False, so
            # compute_3d_position read it as a RELATIVE depth and produced
            #     0.5 + (1 - 2.0) * (10.0 - 0.5) = -9.0 m
            # for every frame: a negative distance that clamped dist_norm to
            # 1.0 and collapsed every source onto the listener, silently.
            fallback = 2.0 if self.is_metric else 0.5
            for f in raw_frames:
                f.depth_m = fallback
            return raw_frames

        # ── frame_cache가 첨부돼 있으면 1-pass 경로 사용 ──────────────────────
        cached = getattr(raw_frames[0], "_frame_cache", None) if raw_frames else None
        depth_stride = getattr(raw_frames[0], "_depth_stride", 1) if raw_frames else 1

        if cached is not None:
            # frame_cache는 depth_stride 간격으로 저장된 프레임만 있음
            # 없는 프레임 인덱스는 가장 가까운 이전 캐시 프레임으로 대체
            sorted_keys = sorted(cached.keys())

            last_depth_val: float = 0.5  # fallback
            last_key: Optional[int] = None

            for f in raw_frames:
                # 가장 가까운 이전(≤) 캐시 키 찾기
                import bisect
                pos = bisect.bisect_right(sorted_keys, f.frame_idx)
                if pos > 0:
                    best_key = sorted_keys[pos - 1]
                else:
                    best_key = sorted_keys[0] if sorted_keys else None

                if best_key is not None and best_key != last_key:
                    frame = cached[best_key]
                    depth_val, _ = estimate_depth_at_bbox(
                        frame, f.cx, f.cy, f.bbox[2], f.bbox[3],
                        self.depth_fn, self._midas_bundle, self.is_metric,
                    )
                    last_depth_val = float(depth_val)
                    last_key = best_key

                f.depth_m = last_depth_val

            n_computed = len(sorted_keys)
            print(f"[depth] 1-pass cache: {n_computed} MiDaS calls for {len(raw_frames)} frames "
                  f"(stride≈{depth_stride})")
            return raw_frames

        # ── fallback: VideoCapture 재오픈 (기존 방식) ─────────────────────────
        cap = cv2.VideoCapture(video_path)
        frame_map: Dict[int, np.ndarray] = {}

        needed = {f.frame_idx for f in raw_frames}
        fidx = 0
        while needed:
            ret, frame = cap.read()
            if not ret:
                break
            if fidx in needed:
                frame_map[fidx] = frame
                needed.discard(fidx)
            fidx += 1
        cap.release()

        for f in raw_frames:
            frame = frame_map.get(f.frame_idx)
            if frame is None:
                f.depth_m = 2.0
                continue
            depth_val, _ = estimate_depth_at_bbox(
                frame, f.cx, f.cy, f.bbox[2], f.bbox[3],
                self.depth_fn, self._midas_bundle, self.is_metric,
            )
            f.depth_m = float(depth_val)

        return raw_frames

    # ── 2-D → 3-D projection ─────────────────────────────────────────────────

    def _project_3d(
        self,
        raw_frames: List[_TrackFrame],
        K: CameraIntrinsics,
    ) -> List[Dict]:
        """Convert pixel positions + depth to az/el/dist trajectory frames.

        f.depth_m holds the raw depth value from estimate_depth_at_bbox:
          - is_metric=True  → already meters (Metric3D, DA-V2 Metric)
          - is_metric=False → relative [0,1] (DA-V2 relative, MiDaS)
        compute_3d_position handles both via is_metric flag.
        """
        result = []
        # EMA smoothing for bbox w/h to reduce visual jitter in overlay
        ema_alpha = 0.3  # lower = smoother (0.3 = moderate)
        prev_w: Optional[float] = None
        prev_h: Optional[float] = None

        for f in raw_frames:
            az, el, dist_m, x, y, z = compute_3d_position(
                f.cx, f.cy, f.depth_m, K, self.depth_scale_m,
                is_metric=self.is_metric,
            )
            # EMA on w/h
            raw_w = float(f.bbox[2])
            raw_h = float(f.bbox[3])
            if prev_w is None:
                sm_w, sm_h = raw_w, raw_h
            else:
                sm_w = ema_alpha * raw_w + (1 - ema_alpha) * prev_w
                sm_h = ema_alpha * raw_h + (1 - ema_alpha) * prev_h
            prev_w, prev_h = sm_w, sm_h

            result.append({
                "frame":        f.frame_idx,
                "cx":           float(f.cx),   # raw pixel center (updated after Kalman)
                "cy":           float(f.cy),
                "az":           float(az),
                "el":           float(el),
                "dist_m":       float(dist_m),
                "x":            float(x),
                "y":            float(y),
                "z":            float(z),
                "w":            sm_w,           # EMA-smoothed bbox width
                "h":            sm_h,           # EMA-smoothed bbox height
                "confidence":   float(f.confidence),
                "depth_backend": self.depth_backend,
                # Intrinsics stored for post-Kalman cx/cy back-projection
                "width":        K.width,
                "height":       K.height,
                "fov_deg":      K.fov_deg,
            })
        return result

    # ── variance-gated depth enhancement ──────────────────────────────────────

    def _enhance_depth(self, frames: List[Dict]) -> List[Dict]:
        """
        Apply depth_utils.process_trajectory_depth:
          - bbox-scale proxy depth
          - proxy variance gating
          - blended depth (depth_blended)
          - d_rel computation from metric depth
        """
        try:
            cfg = DepthConfig(
                blend_strategy="metric_default",
                use_bbox_proxy=True,
                proxy_blend_by_confidence=True,
                use_proxy_variance_gating=True,
            )
            enhanced = process_trajectory_depth(frames, cfg)
            print(f"[V2SpatialTracker] Variance-gated depth enhancement applied ({len(enhanced)} frames)")
            return enhanced
        except Exception as e:
            print(f"[V2SpatialTracker] depth enhancement skipped: {e}")
            return frames

    # ── YOLO-World + SAM2 fallback tracking ──────────────────────────────────

    def _track_yoloworld_sam2(
        self,
        video_path: str,
        cls_name: str,
        init_bbox: Optional[Tuple[int, int, int, int]],
        text_query: Optional[str] = None,
        sample_stride: int = 1,
        yw_det_threshold: float = 0.99,  # YOLO-World det_rate 임계값; 이 이하면 SAM2 fallback
        yw_conf: float = 0.05,
        sam2_model: str = "sam2.1_hiera_base_plus",
        _frame_cache: Optional[Dict] = None,   # 외부에서 주입된 frame_cache (fast fallback용)
        _depth_stride: int = 5,
    ) -> List[_TrackFrame]:
        """
        Phase 1: YOLO-World로 전체 클립 빠르게 스캔 (open-vocab, text_query 지원)
        Phase 2: det_rate < yw_det_threshold 구간 → SAM2 video predictor로 보완
        Phase 3: 두 결과 merge → adaptive-K velocity 기반 confidence 재계산

        init_bbox: 사용자가 제공한 첫 프레임 bbox (SAM2 init에 사용).
                   None이면 YOLO-World의 첫 검출 bbox를 SAM2 init으로 사용.
        text_query: open-vocab text query. None이면 cls_name 사용.
        yw_det_threshold: 이 비율 미만이면 해당 구간 SAM2 fallback.
                          전체 det_rate < threshold이면 전 구간 SAM2.
        """
        import tempfile, shutil, torch
        from pathlib import Path as _Path

        tq = text_query if text_query else cls_name
        print(f"[yw_sam2] Phase 1: YOLO-World scan, text='{tq}', threshold={yw_det_threshold:.0%}")

        # ── Phase 1: YOLO-World 전체 프레임 스캔 ─────────────────────────────
        try:
            from ultralytics import YOLO as _YOLO
            yw_model = _YOLO("/home/seung/yolov8s-worldv2.pt", verbose=False)
            yw_model.set_classes([tq])
        except Exception as e:
            print(f"[yw_sam2] YOLO-World load failed: {e}, fallback to adaptive_k")
            return self._track_adaptive_k(video_path, cls_name, init_bbox, sample_stride)

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        yw_dets: List[Optional[Tuple[float, float, float, float]]] = []  # (cx,cy,w,h) or None
        yw_confs: List[float] = []
        # depth stride: depth를 depth_stride 프레임마다만 계산, 중간은 마지막 값 재사용
        depth_stride = _depth_stride
        # 외부에서 frame_cache가 주입된 경우 (fast fallback) → VideoCapture 재오픈 불필요
        frame_cache: Dict[int, np.ndarray] = _frame_cache if _frame_cache is not None else {}
        _cache_injected = _frame_cache is not None

        # init_bbox 있을 때 proximity tracking에 쓸 anchor cx/cy
        _anchor_cx: Optional[float] = None
        _anchor_cy: Optional[float] = None
        if init_bbox is not None:
            ix, iy, iw, ih = init_bbox
            _anchor_cx = float(ix + iw / 2)
            _anchor_cy = float(iy + ih / 2)

        def _iou_bbox(a_cx, a_cy, a_w, a_h, b_cx, b_cy, b_w, b_h) -> float:
            """두 (cx,cy,w,h) bbox의 IoU."""
            ax1, ay1, ax2, ay2 = a_cx - a_w/2, a_cy - a_h/2, a_cx + a_w/2, a_cy + a_h/2
            bx1, by1, bx2, by2 = b_cx - b_w/2, b_cy - b_h/2, b_cx + b_w/2, b_cy + b_h/2
            ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (a_w * a_h) + (b_w * b_h) - inter
            return inter / union if union > 0 else 0.0

        # frame_cache 주입된 경우: 캐시된 프레임으로 YW scan (VideoCapture 재오픈 없음)
        if _cache_injected:
            sorted_ck = sorted(frame_cache.keys())
            import bisect as _bisect2
            for fidx in range(total):
                if fidx % sample_stride != 0:
                    yw_dets.append(None); yw_confs.append(0.0)
                    continue
                # 가장 가까운 캐시 프레임 찾기
                pos = _bisect2.bisect_right(sorted_ck, fidx)
                ck = sorted_ck[pos - 1] if pos > 0 else sorted_ck[0]
                frame = frame_cache[ck]
                results = yw_model.predict(frame, conf=yw_conf, verbose=False)
                # detection 처리 (아래 공통 블록으로 이어짐)
                all_dets_c: List[Tuple[float, float, float, float, float]] = []
                for r in results:
                    if r.boxes is None: continue
                    for box in r.boxes:
                        c = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        all_dets_c.append((float((x1+x2)/2), float((y1+y2)/2),
                                           float(x2-x1), float(y2-y1), c))
                best_bbox, best_conf = None, 0.0
                if all_dets_c:
                    if _anchor_cx is not None:
                        if fidx == 0:
                            best_iou_c = -1.0
                            iw_f, ih_f = float(init_bbox[2]), float(init_bbox[3])
                            for cx_d, cy_d, bw_d, bh_d, c in all_dets_c:
                                iou = _iou_bbox(cx_d, cy_d, bw_d, bh_d, _anchor_cx, _anchor_cy, iw_f, ih_f)
                                if iou > best_iou_c:
                                    best_iou_c = iou; best_bbox = (cx_d, cy_d, bw_d, bh_d); best_conf = c
                            if best_iou_c < 0.05:
                                bd = min(all_dets_c, key=lambda d: (d[0]-_anchor_cx)**2+(d[1]-_anchor_cy)**2)
                                best_bbox = bd[:4]; best_conf = bd[4]
                        else:
                            bd = min(all_dets_c, key=lambda d: (d[0]-_anchor_cx)**2+(d[1]-_anchor_cy)**2)
                            dist_c = ((bd[0]-_anchor_cx)**2+(bd[1]-_anchor_cy)**2)**0.5
                            if dist_c <= cap_w * 0.10:
                                best_bbox = bd[:4]; best_conf = bd[4]
                    else:
                        for cx_d, cy_d, bw_d, bh_d, c in all_dets_c:
                            if c > best_conf: best_conf = c; best_bbox = (cx_d, cy_d, bw_d, bh_d)
                if best_bbox is not None and _anchor_cx is not None:
                    _anchor_cx = 0.7 * best_bbox[0] + 0.3 * _anchor_cx
                    _anchor_cy = 0.7 * best_bbox[1] + 0.3 * _anchor_cy
                yw_dets.append(best_bbox); yw_confs.append(best_conf)
        else:
            # 일반 경로: VideoCapture로 프레임 읽기 + YW predict (기존 로직 그대로)
            cap2 = cv2.VideoCapture(video_path)
            fidx = 0
            while fidx < total:
                ret, frame = cap2.read()
                if not ret:
                    break
                if fidx % sample_stride != 0:
                    yw_dets.append(None)
                    yw_confs.append(0.0)
                    fidx += 1
                    continue

                if fidx % depth_stride == 0:
                    frame_cache[fidx] = frame.copy()

                results = yw_model.predict(frame, conf=yw_conf, verbose=False)

                # 모든 detection 수집
                all_dets: List[Tuple[float, float, float, float, float]] = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        c = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        all_dets.append((float((x1+x2)/2), float((y1+y2)/2),
                                         float(x2-x1), float(y2-y1), c))

                best_bbox, best_conf = None, 0.0
                if all_dets:
                    if _anchor_cx is not None:
                        if fidx == 0:
                            best_iou = -1.0
                            iw_f, ih_f = float(init_bbox[2]), float(init_bbox[3])
                            for cx_d, cy_d, bw_d, bh_d, c in all_dets:
                                iou = _iou_bbox(cx_d, cy_d, bw_d, bh_d,
                                                _anchor_cx, _anchor_cy, iw_f, ih_f)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_bbox = (cx_d, cy_d, bw_d, bh_d)
                                    best_conf = c
                            if best_iou < 0.05:
                                best_bbox = min(all_dets,
                                    key=lambda d: (d[0]-_anchor_cx)**2+(d[1]-_anchor_cy)**2)
                                best_conf = best_bbox[4]
                                best_bbox = best_bbox[:4]
                        else:
                            best_det = min(all_dets,
                                key=lambda d: (d[0]-_anchor_cx)**2+(d[1]-_anchor_cy)**2)
                            nearest_dist = ((best_det[0]-_anchor_cx)**2 +
                                            (best_det[1]-_anchor_cy)**2) ** 0.5
                            _max_dist = cap_w * 0.10
                            if nearest_dist <= _max_dist:
                                best_bbox = best_det[:4]
                                best_conf = best_det[4]
                    else:
                        for cx_d, cy_d, bw_d, bh_d, c in all_dets:
                            if c > best_conf:
                                best_conf = c
                                best_bbox = (cx_d, cy_d, bw_d, bh_d)

                if best_bbox is not None and _anchor_cx is not None:
                    _anchor_cx = 0.7 * best_bbox[0] + 0.3 * _anchor_cx
                    _anchor_cy = 0.7 * best_bbox[1] + 0.3 * _anchor_cy

                yw_dets.append(best_bbox)
                yw_confs.append(best_conf)
                fidx += 1
            cap2.release()

        # align list length to total
        while len(yw_dets) < total:
            yw_dets.append(None)
            yw_confs.append(0.0)

        n_sampled = sum(1 for i, d in enumerate(yw_dets) if i % sample_stride == 0)
        n_detected = sum(1 for i, d in enumerate(yw_dets)
                         if i % sample_stride == 0 and d is not None)
        det_rate = n_detected / n_sampled if n_sampled > 0 else 0.0
        print(f"[yw_sam2] Phase 1 done: det_rate={det_rate:.1%} ({n_detected}/{n_sampled})")

        # ── Phase 2: SAM2 fallback 판단 ───────────────────────────────────────
        # SAM2 init bbox: init_bbox 우선, 없으면 YOLO-World 첫 검출 bbox
        sam2_init: Optional[Tuple[float, float, float, float]] = None
        if init_bbox is not None:
            x, y, w, h = init_bbox
            sam2_init = (float(x + w / 2), float(y + h / 2), float(w), float(h))
        else:
            for i, d in enumerate(yw_dets):
                if d is not None:
                    sam2_init = d
                    break

        # SAM2 실행 조건:
        #   det_rate < yw_det_threshold  → YW 실패 구간 보완 (기존)
        # init_bbox가 있어도 YW가 충분하면 (det_rate >= threshold) SAM2 생략 → 속도 유지
        need_sam2 = det_rate < yw_det_threshold and sam2_init is not None

        sam2_dets: List[Optional[Tuple[float, float, float, float]]] = [None] * total

        if need_sam2:
            fail_mask = [
                (i % sample_stride == 0 and yw_dets[i] is None)
                for i in range(total)
            ]
            n_fail = sum(fail_mask)
            fail_ratio = n_fail / n_sampled if n_sampled > 0 else 0.0
            print(f"[yw_sam2] Phase 2: SAM2 propagate (init_bbox mode), "
                  f"fail_ratio={fail_ratio:.1%} ({n_fail}/{n_sampled})")

            # SAM2 video predictor로 전체 클립 propagate
            # (실패 구간만 처리하려면 SAM2의 state 관리가 복잡하므로 전체 propagate 후 merge)
            try:
                from sam2.build_sam import build_sam2_video_predictor

                sam2_cfgs = {
                    "sam2.1_hiera_base_plus": (
                        "configs/sam2.1/sam2.1_hiera_b+.yaml",
                        "/home/seung/mmhoa/vid2spatial_v2/weights/sam2.1_hiera_base_plus.pt",
                    ),
                    "sam2_hiera_small": (
                        "configs/sam2/sam2_hiera_s.yaml",
                        "/home/seung/mmhoa/vid2spatial_v2/weights/sam2_hiera_small.pt",
                    ),
                }
                cfg_name, ckpt = sam2_cfgs.get(sam2_model, sam2_cfgs["sam2.1_hiera_base_plus"])

                # 임시 디렉토리에 심볼릭 링크
                tmp = _Path(tempfile.mkdtemp(prefix="yw_sam2_"))
                try:
                    # 원본 프레임 추출
                    cap2 = cv2.VideoCapture(video_path)
                    frame_paths: List[_Path] = []
                    fi2 = 0
                    while fi2 < total:
                        ok, frm = cap2.read()
                        if not ok:
                            break
                        fpath = tmp / f"{fi2:06d}.jpg"
                        cv2.imwrite(str(fpath), frm)
                        frame_paths.append(fpath)
                        fi2 += 1
                    cap2.release()

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    predictor = build_sam2_video_predictor(cfg_name, ckpt, device=device)

                    cx0, cy0, w0, h0 = sam2_init
                    x1s, y1s = cx0 - w0/2, cy0 - h0/2
                    x2s, y2s = cx0 + w0/2, cy0 + h0/2

                    # SAM2 init frame: init_bbox가 있으면 frame 0, 없으면 첫 YW 검출 frame
                    if init_bbox is not None:
                        sam2_init_frame = 0
                    else:
                        sam2_init_frame = next(
                            (i for i, d in enumerate(yw_dets) if d is not None), 0
                        )

                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        state = predictor.init_state(video_path=str(tmp))
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=sam2_init_frame,
                            obj_id=1,
                            box=np.array([x1s, y1s, x2s, y2s], dtype=np.float32),
                        )
                        for fi3, obj_ids, masks in predictor.propagate_in_video(state):
                            ids_list = (obj_ids.tolist()
                                        if hasattr(obj_ids, "tolist") else list(obj_ids))
                            if 1 in ids_list:
                                idx = ids_list.index(1)
                                mask = masks[idx].cpu().numpy().squeeze()
                                if mask.ndim > 2:
                                    mask = mask[0]
                                ys, xs = np.where(mask > 0.5)
                                if len(xs) > 0:
                                    cx_s = float(np.mean(xs))
                                    cy_s = float(np.mean(ys))
                                    bw_s = float(xs.max() - xs.min()) if len(xs) > 1 else 10.0
                                    bh_s = float(ys.max() - ys.min()) if len(ys) > 1 else 10.0
                                    if fi3 < total:
                                        sam2_dets[fi3] = (cx_s, cy_s, bw_s, bh_s)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

                n_s2 = sum(1 for d in sam2_dets if d is not None)
                print(f"[yw_sam2] SAM2 produced {n_s2}/{total} detections")

            except Exception as e:
                print(f"[yw_sam2] SAM2 failed: {e}, using YOLO-World only")

        # ── Phase 3: Merge YW + SAM2 ─────────────────────────────────────────
        # 우선순위 (init_bbox 있을 때): SAM2 우선, YW는 SAM2 실패 시 보완
        #   - SAM2는 GT init_bbox로 propagate → 항상 올바른 object를 추적
        #   - YW는 open-vocab이라 같은 카테고리 다른 object를 검출할 수 있음
        # 우선순위 (init_bbox 없을 때): YW 우선, SAM2는 YW 실패 구간만 보완
        use_sam2_primary = (init_bbox is not None) and any(d is not None for d in sam2_dets)
        merged_dets: List[Optional[Tuple]] = []
        merged_confs: List[float] = []

        for i in range(total):
            yw = yw_dets[i]
            s2 = sam2_dets[i]
            if use_sam2_primary:
                # SAM2 우선: SAM2가 있으면 SAM2, 없으면 YW
                if s2 is not None:
                    merged_dets.append(s2)
                    merged_confs.append(0.75)  # SAM2는 init_bbox 기반 → 높은 신뢰도
                elif yw is not None:
                    merged_dets.append(yw)
                    merged_confs.append(yw_confs[i])
                else:
                    merged_dets.append(None)
                    merged_confs.append(0.0)
            else:
                # YW 우선: YW가 있으면 YW, 없으면 SAM2
                if yw is not None:
                    merged_dets.append(yw)
                    merged_confs.append(yw_confs[i])
                elif s2 is not None:
                    merged_dets.append(s2)
                    merged_confs.append(0.45)
                else:
                    merged_dets.append(None)
                    merged_confs.append(0.0)

        n_merged = sum(1 for d in merged_dets if d is not None)
        final_det_rate = n_merged / total if total > 0 else 0.0
        print(f"[yw_sam2] Merged: {n_merged}/{total} = {final_det_rate:.1%} det_rate")

        # ── Phase 4: Adaptive-K velocity 기반 confidence + gating ────────────
        # merged_dets를 keyframe 리스트로 변환
        # Jump reject + velocity 기반 현재 K 계산 (adaptive_k와 동일한 로직)
        keyframes: List[tuple] = []
        last_cx: Optional[float] = None
        last_cy: Optional[float] = None
        velocity_history: List[float] = []
        current_k = 5
        jump_reject_count = 0
        low_conf_reject_count = 0
        consecutive_jump_rejects = 0
        MAX_CONSECUTIVE = 5
        frames_since_detect = 0

        sampled_indices = [i for i in range(total) if i % sample_stride == 0]
        frame_indices = sampled_indices

        for fidx in sampled_indices:
            det = merged_dets[fidx]
            conf = merged_confs[fidx]
            frames_since_detect += 1

            if det is None:
                if keyframes:
                    last_kf = keyframes[-1]
                    keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.3, current_k))
                frames_since_detect = 0
                continue

            cx, cy, bw, bh = det
            bbox = (int(cx - bw/2), int(cy - bh/2), int(bw), int(bh))

            # Confidence gating
            if conf < self.low_conf_threshold and keyframes:
                low_conf_reject_count += 1
                last_kf = keyframes[-1]
                keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.3, 1))
                frames_since_detect = 0
                continue

            # Jump reject
            if last_cx is not None:
                dist_px = math.hypot(cx - last_cx, cy - last_cy)
                vel = dist_px / max(frames_since_detect, 1)
                if vel > self.max_velocity_px:
                    jump_reject_count += 1
                    consecutive_jump_rejects += 1
                    if consecutive_jump_rejects >= MAX_CONSECUTIVE:
                        print(f"[yw_sam2] frame {fidx}: {consecutive_jump_rejects} consecutive jumps → unlock")
                        last_cx, last_cy = None, None
                        consecutive_jump_rejects = 0
                        # SAM2가 이 프레임에 대한 det가 있으면 SAM2 위치로 anchor
                        s2_here = sam2_dets[fidx] if fidx < len(sam2_dets) else None
                        if s2_here is not None:
                            cx, cy = s2_here[0], s2_here[1]
                            bw2, bh2 = s2_here[2], s2_here[3]
                            bbox = (int(cx - bw2/2), int(cy - bh2/2), int(bw2), int(bh2))
                        # fall through with (possibly SAM2-corrected) cx/cy
                    else:
                        if keyframes:
                            last_kf = keyframes[-1]
                            keyframes.append((fidx, last_kf[1], last_kf[2], last_kf[3], 0.3, 1))
                        frames_since_detect = 0
                        continue
                else:
                    consecutive_jump_rejects = 0

                velocity_history.append(vel)
                if len(velocity_history) > 5:
                    velocity_history.pop(0)
                avg_vel = float(np.mean(velocity_history))
                v_fast, v_slow = 30.0, 5.0
                if avg_vel > v_fast:
                    current_k = self.k_min
                elif avg_vel < v_slow:
                    current_k = self.k_max
                else:
                    t = (avg_vel - v_slow) / (v_fast - v_slow)
                    current_k = int(self.k_max - t * (self.k_max - self.k_min))
                    current_k = max(self.k_min, min(self.k_max, current_k))

            keyframes.append((fidx, cx, cy, bbox, conf, current_k))
            last_cx, last_cy = cx, cy
            frames_since_detect = 0

        if jump_reject_count or low_conf_reject_count:
            print(f"[yw_sam2] jump_rejects={jump_reject_count}, low_conf_rejects={low_conf_reject_count}")

        if not keyframes:
            # 완전 실패 — center로 fallback
            print("[yw_sam2] No keyframes! Fallback to center.")
            cx0, cy0 = float(cap_w // 2), float(cap_h // 2)
            for i in sampled_indices:
                keyframes.append((i, cx0, cy0, (cap_w//4, cap_h//4, cap_w//2, cap_h//2), 0.1, 5))

        result = self._interpolate_keyframes(keyframes, frame_indices)
        # frame_cache를 _TrackFrame에 첨부해서 _estimate_depth가 VideoCapture 재오픈 없이 사용
        for tf in result:
            tf._frame_cache = frame_cache
            tf._depth_stride = depth_stride
        return result

    def _track_hybrid_dino(
        self,
        video_path: str,
        cls_name: str,
        init_bbox: Optional[Tuple[int, int, int, int]],
        sample_stride: int,
    ) -> List[_TrackFrame]:
        """
        v1 HybridTracker (Grounding-DINO + adaptive-K) wrapper.

        v1의 HybridTracker를 직접 호출해서 2D tracking 결과를 받아온다.
        tracking_method="adaptive_k" — DINO로 주기적 재탐지 + AdaptiveK 간격 조절.
        HybridTracker의 depth 추정은 무시하고, 2D bbox/center만 가져온 뒤
        v2의 depth 추정 + Kalman을 그대로 적용한다.
        """
        import sys
        sys.path.insert(0, "/home/seung/mmhoa/vid2spatial")
        try:
            from vid2spatial_pkg.hybrid_tracker import HybridTracker
        except ImportError as e:
            print(f"[hybrid_dino] HybridTracker import failed: {e}, falling back to v1_bytetrack")
            return self._track_v1_bytetrack(video_path, cls_name, init_bbox, sample_stride)

        text_prompt = cls_name if cls_name else "object"
        print(f"[hybrid_dino] Running DINO+AdaptiveK, prompt='{text_prompt}'")

        try:
            tracker = HybridTracker(
                device="cuda",
                box_threshold=0.15,
                text_threshold=0.1,
                scene_type="auto",
                fov_deg=self.fov_deg,
            )
            result = tracker.track(
                video_path=video_path,
                text_prompt=text_prompt,
                sample_stride=sample_stride,
                tracking_method="adaptive_k",
                detect_scene_cuts=True,
                detect_zoom=True,
            )
        except Exception as e:
            print(f"[hybrid_dino] HybridTracker failed: {e}, falling back to v1_bytetrack")
            return self._track_v1_bytetrack(video_path, cls_name, init_bbox, sample_stride)

        # HybridTrackingResult → _TrackFrame 리스트
        raw: List[_TrackFrame] = []
        for f in result.frames:
            cx, cy = f.center
            x, y, w, h = f.bbox
            raw.append(_TrackFrame(
                frame_idx=f.frame_idx,
                cx=float(cx), cy=float(cy),
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=float(f.confidence),
            ))

        print(f"[hybrid_dino] DINO+AdaptiveK: {len(raw)} frames tracked")
        return raw

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _yolo_detect(
        yolo,
        frame: np.ndarray,
        cls_name: str,
    ) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """Run YOLO on a single frame; return (bbox_xywh, conf) or None."""
        results = yolo.track(frame, persist=True, verbose=False)
        best_conf = 0.0
        best_bbox = None

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                name = yolo.names[int(box.cls[0])]
                if cls_name and name != cls_name:
                    continue
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    bx = int(x1); by = int(y1)
                    bw = int(x2 - x1); bh = int(y2 - y1)
                    best_bbox = (bx, by, bw, bh)

        if best_bbox is None:
            return None
        return best_bbox, best_conf

    def _track_v1_bytetrack(
        self,
        video_path: str,
        cls_name: str,
        init_bbox: Optional[Tuple[int, int, int, int]],
        sample_stride: int,
    ) -> List[_TrackFrame]:
        """
        v1-style tracking: YOLO+ByteTrack at conf=0.25, with tm_track fallback.

        v1과 동일한 2D 트래킹 로직을 사용한다:
          - yolo_bytetrack_traj() conf=0.25 (gating 없음, ByteTrack이 association 담당)
          - 탐지 실패 시 tm_track() template matching fallback
          - confidence = YOLO box.conf (실제 탐지값), fallback은 0.5 고정

        결과는 _TrackFrame 리스트로 반환 → 이후 v2의 depth/_project_3d/_enhance_depth/Kalman 적용.
        """
        print(f"[v1_bytetrack] Running YOLO+ByteTrack (conf=0.25) on {video_path}")

        # ── YOLO + ByteTrack (1-pass: track + depth frame_cache 동시 수집) ────
        depth_stride = 5  # MiDaS는 5프레임마다만 계산
        frame_cache: Dict[int, np.ndarray] = {}

        try:
            from ultralytics import YOLO as _YOLO11
            bt_model = _YOLO11("yolo11n.pt")
            bt_cap = cv2.VideoCapture(video_path)
            raw_yolo: List[dict] = []
            fidx = 0
            while True:
                ok, frame = bt_cap.read()
                if not ok:
                    break
                if fidx % depth_stride == 0:
                    frame_cache[fidx] = frame.copy()
                if fidx % sample_stride == 0:
                    results = bt_model.track(frame, persist=True, verbose=False, conf=0.25)
                    for r in results:
                        if r.boxes is None or r.boxes.id is None:
                            continue
                        for box in r.boxes:
                            if bt_model.names[int(box.cls[0])].lower() != cls_name.lower():
                                continue
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                            raw_yolo.append({
                                "frame": fidx,
                                "x": x, "y": y, "w": w, "h": h,
                                "cx": x + w//2, "cy": y + h//2,
                                "track_id": int(box.id[0]),
                                "conf": float(box.conf[0]),
                            })
                fidx += 1
            bt_cap.release()
            print(f"[v1_bytetrack] 1-pass done: {fidx}f, cache={len(frame_cache)} depth frames")
        except Exception as e:
            print(f"[v1_bytetrack] 1-pass failed ({e}), fallback to yolo_bytetrack_traj")
            frame_cache = {}
            raw_yolo = yolo_bytetrack_traj(
                video_path, cls_name=cls_name,
                select_track_id=None, sample_stride=sample_stride, conf=0.25,
            )

        # ── COCO 클래스 없음 감지 → YW-only fallback (fast mode) ─────────────
        if not raw_yolo and frame_cache:
            print(f"[v1_bytetrack] YOLO 0-det (non-COCO class '{cls_name}') → YW-only fallback")
            return self._track_yoloworld_sam2(
                video_path, cls_name, init_bbox,
                text_query=cls_name,
                sample_stride=sample_stride,
                yw_det_threshold=1.01,  # SAM2 비활성: YW-only (빠름)
                yw_conf=0.05,
                _frame_cache=frame_cache,
                _depth_stride=depth_stride,
            )

        if raw_yolo:
            # ── Track ID 고정: init_bbox가 있으면 첫 프레임 IOU로 target 선택 ──
            locked_track_id: Optional[int] = None
            if init_bbox is not None:
                from .vision import _iou
                first_frame_entries = [e for e in raw_yolo if e["frame"] == raw_yolo[0]["frame"]]
                if first_frame_entries:
                    best_iou, best_id = 0.0, None
                    for e in first_frame_entries:
                        cand = (e["x"], e["y"], e["w"], e["h"])
                        iou = _iou(cand, init_bbox)
                        if iou > best_iou:
                            best_iou, best_id = iou, e.get("track_id")
                    if best_id is not None and best_iou > 0.05:
                        locked_track_id = best_id
                        print(f"[v1_bytetrack] locked track_id={locked_track_id} (IOU={best_iou:.2f})")
                    else:
                        # No IOU match — fall back to closest center
                        ibx = init_bbox[0] + init_bbox[2]/2
                        iby = init_bbox[1] + init_bbox[3]/2
                        best_dist, best_id = float('inf'), None
                        for e in first_frame_entries:
                            dist = ((e["cx"]-ibx)**2 + (e["cy"]-iby)**2)**0.5
                            if dist < best_dist:
                                best_dist, best_id = dist, e.get("track_id")
                        # Bound the fallback. Unbounded, "closest centre" is
                        # "any detection anywhere": with one detection in the
                        # frame EVERY init_bbox locks onto it, however far
                        # away. Measured: two multi-source boxes 314.8 px
                        # apart in a 640x360 frame both locked track_id=2 and
                        # produced byte-identical automation files, with the
                        # distance printed but never acted on.
                        gate_px = _center_lock_gate_px(init_bbox, frame_cache)
                        if best_id is None or best_dist > gate_px:
                            raise NoTrackForInitBBoxError(
                                f"init_bbox {tuple(init_bbox)} matches no "
                                f"detection: best IOU {best_iou:.2f} and the "
                                f"nearest detected centre is {best_dist:.1f} px "
                                f"away (gate {gate_px:.1f} px). The tracker "
                                f"would otherwise lock onto an unrelated "
                                f"object and report nothing. Draw the box on a "
                                f"detected object, or use method='yw_sam2' "
                                f"which segments the box directly.")
                        locked_track_id = best_id
                        print(f"[v1_bytetrack] locked track_id={locked_track_id} "
                              f"by center dist={best_dist:.1f}px "
                              f"(gate {gate_px:.1f}px)")

            # Build frame→entry map
            # locked_track_id 있을 때: 해당 track만 수집.
            # locked_track_id가 없는 프레임(occlusion 후 ID 재할당)에서는
            # 마지막 알려진 위치와 size에 가장 가까운 candidate를 재선택 (re-lock).
            frame_map: Dict[int, dict] = {}
            if locked_track_id is not None:
                # Step A: 먼저 locked_track_id 항목만 수집
                for entry in raw_yolo:
                    fi = entry["frame"]
                    if entry.get("track_id") == locked_track_id:
                        frame_map[fi] = entry

                # Step B: locked_track_id가 없는 구간은 center 거리로 re-lock
                # (ByteTrack이 occlusion 후 새 ID를 할당한 경우 대응)
                all_fis = sorted({e["frame"] for e in raw_yolo})
                by_frame: Dict[int, List[dict]] = {}
                for e in raw_yolo:
                    by_frame.setdefault(e["frame"], []).append(e)

                last_cx:       Optional[float] = None
                last_cy:       Optional[float] = None
                last_w:        Optional[float] = None
                last_h:        Optional[float] = None
                last_good:     Optional[int]   = None  # last frame where locked track was found
                gap_start:     Optional[int]   = None  # first missing frame in current gap
                # Re-lock only within moderate gaps (≤ MAX_RELOCK_GAP frames).
                # After a long occlusion the target could be anywhere, so it's safer
                # to hold-last rather than risk jumping to a wrong object.
                MAX_RELOCK_GAP = 30    # frames; ~1 second at 30fps
                RELOCK_MAX_PX  = 60.0  # absolute cap on search radius

                for fi in all_fis:
                    if fi in frame_map:
                        e = frame_map[fi]
                        last_cx   = float(e["cx"])
                        last_cy   = float(e["cy"])
                        last_w    = float(e["w"])
                        last_h    = float(e["h"])
                        last_good = fi
                        gap_start = None  # gap ended
                    elif last_cx is not None:
                        # No locked track in this frame → try re-lock if gap is short
                        if gap_start is None:
                            gap_start = fi
                        gap = fi - gap_start + 1
                        if gap > MAX_RELOCK_GAP:
                            continue   # gap too long — hold-last handled below

                        # Scale search radius by gap length, capped at RELOCK_MAX_PX
                        diag = math.hypot(last_w or 50, last_h or 50)
                        max_dist = min(diag * (gap / MAX_RELOCK_GAP), RELOCK_MAX_PX)

                        candidates = by_frame.get(fi, [])
                        best_dist, best_e = float("inf"), None
                        for c in candidates:
                            # Size similarity guard: within 2× of last size
                            if last_w is not None and last_h is not None:
                                cw, ch = float(c["w"]), float(c["h"])
                                if cw > 2 * last_w or ch > 2 * last_h:
                                    continue
                                if last_w > 2 * cw or last_h > 2 * ch:
                                    continue
                            d = math.hypot(float(c["cx"]) - last_cx,
                                           float(c["cy"]) - last_cy)
                            if d < best_dist:
                                best_dist, best_e = d, c
                        if best_e is not None and best_dist <= max_dist:
                            # Accept re-lock; lower confidence to flag uncertainty
                            relocked = dict(best_e)
                            relocked["conf"] = min(float(best_e.get("conf", 0.5)), 0.45)
                            frame_map[fi] = relocked
                            new_id = best_e.get("track_id")
                            if new_id != locked_track_id:
                                print(f"[v1_bytetrack] re-lock frame={fi}: "
                                      f"tid {locked_track_id}→{new_id} "
                                      f"(dist={best_dist:.1f}px, max={max_dist:.0f}px, gap={gap})")
                                locked_track_id = new_id  # update for subsequent frames
                            last_cx   = float(best_e["cx"])
                            last_cy   = float(best_e["cy"])
                            last_w    = float(best_e["w"])
                            last_h    = float(best_e["h"])
                            last_good = fi
                            gap_start = None  # reset gap counter
            else:
                # 기존 동작 (init_bbox 없을 때): best-conf per frame
                for entry in raw_yolo:
                    fi = entry["frame"]
                    if fi not in frame_map or entry.get("conf", 0) > frame_map[fi].get("conf", 0):
                        frame_map[fi] = entry

            # Collect all sampled frame indices (stride)
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            sampled = list(range(0, total, sample_stride))

            result: List[_TrackFrame] = []
            last_entry: Optional[dict] = None

            for fi in sampled:
                entry = frame_map.get(fi)
                if entry is not None:
                    x, y, w, h = entry["x"], entry["y"], entry["w"], entry["h"]
                    cx, cy = entry["cx"], entry["cy"]
                    # Use box.conf if present, else 0.6 (ByteTrack succeeded)
                    conf = float(entry.get("conf", 0.6))
                    last_entry = entry
                elif last_entry is not None:
                    # Detection miss: hold last position, lower confidence
                    x, y, w, h = last_entry["x"], last_entry["y"], last_entry["w"], last_entry["h"]
                    cx, cy = last_entry["cx"], last_entry["cy"]
                    conf = 0.3
                else:
                    continue

                result.append(_TrackFrame(
                    frame_idx=fi,
                    cx=float(cx), cy=float(cy),
                    bbox=(int(x), int(y), int(w), int(h)),
                    confidence=conf,
                ))

            if result:
                print(f"[v1_bytetrack] YOLO succeeded: {len(result)} frames tracked")
                # 1-pass: frame_cache attach → _estimate_depth가 VideoCapture 재오픈 안 함
                if frame_cache:
                    for tf in result:
                        tf._frame_cache = frame_cache
                        tf._depth_stride = depth_stride
                return result

        # ── fallback: template matching (frame_cache 우선, 없으면 VideoCapture) ──
        print(f"[v1_bytetrack] YOLO failed, falling back to tm_track")

        if frame_cache:
            # frame_cache 기반 인라인 TM (VideoCapture 재오픈 없음)
            sorted_cache_keys = sorted(frame_cache.keys())
            total_cached = max(sorted_cache_keys) + 1 if sorted_cache_keys else 0

            if init_bbox is None:
                first_frame = frame_cache[sorted_cache_keys[0]]
                H0, W0 = first_frame.shape[:2]
                w0, h0 = W0 // 5, H0 // 5
                init_bbox = (W0 // 2 - w0 // 2, H0 // 2 - h0 // 2, w0, h0)

            x0, y0, w0, h0 = init_bbox
            first = frame_cache[sorted_cache_keys[0]]
            gray0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
            template = gray0[y0:y0+h0, x0:x0+w0]
            bx, by, bw, bh = x0, y0, w0, h0

            result = []
            all_fidxs = list(range(0, max(sorted_cache_keys) + depth_stride, sample_stride))
            import bisect as _bisect
            for fi in all_fidxs:
                # 가장 가까운 캐시 프레임 찾기
                pos = _bisect.bisect_right(sorted_cache_keys, fi)
                ck = sorted_cache_keys[pos - 1] if pos > 0 else sorted_cache_keys[0]
                frame = frame_cache[ck]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                try:
                    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > 0.3:
                        bx, by = max_loc
                except Exception:
                    pass
                cx, cy = bx + bw // 2, by + bh // 2
                result.append(_TrackFrame(
                    frame_idx=fi, cx=float(cx), cy=float(cy),
                    bbox=(int(bx), int(by), int(bw), int(bh)),
                    confidence=0.5,
                ))
            # frame_cache attach
            for tf in result:
                tf._frame_cache = frame_cache
                tf._depth_stride = depth_stride
            print(f"[v1_bytetrack] tm_track (cache-based): {len(result)} frames")
            return result

        # frame_cache 없으면 원래 VideoCapture 기반 tm_track
        if init_bbox is None:
            cap = cv2.VideoCapture(video_path)
            ret, ff = cap.read()
            cap.release()
            if ret:
                H0, W0 = ff.shape[:2]
                w0, h0 = W0 // 5, H0 // 5
                init_bbox = (W0 // 2 - w0 // 2, H0 // 2 - h0 // 2, w0, h0)
            else:
                raise RuntimeError("[v1_bytetrack] Cannot read first frame for tm_track fallback")

        tm_raw = tm_track(video_path, init_bbox, sample_stride)
        result = []
        for entry in tm_raw:
            fi = entry["frame"]
            x, y, w, h = entry["x"], entry["y"], entry["w"], entry["h"]
            cx, cy = x + w / 2.0, y + h / 2.0
            result.append(_TrackFrame(
                frame_idx=fi, cx=cx, cy=cy,
                bbox=(int(x), int(y), int(w), int(h)),
                confidence=float(entry.get("conf", 0.5)),
            ))

        print(f"[v1_bytetrack] tm_track fallback: {len(result)} frames")
        return result

    @staticmethod
    def _interpolate_keyframes(
        keyframes: List[tuple],
        frame_indices: List[int],
    ) -> List[_TrackFrame]:
        """Linear interpolation between keyframes."""
        kf_dict = {kf[0]: kf for kf in keyframes}
        kf_sorted = sorted(kf_dict.keys())

        result: List[_TrackFrame] = []
        for fidx in frame_indices:
            if fidx in kf_dict:
                kf = kf_dict[fidx]
                result.append(_TrackFrame(
                    frame_idx=fidx,
                    cx=kf[1], cy=kf[2],
                    bbox=kf[3], confidence=kf[4],
                ))
                continue

            prev_idx = next((k for k in reversed(kf_sorted) if k < fidx), None)
            next_idx = next((k for k in kf_sorted           if k > fidx), None)

            if prev_idx is not None and next_idx is not None:
                prev_kf = kf_dict[prev_idx]
                next_kf = kf_dict[next_idx]
                t = (fidx - prev_idx) / (next_idx - prev_idx)

                cx = prev_kf[1] * (1 - t) + next_kf[1] * t
                cy = prev_kf[2] * (1 - t) + next_kf[2] * t
                px, py, pw, ph = prev_kf[3]
                nx, ny, nw, nh = next_kf[3]
                bbox = (
                    int(px * (1-t) + nx * t),
                    int(py * (1-t) + ny * t),
                    int(pw * (1-t) + nw * t),
                    int(ph * (1-t) + nh * t),
                )
                conf = min(prev_kf[4], next_kf[4]) * 0.9

            elif prev_idx is not None:
                kf = kf_dict[prev_idx]
                cx, cy, bbox, conf = kf[1], kf[2], kf[3], kf[4] * 0.8
            elif next_idx is not None:
                kf = kf_dict[next_idx]
                cx, cy, bbox, conf = kf[1], kf[2], kf[3], kf[4] * 0.8
            else:
                continue

            result.append(_TrackFrame(fidx, cx, cy, bbox, conf))

        return result

    def _expand_to_full_density(
        self,
        traj_frames: List[Dict],
        W: int,
        H: int,
    ) -> List[Dict]:
        """Expand sparse keyframes (stride=N) to per-frame (stride=1) by linear
        interpolation. This ensures smooth overlay rendering and lets the Kalman
        smoother work on full-resolution data instead of decimated keyframes.

        Non-spatial fields (depth_backend, proxy_stability …) are copied from
        the nearest keyframe.
        """
        if not traj_frames:
            return traj_frames

        fids = [f["frame"] for f in traj_frames]
        stride = fids[1] - fids[0] if len(fids) > 1 else 1
        if stride <= 1:
            return traj_frames   # already full-density

        first, last = fids[0], fids[-1]
        all_fids = list(range(first, last + 1))

        # Build lookup from existing sparse frames
        kf_by_fid = {f["frame"]: f for f in traj_frames}
        kf_sorted  = sorted(kf_by_fid.keys())

        def _lerp(a, b, t):
            return a + (b - a) * t

        result = []
        for fi in all_fids:
            if fi in kf_by_fid:
                result.append(kf_by_fid[fi])
                continue

            # Find bracketing keyframes
            prev_fi = next((k for k in reversed(kf_sorted) if k < fi), None)
            next_fi = next((k for k in kf_sorted if k > fi), None)

            if prev_fi is None or next_fi is None:
                src = kf_by_fid.get(prev_fi or next_fi)
                if src:
                    result.append({**src, "frame": fi, "interpolated": True})
                continue

            p = kf_by_fid[prev_fi]
            n = kf_by_fid[next_fi]
            t = (fi - prev_fi) / (next_fi - prev_fi)

            # Interpolate spatial fields
            interp: Dict = {
                "frame":    fi,
                "cx":       _lerp(p["cx"],  n["cx"],  t),
                "cy":       _lerp(p["cy"],  n["cy"],  t),
                "az":       _lerp(p["az"],  n["az"],  t),
                "el":       _lerp(p["el"],  n["el"],  t),
                "dist_m":   _lerp(p["dist_m"], n["dist_m"], t),
                "w":        _lerp(p.get("w", 50), n.get("w", 50), t),
                "h":        _lerp(p.get("h", 50), n.get("h", 50), t),
                "confidence": min(p.get("confidence", 0.5),
                                  n.get("confidence", 0.5)) * 0.95,
                "interpolated": True,
            }
            # Copy non-spatial fields from nearest keyframe
            nearest = p if t <= 0.5 else n
            for key in ("x", "y", "z", "depth_backend", "depth_proxy",
                        "depth_blended", "d_rel", "proxy_stability",
                        "dist_m_raw", "v_az", "v_el", "v_dist"):
                if key in nearest:
                    interp.setdefault(key, nearest[key])
            result.append(interp)

        return result

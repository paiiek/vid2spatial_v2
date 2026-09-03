"""
Spatial audio pipeline orchestration.
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import librosa
import numpy as np

from .config import PipelineConfig
from .vision import compute_trajectory_3d

# Try to import refactored version if available
try:
    from .vision_refactored import compute_trajectory_3d_refactored
    USE_REFACTORED_VISION = True
except ImportError:
    compute_trajectory_3d_refactored = None
    USE_REFACTORED_VISION = False

from .foa_render import (
    interpolate_angles_distance,
    smooth_limit_angles,
    apply_distance_gain_lpf,
    encode_mono_to_foa,
    write_foa_wav,
    foa_to_stereo,
    foa_to_binaural,
    foa_to_binaural_sofa,
)


class SpatialAudioPipeline:
    """
    End-to-end pipeline for creating spatial audio from video and mono audio.

    This class orchestrates the complete workflow:
    1. Vision processing: object tracking + depth estimation → 3D trajectory
    2. Audio preprocessing: room IR convolution
    3. Spatial rendering: FOA encoding with distance effects
    4. Output: FOA, stereo, binaural formats
    """

    def __init__(
        self,
        config: PipelineConfig,
        use_refactored_vision: bool = True,
        use_v2_tracker: bool = True,
        use_depth_enhance: bool = True,
        depth_priority: Optional[tuple] = None,
    ):
        """
        Initialize pipeline with configuration.

        Args:
            config: Complete pipeline configuration
            use_refactored_vision: Whether to use refactored vision module (default: True)
            use_v2_tracker: Use V2SpatialTracker (adaptive-K, confidence gating,
                            variance-gated depth, adaptive Kalman).  Default True.
                            Falls back to legacy compute_trajectory_3d on import error.
            use_depth_enhance: Run variance-gated depth blending (_enhance_depth).
                               Set False to use raw MiDaS dist_m only.
            depth_priority: Override depth field lookup order in audio render.
                            Default: ("depth_render","depth_blended","dist_m_raw","dist_m").
                            E.g. ("dist_m",) to skip blended/render fields.
        """
        self.config = config
        self.use_refactored_vision = use_refactored_vision and USE_REFACTORED_VISION
        self.use_v2_tracker = use_v2_tracker
        self.use_depth_enhance = use_depth_enhance
        self.depth_priority = depth_priority
        self._sam2_mask_fn: Optional[Callable] = None
        self._depth_fn: Optional[Callable] = None
        self._trajectory: Optional[Dict[str, Any]] = None

    def _build_sam2_predictor(self) -> Optional[Callable]:
        """Build SAM2 predictor if refinement is enabled."""
        if not self.config.vision.refinement.enabled:
            return None

        if self.config.vision.refinement.method != 'sam2':
            return None

        try:
            from .sam2_adapter import build_sam2_predictor
            predictor = build_sam2_predictor(
                checkpoint_path=self.config.vision.refinement.sam_ckpt,
                config_path=None,
                model_id=self.config.vision.refinement.sam2_model_id
            )
            return predictor
        except Exception as e:
            print(f'[warn] SAM2 integration failed: {e}')
            print('[warn] falling back to grabcut center refinement')
            self.config.vision.refinement.method = 'grabcut'
            return None

    def _build_depth_predictor(self) -> Optional[Callable]:
        """Build depth predictor.

        Always built when room.backend == 'visual', because _apply_visual_room_ir
        needs self._depth_fn to estimate room geometry from the first video frame.
        When use_adapter is False and backend is not visual, skip (depth will be
        handled inside compute_trajectory via initialize_depth_backend).
        """
        needs_depth_for_room = (self.config.room.backend == "visual")
        if not self.config.vision.depth.use_adapter and not needs_depth_for_room:
            return None

        backend = self.config.vision.depth.backend

        # For visual room estimation, prefer metric3d → depth_anything_v2 → fallback
        if needs_depth_for_room and backend in ("auto", "metric3d"):
            try:
                from .vision import get_metric3d_estimator
                est = get_metric3d_estimator(
                    model_size=self.config.vision.depth.model_size,
                    focal_length_px=self.config.vision.depth.focal_length_px,
                )
                print('[info] Depth predictor (Metric3D v2) built for visual room estimation')
                return est.infer
            except Exception as e:
                print(f'[warn] Metric3D init failed for visual room: {e}, trying DA-V2')

        try:
            from .depth_anything_adapter import build_depth_predictor
            fn = build_depth_predictor()
            print('[info] Depth predictor (DA-V2) built for visual room estimation')
            return fn
        except Exception as e:
            print(f'[warn] depth adapter init failed: {e}')
            return None

    @staticmethod
    def _clip_outlier_distances(frames: list, iqr_k: float = 5.0) -> list:
        """Clip extreme dist_m outliers using IQR-based fence.

        M3D occasionally produces runaway values (e.g. 212m for a nearby dog)
        when the depth model fails on a frame.  We use a conservative IQR fence
        (default k=5) so only genuine outliers are clamped; normal dynamic range
        (e.g. 0.5–20m) is preserved.

        Applies in-place clipping on the 'dist_m' key (and dist_m_raw /
        depth_render / depth_blended if present).
        """
        _depth_keys = ("depth_render", "depth_blended", "dist_m_raw", "dist_m")
        for key in _depth_keys:
            vals = [f[key] for f in frames if key in f]
            if len(vals) < 4:
                continue
            q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            iqr = q3 - q1
            if iqr < 0.01:
                continue
            hi = q3 + iqr_k * iqr
            clipped = 0
            for f in frames:
                if key in f and f[key] > hi:
                    f[key] = hi
                    clipped += 1
            if clipped:
                print(f'[info] Outlier clipping: {key} clamped {clipped} frames to ≤{hi:.2f}m (IQR fence k={iqr_k})')
        return frames

    def _resolve_camera_fov(self):
        """Resolve the horizontal FOV to use, with provenance (gap item A2).

        Reads the container metadata unless the user pinned a FOV explicitly.
        Falls back to config.vision.camera.fov_deg (60 deg by default) with a
        loud warning, and records where the number came from so a downstream
        result is auditable.
        """
        from .camera_intrinsics import resolve_fov

        cam = self.config.vision.camera
        info = resolve_fov(
            self.config.video_path,
            explicit_fov_deg=cam.fov_deg if cam.fov_explicit else None,
            focal_35mm=cam.focal_35mm,
            default_fov_deg=cam.fov_deg,
            use_metadata=cam.fov_from_metadata,
        )
        cam.fov_deg = float(info.fov_deg)
        cam.fov_source = info.source
        cam.fov_detail = info.detail
        if not info.is_default:
            print(f'[info] camera FOV {info.fov_deg:.2f} deg from {info.source} '
                  f'({info.detail})')
        return info

    def _stamp_intrinsics(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Record FOV provenance (and camera yaw, when estimated) in the traj."""
        cam = self.config.vision.camera
        intr = traj.setdefault("intrinsics", {})
        intr["fov_deg"] = float(cam.fov_deg)
        intr["fov_source"] = cam.fov_source
        intr["fov_detail"] = cam.fov_detail
        intr["motion_mode"] = cam.motion_mode
        return traj

    def _apply_camera_motion(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate camera yaw and, in world_frame mode, subtract it (A3)."""
        cam = self.config.vision.camera
        if cam.motion_mode == "camera_frame" and not cam.estimate_camera_motion:
            return traj
        from .camera_motion import annotate_trajectory_with_camera_motion
        return annotate_trajectory_with_camera_motion(
            traj, self.config.video_path,
            fov_deg=cam.fov_deg,
            mode=cam.motion_mode,
            sample_stride=cam.sample_stride,
        )

    def _compute_trajectory(self) -> Dict[str, Any]:
        """
        Compute or load 3D trajectory from video.

        Returns:
            Trajectory dict with 'intrinsics' and 'frames' keys
        """
        # Check for precomputed trajectory
        if self.config.trajectory_json and os.path.exists(self.config.trajectory_json):
            print(f'[info] Loading precomputed trajectory from {self.config.trajectory_json}')
            with open(self.config.trajectory_json, "r") as f:
                traj = json.load(f)

            # Outlier clipping (handles M3D runaway values in stored JSONs)
            traj['frames'] = self._clip_outlier_distances(traj['frames'])

            # Kalman smoothing also applies to precomputed trajectories so that
            # stored raw trajectories (e.g. from a previous run without smoothing)
            # still benefit from temporal noise reduction.
            if self.config.spatial.use_kalman_smoothing:
                print('[info] Applying Kalman filter temporal smoothing to precomputed trajectory...')
                from .temporal_smoother import smooth_trajectory_batch
                import cv2
                cap = cv2.VideoCapture(self.config.video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps <= 0:
                    fps = 30.0
                traj['frames'] = smooth_trajectory_batch(
                    traj['frames'],
                    fps=fps,
                    process_noise=0.01,
                    measurement_noise=0.1,
                )

            # FOV provenance for a precomputed trajectory comes from the file
            # itself, not from the container (A2).
            intr = traj.setdefault("intrinsics", {})
            intr.setdefault("fov_deg", self.config.vision.camera.fov_deg)
            intr.setdefault("fov_source", "trajectory_json")
            self.config.vision.camera.fov_deg = float(intr["fov_deg"])
            self.config.vision.camera.fov_source = str(intr["fov_source"])
            traj = self._apply_camera_motion(traj)

            # Save post-processed trajectory to output path (if requested)
            if self.config.output.trajectory_path:
                print(f'[info] Saving processed trajectory to {self.config.output.trajectory_path}')
                with open(self.config.output.trajectory_path, "w") as fh:
                    json.dump(traj, fh, indent=2)

            return traj

        # Resolve the camera FOV from container metadata before any projection
        # happens; everything downstream reads config.vision.camera.fov_deg (A2).
        self._resolve_camera_fov()

        # Build vision components
        print('[info] Computing trajectory from video...')
        if self._sam2_mask_fn is None:
            self._sam2_mask_fn = self._build_sam2_predictor()

        if self._depth_fn is None:
            self._depth_fn = self._build_depth_predictor()

        # ── V2SpatialTracker path ──────────────────────────────────────────
        # Applies adaptive-K, confidence gating, jump reject, variance-gated
        # depth blending, and adaptive Kalman — all in one place.
        if self.use_v2_tracker:
            try:
                from .v2_spatial_tracker import V2SpatialTracker

                # Pass depth_fn=None so V2SpatialTracker initializes its own
                # depth backend and correctly sets is_metric.
                # (Passing self._depth_fn would cause initialize_depth_backend
                # to short-circuit with is_metric=False even for metric backends.)
                tracker = V2SpatialTracker(
                    depth_fn=None,   # tracker loads its own depth backend
                    fov_deg=self.config.vision.camera.fov_deg,
                    depth_backend=self.config.vision.depth.backend,
                )

                tracking_method = self.config.vision.tracking.method
                # "kcf"          → KCF tracker
                # "v1_bytetrack" / "yolo" → v1-style YOLO+ByteTrack
                # "hybrid_dino"  → Grounding-DINO + AdaptiveK
                # "auto"         → speed-based selection (fast→bytetrack, slow→adaptive_k)
                # anything else  → adaptive-K YOLO
                if tracking_method == "kcf":
                    v2_method = "kcf"
                elif tracking_method in ("yolo", "v1_bytetrack"):
                    v2_method = "v1_bytetrack"
                elif tracking_method == "hybrid_dino":
                    v2_method = "hybrid_dino"
                elif tracking_method == "auto":
                    v2_method = "auto"
                else:
                    v2_method = "adaptive_k"

                print(f'[info] Using V2SpatialTracker (method={v2_method}, '
                      f'depth={self.config.vision.depth.backend})')
                traj = tracker.track(
                    video_path=self.config.video_path,
                    cls_name=self.config.vision.tracking.class_name,
                    init_bbox=self.config.vision.tracking.init_bbox,
                    sample_stride=self.config.vision.camera.sample_stride,
                    method=v2_method,
                    enhance_depth=self.use_depth_enhance,
                )

                # Outlier clipping (in case depth estimator produced runaways)
                traj['frames'] = self._clip_outlier_distances(traj['frames'])

                # Store fps in trajectory so audio renderer maps frame→sample correctly.
                # V2SpatialTracker may already include 'fps' from its track() return.
                # If not (e.g. image sequence where cv2 returns 0), re-read from file.
                if 'fps' not in traj or traj['fps'] <= 0:
                    import cv2 as _cv2
                    _cap = _cv2.VideoCapture(self.config.video_path)
                    _fps = _cap.get(_cv2.CAP_PROP_FPS); _cap.release()
                    if _fps > 0:
                        traj['fps'] = float(_fps)
                    elif '%' in self.config.video_path or self.config.video_path.endswith('.jpg'):
                        # Image sequence (LaSOT etc.) — cv2 returns 0; use 25fps (LaSOT standard)
                        traj['fps'] = 25.0
                    else:
                        traj['fps'] = 30.0

                traj = self._stamp_intrinsics(traj)
                traj = self._apply_camera_motion(traj)

                # Save if requested
                if self.config.output.trajectory_path:
                    print(f'[info] Saving trajectory to {self.config.output.trajectory_path}')
                    with open(self.config.output.trajectory_path, "w") as f:
                        json.dump(traj, f, indent=2)

                return traj

            except Exception as e:
                print(f'[warn] V2SpatialTracker failed ({e}), falling back to legacy vision module')

        # ── Legacy vision path (fallback) ─────────────────────────────────
        # Choose vision implementation
        compute_fn = compute_trajectory_3d_refactored if self.use_refactored_vision else compute_trajectory_3d

        if self.use_refactored_vision:
            print('[info] Using refactored vision module (legacy path)')

        # Compute trajectory
        traj = compute_fn(
            self.config.video_path,
            init_bbox=self.config.vision.tracking.init_bbox,
            fov_deg=self.config.vision.camera.fov_deg,
            sample_stride=self.config.vision.camera.sample_stride,
            method=self.config.vision.tracking.method,
            cls_name=self.config.vision.tracking.class_name,
            refine_center=self.config.vision.refinement.enabled,
            refine_center_method=self.config.vision.refinement.method,
            depth_backend=self.config.vision.depth.backend,
            sam2_mask_fn=self._sam2_mask_fn,
            depth_fn=self._depth_fn,
            sam2_model_id=self.config.vision.refinement.sam2_model_id,
            sam2_cfg=self.config.vision.refinement.sam2_cfg,
            sam2_ckpt=self.config.vision.refinement.sam2_ckpt,
            select_track_id=self.config.vision.tracking.select_track_id,
            smooth_alpha=self.config.vision.tracking.smooth_alpha,
            fallback_center_if_no_bbox=self.config.vision.tracking.fallback_center_if_no_bbox,
            target_color=self.config.vision.tracking.target_color,
            color_tolerance=self.config.vision.tracking.color_tolerance,
            color_min_area=self.config.vision.tracking.color_min_area,
            point_method=self.config.vision.tracking.point_method,
            point_min_brightness=self.config.vision.tracking.point_min_brightness,
            point_template_path=self.config.vision.tracking.point_template_path,
            point_template_threshold=self.config.vision.tracking.point_template_threshold,
            point_use_optical_flow=self.config.vision.tracking.point_use_optical_flow,
            skeleton_joint=self.config.vision.tracking.skeleton_joint,
            skeleton_backend=self.config.vision.tracking.skeleton_backend,
            skeleton_min_visibility=self.config.vision.tracking.skeleton_min_visibility,
            skeleton_smooth_alpha=self.config.vision.tracking.skeleton_smooth_alpha,
            depth_model_size=self.config.vision.depth.model_size,
            focal_length_px=self.config.vision.depth.focal_length_px,
        )

        # Outlier clipping before smoothing (handles M3D runaway values)
        traj['frames'] = self._clip_outlier_distances(traj['frames'])

        # Apply Kalman filter temporal smoothing
        if self.config.spatial.use_kalman_smoothing:
            print('[info] Applying Kalman filter temporal smoothing...')
            from .temporal_smoother import smooth_trajectory_batch

            # Get FPS from video
            import cv2
            cap = cv2.VideoCapture(self.config.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            if fps <= 0:
                fps = 30.0  # Default fallback

            traj['frames'] = smooth_trajectory_batch(
                traj['frames'],
                fps=fps,
                process_noise=0.01,
                measurement_noise=0.1
            )

        # Store fps in trajectory so audio renderer maps frame→sample correctly
        if 'fps' not in traj or traj['fps'] <= 0:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(self.config.video_path)
            _fps = _cap.get(_cv2.CAP_PROP_FPS); _cap.release()
            if _fps > 0:
                traj['fps'] = float(_fps)
            elif '%' in self.config.video_path or self.config.video_path.endswith('.jpg'):
                traj['fps'] = 25.0  # LaSOT image sequence standard
            else:
                traj['fps'] = 30.0

        # Save if requested
        if self.config.output.trajectory_path:
            print(f'[info] Saving trajectory to {self.config.output.trajectory_path}')
            with open(self.config.output.trajectory_path, "w") as f:
                json.dump(traj, f, indent=2)

        return traj

    def _apply_room_ir(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply room impulse response to audio.

        Args:
            audio: Input audio signal
            sr: Sample rate

        Returns:
            Audio with room IR applied
        """
        if self.config.room.disabled or self.config.room.backend in ("none", "brir"):
            return audio.astype(np.float32)

        # Visual room estimation from depth map
        if self.config.room.backend == "visual":
            return self._apply_visual_room_ir(audio, sr)

        Lx, Ly, Lz = self.config.room.dimensions
        mx, my, mz = self.config.room.mic_position

        # FAIR-Play matched IR (recommended based on ablation study)
        if self.config.room.backend == "fairplay":
            from .irgen import fairplay_matched_ir
            print('[info] Using FAIR-Play matched IR (dry acoustics)...')
            rir = fairplay_matched_ir(fs=sr, rt60=self.config.room.rt60)
            return self._convolve_ir(audio, rir)

        # Try PRA backend
        if self.config.room.backend in ("auto", "pra"):
            try:
                from .irgen import synthesize_mono_rir, fft_convolve
                print('[info] Synthesizing room IR with pyroomacoustics...')
                rir = synthesize_mono_rir(
                    (Lx, Ly, Lz),
                    (mx - 1.0, my, mz),  # Source offset
                    (mx, my, mz),  # Mic position
                    fs=sr,
                    rt60=self.config.room.rt60
                )
                return self._convolve_ir(audio, rir)
            except Exception as e:
                print(f'[warn] PRA backend failed: {e}, falling back to FAIR-Play matched IR')
                # Fallback to FAIR-Play matched instead of Schroeder
                from .irgen import fairplay_matched_ir
                print('[info] Using FAIR-Play matched IR (fallback)...')
                rir = fairplay_matched_ir(fs=sr, rt60=self.config.room.rt60)
                return self._convolve_ir(audio, rir)

        # Schroeder IR (legacy, not recommended)
        from .irgen import schroeder_ir
        print('[info] Using Schroeder IR...')
        rir = schroeder_ir(sr, rt60=self.config.room.rt60)
        return self._convolve_ir(audio, rir)

    def _convolve_ir(self, audio: np.ndarray, rir: np.ndarray,
                     wet_mix: float = 1.0) -> np.ndarray:
        """Convolve audio with energy-normalized IR and blend with dry signal.

        Args:
            audio: Dry input signal (float32)
            rir: Room impulse response
            wet_mix: Wet/dry blend ratio [0=dry, 1=full wet]. When < 1.0,
                     the IR is energy-normalized so the wet component has the
                     same RMS as the dry, then blended: out = (1-w)*dry + w*wet.
        """
        from .irgen import fft_convolve
        dry = audio.astype(np.float32)
        wet = fft_convolve(dry, rir)

        # Trim or pad to original length
        if wet.shape[0] >= dry.shape[0]:
            wet = wet[: dry.shape[0]]
        else:
            wet = np.pad(wet, (0, dry.shape[0] - wet.shape[0]))

        if wet_mix >= 1.0:
            return wet

        # Energy-normalize wet to match dry RMS, then blend
        dry_rms = float(np.sqrt(np.mean(dry ** 2))) + 1e-9
        wet_rms = float(np.sqrt(np.mean(wet ** 2))) + 1e-9
        wet_norm = wet * (dry_rms / wet_rms)
        return (1.0 - wet_mix) * dry + wet_mix * wet_norm

    def _apply_visual_room_ir(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply room IR estimated from video's depth map."""
        import cv2
        try:
            from .visual_room_estimator import VisualRoomEstimator

            # Extract first frame for depth-based room estimation
            cap = cv2.VideoCapture(self.config.video_path)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print("[warn] Could not read video frame for room estimation, skipping IR")
                return audio.astype(np.float32)

            # Get depth map from first frame
            if self._depth_fn is not None:
                depth_map = self._depth_fn(frame)
            else:
                print("[warn] No depth function for visual room estimation, using fallback IR")
                from .irgen import fairplay_matched_ir
                rir = fairplay_matched_ir(fs=sr, rt60=self.config.room.rt60)
                return self._convolve_ir(audio, rir)

            # Estimate room and generate IR
            estimator = VisualRoomEstimator(fov_deg=self.config.vision.camera.fov_deg)
            rir, estimate = estimator.estimate_and_generate(
                depth_map, fs=sr,
                fov_deg=self.config.vision.camera.fov_deg,
            )
            wet_mix = self.config.room.visual_wet_mix
            return self._convolve_ir(audio, rir, wet_mix=wet_mix)

        except Exception as e:
            print(f"[warn] Visual room estimation failed: {e}, skipping IR")
            return audio.astype(np.float32)

    def _load_occlusion_timeline(self, T: int) -> Optional[np.ndarray]:
        """
        Load or estimate occlusion timeline.

        Args:
            T: Number of audio samples

        Returns:
            Occlusion values [0, 1] per sample, or None
        """
        if not self.config.occlusion.enabled:
            return None

        # Load from JSON
        if self.config.occlusion.json_path and os.path.exists(self.config.occlusion.json_path):
            print(f'[info] Loading occlusion from {self.config.occlusion.json_path}')
            with open(self.config.occlusion.json_path, "r") as f:
                occ_data = json.load(f)

            if isinstance(occ_data, dict) and "frames" in occ_data:
                idx = np.array([it.get("frame", 0) for it in occ_data["frames"]], np.float32)
                occ = np.array([float(it.get("occ", 0.0)) for it in occ_data["frames"]], np.float32)
                _fps = float(self._trajectory.get("fps", 30.0)) if self._trajectory else 30.0
                _spf = float(self.config.spatial._sr if hasattr(self.config.spatial, '_sr') else 48000) / _fps
                idx_samples = idx * _spf
                s = np.arange(T, dtype=np.float32)
                return np.interp(s, idx_samples, occ).astype(np.float32)

        # Estimate from video
        if self.config.occlusion.estimate:
            try:
                from .occlusion import estimate_occlusion_timeline
                print('[info] Estimating occlusion from video...')
                occ_tl = estimate_occlusion_timeline(
                    self.config.video_path,
                    self._trajectory["frames"],
                    use_depth=True,
                    stride=self.config.vision.camera.sample_stride
                )
                idx = np.array([it.get("frame", 0) for it in occ_tl["frames"]], np.float32)
                occ = np.array([float(it.get("occ", 0.0)) for it in occ_tl["frames"]], np.float32)
                _fps = float(self._trajectory.get("fps", 30.0)) if self._trajectory else 30.0
                _spf = float(self.config.spatial._sr if hasattr(self.config.spatial, '_sr') else 48000) / _fps
                idx_samples = idx * _spf
                s = np.arange(T, dtype=np.float32)
                return np.interp(s, idx_samples, occ).astype(np.float32)
            except Exception as e:
                print(f"[warn] occlusion estimation failed: {e}")

        return None

    def _render_spatial_audio(
        self,
        audio: np.ndarray,
        sr: int,
        trajectory: Dict[str, Any]
    ) -> np.ndarray:
        """
        Render spatial FOA audio from mono audio and trajectory.

        Args:
            audio: Mono audio signal
            sr: Sample rate
            trajectory: 3D trajectory dict

        Returns:
            FOA audio [4, T]
        """
        T = audio.shape[0]

        # Interpolate angles and distance
        print('[info] Interpolating trajectory to audio timeline...')
        fps = float(trajectory.get("fps", trajectory.get("intrinsics", {}).get("fps", 30.0)))
        az_s, el_s, dist_s, d_rel_s = interpolate_angles_distance(
            trajectory["frames"], T=T, sr=sr, fps=fps,
            depth_keys=self.depth_priority,
        )

        # Smooth angles
        print('[info] Smoothing angles...')
        az_s, el_s = smooth_limit_angles(
            az_s, el_s, sr,
            smooth_ms=self.config.spatial.angle_smooth_ms,
            max_deg_per_s=self.config.spatial.max_deg_per_s
        )

        # Apply distance effects
        _use_lm = self.config.spatial.use_learned_mapping
        _lm_path = self.config.spatial.learned_model_path
        if _use_lm:
            print('[info] Applying LEARNED distance-based gain and filtering...')
        else:
            print('[info] Applying distance-based gain and filtering...')
        audio_dist = apply_distance_gain_lpf(
            audio, sr, dist_s, d_rel_s,
            gain_k=self.config.spatial.dist_gain_k,
            lpf_min_hz=self.config.spatial.dist_lpf_min_hz,
            lpf_max_hz=self.config.spatial.dist_lpf_max_hz,
            use_learned_mapping=_use_lm,
            learned_model_path=_lm_path,
        )

        # Load occlusion if needed
        occ_s = self._load_occlusion_timeline(T)

        # Apply reverb coupling if enabled
        if self.config.reverb.enabled:
            print('[info] Applying time-varying reverb...')
            from .foa_render import build_wet_curve_from_dist_occ, apply_timevarying_reverb_mono
            wet = build_wet_curve_from_dist_occ(
                d_rel_s, occ_s,
                wet_min=self.config.reverb.wet_min,
                wet_max=self.config.reverb.wet_max,
                occ_boost=self.config.reverb.wet_occ_boost,
                use_learned_mapping=_use_lm,
                learned_model_path=_lm_path,
            )
            audio_dist = apply_timevarying_reverb_mono(audio_dist, sr, wet, rt60=self.config.reverb.rt60)

        # Encode to FOA
        # Pipeline az convention: az>0 = RIGHT of image (atan2(x,z))
        # AmbiX/SOFA convention: az>0 = LEFT (counterclockwise from front)
        # → must negate before FOA encoding (same as render_foa_from_trajectory)
        print('[info] Encoding to first-order ambisonics...')
        foa = encode_mono_to_foa(audio_dist, -az_s, el_s)

        # Optional FOA AIR convolution
        if self.config.air_foa_path:
            print(f'[info] Applying FOA AIR from {self.config.air_foa_path}')
            import soundfile as sf
            air, sra = sf.read(self.config.air_foa_path, always_2d=True)
            air = air.T.astype(np.float32)

            if air.shape[0] != 4:
                raise ValueError("--air_foa must be 4-channel AmbiX wav")

            if self.config.reverb.enabled:
                from .foa_render import build_wet_curve_from_dist_occ, apply_timevarying_reverb_foa
                wet = build_wet_curve_from_dist_occ(
                    d_rel_s, occ_s,
                    wet_min=self.config.reverb.wet_min,
                    wet_max=self.config.reverb.wet_max,
                    occ_boost=self.config.reverb.wet_occ_boost
                )
                foa = apply_timevarying_reverb_foa(foa, sr, wet, air_foa=air, rt60=self.config.reverb.rt60)
            else:
                from .irgen import fft_convolve
                foa_conv = np.zeros_like(foa)
                for ch in range(4):
                    foa_conv[ch] = fft_convolve(foa[ch], air[ch])[:T]
                foa = foa_conv

        return foa

    def _write_outputs(self, foa: np.ndarray, sr: int):
        """
        Write output files in various formats.

        Args:
            foa: FOA audio [4, T]
            sr: Sample rate
        """
        # Write FOA
        print(f'[info] Writing FOA to {self.config.output.foa_path}')
        write_foa_wav(self.config.output.foa_path, foa, sr)

        # Write stereo if requested
        if self.config.output.stereo_path:
            print(f'[info] Writing stereo to {self.config.output.stereo_path}')
            stereo = foa_to_stereo(foa, sr)
            import soundfile as sf
            sf.write(self.config.output.stereo_path, stereo.T, sr)

        # Write binaural if requested
        if self.config.output.binaural_path:
            print(f'[info] Writing binaural to {self.config.output.binaural_path}')

            if self.config.output.binaural_config.mode == "sofa":
                if not self.config.output.binaural_config.sofa_path:
                    print('[warn] SOFA mode requires --sofa path, falling back to crossfeed')
                    binaural = foa_to_binaural(foa, sr)
                else:
                    binaural = foa_to_binaural_sofa(
                        foa, sr,
                        self.config.output.binaural_config.sofa_path
                    )
            else:
                binaural = foa_to_binaural(foa, sr)

            import soundfile as sf
            sf.write(self.config.output.binaural_path, binaural.T, sr)

    def run(self) -> Dict[str, Any]:
        """
        Run the complete spatial audio pipeline.

        Returns:
            Dictionary with pipeline outputs and metadata
        """
        print('='*60)
        print('Spatial Audio Pipeline')
        print('='*60)

        # Pre-build depth predictor when visual room IR is requested.
        # _compute_trajectory() returns early (without building _depth_fn) when
        # a precomputed trajectory_json is provided, so we must ensure _depth_fn
        # is ready before _apply_room_ir() calls _apply_visual_room_ir().
        if self.config.room.backend == "visual" and not self.config.room.disabled:
            if self._depth_fn is None:
                print('[info] Pre-building depth predictor for visual room estimation...')
                self._depth_fn = self._build_depth_predictor()
                if self._depth_fn is None:
                    print('[warn] Could not build depth predictor; visual room will use fallback IR')

        # Step 1: Compute trajectory
        print('\n[1/4] Computing 3D trajectory from video...')
        self._trajectory = self._compute_trajectory()
        print(f'      → Found {len(self._trajectory["frames"])} trajectory frames')
        if self.config.output.automation_path:
            from .trajectory_export import export_trajectory
            out = export_trajectory(
                self._trajectory, self.config.output.automation_path,
                fps=float(self._trajectory.get('fps') or 30.0),
            )
            print(f'      → Wrote automation export to {out}')

        # Step 2: Load and process audio
        print('\n[2/4] Loading and processing audio...')
        audio, sr = librosa.load(self.config.audio_path, sr=None, mono=True)
        print(f'      → Loaded {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f}s)')

        audio_processed = self._apply_room_ir(audio, sr)

        # Step 3: Render spatial audio
        print('\n[3/4] Rendering spatial audio...')
        foa = self._render_spatial_audio(audio_processed, sr, self._trajectory)
        print(f'      → Generated FOA audio: {foa.shape}')

        # Step 4: Write outputs
        print('\n[4/4] Writing outputs...')
        self._write_outputs(foa, sr)

        print('\n' + '='*60)
        print('Pipeline completed successfully!')
        print('='*60)

        return {
            "trajectory": self._trajectory,
            "sample_rate": sr,
            "duration_sec": len(audio) / sr,
            "num_frames": len(self._trajectory["frames"]),
            "outputs": {
                "foa": self.config.output.foa_path,
                "stereo": self.config.output.stereo_path,
                "binaural": self.config.output.binaural_path,
            }
        }

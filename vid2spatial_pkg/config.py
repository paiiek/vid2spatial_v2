"""
Configuration classes for vid2spatial pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from pathlib import Path


@dataclass
class CameraConfig:
    """Camera and video processing configuration."""
    fov_deg: float = 60.0
    sample_stride: int = 1


@dataclass
class TrackingConfig:
    """Object tracking configuration."""
    method: str = "yolo"  # yolo, kcf, ostrack, sam2, color, point, skeleton
    class_name: str = "person"
    select_track_id: Optional[int] = None
    init_bbox: Optional[Tuple[int, int, int, int]] = None
    fallback_center_if_no_bbox: bool = False
    smooth_alpha: float = 0.2
    ostrack_checkpoint: Optional[str] = None  # Custom OSTrack checkpoint path

    # Color tracking (for hand-drawn sketches, colored markers)
    target_color: Optional[Tuple[int, int, int]] = None  # BGR color (e.g., (255,0,0) for red)
    color_tolerance: int = 30  # HSV tolerance for color matching
    color_min_area: int = 100  # Minimum blob area in pixels

    # Point tracking (for cursor, laser pointer, etc.)
    point_method: str = "brightness"  # brightness, template, goodfeatures
    point_min_brightness: int = 200  # Minimum brightness (0-255)
    point_template_path: Optional[str] = None  # Template image path
    point_template_threshold: float = 0.7  # Template matching threshold
    point_use_optical_flow: bool = True  # Use optical flow for tracking

    # Skeleton tracking (for motion capture, dance, etc.)
    skeleton_joint: str = "right_wrist"  # nose, left_wrist, right_wrist, left_ankle, etc.
    skeleton_backend: str = "mediapipe"  # mediapipe (only backend supported currently)
    skeleton_min_visibility: float = 0.5  # Minimum joint visibility (0-1)
    skeleton_smooth_alpha: float = 0.3  # Smoothing factor (0=max smooth, 1=no smooth)


@dataclass
class DepthConfig:
    """Depth estimation configuration."""
    backend: str = "auto"  # auto, metric3d, midas, depth_anything_v2, none
    use_adapter: bool = False
    model_size: str = "small"  # For depth_anything_v2/metric3d: small, base/large, giant
    focal_length_px: Optional[float] = None  # For metric3d: camera focal length in pixels


@dataclass
class RefinementConfig:
    """Center refinement configuration."""
    enabled: bool = False
    method: str = "grabcut"  # grabcut, sam2
    sam_ckpt: Optional[str] = None
    sam2_model_id: str = "facebook/sam2.1-hiera-base-plus"
    sam2_cfg: Optional[str] = None
    sam2_ckpt: Optional[str] = None


@dataclass
class VisionConfig:
    """Complete vision processing configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)


@dataclass
class RoomConfig:
    """Room acoustics configuration."""
    dimensions: Tuple[float, float, float] = (6.0, 5.0, 3.0)  # Lx, Ly, Lz
    mic_position: Tuple[float, float, float] = (3.0, 2.5, 1.5)  # mx, my, mz
    rt60: float = 0.5  # Reduced from 0.6 based on GT analysis
    backend: str = "auto"  # auto, pra, schroeder, fairplay, none, visual, brir
    disabled: bool = True  # Default disabled: ablation study shows IR degrades performance
    # Visual room IR wet/dry mix (0=dry, 1=full wet).
    # Energy-normalized IR is blended: out = (1-wet_mix)*dry + wet_mix*wet
    visual_wet_mix: float = 0.30  # 30% wet — scene-adaptive reverb without energy blowup


@dataclass
class SpatialConfig:
    """Spatial audio rendering configuration."""
    # Angle smoothing
    angle_smooth_ms: float = 50.0
    max_deg_per_s: Optional[float] = None

    # Distance mapping
    dist_gain_k: float = 1.0
    dist_lpf_min_hz: float = 800.0
    dist_lpf_max_hz: float = 8000.0

    # Learned distance mapping (replaces hardcoded gain/LPF/wet curves)
    # v2: disabled — learned curve has narrow gain range (0.56-0.74) and
    # non-monotonic LPF, making distance perception weaker than hardcoded.
    use_learned_mapping: bool = False
    learned_model_path: Optional[str] = None  # None = use default weights

    # Kalman filter smoothing on trajectory (applied after tracking)
    use_kalman_smoothing: bool = True


@dataclass
class OcclusionConfig:
    """Occlusion handling configuration.

    Only the ``json_path`` route is implemented: an externally produced
    occlusion timeline is loaded and interpolated onto the audio clock.
    ``estimate=True`` would need ``vid2spatial_pkg/occlusion.py``, which has
    never existed -- the pipeline's import of it used to raise into a broad
    ``except Exception`` and degrade to no occlusion with only a warning, so a
    run configured for estimated occlusion silently produced un-occluded audio.
    It now fails at construction time instead.
    """
    enabled: bool = False
    estimate: bool = False
    json_path: Optional[str] = None

    def __post_init__(self):
        if self.estimate:
            raise ValueError(
                "occlusion.estimate=True is not implemented: there is no "
                "vid2spatial_pkg/occlusion.py, so no occlusion timeline can be "
                "estimated from video. Supply occlusion.json_path with a "
                '{"frames": [{"frame": int, "occ": float}, ...]} document instead.'
            )


@dataclass
class ReverbConfig:
    """Reverb configuration."""
    enabled: bool = False
    rt60: float = 0.6
    wet_min: float = 0.05
    wet_max: float = 0.35
    wet_occ_boost: float = 0.10


@dataclass
class BinauralConfig:
    """Binaural rendering configuration."""
    mode: str = "crossfeed"  # crossfeed, sofa
    sofa_path: Optional[str] = None


@dataclass
class OutputConfig:
    """Output file configuration."""
    foa_path: str
    stereo_path: Optional[str] = None
    binaural_path: Optional[str] = None
    binaural_config: BinauralConfig = field(default_factory=BinauralConfig)

    # Optional outputs
    trajectory_path: Optional[str] = None
    # Offline automation export of the tracked trajectory (.csv or .json);
    # see vid2spatial_pkg/trajectory_export.py for the row schema.
    automation_path: Optional[str] = None


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    # Input files
    video_path: str
    audio_path: str

    # Optional precomputed data
    trajectory_json: Optional[str] = None
    air_foa_path: Optional[str] = None
    brir_left_path: Optional[str] = None
    brir_right_path: Optional[str] = None

    # Component configs
    vision: VisionConfig = field(default_factory=VisionConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    occlusion: OcclusionConfig = field(default_factory=OcclusionConfig)
    reverb: ReverbConfig = field(default_factory=ReverbConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(foa_path="out.foa.wav"))

    @classmethod
    def from_args(cls, args):
        """Create config from argparse Namespace."""
        # Parse room and mic
        room_dims = tuple(float(x) for x in args.room.split(","))
        mic_pos = tuple(float(x) for x in args.mic.split(","))

        # Parse init_bbox if provided
        init_bbox = None
        if args.init_bbox:
            parts = [int(float(t)) for t in args.init_bbox.split(",")]
            if len(parts) == 4:
                init_bbox = tuple(parts)

        return cls(
            video_path=args.video,
            audio_path=args.audio,
            trajectory_json=args.traj_json,
            air_foa_path=args.air_foa,
            brir_left_path=args.brir_L,
            brir_right_path=args.brir_R,
            vision=VisionConfig(
                camera=CameraConfig(
                    fov_deg=args.fov_deg,
                    sample_stride=args.stride,
                ),
                tracking=TrackingConfig(
                    method=args.method,
                    class_name=args.cls,
                    select_track_id=args.select_track_id,
                    init_bbox=init_bbox,
                    fallback_center_if_no_bbox=args.fallback_center_box,
                    smooth_alpha=args.smooth_alpha,
                ),
                depth=DepthConfig(
                    backend=args.depth_backend,
                    use_adapter=args.use_depth_adapter,
                ),
                refinement=RefinementConfig(
                    enabled=args.refine_center,
                    method=args.refine_center_method,
                    sam_ckpt=args.sam_ckpt,
                    sam2_model_id=args.sam2_model_id,
                    sam2_cfg=args.sam2_cfg,
                    sam2_ckpt=args.sam2_ckpt,
                ),
            ),
            room=RoomConfig(
                dimensions=room_dims,
                mic_position=mic_pos,
                rt60=args.rt60,
                backend=args.ir_backend,
                disabled=args.no_ir,
                visual_wet_mix=getattr(args, 'visual_wet_mix', 0.30),
            ),
            spatial=SpatialConfig(
                angle_smooth_ms=args.ang_smooth_ms,
                max_deg_per_s=args.max_deg_per_s,
                dist_gain_k=args.dist_gain_k,
                dist_lpf_min_hz=args.dist_lpf_min_hz,
                dist_lpf_max_hz=args.dist_lpf_max_hz,
                use_learned_mapping=getattr(args, 'use_learned_mapping', False),
                use_kalman_smoothing=getattr(args, 'use_kalman_smoothing', True),
            ),
            occlusion=OcclusionConfig(
                enabled=(args.occ_json is not None or args.estimate_occ),
                estimate=args.estimate_occ,
                json_path=args.occ_json,
            ),
            reverb=ReverbConfig(
                enabled=args.reverb_on,
                rt60=args.rev_rt60,
                wet_min=args.rev_wet_min,
                wet_max=args.rev_wet_max,
                wet_occ_boost=args.rev_wet_occ_boost,
            ),
            output=OutputConfig(
                foa_path=args.out_foa,
                stereo_path=args.out_st,
                binaural_path=args.out_bin,
                binaural_config=BinauralConfig(
                    mode=args.binaural_mode,
                    sofa_path=args.sofa,
                ),
                trajectory_path=args.save_traj,
                automation_path=getattr(args, "automation_path", None),
            ),
        )

    def to_dict(self) -> dict:
        """Convert config to dictionary (for YAML export)."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """Create config from dictionary (for YAML import)."""
        # Reconstruct nested dataclasses
        if "vision" in data and isinstance(data["vision"], dict):
            vision_data = data["vision"]
            data["vision"] = VisionConfig(
                camera=CameraConfig(**vision_data.get("camera", {})),
                tracking=TrackingConfig(**vision_data.get("tracking", {})),
                depth=DepthConfig(**vision_data.get("depth", {})),
                refinement=RefinementConfig(**vision_data.get("refinement", {})),
            )

        if "room" in data and isinstance(data["room"], dict):
            data["room"] = RoomConfig(**data["room"])

        if "spatial" in data and isinstance(data["spatial"], dict):
            data["spatial"] = SpatialConfig(**data["spatial"])

        if "occlusion" in data and isinstance(data["occlusion"], dict):
            data["occlusion"] = OcclusionConfig(**data["occlusion"])

        if "reverb" in data and isinstance(data["reverb"], dict):
            data["reverb"] = ReverbConfig(**data["reverb"])

        if "output" in data and isinstance(data["output"], dict):
            output_data = data["output"]
            binaural_cfg = BinauralConfig(**output_data.get("binaural_config", {}))
            output_data["binaural_config"] = binaural_cfg
            data["output"] = OutputConfig(**output_data)

        return cls(**data)

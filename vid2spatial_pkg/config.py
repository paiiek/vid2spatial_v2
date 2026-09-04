"""
Configuration classes for vid2spatial pipeline.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CameraConfig:
    """Camera and video processing configuration."""
    fov_deg: float = 60.0
    sample_stride: int = 1

    # --- FOV provenance (gap item A2) -------------------------------------
    # fov_deg above is only the FALLBACK. When fov_from_metadata is True the
    # pipeline asks camera_intrinsics.resolve_fov() to read the real FOV from
    # the container (sidecar JSON / exiftool / ffprobe) and warns loudly when
    # it has to fall back. The value actually used and where it came from are
    # written into the trajectory JSON as intrinsics.fov_deg / fov_source.
    # OFF by default: turning it on can change the FOV a clip renders with, and
    # therefore every azimuth, so it is an explicit opt-in rather than a silent
    # behaviour change on existing pipelines (2026-09-04 review, MEDIUM).
    fov_from_metadata: bool = False
    fov_explicit: bool = False          # user passed --fov-deg: metadata is skipped
    focal_35mm: Optional[float] = None  # --focal-35mm, converted to a FOV
    fov_source: str = "default"         # filled in by the pipeline at run time
    fov_detail: str = ""

    # --- camera-motion compensation (gap item A3) -------------------------
    # "camera_frame": azimuth is relative to the camera (the listener turns
    #                 with the camera). This is the historical behaviour.
    # "world_frame":  estimated camera yaw is subtracted, so a stationary
    #                 source stays put in the sound field during a pan.
    motion_mode: str = "camera_frame"
    estimate_camera_motion: bool = False  # implied by motion_mode="world_frame"


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
class MultiSourceConfig:
    """Offline multi-object rendering.

    OFFLINE ONLY. Each source is tracked from its own init bbox, rendered
    through the same per-source chain as the single-source path, summed with
    ``foa_render.encode_many_to_foa``, and exported as its own automation file
    with ADM object_id 1..N.

    LIMITATION -- there is no live multi-object OSC path. The bridge keys every
    datagram on the single tracking id "default" and the ADR's
    ``/vid2spatial/obj/{N}/azim`` family is not implemented engine-side, so
    streaming N sources would collapse them onto object 1. Use the automation
    export (one file per object) to drive a DAW instead.
    """
    enabled: bool = False
    # One (x, y, w, h) per source, in the first frame. len() == n_sources.
    init_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Optional per-source mono audio. Empty -> every source uses
    # PipelineConfig.audio_path (useful for geometry tests).
    audio_paths: List[str] = field(default_factory=list)

    @property
    def n_sources(self) -> int:
        return len(self.init_bboxes)

    def __post_init__(self):
        if not self.enabled:
            return
        if self.n_sources < 2:
            raise ValueError(
                f"multi_source.enabled needs at least 2 init_bboxes, got {self.n_sources}")
        if self.audio_paths and len(self.audio_paths) != self.n_sources:
            raise ValueError(
                f"multi_source: {len(self.audio_paths)} audio_paths for "
                f"{self.n_sources} init_bboxes -- give one per source or none")
        for b in self.init_bboxes:
            if len(b) != 4:
                raise ValueError(f"init_bbox must be (x, y, w, h), got {b!r}")


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
    multi_source: MultiSourceConfig = field(default_factory=MultiSourceConfig)
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
                    fov_from_metadata=getattr(args, "fov_from_metadata", True),
                    # An explicitly passed FOV must win over container metadata.
                    # argparse cannot distinguish "user typed 60" from "default
                    # 60", so anything other than the library default counts.
                    fov_explicit=(float(args.fov_deg) != CameraConfig.fov_deg),
                    focal_35mm=getattr(args, "focal_35mm", None),
                    motion_mode=getattr(args, "motion_mode", "camera_frame"),
                    estimate_camera_motion=getattr(args, "estimate_camera_motion", False),
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

        if "multi_source" in data and isinstance(data["multi_source"], dict):
            ms = dict(data["multi_source"])
            ms["init_bboxes"] = [tuple(b) for b in ms.get("init_bboxes", [])]
            data["multi_source"] = MultiSourceConfig(**ms)

        if "reverb" in data and isinstance(data["reverb"], dict):
            data["reverb"] = ReverbConfig(**data["reverb"])

        if "output" in data and isinstance(data["output"], dict):
            output_data = data["output"]
            binaural_cfg = BinauralConfig(**output_data.get("binaural_config", {}))
            output_data["binaural_config"] = binaural_cfg
            data["output"] = OutputConfig(**output_data)

        return cls(**data)


def add_camera_cli_args(parser):
    """Register the 2026-09-04 camera/geometry flags on an argparse parser.

    `PipelineConfig.from_args` reads these off the namespace, but until an entry
    point declares them there is no way to reach `world_frame` rendering or the
    metadata FOV path from a command line. Any front end that builds a
    PipelineConfig should call this on its parser.

    Every flag is opt-in and leaves the default render unchanged.
    """
    g = parser.add_argument_group("camera / geometry (2026-09-04)")
    g.add_argument("--focal-35mm", type=float, default=None, dest="focal_35mm",
                   help="35 mm-equivalent focal length; overrides --fov-deg "
                        "(27 mm is about 67 deg, 50 mm about 40 deg)")
    g.add_argument("--fov-from-metadata", action="store_true",
                   dest="fov_from_metadata",
                   help="read the horizontal FOV from the container "
                        "(sidecar JSON / exiftool / ffprobe) instead of assuming "
                        "%(default)s deg; off by default because it can change "
                        "every azimuth" % {"default": CameraConfig.fov_deg})
    g.add_argument("--motion-mode", choices=("camera_frame", "world_frame"),
                   default="camera_frame", dest="motion_mode",
                   help="camera_frame (default): the listener turns with the "
                        "camera. world_frame: subtract estimated camera yaw so a "
                        "world-static source stays put during a pan")
    g.add_argument("--estimate-camera-motion", action="store_true",
                   dest="estimate_camera_motion",
                   help="estimate and record camera yaw even in camera_frame mode")
    return parser


def add_render_cli_args(parser):
    """Register the 2026-09-04 render flags. All opt-in; see README."""
    g = parser.add_argument_group("render options (2026-09-04)")
    g.add_argument("--hrir-interp", choices=("nearest", "barycentric"),
                   default="nearest", dest="hrir_interp",
                   help="nearest (default, what every shipped stimulus used) or "
                        "barycentric interpolation over the 3 nearest SOFA "
                        "directions")
    g.add_argument("--confidence-gate", action="store_true",
                   dest="confidence_gate",
                   help="freeze azimuth and duck toward diffuse during lost "
                        "tracker episodes")
    g.add_argument("--doppler", action="store_true", dest="doppler",
                   help="pitch-shift by the radial velocity of the source")
    return parser

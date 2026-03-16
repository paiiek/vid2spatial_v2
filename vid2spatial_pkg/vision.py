"""
Refactored vision module with smaller, testable functions.

This module breaks down the monolithic compute_trajectory_3d function into
logical, composable pieces for better maintainability and testing.
"""
import math
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable

# Basic helper classes and functions
# (Previously these were imported from a separate module, now defined inline)

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters for 3D projection."""
    width: int
    height: int
    fov_deg: float

    @property
    def focal_length(self) -> float:
        """Compute focal length from FOV."""
        return (self.width / 2.0) / np.tan(np.radians(self.fov_deg / 2.0))

    @property
    def cx(self) -> float:
        """Principal point x-coordinate."""
        return self.width / 2.0

    @property
    def cy(self) -> float:
        """Principal point y-coordinate."""
        return self.height / 2.0


def pixel_to_ray(px: float, py: float, K: CameraIntrinsics) -> np.ndarray:
    """Convert pixel coordinates to normalized 3D ray."""
    x = (px - K.cx) / K.focal_length
    y = (py - K.cy) / K.focal_length
    z = 1.0
    ray = np.array([x, y, z], dtype=np.float32)
    return ray / np.linalg.norm(ray)


def ray_to_angles(ray: np.ndarray) -> Tuple[float, float]:
    """Convert 3D ray to azimuth and elevation angles.
    Image y-axis is DOWN (pixel_to_ray gives y>0 for below-center).
    Audio/spatial convention: elevation up = positive.
    → negate y before computing elevation.
    """
    x, y, z = ray
    az = np.arctan2(x, z)   # Azimuth: right = positive
    el = np.arcsin(-y)       # Elevation: up = positive (negate image-y)
    return float(az), float(el)


def smooth_series(series: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    if len(series) == 0:
        return series
    smoothed = np.zeros_like(series)
    smoothed[0] = series[0]
    for i in range(1, len(series)):
        smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i-1]
    return smoothed


def _load_midas():
    """Load MiDaS depth estimation model."""
    try:
        import torch
        model_type = "DPT_Large"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas.to(device)

        return midas, transform, device
    except Exception as e:
        print(f"[warn] MiDaS load failed: {e}")
        return None, None, None


def estimate_depth(frame: np.ndarray, midas_bundle: Optional[tuple] = None) -> np.ndarray:
    """Estimate depth map from frame."""
    if midas_bundle is None:
        # Return dummy depth map
        h, w = frame.shape[:2]
        return np.ones((h, w), dtype=np.float32) * 0.5

    model, transform, device = midas_bundle
    if model is None:
        h, w = frame.shape[:2]
        return np.ones((h, w), dtype=np.float32) * 0.5

    import torch
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    # Normalize to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def auto_init_bbox(frame: np.ndarray, cls_name: str = "person", fallback_center: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """Auto-initialize bounding box using YOLO detection."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        results = model(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name_detected = model.names[cls_id]
                if cls_name_detected.lower() == cls_name.lower():
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                    return (x, y, w, h)
    except Exception as e:
        print(f"[warn] YOLO auto-init failed: {e}")

    if fallback_center:
        h, w = frame.shape[:2]
        box_w, box_h = w // 4, h // 4
        return (w//2 - box_w//2, h//2 - box_h//2, box_w, box_h)

    return None


def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """IOU of two (x,y,w,h) boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def resolve_init_bbox(
    frame: np.ndarray,
    cls_name: str = "car",
    text_query: Optional[str] = None,
    user_bbox: Optional[Tuple[int,int,int,int]] = None,
    yolo_world_path: str = "/home/seung/yolov8s-worldv2.pt",
    conf: float = 0.10,
) -> Optional[Tuple[int,int,int,int]]:
    """Resolve the target object bbox on the first frame.

    Priority logic:
      1. text_query + user_bbox 둘 다: YOLO-World로 후보 추출 → user_bbox와 IOU 가장 높은 것
      2. text_query만: YOLO-World로 가장 높은 conf 후보
      3. user_bbox만: 그대로 반환
      4. 둘 다 없음: cls_name으로 YOLO11n 자동 detect (기존 auto_init_bbox 동작)

    Returns (x, y, w, h) or None.
    """
    candidates = []

    if text_query is not None:
        # YOLO-World: open-vocabulary detection with text query
        # user_bbox가 있을 때는 conf 임계값을 낮춰 더 많은 후보를 수집 (IoU 기준 선택이므로)
        yw_conf = 0.01 if user_bbox is not None else conf
        try:
            from ultralytics import YOLO
            model = YOLO(yolo_world_path)
            model.set_classes([text_query])
            results = model(frame, verbose=False, conf=yw_conf)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                    bconf = float(box.conf[0])
                    candidates.append(((int(x1),int(y1),int(x2-x1),int(y2-y1)), bconf))
            if candidates:
                print(f"[resolve_init_bbox] YOLO-World '{text_query}': {len(candidates)} candidates")
        except Exception as e:
            print(f"[resolve_init_bbox] YOLO-World failed: {e}")

    if text_query is not None and user_bbox is not None and candidates:
        # Both given: pick candidate with highest IOU to user_bbox
        best = max(candidates, key=lambda c: _iou(c[0], user_bbox))
        iou_val = _iou(best[0], user_bbox)
        print(f"[resolve_init_bbox] text+bbox: best IOU={iou_val:.2f} bbox={best[0]}")
        if iou_val < 0.05:
            # No good match found — fall back to user_bbox directly
            print(f"[resolve_init_bbox] IoU too low, using user_bbox directly: {user_bbox}")
            return user_bbox
        return best[0]

    if text_query is not None and candidates:
        # Text only: highest conf
        best = max(candidates, key=lambda c: c[1])
        print(f"[resolve_init_bbox] text only: conf={best[1]:.2f} bbox={best[0]}")
        return best[0]

    if user_bbox is not None:
        # User bbox only
        print(f"[resolve_init_bbox] user_bbox: {user_bbox}")
        return user_bbox

    # Fallback: standard YOLO11n cls_name detection
    return auto_init_bbox(frame, cls_name=cls_name)


def tm_track(video_path: str, init_bbox: Tuple[int, int, int, int], sample_stride: int = 1) -> List[Dict]:
    """Track object using simple template matching."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    traj = []
    fidx = 0
    template = None
    x, y, w, h = init_bbox

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if (fidx % sample_stride) != 0:
            fidx += 1
            continue

        if template is None:
            # Initialize with first frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            template = gray[y:y+h, x:x+w]
            cx, cy = x + w // 2, y + h // 2
            traj.append({"frame": fidx, "x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy})
        else:
            # Simple template matching
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > 0.3:  # Threshold for valid match
                    x, y = max_loc
                    cx, cy = x + w // 2, y + h // 2
                    traj.append({"frame": fidx, "x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy})
                else:
                    # Keep previous position if match is weak
                    if traj:
                        prev = traj[-1]
                        traj.append({"frame": fidx, "x": prev["x"], "y": prev["y"],
                                    "w": prev["w"], "h": prev["h"], "cx": prev["cx"], "cy": prev["cy"]})
            except:
                # On error, keep previous position
                if traj:
                    prev = traj[-1]
                    traj.append({"frame": fidx, "x": prev["x"], "y": prev["y"],
                                "w": prev["w"], "h": prev["h"], "cx": prev["cx"], "cy": prev["cy"]})

        fidx += 1

    cap.release()
    return traj


def yolo_bytetrack_traj(video_path: str, cls_name: str = "person",
                        select_track_id: Optional[int] = None,
                        sample_stride: int = 1,
                        conf: float = 0.25) -> List[Dict]:
    """Track object using YOLO + ByteTrack."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        traj = []
        fidx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if (fidx % sample_stride) != 0:
                fidx += 1
                continue

            results = model.track(frame, persist=True, verbose=False, conf=conf)

            for r in results:
                if r.boxes is None or r.boxes.id is None:
                    continue

                boxes = r.boxes
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    cls_name_detected = model.names[cls_id]

                    if cls_name_detected.lower() != cls_name.lower():
                        continue

                    track_id = int(box.id[0])

                    if select_track_id is not None and track_id != select_track_id:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                    cx, cy = x + w // 2, y + h // 2

                    traj.append({
                        "frame": fidx,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "cx": cx,
                        "cy": cy,
                        "track_id": track_id
                    })

                    if select_track_id is not None:
                        break

            fidx += 1

        cap.release()
        return traj

    except Exception as e:
        print(f"[error] YOLO tracking failed: {e}")
        raise


def refine_center_grabcut(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Refine object center using GrabCut segmentation."""
    x, y, w, h = bbox

    # Ensure bbox is within frame
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))

    try:
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        rect = (x, y, w, h)
        cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Find center of mass
        M = cv2.moments(mask2)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            return float(cx), float(cy)
    except Exception as e:
        pass

    # Fallback to bbox center
    return float(x + w/2), float(y + h/2)


# Placeholder for compute_trajectory_3d (basic version)
def compute_trajectory_3d(
    video_path: str,
    init_bbox: Optional[Tuple[int, int, int, int]] = None,
    fov_deg: float = 60.0,
    depth_scale_m: Tuple[float, float] = (1.0, 3.0),
    use_midas: bool = True,
    sample_stride: int = 1,
    method: str = "yolo",
    cls_name: str = "person",
    refine_center: bool = False,
    refine_center_method: str = "grabcut",
    depth_backend: str = "auto",
    sam2_mask_fn: Optional[Callable] = None,
    depth_fn: Optional[Callable] = None,
    fallback_center_if_no_bbox: bool = False,
    select_track_id: Optional[int] = None,
    smooth_alpha: float = 0.2,
    sam2_model_id: Optional[str] = None,
    sam2_cfg: Optional[str] = None,
    sam2_ckpt: Optional[str] = None,
    target_color: Optional[Tuple[int, int, int]] = None,
    color_tolerance: int = 30,
    color_min_area: int = 100,
    point_method: str = "brightness",
    point_min_brightness: int = 200,
    point_template_path: Optional[str] = None,
    point_template_threshold: float = 0.7,
    point_use_optical_flow: bool = True,
    skeleton_joint: str = "right_wrist",
    skeleton_backend: str = "mediapipe",
    skeleton_min_visibility: float = 0.5,
    skeleton_smooth_alpha: float = 0.3,
    depth_model_size: str = "small",
    focal_length_px: Optional[float] = None,
) -> Dict:
    """
    Basic implementation of compute_trajectory_3d.
    Uses compute_trajectory_3d_refactored for actual processing.
    """
    return compute_trajectory_3d_refactored(
        video_path=video_path,
        init_bbox=init_bbox,
        fov_deg=fov_deg,
        depth_scale_m=depth_scale_m,
        use_midas=use_midas,
        sample_stride=sample_stride,
        method=method,
        cls_name=cls_name,
        refine_center=refine_center,
        refine_center_method=refine_center_method,
        depth_backend=depth_backend,
        sam2_mask_fn=sam2_mask_fn,
        depth_fn=depth_fn,
        fallback_center_if_no_bbox=fallback_center_if_no_bbox,
        select_track_id=select_track_id,
        smooth_alpha=smooth_alpha,
        sam2_model_id=sam2_model_id,
        sam2_cfg=sam2_cfg,
        sam2_ckpt=sam2_ckpt,
        target_color=target_color,
        color_tolerance=color_tolerance,
        color_min_area=color_min_area,
        point_method=point_method,
        point_min_brightness=point_min_brightness,
        point_template_path=point_template_path,
        point_template_threshold=point_template_threshold,
        point_use_optical_flow=point_use_optical_flow,
        skeleton_joint=skeleton_joint,
        skeleton_backend=skeleton_backend,
        skeleton_min_visibility=skeleton_min_visibility,
        skeleton_smooth_alpha=skeleton_smooth_alpha,
        depth_model_size=depth_model_size,
        focal_length_px=focal_length_px,
    )


# ============================================================================
# Tracking Initialization
# ============================================================================

def initialize_tracking(
    video_path: str,
    method: str,
    init_bbox: Optional[Tuple[int, int, int, int]] = None,
    cls_name: str = "person",
    select_track_id: Optional[int] = None,
    fallback_center_if_no_bbox: bool = False,
    sample_stride: int = 1,
    sam2_model_id: Optional[str] = None,
    sam2_cfg: Optional[str] = None,
    sam2_ckpt: Optional[str] = None,
    **kwargs  # Additional tracker-specific parameters (e.g., target_color, color_tolerance)
) -> List[Dict]:
    """
    Initialize tracking and return 2D trajectory.

    Args:
        video_path: Path to input video
        method: Tracking method ('yolo', 'kcf', 'sam2')
        init_bbox: Initial bounding box (x, y, w, h)
        cls_name: Object class name for YOLO
        select_track_id: Specific track ID to select
        fallback_center_if_no_bbox: Use center bbox if detection fails
        sample_stride: Frame sampling stride
        sam2_model_id: SAM2 model identifier
        sam2_cfg: SAM2 config path
        sam2_ckpt: SAM2 checkpoint path

    Returns:
        List of tracking records: [{"frame": int, "cx": float, "cy": float, "w": float, "h": float}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ok, first = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read first frame")

    H0, W0 = first.shape[:2]

    # Helper to create center fallback bbox
    def get_fallback_bbox():
        if not fallback_center_if_no_bbox:
            raise RuntimeError("Could not auto-initialize bbox; please provide init_bbox")
        cw, ch = int(W0 * 0.2), int(H0 * 0.2)
        cx0, cy0 = int(W0 * 0.5 - cw * 0.5), int(H0 * 0.5 - ch * 0.5)
        return (cx0, cy0, cw, ch)

    if method == "kcf":
        return _initialize_kcf_tracking(
            video_path, first, init_bbox, get_fallback_bbox, sample_stride
        )
    elif method == "yolo":
        return _initialize_yolo_tracking(
            video_path, first, cls_name, select_track_id, init_bbox, get_fallback_bbox, sample_stride
        )
    elif method == "ostrack":
        return _initialize_ostrack_tracking(
            video_path, first, init_bbox, get_fallback_bbox, sample_stride
        )
    elif method == "sam2":
        return _initialize_sam2_tracking(
            video_path, first, init_bbox, cls_name, select_track_id,
            sample_stride, sam2_model_id, sam2_cfg, sam2_ckpt, get_fallback_bbox
        )
    elif method == "color":
        # Color tracking requires target_color from config
        # Use default red if not provided
        target_color = kwargs.get("target_color")
        if target_color is None:
            target_color = (0, 0, 255)  # Default red in BGR
        color_tolerance = kwargs.get("color_tolerance", 30)
        color_min_area = kwargs.get("color_min_area", 100)
        return _initialize_color_tracking(
            video_path, target_color, color_tolerance, color_min_area, sample_stride
        )
    elif method == "point":
        # Point tracking for cursor, laser pointer, etc.
        point_method = kwargs.get("point_method", "brightness")
        point_min_brightness = kwargs.get("point_min_brightness", 200)
        point_template_path = kwargs.get("point_template_path")
        point_template_threshold = kwargs.get("point_template_threshold", 0.7)
        point_use_optical_flow = kwargs.get("point_use_optical_flow", True)
        return _initialize_point_tracking(
            video_path, point_method, point_min_brightness, point_template_path,
            point_template_threshold, point_use_optical_flow, sample_stride
        )
    elif method == "skeleton":
        # Skeleton tracking for motion capture, dance, etc.
        skeleton_joint = kwargs.get("skeleton_joint", "right_wrist")
        skeleton_backend = kwargs.get("skeleton_backend", "mediapipe")
        skeleton_min_visibility = kwargs.get("skeleton_min_visibility", 0.5)
        skeleton_smooth_alpha = kwargs.get("skeleton_smooth_alpha", 0.3)
        return _initialize_skeleton_tracking(
            video_path, skeleton_joint, skeleton_backend, skeleton_min_visibility,
            skeleton_smooth_alpha, sample_stride
        )
    else:
        raise ValueError(f"Unknown tracking method: {method}")


def _initialize_kcf_tracking(
    video_path: str,
    first_frame: np.ndarray,
    init_bbox: Optional[Tuple[int, int, int, int]],
    get_fallback_bbox: Callable,
    sample_stride: int = 1,
) -> List[Dict]:
    """Initialize KCF tracking."""
    if init_bbox is None:
        init_bbox = auto_init_bbox(first_frame)
        if init_bbox is None:
            init_bbox = get_fallback_bbox()

    try:
        # Try contrib KCF if available, else use template matching
        tracker = getattr(cv2, 'TrackerKCF_create', None)
        if tracker is not None:
            # Validate tracker works
            tr = tracker()
            tr.init(first_frame, init_bbox)
        # Use template matching for full sequence (deterministic)
        return tm_track(video_path, init_bbox, sample_stride)
    except Exception:
        return tm_track(video_path, init_bbox, sample_stride)


def _initialize_yolo_tracking(
    video_path: str,
    first_frame: np.ndarray,
    cls_name: str,
    select_track_id: Optional[int],
    init_bbox: Optional[Tuple[int, int, int, int]],
    get_fallback_bbox: Callable,
    sample_stride: int = 1,
) -> List[Dict]:
    """Initialize YOLO + ByteTrack tracking."""
    traj_2d = yolo_bytetrack_traj(
        video_path, cls_name=cls_name, conf=0.25, select_track_id=select_track_id, sample_stride=sample_stride
    )

    if not traj_2d:
        # Fallback to KCF/template matching
        if init_bbox is None:
            init_bbox = auto_init_bbox(first_frame)
            if init_bbox is None:
                raise RuntimeError("Detection/Tracking failed and no bbox available")
        traj_2d = tm_track(video_path, init_bbox)

    return traj_2d


def _initialize_ostrack_tracking(
    video_path: str,
    first_frame: np.ndarray,
    init_bbox: Optional[Tuple[int, int, int, int]],
    get_fallback_bbox: Callable,
    sample_stride: int = 1,
) -> List[Dict]:
    """Initialize OSTrack tracking."""
    if init_bbox is None:
        init_bbox = auto_init_bbox(first_frame)
        if init_bbox is None:
            init_bbox = get_fallback_bbox()

    try:
        from .ostrack_wrapper import track_with_ostrack
        print("[info] Using OSTrack tracker...")
        return track_with_ostrack(video_path, init_bbox, sample_stride)
    except Exception as e:
        print(f"[warn] OSTrack failed: {e}, falling back to template matching")
        return tm_track(video_path, init_bbox, sample_stride)


def _initialize_sam2_tracking(
    video_path: str,
    first_frame: np.ndarray,
    init_bbox: Optional[Tuple[int, int, int, int]],
    cls_name: str,
    select_track_id: Optional[int],
    sample_stride: int,
    sam2_model_id: Optional[str],
    sam2_cfg: Optional[str],
    sam2_ckpt: Optional[str],
    get_fallback_bbox: Callable,
) -> List[Dict]:
    """Initialize SAM2 tracking."""
    # Get initial bbox if not provided
    if init_bbox is None:
        tmp = yolo_bytetrack_traj(
            video_path, cls_name=cls_name, conf=0.25, select_track_id=select_track_id
        )
        if tmp:
            rec0 = tmp[0]
            init_bbox = (
                int(rec0["cx"] - rec0["w"] * 0.5),
                int(rec0["cy"] - rec0["h"] * 0.5),
                int(rec0["w"]),
                int(rec0["h"])
            )
        else:
            init_bbox = get_fallback_bbox()

    try:
        from .sam2_traj import sam2_video_traj
        return sam2_video_traj(
            video_path, init_bbox,
            model_id=sam2_model_id,
            config_path=sam2_cfg,
            checkpoint_path=sam2_ckpt,
            stride=sample_stride
        )
    except Exception:
        # Fallback to YOLO/template matching
        traj_2d = yolo_bytetrack_traj(
            video_path, cls_name=cls_name, conf=0.25, select_track_id=select_track_id
        )
        if not traj_2d:
            traj_2d = tm_track(video_path, init_bbox)
        return traj_2d


def _initialize_color_tracking(
    video_path: str,
    target_color: Tuple[int, int, int],
    color_tolerance: int,
    color_min_area: int,
    sample_stride: int,
) -> List[Dict]:
    """
    Initialize color-based tracking.

    Args:
        video_path: Path to video
        target_color: Target color in BGR format (e.g., (255, 0, 0) for red)
        color_tolerance: HSV tolerance for color matching
        color_min_area: Minimum blob area in pixels
        sample_stride: Frame sampling stride

    Returns:
        Trajectory list
    """
    from .color_tracker import color_track
    return color_track(
        video_path=video_path,
        target_color=target_color,
        color_tolerance=color_tolerance,
        min_area=color_min_area,
        sample_stride=sample_stride,
        verbose=False
    )


def _initialize_point_tracking(
    video_path: str,
    point_method: str,
    point_min_brightness: int,
    point_template_path: Optional[str],
    point_template_threshold: float,
    point_use_optical_flow: bool,
    sample_stride: int,
) -> List[Dict]:
    """
    Initialize point-based tracking (cursor, laser pointer).

    Args:
        video_path: Path to video
        point_method: Detection method ("brightness", "template", "goodfeatures")
        point_min_brightness: Minimum brightness for bright point detection
        point_template_path: Path to cursor template image (for template method)
        point_template_threshold: Template matching threshold
        point_use_optical_flow: Use optical flow for tracking
        sample_stride: Frame sampling stride

    Returns:
        Trajectory list
    """
    from .point_tracker import point_track
    return point_track(
        video_path=video_path,
        method=point_method,
        min_brightness=point_min_brightness,
        template_path=point_template_path,
        template_threshold=point_template_threshold,
        sample_stride=sample_stride,
        use_optical_flow=point_use_optical_flow,
        verbose=False
    )


def _initialize_skeleton_tracking(
    video_path: str,
    skeleton_joint: str,
    skeleton_backend: str,
    skeleton_min_visibility: float,
    skeleton_smooth_alpha: float,
    sample_stride: int,
) -> List[Dict]:
    """
    Initialize skeleton-based tracking (motion capture, dance).

    Args:
        video_path: Path to video
        skeleton_joint: Joint to track (e.g., "right_wrist", "nose", "left_ankle")
        skeleton_backend: Pose estimation backend ("mediapipe")
        skeleton_min_visibility: Minimum joint visibility threshold (0-1)
        skeleton_smooth_alpha: Smoothing factor (0=max smooth, 1=no smooth)
        sample_stride: Frame sampling stride

    Returns:
        Trajectory list
    """
    from .skeleton_tracker import skeleton_track
    return skeleton_track(
        video_path=video_path,
        joint_name=skeleton_joint,
        backend=skeleton_backend,
        min_visibility=skeleton_min_visibility,
        sample_stride=sample_stride,
        smooth_alpha=skeleton_smooth_alpha,
        verbose=False
    )


# ============================================================================
# Depth Estimation
# ============================================================================

# Global metric depth estimator (singleton for efficiency)
_METRIC_DEPTH_ESTIMATOR = None
_METRIC3D_ESTIMATOR = None


def get_metric_depth_estimator(scene_type: str = "auto", model_size: str = "small", device: str = "cuda"):
    """Get or create singleton metric depth estimator."""
    global _METRIC_DEPTH_ESTIMATOR
    if _METRIC_DEPTH_ESTIMATOR is None:
        try:
            from .depth_metric import MetricDepthEstimator
            _METRIC_DEPTH_ESTIMATOR = MetricDepthEstimator(
                scene_type=scene_type,
                model_size=model_size,
                device=device,
            )
        except Exception as e:
            print(f"[warn] Failed to load MetricDepthEstimator: {e}")
            _METRIC_DEPTH_ESTIMATOR = None
    return _METRIC_DEPTH_ESTIMATOR


def get_metric3d_estimator(model_size: str = "small", device: str = "cuda",
                           focal_length_px: Optional[float] = None):
    """Get or create singleton Metric3D v2 estimator."""
    global _METRIC3D_ESTIMATOR
    if _METRIC3D_ESTIMATOR is None:
        try:
            from .depth_metric3d import Metric3DEstimator
            _METRIC3D_ESTIMATOR = Metric3DEstimator(
                model_size=model_size,
                device=device,
                focal_length_px=focal_length_px,
            )
        except Exception as e:
            print(f"[warn] Failed to load Metric3DEstimator: {e}")
            _METRIC3D_ESTIMATOR = None
    return _METRIC3D_ESTIMATOR


def initialize_depth_backend(
    depth_backend: str,
    use_midas: bool = True,
    depth_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    scene_type: str = "auto",
    model_size: str = "small",
    focal_length_px: Optional[float] = None,
) -> Tuple[Optional[Callable], Optional[tuple], bool]:
    """
    Initialize depth estimation backend.

    Args:
        depth_backend: Backend choice ('auto', 'metric3d', 'metric', 'midas', 'depth_anything_v2')
                       'auto' tries metric3d → metric → DA-V2 → MiDaS
        use_midas: Whether to use MiDaS as fallback
        depth_fn: Custom depth function
        scene_type: Scene type for metric depth ('indoor', 'outdoor', 'auto')
        model_size: Model size ('small', 'large', 'giant')
        focal_length_px: Camera focal length in pixels (for metric3d)

    Returns:
        Tuple of (depth_fn, midas_bundle, is_metric)
        is_metric: True if depth_fn returns meters directly, False if relative [0,1]
    """
    if depth_fn is not None:
        # User provided custom depth function - assume it's relative unless specified
        return depth_fn, None, False

    # Try Metric3D v2 (preferred for v2 - universal metric depth)
    if depth_backend in ("auto", "metric3d"):
        try:
            estimator = get_metric3d_estimator(
                model_size=model_size, device="cuda",
                focal_length_px=focal_length_px,
            )
            if estimator is not None and estimator.model is not None:
                def metric3d_fn(image: np.ndarray) -> np.ndarray:
                    return estimator.infer(image)
                print(f"[info] Using Metric3D v2 backend (model_size={model_size}) - returns meters")
                return metric3d_fn, None, True
        except Exception as e:
            print(f"[warn] Metric3D v2 failed: {e}, trying fallback")

    # Try Depth Anything V2 Metric (returns meters)
    if depth_backend in ("auto", "metric"):
        try:
            from .depth_metric import MetricDepthEstimator
            estimator = get_metric_depth_estimator(scene_type=scene_type, model_size="small", device="cuda")
            if estimator is not None and estimator.model is not None:
                def metric_depth_fn(image: np.ndarray) -> np.ndarray:
                    return estimator.infer(image)
                print(f"[info] Using Metric Depth backend (scene_type={scene_type}) - returns meters")
                return metric_depth_fn, None, True
        except Exception as e:
            print(f"[warn] Metric Depth failed: {e}, falling back to relative depth")

    # Try Depth Anything V2 (relative depth)
    if depth_backend in ("auto", "depth_anything_v2"):
        try:
            from .depth_anything_v2 import create_depth_anything_v2_backend
            depth_fn = create_depth_anything_v2_backend(
                model_size="small",
                device="cuda"
            )
            print("[info] Using Depth Anything V2 backend (relative depth)")
            return depth_fn, None, False
        except Exception:
            pass
        # Fallback: use depth_anything_adapter which handles the local repo path
        try:
            from .depth_anything_adapter import build_depth_predictor as _build_da
            depth_fn = _build_da(device="cuda", backend="depth_anything_v2")
            print("[info] Using Depth Anything V2 backend via adapter (relative depth)")
            return depth_fn, None, False
        except Exception as e:
            print(f"[warn] Depth Anything V2 failed: {e}, falling back to MiDaS")

    # Load MiDaS (relative depth)
    if depth_backend in ("auto", "midas") and use_midas:
        try:
            device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
            midas_bundle = _load_midas()
            print("[info] Using MiDaS backend (relative depth)")
            return None, midas_bundle, False
        except Exception:
            pass

    print("[warn] No depth backend available, using fallback")
    return None, None, False


def estimate_depth_at_bbox(
    frame: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    depth_fn: Optional[Callable],
    midas_bundle: Optional[tuple],
    is_metric: bool = False,
) -> Tuple[float, bool]:
    """
    Estimate depth at bounding box center.

    Args:
        frame: Input frame (BGR)
        cx, cy: Center coordinates
        w, h: Bounding box size
        depth_fn: Custom depth function
        midas_bundle: MiDaS model bundle
        is_metric: Whether depth_fn returns meters directly

    Returns:
        Tuple of (depth_value, is_metric)
        - If is_metric=True: depth in meters
        - If is_metric=False: relative depth [0, 1]
    """
    H, W = frame.shape[:2]

    # Use custom depth function if provided
    if depth_fn is not None:
        try:
            dmap = depth_fn(frame)
            dmap = dmap.astype(np.float32)

            if is_metric:
                # Metric depth - extract directly without normalization
                depth_val = _extract_depth_from_bbox_raw(dmap, cx, cy, w, h, W, H)
                return depth_val, True
            else:
                # Relative depth - normalize to [0, 1]
                dmap -= dmap.min()
                if dmap.max() > 1e-8:
                    dmap /= dmap.max()
                return _extract_depth_from_bbox_raw(dmap, cx, cy, w, h, W, H), False
        except Exception as e:
            print(f"[warn] Depth estimation failed: {e}")
            return 2.0 if is_metric else 0.5, is_metric

    # Use MiDaS if available (relative depth)
    if midas_bundle is not None:
        dmap = estimate_depth(frame, midas_bundle)
        return _extract_depth_from_bbox_raw(dmap, cx, cy, w, h, W, H), False

    # Default fallback
    return 2.0, True  # Default 2 meters


def _extract_depth_from_bbox_raw(
    dmap: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    W: int,
    H: int,
) -> float:
    """Extract median depth from bounding box region (no normalization)."""
    x0 = max(0, int(cx - w * 0.25))
    y0 = max(0, int(cy - h * 0.25))
    x1 = min(W, int(cx + w * 0.25))
    y1 = min(H, int(cy + h * 0.25))

    box = dmap[y0:y1, x0:x1]
    if box.size > 0:
        return float(np.median(box))
    else:
        return float(dmap[min(H-1, int(cy)), min(W-1, int(cx))])


# ============================================================================
# Center Refinement
# ============================================================================

def refine_object_center(
    frame: np.ndarray,
    rec: Dict,
    refine_center: bool,
    refine_center_method: str,
    sam2_mask_fn: Optional[Callable],
) -> Tuple[float, float]:
    """
    Refine object center using GrabCut or SAM2.

    Args:
        frame: Input frame (BGR)
        rec: Tracking record {"cx", "cy", "w", "h"}
        refine_center: Whether to refine
        refine_center_method: Method ('grabcut', 'sam2')
        sam2_mask_fn: SAM2 mask function

    Returns:
        Tuple of (refined_cx, refined_cy)
    """
    cx, cy = rec["cx"], rec["cy"]

    if not refine_center:
        return cx, cy

    if refine_center_method == "grabcut":
        bbox = (
            rec["cx"] - rec["w"] * 0.5,
            rec["cy"] - rec["h"] * 0.5,
            rec["w"],
            rec["h"]
        )
        return refine_center_grabcut(frame, bbox)

    elif refine_center_method == "sam2":
        if sam2_mask_fn is None:
            raise RuntimeError(
                "refine_center_method='sam2' requires sam2_mask_fn(frame_bgr, bbox)->mask"
            )
        bbox = (
            rec["cx"] - rec["w"] * 0.5,
            rec["cy"] - rec["h"] * 0.5,
            rec["w"],
            rec["h"]
        )
        mask = sam2_mask_fn(frame, bbox)
        ys, xs = np.nonzero(mask.astype(np.uint8))
        if len(xs) > 0:
            return float(xs.mean()), float(ys.mean())
        return cx, cy

    else:
        raise ValueError(f"Unknown refine_center_method: {refine_center_method}")


# ============================================================================
# 3D Position Computation
# ============================================================================

def compute_3d_position(
    cx: float,
    cy: float,
    depth_value: float,
    K: CameraIntrinsics,
    depth_scale_m: Tuple[float, float],
    is_metric: bool = False,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute 3D position from 2D center and depth.

    Args:
        cx, cy: Center coordinates in pixels
        depth_value: Depth value (meters if is_metric=True, else relative [0,1])
        K: Camera intrinsics
        depth_scale_m: (near, far) metric depth range (only used if is_metric=False)
        is_metric: Whether depth_value is already in meters

    Returns:
        Tuple of (az, el, dist_m, x, y, z)
    """
    # Convert pixel to ray
    ray = pixel_to_ray(cx, cy, K)
    az, el = ray_to_angles(ray)

    # Get distance in meters
    if is_metric:
        # Direct metric depth - use as-is
        dist_m = float(np.clip(depth_value, 0.3, 50.0))  # Clip to reasonable range
    else:
        # Map relative depth to metric distance (legacy heuristic)
        near, far = depth_scale_m
        dist_m = near + (1.0 - depth_value) * (far - near)

    # Compute 3D position
    pos = ray * dist_m
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

    return float(az), float(el), float(dist_m), x, y, z


# ============================================================================
# Frame Processing
# ============================================================================

def process_trajectory_frames(
    video_path: str,
    traj_2d: List[Dict],
    K: CameraIntrinsics,
    sample_stride: int,
    depth_fn: Optional[Callable],
    midas_bundle: Optional[tuple],
    depth_scale_m: Tuple[float, float],
    refine_center: bool,
    refine_center_method: str,
    sam2_mask_fn: Optional[Callable],
    is_metric: bool = False,
) -> List[Dict]:
    """
    Process video frames to compute 3D trajectory.

    Args:
        video_path: Path to input video
        traj_2d: 2D tracking trajectory
        K: Camera intrinsics
        sample_stride: Frame sampling stride
        depth_fn: Depth estimation function
        midas_bundle: MiDaS model bundle
        depth_scale_m: Depth scale range (only used if is_metric=False)
        refine_center: Whether to refine centers
        refine_center_method: Refinement method
        sam2_mask_fn: SAM2 mask function
        is_metric: Whether depth_fn returns meters directly

    Returns:
        List of 3D trajectory frames
    """
    result_frames = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fidx = 0
    i2d = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Skip frames based on stride
        if (fidx % sample_stride) != 0:
            fidx += 1
            continue

        # Check if we have tracking data for this frame
        if i2d >= len(traj_2d):
            break

        rec = traj_2d[i2d]
        if rec["frame"] < fidx:
            i2d += 1
            continue

        # Refine center if needed
        cx, cy = refine_object_center(
            frame, rec, refine_center, refine_center_method, sam2_mask_fn
        )

        # Estimate depth
        depth_value, depth_is_metric = estimate_depth_at_bbox(
            frame, cx, cy, rec["w"], rec["h"], depth_fn, midas_bundle, is_metric
        )

        # Compute 3D position
        az, el, dist_m, x, y, z = compute_3d_position(
            cx, cy, depth_value, K, depth_scale_m, is_metric=depth_is_metric
        )

        result_frames.append({
            "frame": int(fidx),
            "az": az,
            "el": el,
            "dist_m": dist_m,
            "x": x,
            "y": y,
            "z": z,
        })

        fidx += 1

    cap.release()
    return result_frames


# ============================================================================
# Post-Processing
# ============================================================================

def smooth_trajectory(
    frames: List[Dict],
    smooth_alpha: float = 0.2,
) -> List[Dict]:
    """
    Smooth trajectory angles using exponential moving average.

    Args:
        frames: List of trajectory frames
        smooth_alpha: Smoothing factor (0 = no smoothing, 1 = no memory)

    Returns:
        Smoothed trajectory frames
    """
    if not frames:
        return frames

    # Extract angles
    azs = np.array([r["az"] for r in frames], dtype=np.float32)
    els = np.array([r["el"] for r in frames], dtype=np.float32)

    # Smooth
    azs_s = smooth_series(azs, alpha=smooth_alpha)
    els_s = smooth_series(els, alpha=smooth_alpha)

    # Update frames
    for i, r in enumerate(frames):
        r["az"] = float(azs_s[i])
        r["el"] = float(els_s[i])

    return frames


# ============================================================================
# Main Refactored Function
# ============================================================================

def compute_trajectory_3d_refactored(
    video_path: str,
    init_bbox: Optional[Tuple[int, int, int, int]] = None,
    fov_deg: float = 60.0,
    depth_scale_m: Tuple[float, float] = (1.0, 3.0),
    use_midas: bool = True,
    sample_stride: int = 1,
    method: str = "yolo",
    cls_name: str = "person",
    refine_center: bool = True,
    refine_center_method: str = "grabcut",
    depth_backend: str = "auto",
    sam2_mask_fn: Optional[Callable[[np.ndarray, Tuple[float, float, float, float]], np.ndarray]] = None,
    depth_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    fallback_center_if_no_bbox: bool = False,
    select_track_id: Optional[int] = None,
    smooth_alpha: float = 0.2,
    sam2_model_id: Optional[str] = None,
    sam2_cfg: Optional[str] = None,
    sam2_ckpt: Optional[str] = None,
    target_color: Optional[Tuple[int, int, int]] = None,
    color_tolerance: int = 30,
    color_min_area: int = 100,
    point_method: str = "brightness",
    point_min_brightness: int = 200,
    point_template_path: Optional[str] = None,
    point_template_threshold: float = 0.7,
    point_use_optical_flow: bool = True,
    skeleton_joint: str = "right_wrist",
    skeleton_backend: str = "mediapipe",
    skeleton_min_visibility: float = 0.5,
    skeleton_smooth_alpha: float = 0.3,
    depth_model_size: str = "small",
    focal_length_px: Optional[float] = None,
) -> Dict:
    """
    Track a single object and estimate a time-sequenced 3D trajectory.

    This is a refactored version of compute_trajectory_3d with the same interface
    but better internal structure for maintainability and testing.

    Args:
        video_path: Path to input video
        init_bbox: Initial bounding box (x, y, w, h)
        fov_deg: Camera horizontal FOV in degrees
        depth_scale_m: (near, far) metric depth range in meters
        use_midas: Whether to use MiDaS for depth estimation
        sample_stride: Process every Nth frame
        method: Tracking method ('yolo', 'kcf', 'sam2', 'color', 'point', 'skeleton')
        cls_name: Object class name for YOLO
        refine_center: Whether to refine object centers
        refine_center_method: Center refinement method ('grabcut', 'sam2')
        depth_backend: Depth estimation backend ('auto', 'metric3d', 'midas', 'depth_anything_v2')
        sam2_mask_fn: SAM2 mask function for center refinement
        depth_fn: Custom depth estimation function
        fallback_center_if_no_bbox: Use center bbox if detection fails
        select_track_id: Specific track ID to select (YOLO)
        smooth_alpha: Smoothing factor for angles
        sam2_model_id: SAM2 model identifier
        sam2_cfg: SAM2 config path
        sam2_ckpt: SAM2 checkpoint path
        target_color: Target color for color tracking (BGR format, e.g., (255, 0, 0) for red)
        color_tolerance: HSV tolerance for color matching (default: 30)
        color_min_area: Minimum blob area in pixels (default: 100)
        point_method: Point detection method ('brightness', 'template', 'goodfeatures')
        point_min_brightness: Minimum brightness for bright point detection (0-255)
        point_template_path: Path to cursor template image (for template method)
        point_template_threshold: Template matching threshold (0-1)
        point_use_optical_flow: Use optical flow for point tracking
        skeleton_joint: Joint to track for skeleton tracking (e.g., "right_wrist", "nose")
        skeleton_backend: Pose estimation backend ('mediapipe')
        skeleton_min_visibility: Minimum joint visibility threshold (0-1)
        skeleton_smooth_alpha: Smoothing factor for skeleton tracking (0=max smooth, 1=no smooth)
        depth_model_size: Model size for depth backend ('small', 'large', 'giant')
        focal_length_px: Camera focal length in pixels (for metric3d)

    Returns:
        Dict with:
            - intrinsics: {"width": int, "height": int, "fov_deg": float}
            - frames: List[{"frame": int, "az": float, "el": float, "dist_m": float, "x": float, "y": float, "z": float}]
    """
    # 1. Get video dimensions and create camera intrinsics
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    K = CameraIntrinsics(width=W, height=H, fov_deg=fov_deg)

    # 2. Initialize tracking
    traj_2d = initialize_tracking(
        video_path=video_path,
        method=method,
        init_bbox=init_bbox,
        cls_name=cls_name,
        select_track_id=select_track_id,
        fallback_center_if_no_bbox=fallback_center_if_no_bbox,
        sample_stride=sample_stride,
        sam2_model_id=sam2_model_id,
        sam2_cfg=sam2_cfg,
        sam2_ckpt=sam2_ckpt,
        target_color=target_color,
        color_tolerance=color_tolerance,
        color_min_area=color_min_area,
        point_method=point_method,
        point_min_brightness=point_min_brightness,
        point_template_path=point_template_path,
        point_template_threshold=point_template_threshold,
        point_use_optical_flow=point_use_optical_flow,
        skeleton_joint=skeleton_joint,
        skeleton_backend=skeleton_backend,
        skeleton_min_visibility=skeleton_min_visibility,
        skeleton_smooth_alpha=skeleton_smooth_alpha,
    )

    # 3. Initialize depth estimation
    depth_fn_init, midas_bundle, is_metric = initialize_depth_backend(
        depth_backend=depth_backend,
        use_midas=use_midas,
        depth_fn=depth_fn,
        model_size=depth_model_size,
        focal_length_px=focal_length_px,
    )

    # Use initialized depth function if custom one not provided
    if depth_fn is None:
        depth_fn = depth_fn_init

    # 4. Process frames to compute 3D trajectory
    result_frames = process_trajectory_frames(
        video_path=video_path,
        traj_2d=traj_2d,
        K=K,
        sample_stride=sample_stride,
        depth_fn=depth_fn,
        midas_bundle=midas_bundle,
        depth_scale_m=depth_scale_m,
        refine_center=refine_center,
        refine_center_method=refine_center_method,
        sam2_mask_fn=sam2_mask_fn,
        is_metric=is_metric,
    )

    # 5. Smooth trajectory
    result_frames = smooth_trajectory(result_frames, smooth_alpha=smooth_alpha)

    # 6. Return result
    return {
        "intrinsics": {"width": W, "height": H, "fov_deg": float(fov_deg)},
        "frames": result_frames,
    }


__all__ = [
    # Main function
    'compute_trajectory_3d_refactored',
    # Tracking
    'initialize_tracking',
    # Depth estimation
    'initialize_depth_backend',
    'estimate_depth_at_bbox',
    # Center refinement
    'refine_object_center',
    # 3D computation
    'compute_3d_position',
    # Frame processing
    'process_trajectory_frames',
    # Post-processing
    'smooth_trajectory',
]

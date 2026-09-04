"""
Skeleton-based tracking for Vid2Spatial using pose estimation.

Use cases: Dance performances, motion capture, body movement spatial audio
Example: Dancer's hand position → Track trajectory

Supports MediaPipe 0.10+ (Tasks API) and legacy versions.
"""

import cv2
import numpy as np
import os
import urllib.request
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path


# Joint mapping for MediaPipe pose landmarks
MEDIAPIPE_JOINTS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


# Model download URLs
POSE_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def _get_model_path(model_complexity: int = 1) -> str:
    """Get path to pose landmarker model, downloading if needed."""
    model_names = {0: "lite", 1: "full", 2: "heavy"}
    model_name = model_names.get(model_complexity, "lite")

    cache_dir = Path.home() / ".cache" / "mediapipe"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / f"pose_landmarker_{model_name}.task"

    if not model_path.exists():
        url = POSE_MODEL_URLS[model_name]
        print(f"[info] Downloading pose model ({model_name})...")
        urllib.request.urlretrieve(url, str(model_path))
        print(f"[info] Saved to {model_path}")

    return str(model_path)


class MediaPipeTasksPose:
    """MediaPipe 0.10+ Tasks API wrapper for pose detection."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        import warnings
        warnings.filterwarnings('ignore')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        model_path = _get_model_path(model_complexity)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.pose_landmarks = None  # Compatibility with old API

    def process(self, frame_rgb: np.ndarray) -> 'MediaPipeTasksPose':
        """Process frame and return self with pose_landmarks set."""
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            # Convert to compatible format
            self.pose_landmarks = _TasksLandmarksWrapper(result.pose_landmarks[0])
        else:
            self.pose_landmarks = None

        return self

    def close(self):
        """Release resources."""
        if self.detector:
            self.detector.close()


class _TasksLandmarksWrapper:
    """Wrapper to make Tasks API landmarks compatible with old API."""

    def __init__(self, landmarks):
        self.landmark = landmarks


def initialize_mediapipe_pose(model_complexity: int = 1, min_detection_confidence: float = 0.5):
    """
    Initialize MediaPipe Pose detector.

    Args:
        model_complexity: 0=lite, 1=full, 2=heavy
        min_detection_confidence: Minimum confidence threshold

    Returns:
        pose: MediaPipe Pose object (Tasks API or legacy)
        None if MediaPipe not available
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import mediapipe as mp

        # Try new Tasks API first (MediaPipe 0.10+)
        if hasattr(mp, 'tasks'):
            print("[info] Using MediaPipe Tasks API (0.10+)")
            return MediaPipeTasksPose(model_complexity, min_detection_confidence)

        # Fall back to legacy solutions API
        if hasattr(mp, 'solutions'):
            print("[info] Using MediaPipe Solutions API (legacy)")
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence
            )
            return pose

        print("[error] MediaPipe API not found")
        return None

    except ImportError as e:
        print(f"[error] MediaPipe not installed: {e}")
        print("  Install with: pip install mediapipe")
        return None
    except Exception as e:
        print(f"[error] MediaPipe initialization failed: {e}")
        return None


def extract_joint_position(
    pose_landmarks,
    joint_name: str,
    frame_width: int,
    frame_height: int
) -> Optional[Tuple[float, float, float]]:
    """
    Extract joint position from pose landmarks.

    Args:
        pose_landmarks: MediaPipe pose landmarks
        joint_name: Joint name (e.g., "left_wrist", "nose")
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        (x, y, visibility) tuple or None if joint not detected
    """
    if pose_landmarks is None:
        return None

    joint_idx = MEDIAPIPE_JOINTS.get(joint_name)
    if joint_idx is None:
        raise ValueError(f"Unknown joint: {joint_name}. Available: {list(MEDIAPIPE_JOINTS.keys())}")

    landmark = pose_landmarks.landmark[joint_idx]

    # Convert normalized coordinates to pixel coordinates
    x = landmark.x * frame_width
    y = landmark.y * frame_height
    visibility = landmark.visibility

    return (float(x), float(y), float(visibility))


def smooth_joint_trajectory(
    trajectory: List[Dict],
    alpha: float = 0.3
) -> List[Dict]:
    """
    Apply exponential smoothing to joint trajectory.

    Args:
        trajectory: List of tracking records
        alpha: Smoothing factor (0=max smoothing, 1=no smoothing)

    Returns:
        Smoothed trajectory
    """
    if len(trajectory) < 2:
        return trajectory

    smoothed = []
    prev_x, prev_y = trajectory[0]["cx"], trajectory[0]["cy"]

    for i, rec in enumerate(trajectory):
        if i == 0:
            smoothed.append(rec.copy())
        else:
            # Exponential moving average
            smooth_x = alpha * rec["cx"] + (1 - alpha) * prev_x
            smooth_y = alpha * rec["cy"] + (1 - alpha) * prev_y

            rec_smooth = rec.copy()
            rec_smooth["cx"] = smooth_x
            rec_smooth["cy"] = smooth_y

            smoothed.append(rec_smooth)

            prev_x, prev_y = smooth_x, smooth_y

    return smoothed


def preprocess_frame_for_lighting(
    frame: np.ndarray,
    mode: str = "auto"
) -> np.ndarray:
    """
    Preprocess frame to handle challenging lighting conditions.

    Args:
        frame: BGR image
        mode: Preprocessing mode:
            - "auto": Detect and apply appropriate preprocessing
            - "normalize": Color normalization for colored lighting
            - "clahe": Histogram equalization for low contrast
            - "none": No preprocessing

    Returns:
        Preprocessed frame (RGB for MediaPipe)
    """
    if mode == "none":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if mode == "auto":
        # Analyze frame to determine best preprocessing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_std = np.std(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])

        if s_mean > 100 and h_std < 30:
            # Strong colored lighting (stage lights, etc.)
            mode = "normalize"
        elif v_std < 40:
            # Low contrast (dark scenes, fog)
            mode = "clahe"
        else:
            # Normal conditions
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if mode == "normalize":
        # Convert to LAB color space for better color invariance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE only to L channel (luminance)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Reduce color cast by normalizing a,b channels
        a = cv2.normalize(a, None, 100, 155, cv2.NORM_MINMAX)
        b = cv2.normalize(b, None, 100, 155, cv2.NORM_MINMAX)

        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    elif mode == "clahe":
        # Apply CLAHE for low contrast scenes
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def skeleton_track(
    video_path: str,
    joint_name: str = "right_wrist",
    backend: str = "mediapipe",
    min_visibility: float = 0.5,
    sample_stride: int = 1,
    smooth_alpha: float = 0.3,
    verbose: bool = False,
    preprocess: str = "auto",
) -> List[Dict]:
    """
    Track skeleton joint in video using pose estimation.

    Args:
        video_path: Path to video file
        joint_name: Joint to track (see MEDIAPIPE_JOINTS for options)
        backend: Pose estimation backend ("mediapipe" only for now)
        min_visibility: Minimum visibility threshold (0-1)
        sample_stride: Process every Nth frame
        smooth_alpha: Smoothing factor (0=max smooth, 1=no smooth)
        verbose: Print progress messages
        preprocess: Frame preprocessing mode for challenging lighting:
            - "auto": Auto-detect and apply appropriate preprocessing
            - "normalize": Color normalization for colored stage lights
            - "clahe": Histogram equalization for low contrast/dark scenes
            - "none": No preprocessing (fastest)

    Returns:
        List of tracking results:
        [
            {
                "frame": int,
                "cx": float,        # Joint X position
                "cy": float,        # Joint Y position
                "w": float,         # Bounding box width (fixed)
                "h": float,         # Bounding box height (fixed)
                "visibility": float # Joint visibility (0-1)
            },
            ...
        ]

    Example:
        >>> # Track dancer's right wrist
        >>> trajectory = skeleton_track(
        ...     "dance_performance.mp4",
        ...     joint_name="right_wrist",
        ...     min_visibility=0.5
        ... )

        >>> # Track in colored lighting (stage performance)
        >>> trajectory = skeleton_track(
        ...     "concert.mp4",
        ...     joint_name="nose",
        ...     preprocess="normalize"  # Handle colored stage lights
        ... )

        >>> # Track head movement (nose)
        >>> trajectory = skeleton_track(
        ...     "performance.mp4",
        ...     joint_name="nose",
        ...     smooth_alpha=0.2  # More smoothing
        ... )
    """
    video_path = str(Path(video_path).resolve())

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Validate joint name
    if joint_name not in MEDIAPIPE_JOINTS:
        raise ValueError(
            f"Unknown joint: {joint_name}\n"
            f"Available joints: {', '.join(MEDIAPIPE_JOINTS.keys())}"
        )

    # Initialize pose estimator
    if backend == "mediapipe":
        pose = initialize_mediapipe_pose()
        if pose is None:
            raise RuntimeError("MediaPipe initialization failed")
    else:
        raise ValueError(f"Unknown backend: {backend}. Only 'mediapipe' supported.")

    if verbose:
        print(f"SkeletonTracker: Tracking '{joint_name}' using {backend}")
        print(f"  Min visibility: {min_visibility}")
        print(f"  Smooth alpha: {smooth_alpha}")
        print(f"  Preprocessing: {preprocess}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    trajectory = []
    frame_idx = 0
    tracked_frames = 0
    failed_frames = 0
    preprocess_used = {"none": 0, "normalize": 0, "clahe": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample stride
        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue

        # Apply preprocessing for challenging lighting
        frame_rgb = preprocess_frame_for_lighting(frame, preprocess)

        # Process frame
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Extract joint position
            joint_pos = extract_joint_position(
                results.pose_landmarks,
                joint_name,
                width,
                height
            )

            if joint_pos is not None:
                x, y, visibility = joint_pos

                # Check visibility threshold
                if visibility >= min_visibility:
                    trajectory.append({
                        "frame": frame_idx,
                        "cx": x,
                        "cy": y,
                        "w": 10.0,  # Fixed size for joint
                        "h": 10.0,
                        "visibility": visibility,
                    })
                    tracked_frames += 1

                    if verbose and tracked_frames % 30 == 0:
                        print(f"  Frame {frame_idx}: Joint at ({x:.1f}, {y:.1f}), vis={visibility:.2f}")
                else:
                    failed_frames += 1
            else:
                failed_frames += 1
        else:
            failed_frames += 1

        frame_idx += 1

    cap.release()
    pose.close()

    if verbose:
        total_frames = frame_idx
        print(f"\nSkeletonTracker: Complete")
        print(f"  Total frames: {total_frames}")
        print(f"  Tracked: {tracked_frames}")
        print(f"  Failed: {failed_frames}")
        print(f"  Success rate: {100 * tracked_frames / max(total_frames, 1):.1f}%")

    if not trajectory:
        raise RuntimeError(
            f"SkeletonTracker: No joints found!\n"
            f"  Joint: {joint_name}\n"
            f"  Min visibility: {min_visibility}\n"
            f"  Suggestions:\n"
            f"  - Decrease min_visibility (current: {min_visibility})\n"
            f"  - Check if person is visible in video\n"
            f"  - Try different joint (e.g., 'nose' is usually most visible)\n"
            f"  - Ensure good lighting and clear view of body"
        )

    # Apply smoothing
    if smooth_alpha < 1.0:
        trajectory = smooth_joint_trajectory(trajectory, alpha=smooth_alpha)

    return trajectory


def get_available_joints() -> List[str]:
    """
    Get list of available joint names.

    Returns:
        List of joint names
    """
    return list(MEDIAPIPE_JOINTS.keys())


def visualize_skeleton(
    video_path: str,
    output_path: str,
    joint_name: str = "right_wrist",
    show_skeleton: bool = True
):
    """
    Visualize skeleton tracking on video (for debugging).

    Args:
        video_path: Input video path
        output_path: Output video path
        joint_name: Joint to highlight
        show_skeleton: Whether to draw full skeleton
    """
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose = initialize_mediapipe_pose()
    if pose is None:
        raise RuntimeError("MediaPipe not available")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    joint_idx = MEDIAPIPE_JOINTS[joint_name]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            if show_skeleton:
                # Draw full skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # Highlight tracked joint
            landmark = results.pose_landmarks.landmark[joint_idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)  # Red circle
            cv2.putText(
                frame,
                joint_name,
                (x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        writer.write(frame)

    cap.release()
    writer.release()
    pose.close()

    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Skeleton-based tracking for Vid2Spatial")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--joint", default="right_wrist", help=f"Joint to track (default: right_wrist)")
    parser.add_argument("--min-visibility", type=float, default=0.5, help="Min visibility (0-1, default: 0.5)")
    parser.add_argument("--smooth", type=float, default=0.3, help="Smoothing alpha (0-1, default: 0.3)")
    parser.add_argument("--list-joints", action="store_true", help="List available joints")
    parser.add_argument("--visualize", help="Save visualization video to path")
    parser.add_argument("--verbose", action="store_true", help="Print progress")

    args = parser.parse_args()

    if args.list_joints:
        print("Available joints:")
        for joint in get_available_joints():
            print(f"  - {joint}")
        exit(0)

    if args.visualize:
        print(f"Creating visualization...")
        visualize_skeleton(
            args.video,
            args.visualize,
            joint_name=args.joint,
            show_skeleton=True
        )
        exit(0)

    # Track skeleton
    print(f"Tracking '{args.joint}' in {args.video}...")
    trajectory = skeleton_track(
        args.video,
        joint_name=args.joint,
        min_visibility=args.min_visibility,
        smooth_alpha=args.smooth,
        verbose=args.verbose
    )

    print(f"\n✓ Tracked {len(trajectory)} frames")
    print(f"  First position: ({trajectory[0]['cx']:.1f}, {trajectory[0]['cy']:.1f})")
    print(f"  Last position: ({trajectory[-1]['cx']:.1f}, {trajectory[-1]['cy']:.1f})")
    print(f"  Avg visibility: {np.mean([t['visibility'] for t in trajectory]):.2f}")

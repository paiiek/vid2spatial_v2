"""
Point-based tracking for Vid2Spatial.

Use case: Cursor tracking, laser pointer tracking, screen recordings
Example: Mouse cursor in DAW screen recording → Track trajectory
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def detect_bright_point(
    frame: np.ndarray,
    min_brightness: int = 200,
    max_radius: int = 50,
    blur_size: int = 5
) -> Optional[Tuple[float, float]]:
    """
    Detect brightest point in frame (laser pointer, cursor).

    Args:
        frame: Input frame (BGR)
        min_brightness: Minimum brightness threshold (0-255)
        max_radius: Maximum search radius from previous position
        blur_size: Gaussian blur kernel size (odd number)

    Returns:
        (cx, cy) coordinates or None if no bright point found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    if blur_size > 1:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # Find brightest point
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

    # Check if bright enough
    if max_val < min_brightness:
        return None

    cx, cy = max_loc
    return (float(cx), float(cy))


def detect_cursor_template(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.7
) -> Optional[Tuple[float, float, float, float]]:
    """
    Detect cursor using template matching.

    Args:
        frame: Input frame (BGR or grayscale)
        template: Cursor template image
        threshold: Matching threshold (0-1)

    Returns:
        (cx, cy, w, h) or None if no match
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame

    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template

    # Template matching
    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        return None

    # Get template dimensions
    h, w = template_gray.shape

    # Calculate center
    cx = max_loc[0] + w / 2
    cy = max_loc[1] + h / 2

    return (float(cx), float(cy), float(w), float(h))


def detect_good_features(
    frame: np.ndarray,
    max_corners: int = 1,
    quality_level: float = 0.01,
    min_distance: int = 10
) -> Optional[Tuple[float, float]]:
    """
    Detect corner features (good for cursor tip detection).

    Args:
        frame: Input frame (BGR)
        max_corners: Maximum number of corners to detect
        quality_level: Quality threshold (0-1)
        min_distance: Minimum distance between corners

    Returns:
        (cx, cy) of best corner or None
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect corners
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7
    )

    if corners is None or len(corners) == 0:
        return None

    # Return first (best) corner
    cx, cy = corners[0][0]
    return (float(cx), float(cy))


def track_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_points: np.ndarray,
    win_size: Tuple[int, int] = (15, 15),
    max_level: int = 2
) -> Optional[np.ndarray]:
    """
    Track points using Lucas-Kanade optical flow.

    Args:
        prev_gray: Previous frame (grayscale)
        curr_gray: Current frame (grayscale)
        prev_points: Previous point locations (Nx1x2)
        win_size: Window size for optical flow
        max_level: Pyramid levels

    Returns:
        New point locations or None if tracking failed
    """
    # Lucas-Kanade parameters
    lk_params = dict(
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Calculate optical flow
    new_points, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **lk_params
    )

    # Check if tracking succeeded
    if new_points is None or status is None:
        return None

    # Filter good points
    good_new = new_points[status == 1]

    if len(good_new) == 0:
        return None

    return good_new


def point_track(
    video_path: str,
    method: str = "brightness",
    min_brightness: int = 200,
    template_path: Optional[str] = None,
    template_threshold: float = 0.7,
    sample_stride: int = 1,
    use_optical_flow: bool = True,
    verbose: bool = False,
    **kwargs
) -> List[Dict]:
    """
    Track point/cursor in video using various methods.

    Args:
        video_path: Path to video file
        method: Detection method ("brightness", "template", "goodfeatures")
        min_brightness: Minimum brightness for bright point detection (0-255)
        template_path: Path to cursor template image (for template method)
        template_threshold: Template matching threshold (0-1)
        sample_stride: Process every Nth frame (default: 1 = all frames)
        use_optical_flow: Use optical flow for inter-frame tracking (default: True)
        verbose: Print progress messages
        **kwargs: Additional method-specific parameters

    Returns:
        List of tracking results:
        [
            {
                "frame": int,
                "cx": float,  # Center X in pixels
                "cy": float,  # Center Y in pixels
                "w": float,   # Width (or 1.0 for point)
                "h": float,   # Height (or 1.0 for point)
            },
            ...
        ]

    Methods:
        - "brightness": Detect brightest point (laser pointer, bright cursor)
        - "template": Template matching (requires template image)
        - "goodfeatures": Corner detection (cursor tip, sharp points)

    Example:
        >>> # Track laser pointer
        >>> trajectory = point_track(
        ...     "laser_presentation.mp4",
        ...     method="brightness",
        ...     min_brightness=220
        ... )

        >>> # Track cursor with template
        >>> trajectory = point_track(
        ...     "screen_recording.mp4",
        ...     method="template",
        ...     template_path="cursor.png",
        ...     template_threshold=0.8
        ... )
    """
    video_path = str(Path(video_path).resolve())

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Load template if needed
    template = None
    if method == "template":
        if template_path is None:
            raise ValueError("template_path required for template method")
        template = cv2.imread(str(template_path))
        if template is None:
            raise FileNotFoundError(f"Template not found: {template_path}")

    if verbose:
        print(f"PointTracker: Tracking using '{method}' method")
        if method == "brightness":
            print(f"  Min brightness: {min_brightness}")
        elif method == "template":
            print(f"  Template: {template_path}")
            print(f"  Threshold: {template_threshold}")
        print(f"  Use optical flow: {use_optical_flow}")

    trajectory = []
    frame_idx = 0
    processed_frames = 0
    failed_frames = 0

    # For optical flow tracking
    prev_gray = None
    prev_point = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample stride
        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue

        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try optical flow first if enabled and we have previous point
        tracked_by_flow = False
        if use_optical_flow and prev_gray is not None and prev_point is not None:
            prev_pts = np.array([[prev_point]], dtype=np.float32)
            new_pts = track_optical_flow(prev_gray, gray, prev_pts)

            if new_pts is not None and len(new_pts) > 0:
                cx, cy = new_pts[0]
                # Use previous dimensions
                if trajectory:
                    w, h = trajectory[-1]["w"], trajectory[-1]["h"]
                else:
                    w, h = 1.0, 1.0

                trajectory.append({
                    "frame": frame_idx,
                    "cx": float(cx),
                    "cy": float(cy),
                    "w": float(w),
                    "h": float(h),
                })
                processed_frames += 1
                tracked_by_flow = True

                # Update for next iteration
                prev_point = (cx, cy)

        # If optical flow failed or not used, detect from scratch
        if not tracked_by_flow:
            result = None

            if method == "brightness":
                blur_size = kwargs.get("blur_size", 5)
                point = detect_bright_point(frame, min_brightness, blur_size=blur_size)
                if point is not None:
                    cx, cy = point
                    result = (cx, cy, 1.0, 1.0)

            elif method == "template":
                result = detect_cursor_template(frame, template, template_threshold)

            elif method == "goodfeatures":
                quality_level = kwargs.get("quality_level", 0.01)
                min_distance = kwargs.get("min_distance", 10)
                point = detect_good_features(frame, quality_level=quality_level, min_distance=min_distance)
                if point is not None:
                    cx, cy = point
                    result = (cx, cy, 1.0, 1.0)

            else:
                raise ValueError(f"Unknown method: {method}")

            if result is not None:
                cx, cy, w, h = result
                trajectory.append({
                    "frame": frame_idx,
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h,
                })
                processed_frames += 1

                # Update for optical flow
                prev_point = (cx, cy)

                if verbose and processed_frames % 30 == 0:
                    print(f"  Frame {frame_idx}: Found point at ({cx:.1f}, {cy:.1f})")
            else:
                failed_frames += 1
                # Lost tracking, reset optical flow
                prev_point = None

        # Update previous frame
        prev_gray = gray
        frame_idx += 1

    cap.release()

    if verbose:
        print(f"\nPointTracker: Complete")
        print(f"  Total frames: {frame_idx}")
        print(f"  Tracked: {processed_frames}")
        print(f"  Failed: {failed_frames}")
        print(f"  Success rate: {100 * processed_frames / max(frame_idx, 1):.1f}%")

    if not trajectory:
        raise RuntimeError(
            f"PointTracker: No points found!\n"
            f"  Method: {method}\n"
            f"  Suggestions:\n"
            f"  - For brightness: Decrease min_brightness (current: {min_brightness})\n"
            f"  - For template: Decrease template_threshold (current: {template_threshold})\n"
            f"  - For goodfeatures: Decrease quality_level\n"
            f"  - Check if video contains visible cursor/pointer"
        )

    return trajectory


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Point-based tracking for Vid2Spatial")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--method", default="brightness", choices=["brightness", "template", "goodfeatures"],
                        help="Detection method (default: brightness)")
    parser.add_argument("--min-brightness", type=int, default=200, help="Min brightness (0-255, default: 200)")
    parser.add_argument("--template", help="Cursor template image path (for template method)")
    parser.add_argument("--template-threshold", type=float, default=0.7, help="Template threshold (0-1, default: 0.7)")
    parser.add_argument("--no-optical-flow", action="store_true", help="Disable optical flow tracking")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    # Track
    print(f"Tracking points in {args.video}...")
    trajectory = point_track(
        args.video,
        method=args.method,
        min_brightness=args.min_brightness,
        template_path=args.template,
        template_threshold=args.template_threshold,
        use_optical_flow=not args.no_optical_flow,
        verbose=args.verbose
    )

    print(f"\n✓ Tracked {len(trajectory)} frames")
    print(f"  First point: ({trajectory[0]['cx']:.1f}, {trajectory[0]['cy']:.1f})")
    print(f"  Last point: ({trajectory[-1]['cx']:.1f}, {trajectory[-1]['cy']:.1f})")

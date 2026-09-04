"""
Color-based tracking for Vid2Spatial.

Use case: Hand-drawn sketches, colored markers, creative trajectory videos
Example: Red dot moving on paper → Track trajectory
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path


def detect_dominant_color(
    video_path: str,
    num_samples: int = 10,
    min_saturation: int = 100,
    min_value: int = 100,
) -> Optional[Tuple[int, int, int]]:
    """
    Auto-detect the most saturated/vibrant color in the video.

    Useful when target color is unknown. Finds the most prominent
    saturated color that could be a marker or object of interest.

    Args:
        video_path: Path to video file
        num_samples: Number of frames to sample
        min_saturation: Minimum saturation to consider (0-255)
        min_value: Minimum brightness to consider (0-255)

    Returns:
        BGR color tuple or None if no dominant color found
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None

    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    hue_votes = []

    for fidx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find saturated and bright pixels
        mask = (hsv[:, :, 1] > min_saturation) & (hsv[:, :, 2] > min_value)

        if mask.sum() > 100:  # Need at least some pixels
            h_vals = hsv[:, :, 0][mask]
            # Get histogram of hue values
            hist, bins = np.histogram(h_vals, bins=36, range=(0, 180))
            dominant_bin = np.argmax(hist)
            hue_votes.append(bins[dominant_bin] + 2.5)  # Center of bin

    cap.release()

    if not hue_votes:
        return None

    # Find most common hue
    avg_hue = int(np.median(hue_votes))

    # Convert back to BGR
    # Create a pixel with this hue at high saturation/value
    hsv_pixel = np.uint8([[[avg_hue, 255, 255]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)

    return tuple(int(x) for x in bgr_pixel[0, 0])


def check_color_in_samples(
    video_path: str,
    hsv_ranges: Tuple,
    min_area: int = 20,
    num_samples: int = 5,
) -> Tuple[bool, int]:
    """
    Quick check if target color exists in sample frames.

    Returns:
        (found, count) - whether color found and in how many frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return False, 0

    frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    found_count = 0

    for fidx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        result = find_largest_color_blob(frame, hsv_ranges, min_area)
        if result is not None:
            found_count += 1

    cap.release()

    return found_count > 0, found_count


def bgr_to_hsv_range(
    bgr_color: Tuple[int, int, int],
    tolerance: int = 30
) -> Tuple[np.ndarray, ...]:
    """
    Convert BGR color to HSV range for masking.

    Args:
        bgr_color: Target color in BGR format (e.g., (255, 0, 0) for red, (0, 0, 255) for blue)
        tolerance: HSV tolerance for color matching

    Returns:
        (lower_bound, upper_bound) for normal colors
        (lower1, upper1, lower2, upper2) for red (wraps around H=0/180)
    """
    # Convert BGR to HSV
    bgr_array = np.uint8([[bgr_color]])
    hsv_color = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)[0][0]

    h, s, v = hsv_color

    # Red in HSV is at H~0 or H~180 (wraps around)
    # Blue is at H~120, Green is at H~60

    # Check if this is red color (H close to 0 or 180)
    if h < tolerance or h > (180 - tolerance):
        # Red wraps: use two ranges
        lower1 = np.array([0, max(s - tolerance, 50), max(v - tolerance, 50)])
        upper1 = np.array([min(h + tolerance, tolerance), 255, 255])
        lower2 = np.array([max(180 - tolerance, 180 - tolerance + h), max(s - tolerance, 50), max(v - tolerance, 50)])
        upper2 = np.array([180, 255, 255])
        return (lower1, upper1, lower2, upper2)  # Return two ranges
    else:
        # Normal case (blue, green, etc.)
        lower = np.array([max(h - tolerance, 0), max(s - tolerance, 50), max(v - tolerance, 50)])
        upper = np.array([min(h + tolerance, 180), 255, 255])
        return (lower, upper)


def find_largest_color_blob(
    frame: np.ndarray,
    hsv_ranges: Tuple,
    min_area: int = 100
) -> Optional[Tuple[float, float, float, float]]:
    """
    Find largest colored blob in frame.

    Args:
        frame: BGR image
        hsv_ranges: Output from bgr_to_hsv_range()
        min_area: Minimum contour area to consider

    Returns:
        (cx, cy, w, h) or None if no blob found
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask
    if len(hsv_ranges) == 4:
        # Red color wrapping case: combine two masks
        lower1, upper1, lower2, upper2 = hsv_ranges
        mask1 = cv2.inRange(hsv, lower1.astype(np.uint8), upper1.astype(np.uint8))
        mask2 = cv2.inRange(hsv, lower2.astype(np.uint8), upper2.astype(np.uint8))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        # Normal case
        lower, upper = hsv_ranges
        mask = cv2.inRange(hsv, lower.astype(np.uint8), upper.astype(np.uint8))

    # Morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < min_area:
        return None

    # Compute moments for center of mass
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    return (cx, cy, float(w), float(h))


def color_track(
    video_path: str,
    target_color: Optional[Union[Tuple[int, int, int], str]] = None,  # BGR or "auto"
    color_tolerance: int = 30,
    min_area: int = 20,  # Lowered default for small objects
    sample_stride: int = 1,
    verbose: bool = False,
    auto_detect: bool = False,
    early_check: bool = True,
    **kwargs
) -> List[Dict]:
    """
    Track colored point/marker in video.

    Use case: Hand-drawn sketches with colored markers, creative trajectory videos

    Args:
        video_path: Path to video file
        target_color: Target color in BGR format, "auto" for auto-detection,
                      or None (will auto-detect if auto_detect=True).
                      Default is (0, 0, 255) red if not auto-detecting.
        color_tolerance: HSV tolerance for color matching (default: 30)
        min_area: Minimum blob area in pixels (default: 20, lowered for small objects)
        sample_stride: Process every Nth frame (default: 1 = all frames)
        verbose: Print progress messages
        auto_detect: If True and target_color is None, auto-detect dominant color
        early_check: Check sample frames before full processing (default: True)
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        List of tracking results:
        [
            {
                "frame": int,
                "cx": float,  # Center X in pixels
                "cy": float,  # Center Y in pixels
                "w": float,   # Bounding box width
                "h": float,   # Bounding box height
            },
            ...
        ]

    Example:
        >>> # Track red marker in hand-drawn sketch
        >>> trajectory = color_track(
        ...     "sketch.mp4",
        ...     target_color=(0, 0, 255),  # Red in BGR
        ...     color_tolerance=40,
        ...     min_area=50
        ... )
        >>> print(f"Tracked {len(trajectory)} frames")

        >>> # Auto-detect color
        >>> trajectory = color_track(
        ...     "colored_marker.mp4",
        ...     auto_detect=True,  # Automatically find dominant color
        ... )
    """
    video_path = str(Path(video_path).resolve())

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Handle auto-detection
    if target_color == "auto" or (target_color is None and auto_detect):
        if verbose:
            print("ColorTracker: Auto-detecting dominant color...")
        detected = detect_dominant_color(video_path)
        if detected is None:
            raise RuntimeError(
                "ColorTracker: Auto-detection failed!\n"
                "  Could not find any saturated colors in video.\n"
                "  Please specify target_color manually."
            )
        target_color = detected
        if verbose:
            print(f"  Detected color: {target_color} (BGR)")
    elif target_color is None:
        # Default to red if no color specified and auto_detect=False
        target_color = (0, 0, 255)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Get video info for adaptive min_area
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Adaptive min_area based on video resolution (if using default)
    if min_area == 20:  # Default value
        # 0.001% of frame area, minimum 10 pixels
        adaptive_min = max(10, int(width * height * 0.00001))
        min_area = max(min_area, adaptive_min)

    # Get HSV ranges for target color
    hsv_ranges = bgr_to_hsv_range(target_color, color_tolerance)

    # Early check: verify color exists in sample frames
    if early_check:
        found, count = check_color_in_samples(video_path, hsv_ranges, min_area, num_samples=5)
        if not found:
            # Try with lower min_area
            found2, count2 = check_color_in_samples(video_path, hsv_ranges, min_area=5, num_samples=5)
            if found2:
                suggestion = f"Try reducing min_area to {max(5, min_area // 2)}"
            else:
                suggestion = "Color not visible in video"

            raise RuntimeError(
                f"ColorTracker: Target color not detected in sample frames!\n"
                f"  Color: {target_color} (BGR)\n"
                f"  Tolerance: {color_tolerance}\n"
                f"  Min area: {min_area}\n"
                f"  Resolution: {width}x{height}\n"
                f"Quick check: {suggestion}\n"
                f"Suggestions:\n"
                f"  - Try auto_detect=True to find dominant color\n"
                f"  - Increase color_tolerance (current: {color_tolerance})\n"
                f"  - Decrease min_area (current: {min_area})\n"
                f"  - Verify target_color is in BGR format (not RGB!)"
            )
        elif verbose:
            print(f"  Early check: Color found in {count}/5 sample frames")

    # Re-open video for full processing
    cap = cv2.VideoCapture(video_path)

    if verbose:
        print(f"ColorTracker: Tracking color {target_color} (BGR)")
        print(f"  Tolerance: {color_tolerance}")
        print(f"  Min area: {min_area}")
        print(f"  HSV ranges: {hsv_ranges}")

    trajectory = []
    frame_idx = 0
    processed_frames = 0
    failed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample stride
        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue

        # Find colored blob
        result = find_largest_color_blob(frame, hsv_ranges, min_area)

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

            if verbose and processed_frames % 30 == 0:
                print(f"  Frame {frame_idx}: Found blob at ({cx:.1f}, {cy:.1f}), size {w:.0f}x{h:.0f}")
        else:
            failed_frames += 1

        frame_idx += 1

    cap.release()

    if verbose:
        print(f"\nColorTracker: Complete")
        print(f"  Total frames: {frame_idx}")
        print(f"  Tracked: {processed_frames}")
        print(f"  Failed: {failed_frames}")
        print(f"  Success rate: {100 * processed_frames / max(frame_idx, 1):.1f}%")

    if not trajectory:
        raise RuntimeError(
            f"ColorTracker: No colored blobs found!\n"
            f"  Color: {target_color} (BGR)\n"
            f"  Tolerance: {color_tolerance}\n"
            f"  Min area: {min_area}\n"
            f"Suggestions:\n"
            f"  - Increase color_tolerance (current: {color_tolerance})\n"
            f"  - Decrease min_area (current: {min_area})\n"
            f"  - Check if target_color is correct (remember BGR format!)\n"
            f"  - Ensure video has visible colored marker"
        )

    return trajectory


def visualize_color_tracking(
    video_path: str,
    trajectory: List[Dict],
    output_path: Optional[str] = None,
    target_color: Tuple[int, int, int] = (255, 0, 0),
    show_mask: bool = True
) -> None:
    """
    Visualize color tracking results.

    Args:
        video_path: Path to original video
        trajectory: Output from color_track()
        output_path: Path to save visualization video (optional)
        target_color: Target color for mask visualization
        show_mask: Show HSV mask overlay
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create trajectory lookup
    traj_dict = {t["frame"]: t for t in trajectory}

    hsv_ranges = bgr_to_hsv_range(target_color, 30)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()

        # Show mask overlay
        if show_mask:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if len(hsv_ranges) == 4:
                lower1, upper1, lower2, upper2 = hsv_ranges
                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                lower, upper = hsv_ranges
                mask = cv2.inRange(hsv, lower, upper)

            # Overlay mask in green
            vis[mask > 0] = (0, 255, 0)
            vis = cv2.addWeighted(frame, 0.7, vis, 0.3, 0)

        # Draw tracking result
        if frame_idx in traj_dict:
            t = traj_dict[frame_idx]
            cx, cy = int(t["cx"]), int(t["cy"])
            w, h = int(t["w"]), int(t["h"])

            # Draw center
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

            # Draw bounding box
            x = int(cx - w / 2)
            y = int(cy - h / 2)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Draw frame number
            cv2.putText(vis, f"Frame {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer:
            writer.write(vis)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"Visualization saved: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Color-based tracking for Vid2Spatial")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--color", default="255,0,0", help="Target color in BGR (default: 255,0,0 = red)")
    parser.add_argument("--tolerance", type=int, default=30, help="Color tolerance (default: 30)")
    parser.add_argument("--min-area", type=int, default=100, help="Min blob area (default: 100)")
    parser.add_argument("--output", help="Output visualization video (optional)")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    # Parse color
    color_parts = [int(x) for x in args.color.split(",")]
    if len(color_parts) != 3:
        raise ValueError("Color must be 3 comma-separated integers (B,G,R)")
    target_color = tuple(color_parts)

    # Track
    print(f"Tracking color {target_color} in {args.video}...")
    trajectory = color_track(
        args.video,
        target_color=target_color,
        color_tolerance=args.tolerance,
        min_area=args.min_area,
        verbose=args.verbose
    )

    print(f"\n✓ Tracked {len(trajectory)} frames")

    # Visualize
    if args.output:
        print(f"\nGenerating visualization...")
        visualize_color_tracking(args.video, trajectory, args.output, target_color)

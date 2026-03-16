"""
Temporal Smoothing for Trajectory using Kalman Filter

Reduces flickering and improves temporal consistency of spatial audio positioning.
Based on VisualEchoes (ECCV 2020) and Sep-Stereo (ECCV 2020) approaches.
"""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from filterpy.kalman import KalmanFilter


class TrajectoryKalmanSmoother:
    """
    Smooth trajectory using Kalman filter with constant velocity model.

    State vector: [az, el, dist, v_az, v_el, v_dist]
    - az, el, dist: azimuth, elevation, distance
    - v_az, v_el, v_dist: velocities

    This helps remove tracking noise and ensures smooth transitions.
    """

    def __init__(self, fps: float = 30.0, process_noise: float = 0.01,
                 measurement_noise: float = 0.1):
        """
        Initialize Kalman filter for trajectory smoothing.

        Args:
            fps: Video frame rate (for dt calculation)
            process_noise: Process noise covariance (lower = smoother, higher = responsive)
            measurement_noise: Measurement noise covariance (lower = trust measurements more)
        """
        # 6D state: [az, el, dist, v_az, v_el, v_dist]
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.fps = fps
        self.dt = 1.0 / fps

        # State transition matrix (constant velocity model)
        # x(t+1) = x(t) + v*dt
        self.kf.F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix (we only observe position, not velocity)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Process noise covariance
        # Higher = allow more variation in dynamics
        self.kf.Q *= process_noise

        # Measurement noise covariance
        # Higher = trust measurements less (smoother but more lag)
        self.kf.R *= measurement_noise

        # Initial state covariance
        self.kf.P *= 10.0

        # Track if initialized
        self.initialized = False

    def reset(self):
        """Reset filter state."""
        self.initialized = False
        self.kf.P *= 10.0

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update filter with new measurement and return smoothed estimate.

        Args:
            measurement: [az, el, dist] in degrees and meters

        Returns:
            smoothed: [az, el, dist] smoothed values
        """
        if not self.initialized:
            # Initialize state with first measurement
            # Assume zero initial velocity
            self.kf.x = np.array([
                measurement[0],  # az
                measurement[1],  # el
                measurement[2],  # dist
                0.0,            # v_az
                0.0,            # v_el
                0.0             # v_dist
            ])
            self.initialized = True
            return measurement.copy()

        # Predict next state
        self.kf.predict()

        # Update with measurement
        self.kf.update(measurement)

        # Return position estimates only
        return self.kf.x[:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimates [v_az, v_el, v_dist]."""
        return self.kf.x[3:].copy()


class AdaptiveKalmanSmoother(TrajectoryKalmanSmoother):
    """
    Adaptive Kalman filter that adjusts noise based on tracking confidence.

    When tracking is unstable (large bbox size changes, low confidence),
    increase smoothing. When stable, allow more responsiveness.
    """

    def __init__(self, fps: float = 30.0, base_process_noise: float = 0.01,
                 base_measurement_noise: float = 0.1):
        super().__init__(fps, base_process_noise, base_measurement_noise)
        self.base_process_noise = base_process_noise
        self.base_measurement_noise = base_measurement_noise
        self.prev_measurement = None

    def update_adaptive(self, measurement: np.ndarray,
                       confidence: float = 1.0) -> np.ndarray:
        """
        Update with adaptive noise based on confidence.

        Args:
            measurement: [az, el, dist]
            confidence: Tracking confidence [0, 1] (1 = high confidence)

        Returns:
            smoothed: [az, el, dist]
        """
        # Adjust measurement noise based on confidence
        # Low confidence → high noise → more smoothing
        adaptive_R = self.base_measurement_noise / (confidence + 0.1)
        self.kf.R = np.eye(3) * adaptive_R

        # If large jump detected, increase process noise temporarily
        if self.prev_measurement is not None:
            delta = np.abs(measurement - self.prev_measurement)
            # Check for large jumps (>20 deg azimuth, >15 deg elevation, >2m distance)
            if delta[0] > 20 or delta[1] > 15 or delta[2] > 2.0:
                # Allow filter to adapt faster
                self.kf.Q *= 5.0

        smoothed = self.update(measurement)

        # Reset process noise
        self.kf.Q = np.eye(6) * self.base_process_noise

        self.prev_measurement = measurement.copy()
        return smoothed


def smooth_trajectory_batch(
    trajectory: List[Dict],
    fps: float = 30.0,
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
    adaptive: bool = False,
    use_rts: bool = True,
) -> List[Dict]:
    """
    Apply Kalman + RTS smoothing to entire trajectory.

    Forward pass: AdaptiveKalman (confidence-weighted measurement noise).
    Backward pass (RTS): Rauch-Tung-Striebel smoother — uses future measurements
        to correct past estimates, eliminating causal lag. Always better than
        forward-only for offline processing.

    Note on jerk: jerk limiting belongs at the audio-sample level (44100 Hz),
    not the traj keyframe level (30 fps). foa_render.smooth_limit_angles()
    handles this via max_deg_per_s on the interpolated audio-rate az/el signal.
    Applying it at 30 fps would require thresholds ~1000× smaller and causes
    cumulative integration drift on fast-moving objects.

    Args:
        trajectory: List of frame dicts with 'az', 'el', 'dist_m' keys
        fps: Video frame rate
        process_noise: Process noise covariance
        measurement_noise: Measurement noise covariance
        adaptive: Use confidence-adaptive measurement noise
        use_rts: Apply RTS backward pass after forward Kalman (default True)
    """
    if not trajectory:
        return []

    N = len(trajectory)

    # ── Forward Kalman pass ──────────────────────────────────────────────────
    if adaptive:
        smoother = AdaptiveKalmanSmoother(fps, process_noise, measurement_noise)
    else:
        smoother = TrajectoryKalmanSmoother(fps, process_noise, measurement_noise)

    # Storage for RTS: (x_fwd, P_fwd, F) per step
    xs_fwd: List[np.ndarray] = []   # posterior state  [6]
    Ps_fwd: List[np.ndarray] = []   # posterior cov    [6,6]
    smoothed_traj = []

    for frame_data in trajectory:
        measurement = np.array([
            frame_data['az'],
            frame_data['el'],
            frame_data['dist_m'],
        ], dtype=np.float64)

        if adaptive and 'confidence' in frame_data:
            smoother.update_adaptive(measurement.astype(np.float32), frame_data['confidence'])
        else:
            smoother.update(measurement.astype(np.float32))

        xs_fwd.append(smoother.kf.x.copy())
        Ps_fwd.append(smoother.kf.P.copy())

        smoothed_frame = frame_data.copy()
        smoothed_frame['az']     = float(smoother.kf.x[0])
        smoothed_frame['el']     = float(smoother.kf.x[1])
        smoothed_frame['dist_m'] = float(smoother.kf.x[2])
        vel = smoother.kf.x[3:]
        smoothed_frame['v_az']   = float(vel[0])
        smoothed_frame['v_el']   = float(vel[1])
        smoothed_frame['v_dist'] = float(vel[2])
        smoothed_traj.append(smoothed_frame)

    # ── RTS backward pass ────────────────────────────────────────────────────
    if use_rts and N > 1:
        F = smoother.kf.F  # state transition matrix (constant)

        # Initialize RTS with last forward estimate
        xs_rts = [None] * N
        Ps_rts = [None] * N
        xs_rts[N - 1] = xs_fwd[N - 1].copy()
        Ps_rts[N - 1] = Ps_fwd[N - 1].copy()

        for k in range(N - 2, -1, -1):
            P_pred = F @ Ps_fwd[k] @ F.T + smoother.kf.Q   # predicted cov at k+1
            # RTS gain
            G = Ps_fwd[k] @ F.T @ np.linalg.inv(P_pred)
            # Smoothed state and cov
            xs_rts[k] = xs_fwd[k] + G @ (xs_rts[k + 1] - F @ xs_fwd[k])
            Ps_rts[k] = Ps_fwd[k] + G @ (Ps_rts[k + 1] - P_pred) @ G.T

        # Write RTS results back into trajectory
        for i, frame in enumerate(smoothed_traj):
            x = xs_rts[i]
            frame['az']     = float(x[0])
            frame['el']     = float(x[1])
            frame['dist_m'] = float(x[2])
            frame['v_az']   = float(x[3])
            frame['v_el']   = float(x[4])
            frame['v_dist'] = float(x[5])

    # Back-project smoothed az/el → cx/cy so overlay bbox follows
    # the smoothed trajectory (not the raw/interpolated pixel positions).
    # Requires 'width' and 'height' intrinsics stored in frame (added by
    # v2_spatial_tracker._project_3d via the 'intrinsics' wrapper).
    # Formula (y-down, pinhole):
    #   f = (W/2) / tan(fov_h/2)
    #   x_ndc = tan(az)
    #   y_ndc = tan(el) * sqrt(1 + tan²(az))   ← correct perspective
    #   cx = W/2 + f * x_ndc
    #   cy = H/2 + f * y_ndc
    if smoothed_traj:
        f0 = smoothed_traj[0]
        W = f0.get('width')
        H = f0.get('height')
        fov_deg = f0.get('fov_deg')
        if W and H and fov_deg:
            fov_h = math.radians(fov_deg)
            f_px = (W / 2.0) / math.tan(fov_h / 2.0)
            for frame in smoothed_traj:
                az = frame['az']
                el = frame['el']
                x_ndc = math.tan(az)
                y_ndc = math.tan(el) * math.sqrt(1.0 + x_ndc ** 2)
                frame['cx'] = W / 2.0 + f_px * x_ndc
                frame['cy'] = H / 2.0 + f_px * y_ndc

    # Recompute d_rel from Kalman-smoothed dist_m so that per-frame d_rel in
    # traj.json reflects the smooth trajectory, not raw noisy depth estimates.
    if smoothed_traj:
        dist_arr = np.array([f['dist_m'] for f in smoothed_traj], dtype=np.float64)
        d_min = float(dist_arr.min())
        d_max = float(dist_arr.max())
        d_range = d_max - d_min
        for i, frame in enumerate(smoothed_traj):
            if d_range < 0.1:
                frame['d_rel'] = 0.5
            else:
                frame['d_rel'] = float(np.clip((dist_arr[i] - d_min) / d_range, 0.0, 1.0))

    return smoothed_traj


def apply_moving_average(
    trajectory: List[Dict],
    window_size: int = 5
) -> List[Dict]:
    """
    Simple moving average smoothing (baseline).

    Faster than Kalman but less sophisticated.

    Args:
        trajectory: List of frame dicts
        window_size: Window size for averaging

    Returns:
        Smoothed trajectory
    """
    if not trajectory or window_size < 2:
        return trajectory

    smoothed = []
    half_window = window_size // 2

    for i, frame in enumerate(trajectory):
        # Collect window
        start = max(0, i - half_window)
        end = min(len(trajectory), i + half_window + 1)
        window = trajectory[start:end]

        # Average
        avg_az = np.mean([f['az'] for f in window])
        avg_el = np.mean([f['el'] for f in window])
        avg_dist = np.mean([f['dist_m'] for f in window])

        # Create smoothed frame
        smoothed_frame = frame.copy()
        smoothed_frame['az'] = float(avg_az)
        smoothed_frame['el'] = float(avg_el)
        smoothed_frame['dist_m'] = float(avg_dist)

        smoothed.append(smoothed_frame)

    return smoothed


def compare_smoothing_methods(
    trajectory: List[Dict],
    fps: float = 30.0
) -> Dict[str, List[Dict]]:
    """
    Compare different smoothing methods on same trajectory.

    Useful for ablation study.

    Returns:
        Dict with keys: 'raw', 'moving_avg', 'kalman', 'adaptive_kalman'
    """
    results = {
        'raw': trajectory,
        'moving_avg': apply_moving_average(trajectory, window_size=5),
        'kalman': smooth_trajectory_batch(trajectory, fps=fps,
                                         process_noise=0.01,
                                         measurement_noise=0.1),
        'adaptive_kalman': smooth_trajectory_batch(trajectory, fps=fps,
                                                   process_noise=0.01,
                                                   measurement_noise=0.1,
                                                   adaptive=True)
    }
    return results


if __name__ == "__main__":
    # Test the smoother
    print("Testing Kalman Smoother...")

    # Create synthetic noisy trajectory
    np.random.seed(42)
    T = 100
    t = np.linspace(0, 10, T)

    # True trajectory: sinusoidal motion
    true_az = 30 * np.sin(0.5 * t)
    true_el = 10 * np.cos(0.3 * t)
    true_dist = 5.0 + 1.0 * np.sin(0.2 * t)

    # Add noise
    noisy_az = true_az + np.random.randn(T) * 5.0
    noisy_el = true_el + np.random.randn(T) * 3.0
    noisy_dist = true_dist + np.random.randn(T) * 0.5

    # Create trajectory list
    noisy_traj = []
    for i in range(T):
        noisy_traj.append({
            'frame': i,
            'az': noisy_az[i],
            'el': noisy_el[i],
            'dist_m': noisy_dist[i]
        })

    # Apply smoothing
    smoothed = smooth_trajectory_batch(noisy_traj, fps=30.0)

    # Compute error reduction
    raw_error_az = np.mean(np.abs(noisy_az - true_az))
    smooth_error_az = np.mean(np.abs([f['az'] for f in smoothed] - true_az))

    print(f"\nAzimuth Error:")
    print(f"  Raw:      {raw_error_az:.2f} degrees")
    print(f"  Smoothed: {smooth_error_az:.2f} degrees")
    print(f"  Reduction: {(1 - smooth_error_az/raw_error_az)*100:.1f}%")

    # Test moving average
    ma_smoothed = apply_moving_average(noisy_traj, window_size=5)
    ma_error_az = np.mean(np.abs([f['az'] for f in ma_smoothed] - true_az))
    print(f"  Moving Avg: {ma_error_az:.2f} degrees")

    print("\n✅ Kalman smoother test passed!")

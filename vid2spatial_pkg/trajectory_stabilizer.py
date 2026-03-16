"""
Trajectory Stabilizer using Kalman Filter.

Provides smooth, physically-plausible trajectories by:
1. Modeling object motion with position + velocity state
2. Filtering measurement noise
3. Predicting through occlusions
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class KalmanState:
    """Kalman filter state for 3D trajectory."""
    # State: [az, el, dist, vaz, vel, vdist]
    x: np.ndarray  # State vector
    P: np.ndarray  # Covariance matrix


class TrajectoryKalmanFilter:
    """
    Kalman filter for 3D spatial trajectory stabilization.

    State vector: [az, el, dist, v_az, v_el, v_dist]
    - az, el: angles in radians
    - dist: distance in meters
    - v_*: velocities (change per frame)
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        dt: float = 1.0,  # Time step (frames)
    ):
        """
        Initialize Kalman filter.

        Args:
            process_noise: How much we expect the object to accelerate
            measurement_noise: How noisy the measurements are
            dt: Time step between frames
        """
        self.dt = dt

        # State dimension: [az, el, dist, vaz, vel, vdist]
        self.n_state = 6
        self.n_meas = 3  # [az, el, dist]

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],   # az = az + vaz * dt
            [0, 1, 0, 0,  dt, 0],   # el = el + vel * dt
            [0, 0, 1, 0,  0,  dt],  # dist = dist + vdist * dt
            [0, 0, 0, 1,  0,  0],   # vaz = vaz
            [0, 0, 0, 0,  1,  0],   # vel = vel
            [0, 0, 0, 0,  0,  1],   # vdist = vdist
        ])

        # Measurement matrix (we observe az, el, dist)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        # Process noise covariance
        # Higher values = trust measurements more, allow faster changes
        q = process_noise
        self.Q = np.array([
            [q*dt**2, 0, 0, q*dt, 0, 0],
            [0, q*dt**2, 0, 0, q*dt, 0],
            [0, 0, q*dt**2, 0, 0, q*dt],
            [q*dt, 0, 0, q, 0, 0],
            [0, q*dt, 0, 0, q, 0],
            [0, 0, q*dt, 0, 0, q],
        ]) * 0.1

        # Measurement noise covariance
        # Higher values = trust predictions more, smoother output
        r = measurement_noise
        self.R = np.diag([r, r, r * 10])  # dist has more noise

        self.state: Optional[KalmanState] = None

    def initialize(self, az: float, el: float, dist: float):
        """Initialize filter with first measurement."""
        x = np.array([az, el, dist, 0, 0, 0])
        P = np.eye(self.n_state) * 1.0  # Initial uncertainty
        P[3:, 3:] *= 0.1  # Lower uncertainty for velocities initially
        self.state = KalmanState(x=x, P=P)

    def predict(self) -> Tuple[float, float, float]:
        """Predict next state."""
        if self.state is None:
            raise ValueError("Filter not initialized")

        # Predict state
        x_pred = self.F @ self.state.x
        P_pred = self.F @ self.state.P @ self.F.T + self.Q

        self.state.x = x_pred
        self.state.P = P_pred

        return x_pred[0], x_pred[1], x_pred[2]

    def update(self, az: float, el: float, dist: float) -> Tuple[float, float, float]:
        """
        Update filter with new measurement.

        Returns:
            Filtered (az, el, dist)
        """
        if self.state is None:
            self.initialize(az, el, dist)
            return az, el, dist

        # Predict
        x_pred = self.F @ self.state.x
        P_pred = self.F @ self.state.P @ self.F.T + self.Q

        # Measurement
        z = np.array([az, el, dist])

        # Kalman gain
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update
        y = z - self.H @ x_pred  # Innovation
        x_new = x_pred + K @ y
        P_new = (np.eye(self.n_state) - K @ self.H) @ P_pred

        self.state.x = x_new
        self.state.P = P_new

        return x_new[0], x_new[1], x_new[2]

    def get_velocity(self) -> Tuple[float, float, float]:
        """Get current velocity estimate."""
        if self.state is None:
            return 0, 0, 0
        return self.state.x[3], self.state.x[4], self.state.x[5]


class OneEuroFilter:
    """
    One Euro Filter for adaptive smoothing.

    - Slow movements: heavy smoothing (low cutoff)
    - Fast movements: light smoothing (high cutoff) to reduce lag
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        """
        Args:
            min_cutoff: Minimum cutoff frequency (lower = smoother)
            beta: Speed coefficient (higher = less lag for fast movements)
            d_cutoff: Cutoff frequency for derivative
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def _smoothing_factor(self, cutoff: float, dt: float) -> float:
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Filter input signal.

        Args:
            x: Input vector [az, el, dist]
            t: Timestamp (frame number)

        Returns:
            Filtered output
        """
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            dt = 1.0

        # Estimate derivative
        dx = (x - self.x_prev) / dt

        # Smooth derivative
        a_d = self._smoothing_factor(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Adaptive cutoff based on speed
        speed = np.linalg.norm(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Smooth signal
        a = self._smoothing_factor(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


class TrajectoryStabilizer:
    """
    High-level trajectory stabilizer with multiple filter options.
    """

    def __init__(
        self,
        method: str = "kalman",
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        """
        Args:
            method: "kalman", "one_euro", or "ema"
            process_noise: For Kalman filter
            measurement_noise: For Kalman filter
        """
        self.method = method

        if method == "kalman":
            self.filter = TrajectoryKalmanFilter(
                process_noise=process_noise,
                measurement_noise=measurement_noise,
            )
        elif method == "one_euro":
            self.filter = OneEuroFilter(
                min_cutoff=1.0,
                beta=0.007,
            )
        else:
            self.filter = None  # EMA handled separately

        self.ema_alpha = 0.3

    def stabilize_trajectory(
        self,
        trajectory: List[Dict],
        smooth_xyz: bool = True,
    ) -> List[Dict]:
        """
        Stabilize a trajectory.

        Args:
            trajectory: List of dicts with 'az', 'el', 'dist_m', 'x', 'y', 'z'
            smooth_xyz: Also smooth XYZ coordinates

        Returns:
            Stabilized trajectory
        """
        if not trajectory:
            return trajectory

        stabilized = []

        if self.method == "kalman":
            kf = self.filter
            kf.state = None  # Reset

            for t in trajectory:
                az, el, dist = kf.update(t['az'], t['el'], t['dist_m'])

                # Recompute XYZ from filtered angles/distance
                x = dist * np.sin(az) * np.cos(el)
                y = dist * np.sin(el)
                z = dist * np.cos(az) * np.cos(el)

                stabilized.append({
                    'frame': t['frame'],
                    'az': az,
                    'el': el,
                    'dist_m': dist,
                    'x': x,
                    'y': y,
                    'z': z,
                })

        elif self.method == "one_euro":
            oef = self.filter
            oef.x_prev = None  # Reset

            for i, t in enumerate(trajectory):
                x_in = np.array([t['az'], t['el'], t['dist_m']])
                x_out = oef.filter(x_in, float(i))

                az, el, dist = x_out
                x = dist * np.sin(az) * np.cos(el)
                y = dist * np.sin(el)
                z = dist * np.cos(az) * np.cos(el)

                stabilized.append({
                    'frame': t['frame'],
                    'az': az,
                    'el': el,
                    'dist_m': dist,
                    'x': x,
                    'y': y,
                    'z': z,
                })

        else:
            # EMA fallback
            prev = trajectory[0]
            stabilized.append(prev.copy())

            alpha = self.ema_alpha
            for t in trajectory[1:]:
                smoothed = {
                    'frame': t['frame'],
                    'az': prev['az'] * (1 - alpha) + t['az'] * alpha,
                    'el': prev['el'] * (1 - alpha) + t['el'] * alpha,
                    'dist_m': prev['dist_m'] * (1 - alpha) + t['dist_m'] * alpha,
                    'x': prev['x'] * (1 - alpha) + t['x'] * alpha,
                    'y': prev['y'] * (1 - alpha) + t['y'] * alpha,
                    'z': prev['z'] * (1 - alpha) + t['z'] * alpha,
                }
                stabilized.append(smoothed)
                prev = smoothed

        return stabilized


def stabilize_trajectory_3d(
    trajectory_3d: Dict,
    method: str = "kalman",
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
) -> Dict:
    """
    Convenience function to stabilize a trajectory_3d dict.

    Args:
        trajectory_3d: Output from HybridTrackingResult.get_trajectory_3d()
        method: "kalman", "one_euro", or "ema"
        process_noise: Kalman process noise (higher = more responsive)
        measurement_noise: Kalman measurement noise (higher = smoother)

    Returns:
        Stabilized trajectory_3d dict
    """
    stabilizer = TrajectoryStabilizer(
        method=method,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )

    # Convert frames to expected format
    frames = trajectory_3d.get('frames', [])

    stabilized_frames = stabilizer.stabilize_trajectory(frames)

    return {
        'intrinsics': trajectory_3d.get('intrinsics', {}),
        'frames': stabilized_frames,
        'stabilization': {
            'method': method,
            'process_noise': process_noise,
            'measurement_noise': measurement_noise,
        }
    }


class RTSSmoother:
    """
    Rauch-Tung-Striebel (RTS) Smoother for optimal trajectory smoothing.

    Two-pass algorithm:
    1. Forward pass: Standard Kalman filter
    2. Backward pass: Smooth using future information

    This produces optimal smoothing when the entire trajectory is available
    (offline processing), reducing jitter while preserving true motion.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        dt: float = 1.0,
    ):
        self.dt = dt
        self.n_state = 6
        self.n_meas = 3

        # State transition matrix
        self.F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1],
        ])

        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        # Process noise
        q = process_noise
        self.Q = np.array([
            [q*dt**2, 0, 0, q*dt, 0, 0],
            [0, q*dt**2, 0, 0, q*dt, 0],
            [0, 0, q*dt**2, 0, 0, q*dt],
            [q*dt, 0, 0, q, 0, 0],
            [0, q*dt, 0, 0, q, 0],
            [0, 0, q*dt, 0, 0, q],
        ]) * 0.1

        # Measurement noise
        r = measurement_noise
        self.R = np.diag([r, r, r * 10])

    def smooth(self, measurements: np.ndarray) -> np.ndarray:
        """
        Apply RTS smoothing to measurements.

        Args:
            measurements: (N, 3) array of [az, el, dist] measurements

        Returns:
            (N, 3) array of smoothed [az, el, dist]
        """
        N = len(measurements)
        if N < 2:
            return measurements

        # Storage for forward pass
        x_fwd = np.zeros((N, self.n_state))
        P_fwd = np.zeros((N, self.n_state, self.n_state))
        x_pred = np.zeros((N, self.n_state))
        P_pred = np.zeros((N, self.n_state, self.n_state))

        # Initialize
        x_fwd[0, :3] = measurements[0]
        x_fwd[0, 3:] = 0  # Zero initial velocity
        P_fwd[0] = np.eye(self.n_state) * 1.0

        # Forward pass (Kalman filter)
        for k in range(1, N):
            # Predict
            x_pred[k] = self.F @ x_fwd[k-1]
            P_pred[k] = self.F @ P_fwd[k-1] @ self.F.T + self.Q

            # Update
            z = measurements[k]
            y = z - self.H @ x_pred[k]
            S = self.H @ P_pred[k] @ self.H.T + self.R
            K = P_pred[k] @ self.H.T @ np.linalg.inv(S)

            x_fwd[k] = x_pred[k] + K @ y
            P_fwd[k] = (np.eye(self.n_state) - K @ self.H) @ P_pred[k]

        # Backward pass (RTS smoother)
        x_smooth = np.zeros((N, self.n_state))
        P_smooth = np.zeros((N, self.n_state, self.n_state))

        x_smooth[N-1] = x_fwd[N-1]
        P_smooth[N-1] = P_fwd[N-1]

        for k in range(N-2, -1, -1):
            # RTS gain
            C = P_fwd[k] @ self.F.T @ np.linalg.inv(P_pred[k+1])

            # Smooth
            x_smooth[k] = x_fwd[k] + C @ (x_smooth[k+1] - x_pred[k+1])
            P_smooth[k] = P_fwd[k] + C @ (P_smooth[k+1] - P_pred[k+1]) @ C.T

        # Return only position (az, el, dist)
        return x_smooth[:, :3]


def rts_smooth_trajectory(
    trajectory: List[Dict],
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
) -> List[Dict]:
    """
    Apply RTS smoothing to a trajectory.

    This is the recommended smoothing method for offline processing
    as it uses both past and future information.

    Args:
        trajectory: List of dicts with 'frame', 'az', 'el', 'dist_m'
        process_noise: Higher = more responsive, lower = smoother
        measurement_noise: Higher = smoother output

    Returns:
        Smoothed trajectory
    """
    if len(trajectory) < 2:
        return trajectory

    # Extract measurements
    measurements = np.array([
        [t['az'], t['el'], t['dist_m']] for t in trajectory
    ])

    # Apply RTS smoother
    smoother = RTSSmoother(
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )
    smoothed = smoother.smooth(measurements)

    # Rebuild trajectory with explicit field naming:
    # - dist_m_raw: original metric depth (preserved)
    # - dist_m: smoothed metric depth (for trajectory analysis)
    # - depth_render: value for FOA rendering (= depth_blended or dist_m_raw)
    result = []
    # Fields that are recomputed (not preserved from original)
    computed_fields = {'frame', 'az', 'el', 'dist_m', 'x', 'y', 'z'}

    for i, t in enumerate(trajectory):
        az, el, dist = smoothed[i]

        # Recompute XYZ from smoothed values
        x = dist * np.sin(az) * np.cos(el)
        y = dist * np.sin(el)
        z = dist * np.cos(az) * np.cos(el)

        # Get raw metric depth (before smoothing)
        dist_m_raw = t.get('dist_m', dist)

        smoothed_frame = {
            'frame': t['frame'],
            'az': float(az),
            'el': float(el),
            'dist_m': float(dist),  # smoothed
            'dist_m_raw': float(dist_m_raw),  # original metric
            'x': float(x),
            'y': float(y),
            'z': float(z),
        }

        # Preserve additional fields (depth_blended, d_rel, confidence, w, h, etc.)
        for key, value in t.items():
            if key not in computed_fields:
                smoothed_frame[key] = value

        # depth_render: the value FOA renderer should use
        # Use RTS-smoothed dist_m to avoid piecewise-linear artifacts
        # from depth_stride interpolation (depth_blended is unsmoothed)
        smoothed_frame['depth_render'] = float(dist)

        result.append(smoothed_frame)

    return result


__all__ = [
    'TrajectoryKalmanFilter',
    'OneEuroFilter',
    'TrajectoryStabilizer',
    'stabilize_trajectory_3d',
    'RTSSmoother',
    'rts_smooth_trajectory',
]

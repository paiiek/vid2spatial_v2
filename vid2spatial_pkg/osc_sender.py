"""
OSC Spatial Parameter Sender.

Streams spatial trajectory parameters to DAW via OSC for:
- Real-time authoring/preview
- Automation control
- Live performance

Parameters sent:
- azimuth (degrees, -180 to 180)
- elevation (degrees, -90 to 90)
- distance (normalized 0-1, optionally meters)
- velocity (degrees/second)
- timecode (seconds)
"""

import time
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class OSCConfig:
    """OSC sender configuration."""
    host: str = "127.0.0.1"
    port: int = 9000
    address_prefix: str = "/vid2spatial"
    distance_mode: str = "normalized"  # "normalized" (0-1) or "meters"
    distance_max_m: float = 10.0  # Max distance for normalization
    send_velocity: bool = True
    send_timecode: bool = True


class OSCSpatialSender:
    """
    Send spatial parameters via OSC.

    Usage:
        sender = OSCSpatialSender(host="127.0.0.1", port=9000)
        sender.connect()

        # Single frame
        sender.send_frame(az_deg=45.0, el_deg=10.0, dist_m=2.5)

        # Stream trajectory
        sender.stream_trajectory(trajectory, fps=30)

        sender.disconnect()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        address_prefix: str = "/vid2spatial",
        distance_mode: str = "normalized",
        distance_max_m: float = 10.0,
    ):
        """
        Initialize OSC sender.

        Args:
            host: OSC server host
            port: OSC server port
            address_prefix: OSC address prefix
            distance_mode: "normalized" (0-1, 1=near) or "meters"
            distance_max_m: Max distance for normalization
        """
        self.config = OSCConfig(
            host=host,
            port=port,
            address_prefix=address_prefix,
            distance_mode=distance_mode,
            distance_max_m=distance_max_m,
        )
        self.client = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to OSC server."""
        try:
            from pythonosc import udp_client
            self.client = udp_client.SimpleUDPClient(
                self.config.host,
                self.config.port
            )
            self._connected = True
            print(f"[OSC] Connected to {self.config.host}:{self.config.port}")
            return True
        except ImportError:
            print("[OSC] python-osc not installed. Run: pip install python-osc")
            return False
        except Exception as e:
            print(f"[OSC] Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from OSC server."""
        self.client = None
        self._connected = False
        print("[OSC] Disconnected")

    def _normalize_distance(self, dist_m: float) -> float:
        """Convert meters to normalized 0-1 (1=near, 0=far)."""
        # Inverse relationship: closer = higher value
        normalized = 1.0 - min(dist_m / self.config.distance_max_m, 1.0)
        return max(0.0, min(1.0, normalized))

    def send_frame(
        self,
        az_deg: float,
        el_deg: float,
        dist_m: float,
        velocity_deg_s: float = 0.0,
        timecode_s: float = 0.0,
        frame_idx: int = 0,
    ):
        """
        Send single frame spatial parameters.

        Args:
            az_deg: Azimuth in degrees (-180 to 180)
            el_deg: Elevation in degrees (-90 to 90)
            dist_m: Distance in meters
            velocity_deg_s: Angular velocity in degrees/second
            timecode_s: Timecode in seconds
            frame_idx: Frame index
        """
        if not self._connected or self.client is None:
            return

        prefix = self.config.address_prefix

        # Core spatial parameters
        self.client.send_message(f"{prefix}/azimuth", float(az_deg))
        self.client.send_message(f"{prefix}/elevation", float(el_deg))

        # Distance
        if self.config.distance_mode == "normalized":
            dist_norm = self._normalize_distance(dist_m)
            self.client.send_message(f"{prefix}/distance", float(dist_norm))
        else:
            self.client.send_message(f"{prefix}/distance_m", float(dist_m))

        # Velocity
        if self.config.send_velocity:
            self.client.send_message(f"{prefix}/velocity", float(velocity_deg_s))

        # Timecode
        if self.config.send_timecode:
            self.client.send_message(f"{prefix}/timecode", float(timecode_s))
            self.client.send_message(f"{prefix}/frame", int(frame_idx))

        # Also send bundled message for atomic updates
        self.client.send_message(
            f"{prefix}/spatial",
            [float(az_deg), float(el_deg), float(dist_m), float(velocity_deg_s), float(timecode_s)]
        )

    def send_xyz(self, x: float, y: float, z: float, timecode_s: float = 0.0):
        """
        Send XYZ cartesian coordinates (Atmos-style).

        Args:
            x, y, z: Cartesian coordinates in meters
            timecode_s: Timecode in seconds
        """
        if not self._connected or self.client is None:
            return

        prefix = self.config.address_prefix
        self.client.send_message(f"{prefix}/xyz", [float(x), float(y), float(z)])
        self.client.send_message(f"{prefix}/timecode", float(timecode_s))

    def stream_trajectory(
        self,
        trajectory: List[Dict],
        fps: float = 30.0,
        realtime: bool = True,
        loop: bool = False,
        on_frame: Optional[Callable[[int, Dict], None]] = None,
    ):
        """
        Stream entire trajectory with timing.

        Args:
            trajectory: List of frame dicts with 'az', 'el', 'dist_m', 'frame'
            fps: Frames per second
            realtime: If True, sleep between frames for real-time playback
            loop: If True, loop playback
            on_frame: Optional callback(frame_idx, frame_data)
        """
        if not self._connected:
            if not self.connect():
                return

        print(f"[OSC] Streaming {len(trajectory)} frames at {fps} FPS")

        # Pre-compute velocities
        velocities = self._compute_velocities(trajectory, fps)

        frame_duration = 1.0 / fps

        try:
            while True:
                start_time = time.time()

                for i, frame in enumerate(trajectory):
                    frame_start = time.time()

                    az_deg = np.degrees(frame.get('az', 0))
                    el_deg = np.degrees(frame.get('el', 0))
                    dist_m = frame.get('dist_m', 2.0)
                    timecode_s = frame.get('frame', i) / fps
                    velocity = velocities[i] if i < len(velocities) else 0.0

                    self.send_frame(
                        az_deg=az_deg,
                        el_deg=el_deg,
                        dist_m=dist_m,
                        velocity_deg_s=velocity,
                        timecode_s=timecode_s,
                        frame_idx=frame.get('frame', i),
                    )

                    # Also send XYZ if available
                    if 'x' in frame and 'y' in frame and 'z' in frame:
                        self.send_xyz(frame['x'], frame['y'], frame['z'], timecode_s)

                    if on_frame:
                        on_frame(i, frame)

                    if realtime:
                        elapsed = time.time() - frame_start
                        sleep_time = frame_duration - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                if not loop:
                    break

                print(f"[OSC] Looping... (total time: {time.time() - start_time:.2f}s)")

        except KeyboardInterrupt:
            print("\n[OSC] Streaming stopped")

    def _compute_velocities(self, trajectory: List[Dict], fps: float) -> List[float]:
        """Compute angular velocity for each frame."""
        velocities = [0.0]

        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            curr = trajectory[i]

            # Angular difference
            d_az = np.degrees(curr.get('az', 0) - prev.get('az', 0))
            d_el = np.degrees(curr.get('el', 0) - prev.get('el', 0))

            # Total angular velocity (deg/s)
            angular_speed = np.sqrt(d_az**2 + d_el**2) * fps
            velocities.append(angular_speed)

        return velocities


def create_osc_sender(
    host: str = "127.0.0.1",
    port: int = 9000,
    **kwargs
) -> OSCSpatialSender:
    """Factory function for OSC sender."""
    return OSCSpatialSender(host=host, port=port, **kwargs)


# CLI interface
def main():
    """Command-line interface for OSC streaming."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Stream spatial trajectory via OSC")
    parser.add_argument("trajectory_json", help="Path to trajectory JSON file")
    parser.add_argument("--host", default="127.0.0.1", help="OSC host")
    parser.add_argument("--port", type=int, default=9000, help="OSC port")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--distance-mode", choices=["normalized", "meters"],
                        default="normalized", help="Distance format")

    args = parser.parse_args()

    # Load trajectory
    with open(args.trajectory_json) as f:
        data = json.load(f)

    trajectory = data.get("frames", data)
    if isinstance(trajectory, dict):
        trajectory = trajectory.get("frames", [])

    print(f"Loaded {len(trajectory)} frames")

    # Create sender and stream
    sender = OSCSpatialSender(
        host=args.host,
        port=args.port,
        distance_mode=args.distance_mode,
    )

    if sender.connect():
        sender.stream_trajectory(
            trajectory,
            fps=args.fps,
            realtime=True,
            loop=args.loop,
        )
        sender.disconnect()


__all__ = [
    'OSCSpatialSender',
    'OSCConfig',
    'create_osc_sender',
]


if __name__ == "__main__":
    main()

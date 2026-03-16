"""
Visual Room Acoustics Estimator

Estimates room geometry and RT60 from depth maps.
Uses depth statistics + Sabine equation to generate scene-adaptive IRs.

Approach:
1. Depth map → room volume estimate (via median depth × FOV geometry)
2. Depth variance → surface absorption estimate (smooth=reflective, rough=absorptive)
3. Volume + absorption → RT60 via Sabine equation
4. RT60 → pyroomacoustics ShoeBox IR
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RoomEstimate:
    """Estimated room parameters."""
    volume_m3: float         # Room volume in cubic meters
    rt60: float              # Estimated RT60 in seconds
    room_dims: Tuple[float, float, float]  # (Lx, Ly, Lz)
    absorption: float        # Mean absorption coefficient
    scene_class: str         # "small", "medium", "large", "outdoor"
    median_depth: float      # Median depth in meters
    depth_std: float         # Depth standard deviation


class VisualRoomEstimator:
    """Estimate room acoustics from depth map."""

    # Scene classification thresholds
    VOLUME_SMALL = 30.0    # < 30 m³ → small room
    VOLUME_MEDIUM = 100.0  # 30-100 m³ → medium room
    VOLUME_LARGE = 500.0   # 100-500 m³ → large room
    # > 500 m³ → outdoor

    # Absorption presets by scene class
    ABSORPTION_PRESETS = {
        "small": 0.35,    # Living room, studio
        "medium": 0.25,   # Classroom, office
        "large": 0.15,    # Hall, auditorium
        "outdoor": 0.60,  # High absorption (open air equivalent)
    }

    # RT60 clamp range
    RT60_MIN = 0.2
    RT60_MAX = 3.0

    def __init__(self, fov_deg: float = 60.0):
        """
        Args:
            fov_deg: Camera horizontal field of view in degrees
        """
        self.fov_deg = fov_deg

    def estimate_from_depth(
        self,
        depth_map: np.ndarray,
        fov_deg: Optional[float] = None,
    ) -> RoomEstimate:
        """
        Estimate room parameters from a metric depth map.

        Args:
            depth_map: (H, W) depth in meters
            fov_deg: Override FOV if provided

        Returns:
            RoomEstimate with volume, RT60, room dims, etc.
        """
        fov = fov_deg or self.fov_deg
        H, W = depth_map.shape

        # Depth statistics
        valid = depth_map[depth_map > 0.1]
        if valid.size == 0:
            valid = np.array([2.0])

        median_d = float(np.median(valid))
        mean_d = float(np.mean(valid))
        std_d = float(np.std(valid))
        p10 = float(np.percentile(valid, 10))
        p90 = float(np.percentile(valid, 90))

        # Estimate room dimensions from depth + FOV
        fov_rad = np.radians(fov)
        # Width at median depth
        width_est = 2.0 * median_d * np.tan(fov_rad / 2.0)
        # Height estimate (assume similar aspect as image)
        aspect = H / W
        height_est = width_est * aspect
        # Depth of room ≈ range between near and far surfaces
        depth_range = max(p90 - p10, 1.0)
        room_depth = max(depth_range, median_d * 0.5)

        # Room dimensions (Lx=width, Ly=depth, Lz=height)
        Lx = max(width_est, 2.0)
        Ly = max(room_depth, 2.0)
        Lz = max(height_est * 0.6, 2.4)  # scale down, assume ceiling ~2.4m+

        volume = Lx * Ly * Lz

        # Scene classification
        if volume < self.VOLUME_SMALL:
            scene_class = "small"
        elif volume < self.VOLUME_MEDIUM:
            scene_class = "medium"
        elif volume < self.VOLUME_LARGE:
            scene_class = "large"
        else:
            scene_class = "outdoor"

        # Absorption estimation
        # High depth variance → varied surfaces → more absorption
        # Low depth variance → flat surfaces → less absorption
        base_absorption = self.ABSORPTION_PRESETS[scene_class]

        # Adjust by depth variance (normalized by median)
        cv = std_d / (median_d + 1e-6)  # coefficient of variation
        if cv > 0.5:
            # High variance: rough/varied → more absorptive
            absorption = min(base_absorption * 1.3, 0.8)
        elif cv < 0.1:
            # Low variance: smooth/flat → more reflective
            absorption = max(base_absorption * 0.7, 0.05)
        else:
            absorption = base_absorption

        # RT60 via Sabine equation: RT60 = 0.161 * V / A
        # A = total absorption = sum(alpha_i * S_i) ≈ alpha_mean * S_total
        surface_area = 2.0 * (Lx * Ly + Ly * Lz + Lx * Lz)
        A = absorption * surface_area

        if A > 0:
            rt60 = 0.161 * volume / A
        else:
            rt60 = 0.5

        rt60 = float(np.clip(rt60, self.RT60_MIN, self.RT60_MAX))

        return RoomEstimate(
            volume_m3=volume,
            rt60=rt60,
            room_dims=(Lx, Ly, Lz),
            absorption=absorption,
            scene_class=scene_class,
            median_depth=median_d,
            depth_std=std_d,
        )

    def generate_ir(
        self,
        room_estimate: RoomEstimate,
        fs: int = 48000,
        src_offset: Tuple[float, float, float] = (-1.0, 0.0, 0.0),
    ) -> np.ndarray:
        """
        Generate room IR from estimated parameters using pyroomacoustics.

        Args:
            room_estimate: Room estimation result
            fs: Sample rate
            src_offset: Source offset from mic position (x, y, z)

        Returns:
            Mono IR as float32 array
        """
        Lx, Ly, Lz = room_estimate.room_dims

        # Mic at room center
        mx, my, mz = Lx / 2.0, Ly / 2.0, min(1.5, Lz / 2.0)
        # Source offset from mic
        sx = mx + src_offset[0]
        sy = my + src_offset[1]
        sz = mz + src_offset[2]

        # Clamp positions inside room
        sx = float(np.clip(sx, 0.3, Lx - 0.3))
        sy = float(np.clip(sy, 0.3, Ly - 0.3))
        sz = float(np.clip(sz, 0.3, Lz - 0.3))
        mx = float(np.clip(mx, 0.3, Lx - 0.3))
        my = float(np.clip(my, 0.3, Ly - 0.3))
        mz = float(np.clip(mz, 0.3, Lz - 0.3))

        try:
            import pyroomacoustics as pra
            e_absorption, max_order = pra.inverse_sabine(room_estimate.rt60, (Lx, Ly, Lz))
            max_order = min(max_order, 15)

            room = pra.ShoeBox(
                [Lx, Ly, Lz],
                fs=fs,
                materials=pra.Material(e_absorption),
                max_order=max_order,
            )
            room.add_source([sx, sy, sz])
            room.add_microphone([mx, my, mz])
            room.compute_rir()

            rir = room.rir[0][0]
            return np.asarray(rir, dtype=np.float32)

        except Exception as e:
            print(f"[visual_room] PRA failed: {e}, using FAIR-Play matched IR")
            from .irgen import fairplay_matched_ir
            return fairplay_matched_ir(fs=fs, rt60=room_estimate.rt60)

    def estimate_and_generate(
        self,
        depth_map: np.ndarray,
        fs: int = 48000,
        fov_deg: Optional[float] = None,
    ) -> Tuple[np.ndarray, RoomEstimate]:
        """
        One-call convenience: estimate room and generate IR.

        Returns:
            (ir, room_estimate)
        """
        estimate = self.estimate_from_depth(depth_map, fov_deg=fov_deg)
        ir = self.generate_ir(estimate, fs=fs)
        print(f"[visual_room] Scene={estimate.scene_class}, "
              f"Vol={estimate.volume_m3:.0f}m³, RT60={estimate.rt60:.2f}s, "
              f"Abs={estimate.absorption:.2f}")
        return ir, estimate


__all__ = ["VisualRoomEstimator", "RoomEstimate"]

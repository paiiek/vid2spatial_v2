"""
OSTrack wrapper for improved object tracking.

OSTrack provides state-of-the-art tracking performance,
significantly better than KCF and comparable to YOLO+ByteTrack.
"""
import sys
import os
from pathlib import Path
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple


class OSTrackWrapper:
    """Wrapper for OSTrack tracker."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize OSTrack.

        Args:
            checkpoint_path: Path to OSTrack checkpoint
        """
        self.tracker = None
        self.checkpoint_path = checkpoint_path

        # Add OSTrack to path
        ostrack_path = "/home/seung/OSTrack"
        if os.path.exists(ostrack_path):
            sys.path.insert(0, ostrack_path)
            sys.path.insert(0, os.path.join(ostrack_path, 'lib'))

        self._load_tracker()

    def _load_tracker(self):
        """Load OSTrack model."""
        try:
            import torch
            from lib.test.evaluation import Tracker
            from lib.test.parameter.ostrack import parameters

            # Get parameters
            params = parameters("vitb_384_mae_ce_32x4_ep300")

            # Override checkpoint if provided
            if self.checkpoint_path:
                params.checkpoint = self.checkpoint_path
            else:
                # Use default checkpoint
                checkpoint_dir = "/home/seung/OSTrack/checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                default_ckpt = os.path.join(checkpoint_dir, "vitb_384_mae_ce_32x4_ep300.pth.tar")

                if not os.path.exists(default_ckpt):
                    print(f"[info] Downloading OSTrack checkpoint...")
                    self._download_checkpoint(default_ckpt)

                params.checkpoint = default_ckpt

            # Create tracker
            self.tracker = Tracker("ostrack", "vitb_384_mae_ce_32x4_ep300", "video")
            self.tracker.params = params

            print("[info] Loaded OSTrack (ViT-B 384)")

        except Exception as e:
            print(f"[error] Failed to load OSTrack: {e}")
            self.tracker = None

    def _download_checkpoint(self, output_path: str):
        """Download OSTrack checkpoint."""
        import urllib.request

        url = "https://github.com/botaoye/OSTrack/releases/download/v1.0.0/vitb_384_mae_ce_32x4_ep300.pth.tar"

        print(f"[info] Downloading from {url}")
        urllib.request.urlretrieve(url, output_path)
        print(f"[info] Saved to {output_path}")

    def track_video(
        self,
        video_path: str,
        init_bbox: Tuple[int, int, int, int],
        sample_stride: int = 1
    ) -> List[Dict]:
        """
        Track object in video.

        Args:
            video_path: Path to video file
            init_bbox: Initial bounding box (x, y, w, h)
            sample_stride: Frame sampling stride

        Returns:
            List of tracking results: [{"frame": int, "x": int, "y": int, "w": int, "h": int}, ...]
        """
        if self.tracker is None:
            raise RuntimeError("OSTrack not loaded")

        import torch

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        trajectory = []
        fidx = 0
        initialized = False

        x, y, w, h = init_bbox

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if (fidx % sample_stride) != 0:
                fidx += 1
                continue

            if not initialized:
                # Initialize tracker
                self.tracker.initialize(frame, {'init_bbox': [x, y, w, h]})
                trajectory.append({"frame": fidx, "x": x, "y": y, "w": w, "h": h})
                initialized = True
            else:
                # Track
                outputs = self.tracker.track(frame)
                pred_bbox = outputs['target_bbox']

                x, y, w, h = [int(v) for v in pred_bbox]
                trajectory.append({"frame": fidx, "x": x, "y": y, "w": w, "h": h})

            fidx += 1

        cap.release()
        return trajectory


def track_with_ostrack(
    video_path: str,
    init_bbox: Tuple[int, int, int, int],
    sample_stride: int = 1,
    checkpoint_path: Optional[str] = None
) -> List[Dict]:
    """
    Track object using OSTrack.

    Args:
        video_path: Path to video file
        init_bbox: Initial bounding box (x, y, w, h)
        sample_stride: Frame sampling stride
        checkpoint_path: Optional custom checkpoint path

    Returns:
        List of tracking results
    """
    tracker = OSTrackWrapper(checkpoint_path=checkpoint_path)
    return tracker.track_video(video_path, init_bbox, sample_stride)


__all__ = [
    "OSTrackWrapper",
    "track_with_ostrack",
]

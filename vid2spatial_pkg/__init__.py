"""
vid2spatial: Object-driven Dynamic Spatial Audio from Video and Mono Sound

Modules
- vision: object tracking + depth → 3D trajectory
- irgen: room IR synthesis + convolution
- foa_render: mono→FOA time-varying encoder
- train: neural mapper (skeleton)
- evaluate: objective metrics

Conventions
- Angles: azimuth/elevation in radians unless specified
- FOA format: ACN channel order [W, Y, Z, X]. Default normalisation is
  "legacy" = ACN/N3D scaled by 1/sqrt(2), NOT the AmbiX SN3D this line
  used to claim (measured X/W = sqrt(3) on axis). Pass foa_norm="sn3d"
  for true AmbiX. See docs/ISSUES.md I12.
"""

__all__ = [
    "vision",
    "irgen",
    "foa_render",
    "train",
    "evaluate",
]


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
- FOA format: AmbiX (ACN/SN3D) channel order [W, Y, Z, X]
"""

__all__ = [
    "vision",
    "irgen",
    "foa_render",
    "train",
    "evaluate",
]


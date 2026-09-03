"""Deprecated shim: this backend moved to `vid2spatial_pkg.experimental`.

It is reachable but unbenchmarked -- see `experimental/__init__.py` for why it
cannot go through test/run_quant_eval.py. Import from
`vid2spatial_pkg.experimental.point_tracker` instead.
"""
import warnings as _warnings

from .experimental.point_tracker import *  # noqa: F401,F403
from .experimental import point_tracker as _impl

_warnings.warn(
    "vid2spatial_pkg.point_tracker moved to vid2spatial_pkg.experimental.point_tracker "
    "(unbenchmarked experimental backend); update the import.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = getattr(_impl, "__all__", [n for n in dir(_impl) if not n.startswith("_")])

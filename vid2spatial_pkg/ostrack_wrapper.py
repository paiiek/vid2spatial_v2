"""Deprecated shim: this backend moved to `vid2spatial_pkg.experimental`.

It is reachable but unbenchmarked -- see `experimental/__init__.py` for why it
cannot go through test/run_quant_eval.py. Import from
`vid2spatial_pkg.experimental.ostrack_wrapper` instead.
"""
import warnings as _warnings

from .experimental.ostrack_wrapper import *  # noqa: F401,F403
from .experimental import ostrack_wrapper as _impl

_warnings.warn(
    "vid2spatial_pkg.ostrack_wrapper moved to vid2spatial_pkg.experimental.ostrack_wrapper "
    "(unbenchmarked experimental backend); update the import.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = getattr(_impl, "__all__", [n for n in dir(_impl) if not n.startswith("_")])

"""Experimental tracker backends: reachable, but never benchmarked.

These four backends are each imported from exactly one place in ``vision.py``
and appear in no evaluation table in this repo. They are not part of the
measured pipeline (SAM2/YOLO-World -> depth -> pinhole -> FOA), and none of
them can be put through ``test/run_quant_eval.py``, which initialises a
tracker from a bounding box on arbitrary LaSOT categories:

* ``ostrack_wrapper`` needs the OSTrack research tree (``lib.test.evaluation``)
  and a checkpoint. Neither is on this machine, so it always falls back to
  template matching.
* ``skeleton_tracker`` needs MediaPipe pose landmarks and only tracks human
  joints. MediaPipe does not currently import in this environment.
* ``color_tracker`` is initialised from an HSV target colour, not a bbox, and
  is meant for fluorescent markers.
* ``point_tracker`` is initialised from a bright point or a template, not a
  bbox, and is meant for laser pointers.

So a QUANT_EVAL number for any of them would be a number about the fallback
path or about an inapplicable protocol, not about the tracker. They are kept
here, importable and unchanged, so a reader can tell at a glance what is
load-bearing and what is not.

``trajectory_stabilizer`` is deliberately NOT here: it has a unit test
(``test_rts_smoother_than_forward``) and a measured ablation
(``STABILIZATION_PROXY_ABLATION.json``: azimuth jitter 691 -> 17).

Old import paths (``vid2spatial_pkg.color_tracker`` etc.) still work through
shims for one release; they emit a ``DeprecationWarning``.
"""

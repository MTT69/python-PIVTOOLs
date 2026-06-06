"""calibration — unified pinhole calibration package.

A clean-slate, pinhole-only calibration system covering planar + stereo dotboard
and charuco boards. Built in parallel with the legacy ``pivtools_gui.calibration``
code; nothing in the old path is touched until cutover.

Design (see PyPIVTools/docs/calibration-v2/PRD.md and
~/.claude/plans/indexed-launching-lemur.md):

- One coordinate contract (``frames``): detection emits image-down pixels
  (0-based, y-down); world is mm in the user-defined frame; no implicit Y-flips.
- One camera-model family (``camera_model``): pinhole, DaVis-matching fit
  (k1,k2,p1,p2 with k3 fixed, fx==fy, bundled two-stage intrinsics).
- A user-defined world frame (``world_frame``): origin/+X/+Y clicked on camera 1,
  snapped to real dots, resolved to orthogonal board-grid axes.
- One model record (``record``): a single ``.mat`` written into the calibration
  source folder.
- One orchestration pipeline (``pipeline``) and one apply step (``apply``).
"""

from .camera_model import (  # noqa: F401
    CameraModel,
    DistortionModel,
    fit_intrinsics,
    fit_pose,
    back_project_to_plane,
    projection_jacobian,
    reprojection_rms,
)
from . import frames  # noqa: F401

__all__ = [
    "CameraModel",
    "DistortionModel",
    "fit_intrinsics",
    "fit_pose",
    "back_project_to_plane",
    "projection_jacobian",
    "reprojection_rms",
    "frames",
]

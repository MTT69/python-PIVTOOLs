"""calibration2.frames — THE coordinate contract.

Three named frames, no implicit fourth:

- **pixel (image-down):** 0-based, origin top-left, +x right, +y *down*. This is
  what OpenCV uses and what the PIV pipeline saves internally. All compute and
  every OpenCV call speaks this frame.
- **matlab-save:** pixel + 1 (1-based). Used only at the ``.mat`` read/write
  boundary (``coordinates.mat`` stores ``x_centre + 1``, ``y_centre + 1``).
- **world:** millimetres, right-handed, with origin and axes equal to whatever the
  user defined on camera 1 (see ``world_frame``). World conversions are carried by
  ``CameraModel.project`` / ``CameraModel.back_project_to_plane`` — NOT by free
  functions here, so they cannot be applied in the wrong place.

The single rule: **there are no implicit Y-flips anywhere.** World-axis directions
come only from the user-defined frame (or the documented default). Detection always
emits image-down pixels; object points are built directly in the user world frame;
velocities follow from the pixel->world Jacobian. Every ``image_height - y`` and
``x - 1`` lives here and nowhere else.
"""

from __future__ import annotations

import numpy as np

# Bumped whenever the on-disk model/coordinate convention changes. Stamped into
# every saved record so a stale model can never be silently misread.
CONTRACT_VERSION = 1


def matlab_to_pixel(xy: np.ndarray) -> np.ndarray:
    """1-based MATLAB-save coords -> 0-based image-down pixels.

    Parameters
    ----------
    xy : array_like, shape (..., 2) or (...,)
        Coordinates as stored in ``coordinates.mat`` (1-based).

    Returns
    -------
    np.ndarray (float64)
        Image-down pixel coordinates (0-based).
    """
    return np.asarray(xy, dtype=np.float64) - 1.0


def pixel_to_matlab(xy: np.ndarray) -> np.ndarray:
    """0-based image-down pixels -> 1-based MATLAB-save coords."""
    return np.asarray(xy, dtype=np.float64) + 1.0

"""Result dataclasses for Stereo Ensemble PIV (Correlation-of-Correlations)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PIVStereoEnsemblePassResult:
    """Result from a single pass of stereo ensemble CoC PIV.

    All velocity fields are in physical units (m/s) after conversion from
    dewarped pixel displacements.  All stress fields are in (m/s)^2.
    Per-camera displacements are stored in dewarped pixel units for diagnostics.
    """

    # 3D velocity fields (m/s)
    ux: np.ndarray                       # (n_win_y, n_win_x) streamwise
    uy: np.ndarray                       # (n_win_y, n_win_x) wall-normal
    uz: np.ndarray                       # (n_win_y, n_win_x) out-of-plane

    # All 6 Reynolds stresses (m/s)^2 — decoupled via CoC
    UU_stress: np.ndarray                # R_xx
    VV_stress: np.ndarray                # R_yy
    WW_stress: np.ndarray                # R_zz (decoupled from R_xx)
    UV_stress: np.ndarray                # R_xy
    UW_stress: np.ndarray                # R_xz
    VW_stress: np.ndarray                # R_yz

    # CoC diagnostic: cross-camera covariance (m/s)^2
    Sigma_12_xx: np.ndarray              # (n_win_y, n_win_x)

    # Per-camera displacements (dewarped pixels, before physical conversion)
    d1_x: np.ndarray                     # cam1 displacement x
    d1_y: np.ndarray                     # cam1 displacement y
    d2_x: np.ndarray                     # cam2 displacement x
    d2_y: np.ndarray                     # cam2 displacement y

    # Quality metrics
    peakheight: np.ndarray               # average normalised peak height
    nan_reason: np.ndarray               # 0=ok, -1=masked, 10=vel_outlier, 11=stress_outlier
    b_mask: np.ndarray                   # binary mask (0=valid, non-zero=excluded)

    # Stereo geometry metadata
    stereo_angle: float                  # sin(theta) — half-angle between cameras
    mm_per_pixel: float                  # dewarped pixel scale

    # Window grid
    window_size: Tuple[int, int]         # (h, w)
    win_ctrs_x: np.ndarray              # (n_win_x,) or (n_win_y, n_win_x)
    win_ctrs_y: np.ndarray              # (n_win_y,) or (n_win_y, n_win_x)

    # Predictor field (dewarped pixels). Two views are saved per pass:
    #   pred_x/y         — POST-remap, on this pass's window grid
    #                      (n_win_y, n_win_x). This is the field that
    #                      actually warped images for this pass.
    #   padded_pred_x/y  — PRE-remap, on the PREVIOUS pass's window grid
    #                      plus boundary padding. The input to the
    #                      cv2.remap upsampling step. Useful for
    #                      diagnosing upscaling-induced edge artefacts —
    #                      compare against pred_x/y to see what the
    #                      remap did at the mask / FOV boundary.
    # Both are None for pass 0 (no predictor on the first pass).
    pred_x: Optional[np.ndarray] = None
    pred_y: Optional[np.ndarray] = None
    padded_pred_x: Optional[np.ndarray] = None
    padded_pred_y: Optional[np.ndarray] = None


@dataclass
class PIVStereoEnsembleResult:
    """Container for all passes of a stereo ensemble CoC run."""

    passes: List[PIVStereoEnsemblePassResult] = field(default_factory=list)

    def add_pass(self, pass_result: PIVStereoEnsemblePassResult) -> None:
        self.passes.append(pass_result)

    @property
    def num_passes(self) -> int:
        return len(self.passes)

    def get_pass(self, idx: int) -> PIVStereoEnsemblePassResult:
        return self.passes[idx]

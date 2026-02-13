"""
Stereo Ensemble PIV Result Dataclasses

Extends the standard ensemble PIV result structure with stereo-specific fields:
- 3D velocity (uz) and all 6 Reynolds stress components (WW, UW, VW)
- Per-camera dewarped displacements (d1_x, d1_y, d2_x, d2_y)
- CoC diagnostic (Sigma_12_xx)
- Stereo geometry metadata (stereo_angle, mm_per_pixel)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PIVStereoEnsemblePassResult:
    """
    Result from a single stereo ensemble PIV pass.

    Contains 3D velocity fields, all 6 Reynolds stress components,
    per-camera dewarped displacements, and CoC diagnostics.
    """
    # 3D velocity fields (m/s after calibration, pixels before)
    ux_mat: Optional[np.ndarray] = None   # In-plane x velocity
    uy_mat: Optional[np.ndarray] = None   # In-plane y velocity
    uz_mat: Optional[np.ndarray] = None   # Out-of-plane z velocity

    # All 6 Reynolds stress components
    UU_stress: Optional[np.ndarray] = None   # R_xx (in-plane normal, x)
    VV_stress: Optional[np.ndarray] = None   # R_yy (in-plane normal, y)
    WW_stress: Optional[np.ndarray] = None   # R_zz (out-of-plane normal) - from CoC
    UV_stress: Optional[np.ndarray] = None   # R_xy (in-plane shear)
    UW_stress: Optional[np.ndarray] = None   # R_xz (out-of-plane shear, x)
    VW_stress: Optional[np.ndarray] = None   # R_yz (out-of-plane shear, y)

    # Per-camera dewarped displacements (pixels in dewarped domain)
    d1_x: Optional[np.ndarray] = None   # Camera 1 x displacement
    d1_y: Optional[np.ndarray] = None   # Camera 1 y displacement
    d2_x: Optional[np.ndarray] = None   # Camera 2 x displacement
    d2_y: Optional[np.ndarray] = None   # Camera 2 y displacement

    # Per-camera sigma parameters (for stress extraction)
    # Camera 1 AB cross-correlation widths
    sig_AB_x_cam1: Optional[np.ndarray] = None
    sig_AB_y_cam1: Optional[np.ndarray] = None
    sig_AB_xy_cam1: Optional[np.ndarray] = None
    # Camera 1 A autocorrelation widths
    sig_A_x_cam1: Optional[np.ndarray] = None
    sig_A_y_cam1: Optional[np.ndarray] = None
    sig_A_xy_cam1: Optional[np.ndarray] = None

    # Camera 2 AB cross-correlation widths
    sig_AB_x_cam2: Optional[np.ndarray] = None
    sig_AB_y_cam2: Optional[np.ndarray] = None
    sig_AB_xy_cam2: Optional[np.ndarray] = None
    # Camera 2 A autocorrelation widths
    sig_A_x_cam2: Optional[np.ndarray] = None
    sig_A_y_cam2: Optional[np.ndarray] = None
    sig_A_xy_cam2: Optional[np.ndarray] = None

    # CoC diagnostic
    Sigma_12_xx: Optional[np.ndarray] = None  # Cross-camera covariance (R_xx - sin^2(theta)*R_zz)

    # Normalized peak height (average of cam1 + cam2)
    peakheight: Optional[np.ndarray] = None

    # NaN reason codes
    nan_reason: Optional[np.ndarray] = None

    # Combined mask
    b_mask: Optional[np.ndarray] = None

    # Predictor field (for next pass)
    pred_x: Optional[np.ndarray] = None
    pred_y: Optional[np.ndarray] = None

    # Window info
    window_size: Optional[tuple[int, int]] = None
    win_ctrs_x: Optional[np.ndarray] = None
    win_ctrs_y: Optional[np.ndarray] = None

    # Stereo geometry metadata
    stereo_angle: Optional[float] = None     # Half-angle in radians
    mm_per_pixel: Optional[float] = None     # Dewarped pixel size


@dataclass
class PIVStereoEnsembleResult:
    """
    Complete result from stereo ensemble PIV processing across all passes.
    """
    passes: List[PIVStereoEnsemblePassResult] = field(default_factory=list)

    def add_pass(self, pass_result: PIVStereoEnsemblePassResult):
        self.passes.append(pass_result)

    def summary(self) -> str:
        s = f"PIVStereoEnsembleResult with {len(self.passes)} passes:\n"
        for i, p in enumerate(self.passes):
            ux_shape = None if p.ux_mat is None else p.ux_mat.shape
            uz_shape = None if p.uz_mat is None else p.uz_mat.shape
            s += f"  Pass {i + 1}: ux.shape={ux_shape}, uz.shape={uz_shape}\n"
        return s

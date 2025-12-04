from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PIVPassResult:
    n_windows: Optional[np.ndarray] = None
    ux_mat: Optional[np.ndarray] = None
    uy_mat: Optional[np.ndarray] = None
    nan_mask: Optional[np.ndarray] = None
    peak_mag: Optional[np.ndarray] = None
    peak_choice: Optional[np.ndarray] = None
    predictor_field: Optional[np.ndarray] = None
    primary_peak_mag: Optional[np.ndarray] = None
    b_mask: Optional[np.ndarray] = None
    window_size: Optional[tuple[int, int]] = None
    win_ctrs_x: Optional[np.ndarray] = None
    win_ctrs_y: Optional[np.ndarray] = None


@dataclass
class PIVResult:
    passes: List[PIVPassResult] = field(default_factory=list)

    def add_pass(self, pass_result: PIVPassResult):
        self.passes.append(pass_result)

    def summary(self) -> str:
        s = f"PIVResult with {len(self.passes)} passes:\n"
        for i, p in enumerate(self.passes):
            s += (
                f"  Pass {i + 1}: ux.shape="
                f"{None if p.ux_mat is None else p.ux_mat.shape}, "
            )
            s += (
                f"uy.shape={None if p.uy_mat is None else p.uy_mat.shape}\n"
            )
        return s


# --- Ensemble PIV Result Classes ---

@dataclass
class PIVEnsembleBlockResult:
    """
    Result from ensemble PIV correlation for a single block/pass.

    Contains averaged correlation planes and point spreads across all images.
    """
    correlation_plane_mean: Optional[np.ndarray] = None
    predictor_field: Optional[np.ndarray] = None
    point_spread_a_mean: Optional[np.ndarray] = None
    point_spread_b_mean: Optional[np.ndarray] = None
    mean_A_warp: Optional[np.ndarray] = None
    mean_B_warp: Optional[np.ndarray] = None
    vector_mask: Optional[np.ndarray] = None
    n_blocks: Optional[int] = None
    n_win_x: Optional[int] = None
    n_win_y: Optional[int] = None

    def summary(self) -> str:
        return "PIVEnsembleBlockResult(...)"


@dataclass
class PIVEnsemblePassResult:
    """
    Result from a single ensemble PIV pass after Gaussian fitting.

    Contains velocity fields and uncertainty/stress tensor information
    derived from the Levenberg-Marquardt Gaussian fitting.
    """
    # Core velocity fields
    ux_mat: Optional[np.ndarray] = None
    uy_mat: Optional[np.ndarray] = None

    # Amplitude estimates
    uxa: Optional[np.ndarray] = None
    uya: Optional[np.ndarray] = None

    # Stress tensors from predictor displacement variances
    UU_stress: Optional[np.ndarray] = None  # Variance in X
    VV_stress: Optional[np.ndarray] = None  # Variance in Y
    UV_stress: Optional[np.ndarray] = None  # Covariance XY

    # Normalized peak height: AB / sqrt(A * B)
    peakheight: Optional[np.ndarray] = None

    # NaN reason codes (0=valid, 1-6=various failure modes)
    nan_reason: Optional[np.ndarray] = None

    # Sigma parameters from Gaussian fitting
    # Total variances (sig_AB)
    sig_AB_x: Optional[np.ndarray] = None
    sig_AB_y: Optional[np.ndarray] = None
    sig_AB_xy: Optional[np.ndarray] = None

    # Auto-correlation variances (sig_A)
    sig_A_x: Optional[np.ndarray] = None
    sig_A_y: Optional[np.ndarray] = None
    sig_A_xy: Optional[np.ndarray] = None

    # Gaussian offset terms (background levels)
    c_A: Optional[np.ndarray] = None
    c_B: Optional[np.ndarray] = None
    c_AB: Optional[np.ndarray] = None

    # Mask (for consistency with instantaneous)
    b_mask: Optional[np.ndarray] = None

    # Predictor field used for this pass (separate X and Y components)
    pred_x: Optional[np.ndarray] = None
    pred_y: Optional[np.ndarray] = None

    # Window info
    window_size: Optional[tuple[int, int]] = None
    win_ctrs_x: Optional[np.ndarray] = None
    win_ctrs_y: Optional[np.ndarray] = None


@dataclass
class PIVEnsembleResult:
    """
    Complete result from ensemble PIV processing across all passes.
    """
    passes: List[PIVEnsemblePassResult] = field(default_factory=list)

    def add_pass(self, pass_result: PIVEnsemblePassResult):
        self.passes.append(pass_result)

    def summary(self) -> str:
        s = f"PIVEnsembleResult with {len(self.passes)} passes:\n"
        for i, p in enumerate(self.passes):
            s += (
                f"  Pass {i + 1}: ux.shape="
                f"{None if p.ux_mat is None else p.ux_mat.shape}, "
            )
            s += (
                f"uy.shape={None if p.uy_mat is None else p.uy_mat.shape}, "
            )
            s += (
                f"UU_stress.shape={None if p.UU_stress is None else p.UU_stress.shape}\n"
            )
        return s

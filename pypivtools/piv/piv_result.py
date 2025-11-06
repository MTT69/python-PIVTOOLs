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

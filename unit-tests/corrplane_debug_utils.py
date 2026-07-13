"""
Shared helpers for the nan_reason / correlation-plane-dump tests.

Builds a minimal on-disk instantaneous workspace (Config auto-detects the
image shape from real files) and runs the CPU correlator directly.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

from pivtools_cli.piv.piv_backend.cpu_instantaneous import (
    InstantaneousCorrelatorCPU,
)
from pivtools_core.config import Config


def make_config(
    workspace: Path,
    images: np.ndarray,
    piv_overrides: Optional[dict] = None,
) -> Config:
    """Write a one-pair instantaneous workspace and return its Config.

    ``images`` is (2, H, W); the pair is written to disk so Config can
    auto-detect the image shape. ``piv_overrides`` merges into the
    ``instantaneous_piv`` section.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(workspace / "B00001_A.tif"), images[0].astype(np.float32))
    cv2.imwrite(str(workspace / "B00001_B.tif"), images[1].astype(np.float32))

    piv = {
        "window_size": [[64, 64]],
        "overlap": [50],
        "runs": [1],
        "peak_finder": "gauss6",
        "num_peaks": 1,
    }
    piv.update(piv_overrides or {})

    cfg = {
        "paths": {
            "source_paths": [str(workspace)],
            "base_paths": [str(workspace)],
            "camera_count": 1,
            "camera_numbers": [1],
        },
        "images": {
            "num_images": 1,
            "image_format": ["B%05d_A.tif", "B%05d_B.tif"],
            "image_type": "standard",
            "start_index": 1,
            "frame_stride": 0,
            "pair_stride": 1,
            "pairing_preset": "ab_format",
            "vector_format": ["B%05d.mat"],
        },
        "processing": {"backend": "cpu", "omp_threads": 2},
        "batches": {"batch_size": 1},
        "instantaneous_piv": piv,
        "outlier_detection": {
            "enabled": True,
            "methods": [{"type": "median_2d", "threshold": 3.0, "epsilon": 0.1}],
        },
        "infilling": {
            "mid_pass": {"method": "nearest", "parameters": {}},
            "final_pass": {"method": "biharmonic", "parameters": {}},
        },
        "filters": [],
        "masking": {"enabled": False},
    }
    cfg_path = workspace / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return Config(path=str(cfg_path))


def run_correlator(
    config: Config,
    images: np.ndarray,
    vector_masks: Optional[List[np.ndarray]] = None,
):
    """Run correlate_batch on one pair; returns the single PIVResult."""
    batch = images[None].astype(np.float32)  # (1, 2, H, W)
    correlator = InstantaneousCorrelatorCPU(config)
    results = correlator.correlate_batch(
        batch, config=config, vector_masks=vector_masks
    )
    assert len(results) == 1
    return results[0]

"""Per-frame laser-gain normalisation pre-pass.

Estimates one scalar gain per frame by least-squares regression against the
ensemble mean image over unmasked pixels:

    g_i = sum(I_i * ref) / sum(ref**2)        (gain-only, no offset)

and the pipeline then divides each frame by its gain at the head of the
filter chain. The algorithm is the one validated on Cam4 2026-07-27
(``manual_tools/normalise_frame_gains.py``: 33% RMS jitter measured, closed
loop re-probe kappa = 0), lifted onto the lazy ``load_images`` dask array so
all image formats, loop handling and camera-path resolution are reused and
the two data passes run on the live cluster.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)


def compute_frame_gains(
    images: da.Array,
    pixel_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Two-pass per-frame gain estimate against the ensemble mean images.

    Pass 1 computes the per-channel ensemble mean image (float64); masked
    pixels are then zeroed in the mean, which restricts BOTH the regression
    numerator and denominator to unmasked pixels. Pass 2 computes the
    per-frame products sum(I_i * ref) and forms the gains.

    Args:
        images: Lazy image array of shape (n_pairs, 2, H, W); channel 0 is
            frame A, channel 1 is frame B.
        pixel_mask: Boolean (H, W) mask, True = masked (excluded from the
            regression). None means no mask — the regression then runs over
            the full frame, which is logged loudly.

    Returns:
        (gains, provenance): gains is float64 of shape (n_pairs, 2) with
        column 0 = A, column 1 = B; provenance holds the masked mean images,
        denominators and mask bookkeeping for the .npz sidecar.

    Raises:
        ValueError: on mask/image shape mismatch, non-positive regression
            denominator (all-masked or black mean image), or any
            non-finite / non-positive gain.
    """
    n_pairs = int(images.shape[0])
    t0 = time.time()

    # Pass 1: ensemble mean per channel, float64 accumulation.
    ref = images.mean(axis=0, dtype=np.float64).compute()

    if pixel_mask is not None:
        pixel_mask = np.asarray(pixel_mask, dtype=bool)
        if pixel_mask.shape != ref.shape[1:]:
            raise ValueError(
                f"gain normalisation: pixel mask shape {pixel_mask.shape} does "
                f"not match image shape {ref.shape[1:]}"
            )
        ref[:, pixel_mask] = 0.0
        n_masked = int(pixel_mask.sum())
    else:
        logger.warning(
            "GAIN NORMALISATION: no pixel mask — the gain regression runs over "
            "the FULL FRAME, including any non-illuminated / reflection regions"
        )
        n_masked = 0

    denoms = (ref * ref).sum(axis=(1, 2))
    if np.any(denoms <= 0.0):
        raise ValueError(
            f"gain normalisation: non-positive regression denominator "
            f"{denoms.tolist()} — mean image is zero over the unmasked region "
            f"(fully masked or black images)"
        )

    # Pass 2: per-frame products. einsum accumulates in float64 per chunk
    # without materialising a float64 copy of the chunk.
    numer = da.einsum("nchw,chw->nc", images, ref, dtype=np.float64).compute()
    gains = numer / denoms[np.newaxis, :]

    bad = ~np.isfinite(gains) | (gains <= 0.0)
    if np.any(bad):
        bad_frames = np.unique(np.nonzero(bad)[0])
        raise ValueError(
            f"gain normalisation: non-positive or non-finite gain at pair "
            f"indices {bad_frames.tolist()[:20]} — refusing to divide"
        )

    for ch, name in ((0, "A"), (1, "B")):
        g = gains[:, ch]
        logger.info(
            f"  gain {name}: mean {g.mean():.4f}, RMS jitter "
            f"{g.std() / g.mean():.4f}, range [{g.min():.4f}, {g.max():.4f}]"
        )
    logger.info(
        f"  gain pre-pass over {n_pairs} pairs took {time.time() - t0:.1f}s"
    )

    provenance: Dict[str, Any] = {
        "ref_a": ref[0].astype(np.float32),
        "ref_b": ref[1].astype(np.float32),
        "denom_a": float(denoms[0]),
        "denom_b": float(denoms[1]),
        "mask_applied": pixel_mask is not None,
        "n_masked_pixels": n_masked,
        "n_pairs": n_pairs,
    }
    return gains, provenance


def compute_and_save_frame_gains(
    images: da.Array,
    pixel_mask: Optional[np.ndarray],
    output_path: Path,
    camera_num: int,
    source_path: Path,
) -> np.ndarray:
    """Run the gain pre-pass and write the provenance .npz to the run output.

    Args:
        images: Lazy image array of shape (n_pairs, 2, H, W).
        pixel_mask: Boolean (H, W) mask, True = masked, or None.
        output_path: Per-camera run output directory (already created).
        camera_num: Camera number, recorded in the provenance file.
        source_path: Image source path, recorded in the provenance file.

    Returns:
        Gains array, float64 of shape (n_pairs, 2), column 0 = A, 1 = B.
    """
    logger.info(
        f"Gain normalisation pre-pass for camera {camera_num} "
        f"({int(images.shape[0])} pairs)..."
    )
    gains, provenance = compute_frame_gains(images, pixel_mask)

    npz_path = Path(output_path) / "gain_normalisation.npz"
    np.savez(
        npz_path,
        gains_a=gains[:, 0],
        gains_b=gains[:, 1],
        camera=camera_num,
        source_path=str(source_path),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        **provenance,
    )
    logger.info(f"  gain provenance written to {npz_path}")
    return gains

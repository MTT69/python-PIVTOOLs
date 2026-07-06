"""Sub-pixel peak fit for the mean-flow-only ensemble mode (fit_method: peak).

Fits the location of the single dominant peak in each ensemble-averaged AB
cross-correlation plane with a 3-point Gaussian (log-parabolic) interpolation
in x and y. This replaces the joint k-space transfer-function fit when only
the mean displacement is wanted: no AA/BB planes, no Reynolds stresses.

Convention: the planes come from the C ``bulkxcorr2d_accumulate`` family,
where zero displacement puts the peak at 0-based index ``(h//2, w//2)``.
Displacements are returned relative to that centre directly, so the full
path's 1-based ``corr_size/2 + 1`` bookkeeping never enters.

Failure policy (no silent fallbacks): windows whose peak cannot be fitted
get NaN displacement and a nonzero status — there is no parabolic or
centroid fallback. Status codes:

    0  success (or masked window — masked windows are skipped, ux = uy = 0)
    1  peak lies on the plane border (no interior 3-point stencil)
    2  peak or a 4-neighbour is non-finite or <= 0 (log undefined)
    3  degenerate log-parabola (flat top) or non-finite sub-pixel offset

These codes only need to be nonzero on failure; they must not collide with
the downstream nan_reason codes 6 (3/4-window rule), 10/11 (outliers) or
-1 (masked), which are assigned later in the accumulator.
"""

import numpy as np


def fit_windows_peak(R_AB, mask_flat, corr_size):
    """Fit the AB peak location in every window of a flattened plane stack.

    Parameters
    ----------
    R_AB : np.ndarray
        Flattened correlation planes, size ``n_windows * h * w`` in C order
        (the layout produced by finalize_pass), any float dtype.
    mask_flat : np.ndarray of bool
        Per-window vector mask, size ``n_windows``. Masked windows are
        skipped: displacement 0.0, status 0 (the accumulator zeroes and
        tags them via nan_reason afterwards).
    corr_size : tuple of int
        (h, w) of each correlation plane.

    Returns
    -------
    (ux, uy, statuses)
        ``ux``/``uy``: float32 arrays (n_windows,), displacement in pixels
        relative to the zero-shift centre ``(h//2, w//2)``; NaN on failure.
        ``statuses``: int32 array (n_windows,), see module docstring.
    """
    h, w = int(corr_size[0]), int(corr_size[1])
    plane_px = h * w
    if R_AB.size % plane_px != 0:
        raise ValueError(
            f"R_AB size {R_AB.size} is not a multiple of the plane size "
            f"{h}x{w}={plane_px}"
        )
    n_windows = R_AB.size // plane_px
    mask_flat = np.asarray(mask_flat, dtype=bool).ravel()
    if mask_flat.size != n_windows:
        raise ValueError(
            f"mask_flat has {mask_flat.size} entries but there are "
            f"{n_windows} windows"
        )

    # float64 for the logs; -inf where non-finite so argmax ignores NaNs.
    planes = np.asarray(R_AB, dtype=np.float64).reshape(n_windows, h, w)
    finite = np.isfinite(planes)
    search = np.where(finite, planes, -np.inf)

    flat_idx = search.reshape(n_windows, -1).argmax(axis=1)
    iy = flat_idx // w
    ix = flat_idx % w

    statuses = np.zeros(n_windows, dtype=np.int32)
    ux = np.full(n_windows, np.nan, dtype=np.float64)
    uy = np.full(n_windows, np.nan, dtype=np.float64)

    # Border peak (or an all-non-finite plane, whose argmax lands at 0,0).
    on_border = (iy == 0) | (iy == h - 1) | (ix == 0) | (ix == w - 1)
    statuses[on_border] = 1

    idx = np.arange(n_windows)
    # Clip the stencil for border windows so the gathers below stay in
    # bounds; those windows are already marked failed and never read out.
    iy_c = np.clip(iy, 1, h - 2)
    ix_c = np.clip(ix, 1, w - 2)
    C = planes[idx, iy_c, ix_c]
    L = planes[idx, iy_c, ix_c - 1]
    R = planes[idx, iy_c, ix_c + 1]
    U = planes[idx, iy_c - 1, ix_c]
    D = planes[idx, iy_c + 1, ix_c]

    stencil = np.stack([C, L, R, U, D], axis=0)
    bad_values = ~np.all(np.isfinite(stencil) & (stencil > 0), axis=0)
    statuses[bad_values & (statuses == 0)] = 2

    ok = statuses == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        lnC, lnL, lnR = np.log(C), np.log(L), np.log(R)
        lnU, lnD = np.log(U), np.log(D)
        dx = 0.5 * (lnL - lnR) / (lnL + lnR - 2.0 * lnC)
        dy = 0.5 * (lnU - lnD) / (lnU + lnD - 2.0 * lnC)

    degenerate = ok & (~np.isfinite(dx) | ~np.isfinite(dy))
    statuses[degenerate] = 3
    ok = statuses == 0

    ux[ok] = (ix_c[ok] + dx[ok]) - w // 2
    uy[ok] = (iy_c[ok] + dy[ok]) - h // 2

    # Masked windows: skipped, zero displacement, not a failure.
    ux[mask_flat] = 0.0
    uy[mask_flat] = 0.0
    statuses[mask_flat] = 0

    return ux.astype(np.float32), uy.astype(np.float32), statuses

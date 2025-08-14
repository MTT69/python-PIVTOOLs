import dask.array as da
from config import Config
import numpy as np


def filter_images(images: da.Array, config: Config, filters_override=None) -> da.Array:
    """Filter Images

    Applies filters in sequence. For temporal filters ('time', 'POD'), a per-filter
    batch_size can be provided in the filter dict and is enforced by rechunking
    the first axis accordingly before applying the filter on each block.

    Args:
        images (da.Array): Input images of shape (N, 2, H, W)
        config (Config): Configuration object
        filters_override (list|None): Optional list of filter dicts to override config.filters

    Returns:
        da.Array: Filtered images
    """
    filtered = images
    filters_list = filters_override if isinstance(filters_override, list) else config.filters
    print("Config/override filters:", filters_list)

    def ensure_temporal_chunk(arr: da.Array, batch_size: int) -> da.Array:
        # Clamp to actual length (e.g., last batch may be shorter)
        bs_eff = max(1, min(int(batch_size), int(arr.shape[0])))
        # Always rechunk first axis to bs_eff so map_blocks sees the intended window
        return arr.rechunk((bs_eff, arr.shape[1], arr.shape[2], arr.shape[3]))

    for filt in filters_list:
        ftype_raw = filt.get('type', None)
        if ftype_raw is None:
            print("No filter type specified, skipping.")
            continue
        ftype = str(ftype_raw).lower()

        # Determine per-filter batch size (for temporal filters)
        if ftype in ("time", "pod"):
            bs = filt.get('batch_size', config.piv_chunk_size)
            filtered = ensure_temporal_chunk(filtered, bs)

        if ftype == "time":
            print(f"Applying time filter; numblocks={filtered.numblocks[0]}, chunklen={filtered.chunks[0]}")
            filtered = time_filter(filtered)
        elif ftype == "pod":
            print(f"Applying POD filter; numblocks={filtered.numblocks[0]}, chunklen={filtered.chunks[0]}")
            filtered = pod_filter(filtered)
        elif ftype == "null":
            print("Skipping null filter")
            continue
        else:
            print(f"Warning: Filter type '{ftype}' not implemented, skipping.")
    return filtered


def time_filter(images: da.Array) -> da.Array:
    """
    Time filter images

    Args:
        images (da.Array): Dask array containing the images.

    Returns:
        da.Array: Filtered Dask array of images.
    """

    processed_images = images.map_blocks(_subtract_local_min, dtype=images.dtype)
    return processed_images


def _subtract_local_min(chunk):
    if chunk.size == 0:
        return chunk
    frame1_min = chunk[:, 0, :, :].min(axis=0)
    frame2_min = chunk[:, 1, :, :].min(axis=0)
    chunk[:, 0, :, :] -= frame1_min
    chunk[:, 1, :, :] -= frame2_min
    return chunk


def pod_filter_block(block):
    """
    block: numpy array of shape (N, 2, H, W)
    returns: numpy array of same shape, filtered
    """
    N, _, H, W = block.shape
    M1 = block[:, 0].reshape(N, -1).astype(np.float32)
    M2 = block[:, 1].reshape(N, -1).astype(np.float32)

    C1 = M1 @ M1.T
    C2 = M2 @ M2.T
    PSI1, S1, _ = np.linalg.svd(C1, full_matrices=False)
    PSI2, S2, _ = np.linalg.svd(C2, full_matrices=False)

    eps_auto_psi = 0.01
    eps_auto_sigma = 0.01

    def find_auto_mode(PSI, eigvals):
        for i in range(N - 1):
            mean_psi = np.abs(np.mean(PSI[:, i]))
            sig_diff = np.abs(eigvals[i] - eigvals[i + 1]) / eigvals[N // 2]
            if mean_psi < eps_auto_psi and sig_diff < eps_auto_sigma * eigvals[0]:
                return i
        return N

    N1 = find_auto_mode(PSI1, S1)
    N2 = find_auto_mode(PSI2, S2)

    def evaluate_phi_tcoeff(M, PSI, N_auto):
        PHI = []
        TC = []
        for i in range(N_auto):
            phi = M.T @ PSI[:, i]
            phi /= np.linalg.norm(phi)
            PHI.append(phi)
            TC.append(M @ phi)
        return PHI, TC

    PHI1, TC1 = evaluate_phi_tcoeff(M1, PSI1, N1)
    PHI2, TC2 = evaluate_phi_tcoeff(M2, PSI2, N2)

    F1 = M1.copy()
    F2 = M2.copy()
    for j in range(N1):
        F1 -= np.outer(TC1[j], PHI1[j])
    for j in range(N2):
        F2 -= np.outer(TC2[j], PHI2[j])

    filtered = np.stack([
        F1.reshape(N, H, W),
        F2.reshape(N, H, W)
    ], axis=1)

    return filtered.astype(block.dtype)


def pod_filter(images: da.Array) -> da.Array:
    """
    POD filter images

    Args:
        images (da.Array): Dask array containing the images.

    Returns:
        da.Array: Filtered Dask array of images.
    """
    processed_images = images.map_blocks(pod_filter_block, dtype=images.dtype)
    return processed_images

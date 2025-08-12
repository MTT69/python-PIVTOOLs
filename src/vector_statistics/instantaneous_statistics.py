from pathlib import Path
import numpy as np
from config import Config
from post_processing.vector_loading import load_vectors_from_directory
from vector_statistics.paths import get_data_paths
from scipy.io import savemat


def instantaneous_statistics(cam_num: int, config: Config, base):
    """
    Compute mean ux, uy for each instantaneous statistics_extraction entry for a given camera.
    Saves npz in stats_dir.
    Skips duplicated merged runs (only first camera processed for merged).
    """
    print(f"[instantaneous] Starting statistics for cam_num={cam_num}")
    for entry in config.statistics_extraction:
        print(f"[instantaneous] Processing entry: {entry}")
        if entry.get("type") != "instantaneous":
            print("[instantaneous] Skipping entry (not instantaneous type)")
            continue
        endpoint = entry.get("endpoint", "")
        use_merged = entry.get("use_merged", False)
        # Avoid repeating merged computation per camera
        if use_merged and cam_num != config.camera_numbers[0]:
            print(f"[instantaneous] Skipping merged for cam_num={cam_num} (only first camera processes merged)")
            continue
        cam_folder_eff = "Merged" if use_merged else f"Cam{cam_num}"
        print(f"[instantaneous] cam_folder_eff: {cam_folder_eff}")
        paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam_folder=cam_folder_eff,
            type_name="instantaneous",  # renamed from type
            endpoint=endpoint,
            use_merged=use_merged,
        )
        print(f"[instantaneous] Data dir: {paths['data_dir']}")
        if not paths["data_dir"].exists():
            print(f"[instantaneous] Data dir missing: {paths['data_dir']}")
            continue
        paths["stats_dir"].mkdir(parents=True, exist_ok=True)
        print(f"[instantaneous] Loading vectors from {paths['data_dir']}")
        # Load all requested passes at once; config.instantaneous_runs is 1-based
        selected_runs_1based = list(config.instantaneous_runs) if config.instantaneous_runs else []
        print(f"[instantaneous] Selected runs/passes (1-based): {selected_runs_1based}")
        arr = load_vectors_from_directory(paths["data_dir"], config, runs=selected_runs_1based)  # (N,R,3,H,W)
        print(f"[instantaneous] Loaded array shape: {arr.shape}")
        # Components: 0=ux, 1=uy, 2=b_mask
        ux = arr[:, :, 0]  # (N,R,H,W)
        uy = arr[:, :, 1]  # (N,R,H,W)
        bmask = arr[:, :, 2]  # (N,R,H,W)

        print("[instantaneous] Computing mean ux and uy per selected pass")
        mean_ux_all = ux.mean(axis=0).compute()  # (R,H,W)
        mean_uy_all = uy.mean(axis=0).compute()  # (R,H,W)

        # b_masks are identical across time -> take first time instance
        print("[instantaneous] Using b_mask from first time instance per selected pass")
        b_mask_all = bmask[0].compute()  # (R,H,W)

        # Compute Reynolds stresses using moment identities
        print("[instantaneous] Computing Reynolds stresses per selected pass")
        E_ux2 = (ux ** 2).mean(axis=0).compute()     # (R,H,W)
        E_uy2 = (uy ** 2).mean(axis=0).compute()     # (R,H,W)
        E_uxuy = (ux * uy).mean(axis=0).compute()    # (R,H,W)
        uu_all = E_ux2 - mean_ux_all ** 2            # (R,H,W)
        uv_all = E_uxuy - (mean_ux_all * mean_uy_all)
        vv_all = E_uy2 - mean_uy_all ** 2

        # Determine labels (1-based) for selected passes
        if selected_runs_1based:
            pass_labels = selected_runs_1based
        else:
            # No passes specified: assume all passes present in files
            R = mean_ux_all.shape[0]
            pass_labels = list(range(1, R + 1))

        # Build piv_result as n-pass-deep MATLAB struct array; populate only selected passes
        n_passes_cfg = len(config.instantaneous_window_sizes) or mean_ux_all.shape[0]
        print(f"[instantaneous] Building piv_result with n_passes={n_passes_cfg}")

        # Create a structured array with object-typed fields so each element can hold arrays
        dt = np.dtype([("ux", object), ("uy", object), ("b_mask", object),
                       ("uu", object), ("uv", object), ("vv", object)])
        piv_result = np.empty((n_passes_cfg,), dtype=dt)

        # Initialize all passes with empty 0x0 arrays
        empty = np.empty((0, 0), dtype=mean_ux_all.dtype)
        for p in range(n_passes_cfg):
            piv_result["ux"][p] = empty
            piv_result["uy"][p] = empty
            piv_result["b_mask"][p] = empty
            piv_result["uu"][p] = empty
            piv_result["uv"][p] = empty
            piv_result["vv"][p] = empty

        # Fill only the selected passes
        label_to_idx = {lbl: i for i, lbl in enumerate(pass_labels)}  # 1-based label -> local index
        for lbl in pass_labels:
            local_idx = label_to_idx[lbl]
            pass_zero_based = lbl - 1
            if 0 <= pass_zero_based < n_passes_cfg:
                piv_result["ux"][pass_zero_based] = mean_ux_all[local_idx]
                piv_result["uy"][pass_zero_based] = mean_uy_all[local_idx]
                piv_result["b_mask"][pass_zero_based] = b_mask_all[local_idx]
                piv_result["uu"][pass_zero_based] = uu_all[local_idx]
                piv_result["uv"][pass_zero_based] = uv_all[local_idx]
                piv_result["vv"][pass_zero_based] = vv_all[local_idx]
        ## efe plots
        # Save a single file per camera/merged with piv_result and meta
        out_file = paths["stats_dir"] / (f"{'merged' if use_merged else f'Cam{cam_num}'}_mean.mat")
        print(f"[instantaneous] Saving piv_result (means and Reynolds stresses) -> {out_file}")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        savemat(out_file, {
            "piv_result": piv_result,
            "meta": {
                "endpoint": endpoint,
                "use_merged": use_merged,
                "camera": cam_folder_eff,
                "selected_passes": pass_labels,
                "n_passes": int(n_passes_cfg),
                "definitions": "ux=<u>, uy=<v>, uu=<u'^2>, uv=<u'v'>, vv=<v'^2>"
            }
        })
        print(f"[instantaneous] Saved -> {out_file}")

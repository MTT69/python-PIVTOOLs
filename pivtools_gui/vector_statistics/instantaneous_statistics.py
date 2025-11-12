import dask  # added
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from pivtools_core.config import Config
from pivtools_core.paths import get_data_paths
from ..plotting.plot_maker import make_scalar_settings, plot_scalar_field
from pivtools_core.vector_loading import (
    load_coords_from_directory,
    load_vectors_from_directory,
)


def instantaneous_statistics(cam_num: int, config: Config, base):
    """
    Compute mean ux, uy for each instantaneous statistics_extraction entry for a given camera.
    Saves npz in stats_dir.
    Skips duplicated merged runs (only first camera processed for merged).
    """
    print(f"[instantaneous] Starting statistics for cam_num={cam_num}")
    if config.statistics_extraction is None:
        return
    for entry in config.statistics_extraction:
        print(f"[instantaneous] Processing entry: {entry}")
        if entry.get("type") != "instantaneous":
            print("[instantaneous] Skipping entry (not instantaneous type)")
            continue
        endpoint = entry.get("endpoint", "")
        use_merged = entry.get("use_merged", False)
        # Avoid repeating merged computation per camera
        if use_merged and cam_num != config.camera_numbers[0]:
            print(
                f"[instantaneous] Skipping merged for cam_num={cam_num} (only first camera processes merged)"
            )
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
        # Create nested mean_stats directory
        mean_stats_dir = paths["stats_dir"] / "mean_stats"
        mean_stats_dir.mkdir(parents=True, exist_ok=True)
        print(f"[instantaneous] Loading vectors from {paths['data_dir']}")
        # Load all requested passes at once; config.instantaneous_runs is 1-based
        selected_runs_1based = (
            list(config.instantaneous_runs) if config.instantaneous_runs else []
        )
        print(f"[instantaneous] Selected runs/passes (1-based): {selected_runs_1based}")
        arr = load_vectors_from_directory(
            paths["data_dir"], config, runs=selected_runs_1based
        )  # (N,R,3,H,W) or (N,R,4,H,W) if stereo
        # co-rodinates need to be loaded
        print(f"[instantaneous] Loaded array shape: {arr.shape}")

        # Load coordinates for the same selected runs
        coords_x_list, coords_y_list = load_coords_from_directory(
            paths["data_dir"], runs=selected_runs_1based
        )

        # Check for stereo (assume config.stereo is a boolean flag)
        stereo = getattr(config, "stereo", False)
        if stereo:
            if arr.shape[2] < 4:
                print("[instantaneous] Stereo enabled but array has fewer than 4 components; skipping uz stresses")
                stereo = False
            else:
                print("[instantaneous] Stereo enabled; computing uz stresses")

        # Components: 0=ux, 1=uy, 2=uz (if stereo), 3=b_mask (if stereo) or 2=b_mask
        ux = arr[:, :, 0]  # (N,R,H,W)
        uy = arr[:, :, 1]  # (N,R,H,W)
        if stereo:
            uz = arr[:, :, 2]  # (N,R,H,W)
            bmask = arr[:, :, 3]  # (N,R,H,W)
        else:
            bmask = arr[:, :, 2]  # (N,R,H,W)

        print("[instantaneous] Computing mean ux and uy per selected pass")
        # Build lazy reductions
        mean_ux_da = ux.mean(axis=0)  # (R,H,W)
        mean_uy_da = uy.mean(axis=0)  # (R,H,W)

        # b_masks are identical across time -> take first time instance lazily
        print("[instantaneous] Using b_mask from first time instance per selected pass")
        b_mask_da = bmask[0]  # (R,H,W)

        # Compute Reynolds stress ingredients lazily
        print("[instantaneous] Computing Reynolds stresses per selected pass")
        E_ux2_da = (ux**2).mean(axis=0)  # (R,H,W)
        E_uy2_da = (uy**2).mean(axis=0)  # (R,H,W)
        E_uxuy_da = (ux * uy).mean(axis=0)  # (R,H,W)
        if stereo:
            mean_uz_da = uz.mean(axis=0)  # (R,H,W)
            E_uz2_da = (uz**2).mean(axis=0)  # (R,H,W)
            E_uxuz_da = (ux * uz).mean(axis=0)  # (R,H,W)
            E_uyuz_da = (uy * uz).mean(axis=0)  # (R,H,W)

        # Execute all reductions in one graph run
        if stereo:
            mean_ux_all, mean_uy_all, mean_uz_all, b_mask_all, E_ux2, E_uy2, E_uxuy, E_uz2, E_uxuz, E_uyuz = dask.compute(
                mean_ux_da, mean_uy_da, mean_uz_da, b_mask_da, E_ux2_da, E_uy2_da, E_uxuy_da, E_uz2_da, E_uxuz_da, E_uyuz_da
            )
        else:
            mean_ux_all, mean_uy_all, b_mask_all, E_ux2, E_uy2, E_uxuy = dask.compute(
                mean_ux_da, mean_uy_da, b_mask_da, E_ux2_da, E_uy2_da, E_uxuy_da
            )

        # Finish Reynolds stresses on NumPy arrays
        uu_all = E_ux2 - mean_ux_all**2  # (R,H,W)
        uv_all = E_uxuy - (mean_ux_all * mean_uy_all)
        vv_all = E_uy2 - mean_uy_all**2
        if stereo:
            uw_all = E_uxuz - (mean_ux_all * mean_uz_all)
            vw_all = E_uyuz - (mean_uy_all * mean_uz_all)
            ww_all = E_uz2 - mean_uz_all**2

        # Determine labels (1-based) for selected passes
        if selected_runs_1based:
            pass_labels = selected_runs_1based
        else:
            # No passes specified: assume all passes present in files
            R = mean_ux_all.shape[0]
            pass_labels = list(range(1, R + 1))

        # Plot mean scalar fields for each selected pass (ux and uy)
        print("[instantaneous] Generating mean scalar plots for ux and uy")
        for lbl in pass_labels:
            idx = lbl - 1  # aligns with array indexing when all passes selected
            # If a subset was selected, map label to local index
            if selected_runs_1based:
                local_idx = selected_runs_1based.index(lbl)
            else:
                local_idx = idx
            # Build boolean mask
            mask_bool = np.asarray(b_mask_all[local_idx]).astype(bool)

            # Per-pass coordinates if available
            cx = coords_x_list[local_idx] if local_idx < len(coords_x_list) else None
            cy = coords_y_list[local_idx] if local_idx < len(coords_y_list) else None

            # ux
            save_base_ux = mean_stats_dir / f"ux_{lbl}"
            settings_ux = make_scalar_settings(
                config,
                variable="ux",
                run_label=lbl,
                save_basepath=save_base_ux,  # used only for naming below
                variable_units="m/s",
                coords_x=cx,
                coords_y=cy,
            )
            fig_ux, _, _ = plot_scalar_field(
                mean_ux_all[local_idx], mask_bool, settings_ux
            )
            fig_ux.savefig(
                f"{save_base_ux}{config.plot_save_extension}",
                dpi=1200,
                bbox_inches="tight",
            )
            if config.plot_save_pickle:
                import pickle

                with open(f"{save_base_ux}.pkl", "wb") as f:
                    pickle.dump(mean_ux_all[local_idx], f)
            plt.close(fig_ux)

            # uy
            save_base_uy = mean_stats_dir / f"uy_{lbl}"
            settings_uy = make_scalar_settings(
                config,
                variable="uy",
                run_label=lbl,
                save_basepath=save_base_uy,  # used only for naming below
                variable_units="m/s",
                coords_x=cx,
                coords_y=cy,
            )
            fig_uy, _, _ = plot_scalar_field(
                mean_uy_all[local_idx], mask_bool, settings_uy
            )
            fig_uy.savefig(
                f"{save_base_uy}{config.plot_save_extension}",
                dpi=1200,
                bbox_inches="tight",
            )
            if config.plot_save_pickle:
                import pickle

                with open(f"{save_base_uy}.pkl", "wb") as f:
                    pickle.dump(mean_uy_all[local_idx], f)
            plt.close(fig_uy)

        # Build piv_result as n-pass-deep MATLAB struct array; populate only selected passes
        n_passes_cfg = len(config.instantaneous_window_sizes) or mean_ux_all.shape[0]
        print(f"[instantaneous] Building piv_result with n_passes={n_passes_cfg}")
        # Create a structured array with object-typed fields so each element can hold arrays
        dt_fields = [
            ("ux", object),
            ("uy", object),
            ("b_mask", object),
            ("uu", object),
            ("uv", object),
            ("vv", object),
        ]
        if stereo:
            dt_fields.extend(
                [
                    ("uz", object),
                    ("uw", object),
                    ("vw", object),
                    ("ww", object),
                ]
            )
        dt = np.dtype(dt_fields)
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
            if stereo:
                piv_result["uz"][p] = empty
                piv_result["uw"][p] = empty
                piv_result["vw"][p] = empty
                piv_result["ww"][p] = empty

        # Fill only the selected passes
        label_to_idx = {
            lbl: i for i, lbl in enumerate(pass_labels)
        }  # 1-based label -> local index (selected order)
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
                if stereo:
                    piv_result["uz"][pass_zero_based] = mean_uz_all[local_idx]
                    piv_result["uw"][pass_zero_based] = uw_all[local_idx]
                    piv_result["vw"][pass_zero_based] = vw_all[local_idx]
                    piv_result["ww"][pass_zero_based] = ww_all[local_idx]

        # Build coordinates struct array (fields: x, y), aligned to n_passes_cfg; fill only selected passes
        dt_coords = np.dtype([("x", object), ("y", object)])
        coordinates = np.empty((n_passes_cfg,), dtype=dt_coords)
        # Initialize empties
        empty_xy = np.empty((0, 0), dtype=empty.dtype)
        for p in range(n_passes_cfg):
            coordinates["x"][p] = empty_xy
            coordinates["y"][p] = empty_xy
        # Fill selected using the same label order
        for lbl in pass_labels:
            local_idx = label_to_idx[lbl]
            pass_zero_based = lbl - 1
            if 0 <= pass_zero_based < n_passes_cfg and local_idx < len(coords_x_list):
                coordinates["x"][pass_zero_based] = coords_x_list[local_idx]
                coordinates["y"][pass_zero_based] = coords_y_list[local_idx]

        # Save a single file per camera/merged with piv_result, coordinates and meta
        out_file = mean_stats_dir / (
            f"{'merged' if use_merged else f'Cam{cam_num}'}_mean.mat"
        )
        print(
            f"[instantaneous] Saving piv_result (means and Reynolds stresses) -> {out_file}"
        )
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # Save piv_result and meta to main file
        meta_dict = {
            "endpoint": endpoint,
            "use_merged": use_merged,
            "camera": cam_folder_eff,
            "selected_passes": pass_labels,
            "n_passes": int(n_passes_cfg),
            "stereo": stereo,
            "definitions": "ux=<u>, uy=<v>, uu=<u'^2>, uv=<u'v'>, vv=<v'^2>"
            + (", uz=<w>, uw=<u'w'>, vw=<v'w'>, ww=<w'^2>" if stereo else ""),
        }
        savemat(
            out_file,
            {
                "piv_result": piv_result,
                "meta": meta_dict,
            },
        )
        # Save coordinates as a separate file into mean_stats folder
        coords_file = mean_stats_dir / (
            f"{'merged' if use_merged else f'Cam{cam_num}'}_coordinates.mat"
        )
        savemat(coords_file, {"coordinates": coordinates})
        print(f"[instantaneous] Saved -> {out_file}")

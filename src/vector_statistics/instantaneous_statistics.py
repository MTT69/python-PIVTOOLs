import dask  # added
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from config import Config
from paths import get_data_paths
from plotting.plot_maker import make_scalar_settings, plot_scalar_field
from vector_loading import (
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
        )  # (N,R,3,H,W)
        # co-rodinates need to be loaded
        print(f"[instantaneous] Loaded array shape: {arr.shape}")

        # Load coordinates for the same selected runs
        coords_x_list, coords_y_list = load_coords_from_directory(
            paths["data_dir"], runs=selected_runs_1based
        )

        # Components: 0=ux, 1=uy, 2=b_mask
        ux = arr[:, :, 0]  # (N,R,H,W)
        uy = arr[:, :, 1]  # (N,R,H,W)
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

        # Execute all reductions in one graph run
        mean_ux_all, mean_uy_all, b_mask_all, E_ux2, E_uy2, E_uxuy = dask.compute(
            mean_ux_da, mean_uy_da, b_mask_da, E_ux2_da, E_uy2_da, E_uxuy_da
        )

        # Finish Reynolds stresses on NumPy arrays
        uu_all = E_ux2 - mean_ux_all**2  # (R,H,W)
        uv_all = E_uxuy - (mean_ux_all * mean_uy_all)
        vv_all = E_uy2 - mean_uy_all**2

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
        # Check whether video-making is true
        # if config.video_making:
        import time

        start_time = time.time()
        if True:
            import imageio

            print("[instantaneous] Video-making is enabled")
            print("[instantaneous] Generating scalar videos for ux and uy")
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
                cx = (
                    coords_x_list[local_idx] if local_idx < len(coords_x_list) else None
                )
                cy = (
                    coords_y_list[local_idx] if local_idx < len(coords_y_list) else None
                )

                # ux -------------------------------------------------------------------------
                save_base_ux_video = mean_stats_dir / f"ux_video_{lbl}"
                save_base_ux_video_mp4 = mean_stats_dir / f"ux_video_{lbl}.mp4"

                # Select 50 random frame indices from ux (shape: N,R,H,W)
                num_frames = ux.shape[0]
                if num_frames >= 50:
                    random_indices = np.random.choice(
                        num_frames, size=50, replace=False
                    )
                else:
                    random_indices = np.arange(num_frames)

                ux_frames = ux[
                    random_indices, local_idx
                ]  # shape: (50,H,W) or (<num_frames>,H,W)
                # Ensure ux_frames is a NumPy array (compute if it's a Dask array)
                if hasattr(ux_frames, "compute"):
                    ux_frames = ux_frames.compute()
                max_val = np.max(ux_frames)
                min_val = np.min(ux_frames)

                # print(f"[instantaneous] ux max values in 50 random frames: {max_val}")
                # print(f"[instantaneous] ux min values in 50 random frames: {min_val}")

                frames_list = []

                for frame_idx, frame in enumerate(ux[:, local_idx]):
                    settings_ux_frame = make_scalar_settings(
                        config,
                        variable=f"ux_frame_{frame_idx + 1}",
                        run_label=lbl,
                        save_basepath=save_base_ux_video,  # used only for naming below
                        variable_units="m/s",
                        coords_x=cx,
                        coords_y=cy,
                        upper_limit=max_val,
                        lower_limit=min_val,
                    )

                    fig_ux_frame, _, _ = plot_scalar_field(
                        frame, mask_bool, settings_ux_frame
                    )

                    fig_ux_frame.canvas.draw()
                    width, height = fig_ux_frame.canvas.get_width_height()
                    argb = np.frombuffer(
                        fig_ux_frame.canvas.tostring_argb(), dtype=np.uint8
                    )
                    argb = argb.reshape((height, width, 4))
                    # Convert ARGB to RGB
                    rgb = argb[:, :, 1:]  # Discard alpha channel, keep R,G,B

                    frames_list.append(rgb)
                    plt.close(fig_ux_frame)

                # Write frames to video
                imageio.mimsave(
                    save_base_ux_video_mp4,
                    frames_list,
                    format="FFMPEG",
                    codec="libx264",
                )

                print(f"[instantaneous] Saved video: {save_base_ux_video_mp4}")

                # uy -------------------------------------------------------------------------
                save_base_uy_video = mean_stats_dir / f"uy_video_{lbl}"
                save_base_uy_video_mp4 = mean_stats_dir / f"uy_video_{lbl}.mp4"

                # Select 50 random frame indices from uy (shape: N,R,H,W)
                num_frames_uy = uy.shape[0]
                if num_frames_uy >= 50:
                    random_indices_uy = np.random.choice(
                        num_frames_uy, size=50, replace=False
                    )
                else:
                    random_indices_uy = np.arange(num_frames_uy)

                uy_frames = uy[
                    random_indices_uy, local_idx
                ]  # shape: (50,H,W) or (<num_frames>,H,W)
                if hasattr(uy_frames, "compute"):
                    uy_frames = uy_frames.compute()
                max_val_uy = np.max(uy_frames)
                min_val_uy = np.min(uy_frames)

                frames_list_uy = []

                for frame_idx, frame in enumerate(uy[:, local_idx]):
                    settings_uy_frame = make_scalar_settings(
                        config,
                        variable=f"uy_frame_{frame_idx + 1}",
                        run_label=lbl,
                        save_basepath=save_base_uy_video,
                        variable_units="m/s",
                        coords_x=cx,
                        coords_y=cy,
                        upper_limit=max_val_uy,
                        lower_limit=min_val_uy,
                    )

                    fig_uy_frame, _, _ = plot_scalar_field(
                        frame, mask_bool, settings_uy_frame
                    )

                    fig_uy_frame.canvas.draw()
                    width, height = fig_uy_frame.canvas.get_width_height()
                    argb = np.frombuffer(
                        fig_uy_frame.canvas.tostring_argb(), dtype=np.uint8
                    )
                    argb = argb.reshape((height, width, 4))
                    rgb = argb[:, :, 1:]  # Discard alpha channel, keep R,G,B

                    frames_list_uy.append(rgb)
                    plt.close(fig_uy_frame)

                imageio.mimsave(
                    save_base_uy_video_mp4,
                    frames_list_uy,
                    format="FFMPEG",
                    codec="libx264",
                )

                print(f"[instantaneous] Saved video: {save_base_uy_video_mp4}")

        end_time = time.time()
        print(
            f"[instantaneous] Video generation completed in {end_time - start_time:.2f} seconds"
        )

        # Build piv_result as n-pass-deep MATLAB struct array; populate only selected passes
        n_passes_cfg = len(config.instantaneous_window_sizes) or mean_ux_all.shape[0]
        print(f"[instantaneous] Building piv_result with n_passes={n_passes_cfg}")
        # Create a structured array with object-typed fields so each element can hold arrays
        dt = np.dtype(
            [
                ("ux", object),
                ("uy", object),
                ("b_mask", object),
                ("uu", object),
                ("uv", object),
                ("vv", object),
            ]
        )
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
        savemat(
            out_file,
            {
                "piv_result": piv_result,
                "meta": {
                    "endpoint": endpoint,
                    "use_merged": use_merged,
                    "camera": cam_folder_eff,
                    "selected_passes": pass_labels,
                    "n_passes": int(n_passes_cfg),
                    "definitions": "ux=<u>, uy=<v>, uu=<u'^2>, uv=<u'v'>, vv=<v'^2>",
                },
            },
        )
        # Save coordinates as a separate file into mean_stats folder
        coords_file = mean_stats_dir / (
            f"{'merged' if use_merged else f'Cam{cam_num}'}_coordinates.mat"
        )
        savemat(coords_file, {"coordinates": coordinates})
        print(f"[instantaneous] Saved -> {out_file}")

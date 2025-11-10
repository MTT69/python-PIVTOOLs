import numpy as np

from ..config import Config
from ..paths import get_data_paths
from ..vector_loading import load_vectors_from_directory


def ensemble_statistics(cam_num: int, config: Config, base):
    """
    Placeholder ensemble stats: compute global mean (ux, uy) across all frames (same as instantaneous for now).
    Extend later with RMS, turbulence quantities, etc.
    """
    if config.statistics_extraction is None:
        return
    for entry in config.statistics_extraction:
        if entry.get("type") != "ensemble":
            continue
        endpoint = entry.get("endpoint", "")
        use_merged = entry.get("use_merged", False)
        if use_merged and cam_num != config.camera_numbers[0]:
            continue
        cam_folder_eff = "Merged" if use_merged else f"Cam{cam_num}"
        paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam_folder=cam_folder_eff,
            type="ensemble",
            endpoint=endpoint,
            use_merged=use_merged,
        )
        if not paths["data_dir"].exists():
            print(f"[ensemble] Data dir missing: {paths['data_dir']}")
            continue
        paths["stats_dir"].mkdir(parents=True, exist_ok=True)
        arr = load_vectors_from_directory(paths["data_dir"], config)
        ux = arr[:, 0]
        uy = arr[:, 1]
        mean_ux = ux.mean(axis=0).compute()
        mean_uy = uy.mean(axis=0).compute()
        out_file = paths["stats_dir"] / (
            f"{'merged' if use_merged else f'Cam{cam_num}'}_ensemble_mean.npz"
        )
        np.savez_compressed(
            out_file,
            mean_ux=mean_ux,
            mean_uy=mean_uy,
            meta=dict(endpoint=endpoint, use_merged=use_merged, camera=cam_folder_eff),
        )
        print(f"[ensemble] Saved ensemble mean -> {out_file}")

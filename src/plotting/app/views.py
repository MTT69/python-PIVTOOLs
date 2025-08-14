import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, jsonify, request
from scipy.io import loadmat

from config import Config
from paths import get_data_paths
from plotting.plot_maker import make_scalar_settings, plot_scalar_field

config = Config()

vector_plot_bp = Blueprint("vector_plot", __name__, url_prefix="/plot")


@vector_plot_bp.route("/plot_vector", methods=["GET"])
def plot_vector():
    # --- Parse frontend parameters ---
    base_path = request.args.get("base_path")
    frame = request.args.get("frame", default=1, type=int)
    camera = request.args.get("camera", default="1", type=str)
    merged = request.args.get("merged", default="0", type=str)
    endpoint = request.args.get("endpoint", default="", type=str)
    var = request.args.get("var", "ux")
    run = request.args.get("run", default=1, type=int)
    lower_limit = request.args.get("lower_limit", type=float)
    upper_limit = request.args.get("upper_limit", type=float)
    cmap = request.args.get("cmap", default=None, type=str)
    if cmap is not None and (cmap.lower() == "default" or cmap.strip() == ""):
        cmap = None

    try:
        base = base_path
        print(f"[plot_vector] base resolved to: {base}")
    except Exception as e:
        print(f"[plot_vector] Error resolving base_path: {e}")
        return jsonify({"error": "Invalid base_path"}), 400

    cam_folder_eff = f"Cam{camera}"
    use_merged = merged in ("1", "true", "True")
    try:
        paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam_folder=cam_folder_eff,
            type_name="instantaneous",
            endpoint=endpoint,
            use_merged=use_merged,
        )
        vector_fmt = config.vector_format  # e.g. "%05d.mat"
        data_path = Path(paths["data_dir"] / (vector_fmt % frame))
        coords_path = Path(paths["data_dir"] / "coordinates.mat")
    except Exception as e:
        return jsonify({"error": f"Failed to resolve data paths: {e}"}), 400

    try:
        # Load data .mat (expects variable 'piv_result')
        data_mat = loadmat(str(data_path), struct_as_record=False, squeeze_me=True)
        if "piv_result" not in data_mat:
            return (
                jsonify({"error": "Variable 'piv_result' not found in data mat"}),
                400,
            )
        piv_result = data_mat["piv_result"]

        # Find first non-empty run if requested run is empty
        pr = None
        max_runs = 1
        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            max_runs = piv_result.size
            current_run = run
            while current_run <= max_runs:
                pr_candidate = piv_result[current_run - 1]
                try:
                    var_arr_candidate = np.asarray(getattr(pr_candidate, var))
                    if var_arr_candidate.size > 0 and not np.all(
                        np.isnan(var_arr_candidate)
                    ):
                        pr = pr_candidate
                        run = current_run
                        break
                except Exception:
                    pass
                current_run += 1
            if pr is None:
                return (
                    jsonify({"error": f"No non-empty run found for variable {var}"}),
                    400,
                )
        else:
            # Single run; only valid run is 1
            try:
                var_arr_candidate = np.asarray(getattr(piv_result, var))
                if var_arr_candidate.size > 0 and not np.all(
                    np.isnan(var_arr_candidate)
                ):
                    pr = piv_result
                    run = 1
                else:
                    return (
                        jsonify(
                            {"error": f"No non-empty run found for variable {var}"}
                        ),
                        400,
                    )
            except Exception:
                return (
                    jsonify({"error": f"'{var}' not found in piv_result element"}),
                    400,
                )

        # Extract variable and mask
        try:
            var_arr = np.asarray(getattr(pr, var))
        except Exception:
            return jsonify({"error": f"'{var}' not found in piv_result element"}), 400
        try:
            mask_arr = np.asarray(getattr(pr, "b_mask")).astype(bool)
        except Exception:
            mask_arr = np.zeros_like(var_arr, dtype=bool)

        # Load coordinates .mat (expects variable 'coordinates')
        coords_mat = loadmat(str(coords_path), struct_as_record=False, squeeze_me=True)
        if "coordinates" not in coords_mat:
            return (
                jsonify({"error": "Variable 'coordinates' not found in coords mat"}),
                400,
            )
        coords = coords_mat["coordinates"]

        cx = cy = None
        if isinstance(coords, np.ndarray) and coords.dtype == object:
            max_coords_runs = coords.size
            if run < 1 or run > max_coords_runs:
                return (
                    jsonify(
                        {
                            "error": f"run out of range for coordinates (1..{max_coords_runs})"
                        }
                    ),
                    400,
                )
            c_el = coords[run - 1]
            cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)
        else:
            if run != 1:
                return (
                    jsonify({"error": "coordinates contains a single run; use run=1"}),
                    400,
                )
            c_el = coords
            cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)

        # Build settings and plot
        save_basepath = Path("plot_vector_tmp")  # not used for saving here
        settings = make_scalar_settings(
            config,
            variable=var,
            run_label=run,
            save_basepath=save_basepath,
            variable_units="m/s",
            coords_x=cx,
            coords_y=cy,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            cmap=cmap,
        )

        fig, ax, im = plot_scalar_field(var_arr, mask_arr, settings)

        # Render to PNG bytes
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        b64_img = base64.b64encode(buf.read()).decode("utf-8")

        # Optionally include dimensions
        H, W = int(var_arr.shape[0]), int(var_arr.shape[1])
        return jsonify(
            {
                "image": b64_img,
                "meta": {"run": run, "var": var, "width": W, "height": H},
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import base64
import math
import random
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, jsonify, request
from scipy.io import loadmat

from config import Config
from paths import get_data_paths
from plotting.plot_maker import make_scalar_settings, plot_scalar_field
from vector_statistics.instantaneous_statistics import instantaneous_statistics

config = Config()

vector_plot_bp = Blueprint("vector_plot", __name__, url_prefix="/plot")


def cam_folder_key(camera):
    """Convert camera parameter to proper folder name format (Cam1, Cam2, etc.)"""
    if str(camera).lower().startswith("cam"):
        return str(camera)
    return f"Cam{camera}"


def parse_common_params(request):
    """Parse common parameters from request args"""
    base_path = (
        request.args.get("base_path")
        or request.args.get("path")
        or request.args.get("full_path")
    )
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

    use_merged = merged in ("1", "true", "True")

    return {
        "base_path": base_path,
        "frame": frame,
        "camera": camera,
        "merged": merged,
        "use_merged": use_merged,
        "endpoint": endpoint,
        "var": var,
        "run": run,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "cmap": cmap,
    }


def find_non_empty_run(piv_result, var, run=1):
    """Find non-empty run in piv_result for variable var"""
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
    else:
        # Single run; only valid run is 1
        try:
            var_arr_candidate = np.asarray(getattr(piv_result, var))
            if var_arr_candidate.size > 0 and not np.all(np.isnan(var_arr_candidate)):
                pr = piv_result
                run = 1
            else:
                pr = None
        except Exception:
            pr = None

    return pr, run


def extract_coordinates(coords, run):
    """Extract x, y coordinates for the given run"""
    if isinstance(coords, np.ndarray) and coords.dtype == object:
        max_coords_runs = coords.size
        if run < 1 or run > max_coords_runs:
            raise ValueError(f"run out of range for coordinates (1..{max_coords_runs})")
        c_el = coords[run - 1]
        cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)
    else:
        if run != 1:
            raise ValueError("coordinates contains a single run; use run=1")
        c_el = coords
        cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)
    return cx, cy


def extract_var_and_mask(pr, var):
    """Extract variable and mask arrays from piv_result element"""
    try:
        var_arr = np.asarray(getattr(pr, var))
    except Exception:
        raise ValueError(f"'{var}' not found in piv_result element")

    try:
        mask_arr = np.asarray(getattr(pr, "b_mask")).astype(bool)
    except Exception:
        mask_arr = np.zeros_like(var_arr, dtype=bool)

    return var_arr, mask_arr


def create_and_return_plot(var_arr, mask_arr, settings):
    """Create plot and return base64 encoded image with metadata"""
    fig, ax, im = plot_scalar_field(var_arr, mask_arr, settings)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode("utf-8")

    H, W = int(var_arr.shape[0]), int(var_arr.shape[1])
    return b64_img, W, H


@vector_plot_bp.route("/plot_vector", methods=["GET"])
def plot_vector():
    # Parse parameters
    try:
        params = parse_common_params(request)
        if not params["base_path"]:
            return jsonify({"error": "base_path required"}), 400

        base = params["base_path"]
        cam_folder_eff = (
            f"Cam{params['camera']}"
            if not str(params["camera"]).lower().startswith("cam")
            else params["camera"]
        )
    except Exception as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam_folder=cam_folder_eff,
            type_name="instantaneous",
            endpoint=params["endpoint"],
            use_merged=params["use_merged"],
        )
        vector_fmt = config.vector_format
        data_path = Path(paths["data_dir"] / (vector_fmt % params["frame"]))
        coords_path = Path(paths["data_dir"] / "coordinates.mat")
    except Exception as e:
        return jsonify({"error": f"Failed to resolve data paths: {e}"}), 400

    try:
        # Load data .mat
        data_mat = loadmat(str(data_path), struct_as_record=False, squeeze_me=True)
        if "piv_result" not in data_mat:
            return (
                jsonify({"error": "Variable 'piv_result' not found in data mat"}),
                400,
            )
        piv_result = data_mat["piv_result"]

        # Find non-empty run
        pr, effective_run = find_non_empty_run(piv_result, params["var"], params["run"])
        if pr is None:
            return (
                jsonify(
                    {"error": f"No non-empty run found for variable {params['var']}"}
                ),
                400,
            )

        # Extract variable and mask
        try:
            var_arr, mask_arr = extract_var_and_mask(pr, params["var"])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Load coordinates
        coords_mat = loadmat(str(coords_path), struct_as_record=False, squeeze_me=True)
        if "coordinates" not in coords_mat:
            return (
                jsonify({"error": "Variable 'coordinates' not found in coords mat"}),
                400,
            )
        coords = coords_mat["coordinates"]

        # Extract coordinates
        try:
            cx, cy = extract_coordinates(coords, effective_run)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Build settings and plot
        settings = make_scalar_settings(
            config,
            variable=params["var"],
            run_label=effective_run,
            save_basepath=Path("plot_vector_tmp"),
            variable_units="m/s",
            coords_x=cx,
            coords_y=cy,
            lower_limit=params["lower_limit"],
            upper_limit=params["upper_limit"],
            cmap=params["cmap"],
        )

        # Create and return plot
        b64_img, W, H = create_and_return_plot(var_arr, mask_arr, settings)

        return jsonify(
            {
                "image": b64_img,
                "meta": {
                    "run": effective_run,
                    "var": params["var"],
                    "width": W,
                    "height": H,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@vector_plot_bp.route("/plot_stats", methods=["GET"])
def plot_stats():
    """
    Run instantaneous_statistics for the provided base/path and camera,
    then load the generated mean .mat and coordinates and return a PNG
    (base64) of the requested scalar (var) same as /plot_vector.
    """
    # Parse parameters
    try:
        params = parse_common_params(request)
        if not params["base_path"]:
            return jsonify({"error": "base_path/path required"}), 400

        base = params["base_path"]
        cam_num = (
            int(params["camera"])
            if not str(params["camera"]).lower().startswith("cam")
            else int(str(params["camera"]).lstrip("Cam").lstrip("cam"))
        )
    except Exception as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    # Resolve expected stats paths
    try:
        cam_folder_eff = "Merged" if params["use_merged"] else f"Cam{cam_num}"
        paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam_folder=cam_folder_eff,
            type_name="instantaneous",
            endpoint=params["endpoint"],
            use_merged=params["use_merged"],
        )
        mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"
        out_file = mean_stats_dir / (
            f"{'merged' if params['use_merged'] else f'Cam{cam_num}'}_mean.mat"
        )
        coords_file = mean_stats_dir / (
            f"{'merged' if params['use_merged'] else f'Cam{cam_num}'}_coordinates.mat"
        )
    except Exception as e:
        return jsonify({"error": f"failed to resolve mean/coords paths: {e}"}), 400

    # Run stats if needed
    if not (out_file.exists() and coords_file.exists()):
        try:
            instantaneous_statistics(cam_num=cam_num, config=config, base=base)
        except Exception as e:
            return jsonify({"error": f"instantaneous_statistics failed: {e}"}), 500
        if not out_file.exists() or not coords_file.exists():
            return jsonify({"error": "Output files not found after stats run"}), 500

    try:
        # Load mean data and coordinates
        data_mat = loadmat(str(out_file), struct_as_record=False, squeeze_me=True)
        if "piv_result" not in data_mat:
            return (
                jsonify({"error": "Variable 'piv_result' not found in mean mat"}),
                400,
            )
        piv_result = data_mat["piv_result"]

        coords_mat = loadmat(str(coords_file), struct_as_record=False, squeeze_me=True)
        if "coordinates" not in coords_mat:
            return (
                jsonify({"error": "Variable 'coordinates' not found in coords mat"}),
                400,
            )
        coords = coords_mat["coordinates"]

        # Find non-empty run
        pr, effective_run = find_non_empty_run(piv_result, params["var"], params["run"])
        if pr is None:
            return (
                jsonify(
                    {"error": f"No non-empty run found for variable {params['var']}"}
                ),
                400,
            )

        # Extract variable and mask
        try:
            var_arr, mask_arr = extract_var_and_mask(pr, params["var"])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Extract coordinates
        try:
            cx, cy = extract_coordinates(coords, effective_run)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Build settings and plot
        settings = make_scalar_settings(
            config,
            variable=params["var"],
            run_label=effective_run,
            save_basepath=Path("plot_stats_tmp"),
            variable_units="m/s",
            coords_x=cx,
            coords_y=cy,
            lower_limit=params["lower_limit"],
            upper_limit=params["upper_limit"],
            cmap=params["cmap"],
        )

        # Create and return plot
        b64_img, W, H = create_and_return_plot(var_arr, mask_arr, settings)

        return jsonify(
            {
                "image": b64_img,
                "meta": {
                    "run": effective_run,
                    "var": params["var"],
                    "width": W,
                    "height": H,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": f"plotting failed: {e}"}), 500


@vector_plot_bp.route("/check_vars", methods=["GET"])
@vector_plot_bp.route("/check_stat_vars", methods=["GET"])  # legacy alias
def check_vars():
    """Inspect a .mat and return available variable names."""
    mat_path_arg = request.args.get("mat_path", default=None, type=str)
    frame = request.args.get("frame", default=None, type=int)

    # If direct mat_path is provided, prefer it
    if mat_path_arg:
        mat_path = Path(mat_path_arg)
        if not mat_path.exists():
            return jsonify({"error": f"mat_path not found: {mat_path}"}), 404
        source_info = {"type": "mat", "path": str(mat_path)}
    else:
        # Parse parameters for constructing paths
        try:
            params = parse_common_params(request)
            if not params["base_path"]:
                return (
                    jsonify(
                        {"error": "base_path/path required unless mat_path provided"}
                    ),
                    400,
                )

            base = params["base_path"]
            cam_num = (
                int(params["camera"])
                if not str(params["camera"]).lower().startswith("cam")
                else int(str(params["camera"]).lstrip("Cam").lstrip("cam"))
            )
            cam_folder_eff = "Merged" if params["use_merged"] else f"Cam{cam_num}"
        except Exception as e:
            return jsonify({"error": f"Invalid parameters: {e}"}), 400

        # resolve paths
        try:
            paths = get_data_paths(
                base_dir=base,
                num_images=config.num_images,
                cam_folder=cam_folder_eff,
                type_name="instantaneous",
                endpoint=params["endpoint"],
                use_merged=params["use_merged"],
            )
            data_dir = Path(paths["data_dir"])
            mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"
        except Exception as e:
            return jsonify({"error": f"failed to resolve paths: {e}"}), 400

        # If frame specified -> inspect instantaneous vector .mat
        if frame is not None:
            # determine vector format
            vec_fmt = config.vector_format
            if isinstance(vec_fmt, (list, tuple)):
                vec_fmt = vec_fmt[0] if len(vec_fmt) > 0 else "%05d.mat"
            try:
                mat_path = Path(data_dir) / (vec_fmt % frame)
            except Exception as e:
                return (
                    jsonify(
                        {
                            "error": f"failed to format vector filename with frame={frame}: {e}"
                        }
                    ),
                    400,
                )
            if not mat_path.exists():
                return (
                    jsonify({"error": f"instantaneous mat not found: {mat_path}"}),
                    404,
                )
            source_info = {"type": "instantaneous", "path": str(mat_path)}
        else:
            # No frame and no mat_path -> inspect mean stats .mat
            out_file = mean_stats_dir / (
                f"{'merged' if params['use_merged'] else f'Cam{cam_num}'}_mean.mat"
            )
            if not out_file.exists():
                return jsonify({"error": f"mean file not found: {out_file}"}), 404
            mat_path = out_file
            source_info = {"type": "mean", "path": str(mat_path)}

    # Load and inspect mat_path
    try:
        data_mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    except Exception as e:
        return jsonify({"error": f"failed to load mat: {e}"}), 500

    if "piv_result" not in data_mat:
        return jsonify({"error": "Variable 'piv_result' not found in mat"}), 400
    piv_result = data_mat["piv_result"]

    # Find element to inspect
    pr = None
    try:
        if isinstance(piv_result, np.ndarray):
            # pick first non-empty run if possible
            for el in piv_result:
                try:
                    # Try common field candidates
                    for candidate in ("ux", "uy", "b_mask", "uu"):
                        val = getattr(el, candidate, None)
                        if val is not None and getattr(np.asarray(val), "size", 0) > 0:
                            pr = el
                            break
                    if pr is not None:
                        break
                except Exception:
                    continue
            if pr is None and piv_result.size > 0:
                pr = piv_result.flat[0]
        else:
            pr = piv_result
    except Exception:
        pr = piv_result

    # Get available variables
    vars_list = []
    dt = getattr(pr, "dtype", None)
    if dt is not None and getattr(dt, "names", None):
        vars_list = list(dt.names)
    else:
        # fallback introspection
        candidates = []
        for attr in dir(pr):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(pr, attr)
                if isinstance(val, (np.ndarray, list, tuple)):
                    candidates.append(attr)
                else:
                    if hasattr(val, "shape") or getattr(val, "size", None) is not None:
                        candidates.append(attr)
            except Exception:
                continue
        vars_list = sorted(set(candidates))

    return jsonify({"vars": vars_list, "source": source_info})


@vector_plot_bp.route("/check_limits", methods=["GET"])
def check_limits():
    """Check min/max of a variable across random .mat files."""
    try:
        params = parse_common_params(request)
        if not params["base_path"]:
            return jsonify({"error": "base_path/path required"}), 400
        if not params["var"]:
            return jsonify({"error": "var parameter required"}), 400

        base = params["base_path"]
        cam_num = (
            int(params["camera"])
            if not str(params["camera"]).lower().startswith("cam")
            else int(str(params["camera"]).lstrip("Cam").lstrip("cam"))
        )
        cam_folder_eff = "Merged" if params["use_merged"] else f"Cam{cam_num}"
    except Exception as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam_folder=cam_folder_eff,
            type_name="instantaneous",
            endpoint=params["endpoint"],
            use_merged=params["use_merged"],
        )
        data_dir = Path(paths["data_dir"])
    except Exception as e:
        return jsonify({"error": f"failed to resolve data_dir: {e}"}), 400

    if not data_dir.exists():
        return jsonify({"error": f"data directory not found: {data_dir}"}), 404

    # Collect .mat files, exclude coordinates/mean files and folders
    all_mats = [
        p
        for p in sorted(data_dir.glob("*.mat"))
        if not (
            p.name.lower().endswith("_coordinates.mat")
            or p.name.lower().endswith("_mean.mat")
            or p.name == "coordinates.mat"
        )
    ]
    files_total = len(all_mats)
    if files_total == 0:
        return jsonify({"error": f"No .mat files found in {data_dir}"}), 404

    # Sample up to 50 random files
    sample_count = min(files_total, max(1, math.ceil(files_total * 0.25)))
    sampled = (
        random.sample(all_mats, sample_count)
        if files_total > sample_count
        else all_mats
    )

    global_min = None
    global_max = None
    files_checked = 0

    for mat_path in sampled:
        try:
            mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
            if "piv_result" not in mat:
                continue
            piv_result = mat["piv_result"]

            # Collect numeric values for this file
            file_min = None
            file_max = None

            # Process piv_result for min/max values
            if isinstance(piv_result, np.ndarray):
                for el in piv_result:
                    try:
                        arr = np.asarray(getattr(el, params["var"]))
                        vals = np.asarray(arr, dtype=float).ravel()
                        vals = vals[np.isfinite(vals)]
                        if vals.size > 0:
                            local_min = float(np.min(vals))
                            local_max = float(np.max(vals))
                            file_min = (
                                local_min
                                if file_min is None
                                else min(file_min, local_min)
                            )
                            file_max = (
                                local_max
                                if file_max is None
                                else max(file_max, local_max)
                            )
                    except Exception:
                        continue
            else:
                try:
                    arr = np.asarray(getattr(piv_result, params["var"], None))
                    if arr is not None and arr.size > 0:
                        vals = np.asarray(arr, dtype=float).ravel()
                        vals = vals[np.isfinite(vals)]
                        if vals.size > 0:
                            file_min = float(np.min(vals))
                            file_max = float(np.max(vals))
                except Exception:
                    pass

            if file_min is not None:
                files_checked += 1
                global_min = (
                    file_min if global_min is None else min(global_min, file_min)
                )
                global_max = (
                    file_max if global_max is None else max(global_max, file_max)
                )
        except Exception:
            continue

    if files_checked == 0:
        return (
            jsonify(
                {
                    "error": f"No valid values found for var '{params['var']}' in sampled files"
                }
            ),
            404,
        )

    return jsonify(
        {
            "min": float(global_min),
            "max": float(global_max),
            "files_checked": files_checked,
            "files_sampled": len(sampled),
            "files_total": files_total,
            "sampled_files": [p.name for p in sampled],
        }
    )


@vector_plot_bp.route("/get_uncalibrated_image", methods=["GET"])
def get_uncalibrated_image():
    """Return a single uncalibrated PNG by index if present."""
    params = parse_common_params(request)
    basepath_idx = request.args.get("basepath_idx", default=0, type=int)
    idx = request.args.get("index", type=int)

    if idx is None:
        return jsonify({"error": "index required"}), 400

    base = (
        config.base_paths[basepath_idx]
        if hasattr(config, "base_paths") and len(config.base_paths) > basepath_idx
        else params["base_path"]
    )
    if not base:
        return jsonify({"error": "base path not available"}), 400

    cam_folder = cam_folder_key(params["camera"])

    # Use our enhanced get_data_paths with use_uncalibrated=True
    try:
        paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam_folder=cam_folder,
            type_name="instantaneous",
            use_uncalibrated=True,
        )
        data_dir = paths["data_dir"]
    except Exception as e:
        return jsonify({"error": f"Failed to resolve paths: {e}"}), 400

    # Build filename from vector_format
    vector_fmt = config.vector_format
    if isinstance(vector_fmt, (list, tuple)):
        vector_fmt = vector_fmt[0] if vector_fmt else "%05d.mat"
    try:
        name = vector_fmt % idx
    except Exception:
        name = f"{idx:05d}.mat"

    mat_path = data_dir / name
    if not mat_path.exists():
        return jsonify({"error": f"uncalibrated .mat not found: {mat_path}"}), 404

    try:
        data_mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        if "piv_result" not in data_mat:
            return jsonify({"error": "Variable 'piv_result' not found in mat"}), 400
        piv_result = data_mat["piv_result"]

        # Find non-empty run
        pr, effective_run = find_non_empty_run(piv_result, params["var"], params["run"])
        if pr is None:
            return jsonify({"error": f"No non-empty run for var {params['var']}"}), 400

        # Extract variable and mask
        try:
            var_arr, mask_arr = extract_var_and_mask(pr, params["var"])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Build settings and plot (no coordinates for uncalibrated)
        settings = make_scalar_settings(
            config,
            variable=params["var"],
            run_label=effective_run,
            save_basepath=Path("plot_vector_tmp"),
            variable_units="m/s",
            coords_x=None,
            coords_y=None,
            lower_limit=params["lower_limit"],
            upper_limit=params["upper_limit"],
            cmap=params["cmap"],
        )

        # Create and return plot
        b64_img, W, H = create_and_return_plot(var_arr, mask_arr, settings)

        return jsonify(
            {
                "image": b64_img,
                "meta": {
                    "run": effective_run,
                    "var": params["var"],
                    "width": W,
                    "height": H,
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

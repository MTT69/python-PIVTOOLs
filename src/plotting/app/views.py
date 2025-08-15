import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import random
import math
from flask import Blueprint, jsonify, request
from scipy.io import loadmat
from vector_statistics.instantaneous_statistics import instantaneous_statistics

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


@vector_plot_bp.route("/plot_stats", methods=["GET"])
def plot_stats():
    """
    Run instantaneous_statistics for the provided base/path and camera,
    then load the generated mean .mat and coordinates and return a PNG
    (base64) of the requested scalar (var) same as /plot_vector.
    Query params:
      - base_path or path : base directory (full path) passed to instantaneous_statistics
      - camera : camera number or 'CamN' (default '1')
      - merged : '1' or 'true' to use merged data (default '0')
      - endpoint : endpoint string (optional)
      - var : 'ux'|'uy' etc. (default 'ux')
      - run : 1-based run/pass index (default 1)
      - lower_limit, upper_limit : optional floats for color scale
      - cmap : optional colormap string
    """
    # --- Parse frontend parameters (mirror behaviour of /plot_vector) ---
    base_path = request.args.get("base_path") or request.args.get("path") or request.args.get("full_path")
    frame = request.args.get("frame", default=1, type=int)  # not used but keep for compatibility
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

    if not base_path:
        return jsonify({"error": "base_path/path required"}), 400

    try:
        base = base_path
        cam_num = int(camera) if not str(camera).lower().startswith("cam") else int(str(camera).lstrip("Cam").lstrip("cam"))
        use_merged = merged in ("1", "true", "True")
    except Exception as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    # Resolve expected stats paths first so we can skip running expensive computation if already present
    try:
        cam_folder_eff = "Merged" if use_merged else f"Cam{cam_num}"
        paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam_folder=cam_folder_eff,
            type_name="instantaneous",
            endpoint=endpoint,
            use_merged=use_merged,
        )
        mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"
        out_file = mean_stats_dir / (f"{'merged' if use_merged else f'Cam{cam_num}'}_mean.mat")
        coords_file = mean_stats_dir / (f"{'merged' if use_merged else f'Cam{cam_num}'}_coordinates.mat")
    except Exception as e:
        return jsonify({"error": f"failed to resolve mean/coords paths: {e}"}), 400

    # Only run instantaneous_statistics if the required outputs are missing
    if not (out_file.exists() and coords_file.exists()):
        try:
            instantaneous_statistics(cam_num=cam_num, config=config, base=base)
        except Exception as e:
            return jsonify({"error": f"instantaneous_statistics failed: {e}"}), 500
        # Ensure stats were created
        if not out_file.exists():
            return jsonify({"error": f"mean file not found after stats run: {out_file}"}), 500
        if not coords_file.exists():
            return jsonify({"error": f"coordinates file not found after stats run: {coords_file}"}), 500

    try:
        # Load mean data and coordinates
        data_mat = loadmat(str(out_file), struct_as_record=False, squeeze_me=True)
        if "piv_result" not in data_mat:
            return jsonify({"error": "Variable 'piv_result' not found in mean mat"}), 400
        piv_result = data_mat["piv_result"]

        coords_mat = loadmat(str(coords_file), struct_as_record=False, squeeze_me=True)
        if "coordinates" not in coords_mat:
            return jsonify({"error": "Variable 'coordinates' not found in coords mat"}), 400
        coords = coords_mat["coordinates"]

        # Reuse the selection logic from plot_vector to find a non-empty run
        pr = None
        max_runs = 1
        if isinstance(piv_result, np.ndarray) and piv_result.dtype == object:
            max_runs = piv_result.size
            current_run = run
            while current_run <= max_runs:
                pr_candidate = piv_result[current_run - 1]
                try:
                    var_arr_candidate = np.asarray(getattr(pr_candidate, var))
                    if var_arr_candidate.size > 0 and not np.all(np.isnan(var_arr_candidate)):
                        pr = pr_candidate
                        run = current_run
                        break
                except Exception:
                    pass
                current_run += 1
            if pr is None:
                return jsonify({"error": f"No non-empty run found for variable {var}"}), 400
        else:
            try:
                var_arr_candidate = np.asarray(getattr(piv_result, var))
                if var_arr_candidate.size > 0 and not np.all(np.isnan(var_arr_candidate)):
                    pr = piv_result
                    run = 1
                else:
                    return jsonify({"error": f"No non-empty run found for variable {var}"}), 400
            except Exception:
                return jsonify({"error": f"'{var}' not found in piv_result element"}), 400

        # Extract variable and mask
        try:
            var_arr = np.asarray(getattr(pr, var))
        except Exception:
            return jsonify({"error": f"'{var}' not found in piv_result element"}), 400
        try:
            mask_arr = np.asarray(getattr(pr, "b_mask")).astype(bool)
        except Exception:
            mask_arr = np.zeros_like(var_arr, dtype=bool)

        # Extract coordinates for the selected run
        cx = cy = None
        if isinstance(coords, np.ndarray) and coords.dtype == object:
            max_coords_runs = coords.size
            if run < 1 or run > max_coords_runs:
                return jsonify({"error": f"run out of range for coordinates (1..{max_coords_runs})"}), 400
            c_el = coords[run - 1]
            cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)
        else:
            if run != 1:
                return jsonify({"error": "coordinates contains a single run; use run=1"}), 400
            c_el = coords
            cx, cy = np.asarray(c_el.x), np.asarray(c_el.y)

        # Build settings and plot using existing helpers
        save_basepath = Path("plot_stats_tmp")
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

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        b64_img = base64.b64encode(buf.read()).decode("utf-8")

        H, W = int(var_arr.shape[0]), int(var_arr.shape[1])
        return jsonify({"image": b64_img, "meta": {"run": run, "var": var, "width": W, "height": H}})
    except Exception as e:
        return jsonify({"error": f"plotting failed: {e}"}), 500


@vector_plot_bp.route("/check_vars", methods=["GET"])
@vector_plot_bp.route("/check_stat_vars", methods=["GET"])  # legacy alias
def check_vars():
    """
    Inspect a .mat and return available variable names.
    Usage modes (priority):
      1) mat_path=<full path>                        -> inspect that .mat
      2) frame=<int>  + base_path/camera/...         -> inspect instantaneous vector file for that frame
      3) no mat_path/frame -> inspect mean_stats .mat as before

    Query params:
      - base_path (or path/full_path): base directory (required unless mat_path provided)
      - camera: camera number or 'CamN' (default '1')
      - merged: '1'/'true' to use Merged (default '0')
      - endpoint: optional endpoint string
      - frame: optional integer frame index (for instantaneous vector mats)
      - mat_path: optional full path to a .mat file to inspect directly
    Returns: { "vars": [...], "source": { "type": "mean|instantaneous|mat", "path": "<path>" } }
    """
    mat_path_arg = request.args.get("mat_path", default=None, type=str)
    frame = request.args.get("frame", default=None, type=int)
    base_path = request.args.get("base_path") or request.args.get("path") or request.args.get("full_path")
    camera = request.args.get("camera", default="1", type=str)
    merged = request.args.get("merged", default="0", type=str)
    endpoint = request.args.get("endpoint", default="", type=str)

    # If a direct mat_path is provided, prefer it
    if mat_path_arg:
        mat_path = Path(mat_path_arg)
        if not mat_path.exists():
            return jsonify({"error": f"mat_path not found: {mat_path}"}), 404
        source_info = {"type": "mat", "path": str(mat_path)}
    else:
        # mat_path not provided -> need base_path unless frame omitted and mean file found later
        if not base_path:
            return jsonify({"error": "base_path/path required unless mat_path provided"}), 400
        try:
            base = base_path
            cam_num = int(camera) if not str(camera).lower().startswith("cam") else int(str(camera).lstrip("Cam").lstrip("cam"))
            use_merged = merged in ("1", "true", "True")
        except Exception as e:
            return jsonify({"error": f"Invalid parameters: {e}"}), 400

        # resolve data/mean paths
        try:
            cam_folder_eff = "Merged" if use_merged else f"Cam{cam_num}"
            paths = get_data_paths(
                base_dir=base,
                num_images=config.num_images,
                cam_folder=cam_folder_eff,
                type_name="instantaneous",
                endpoint=endpoint,
                use_merged=use_merged,
            )
            data_dir = Path(paths["data_dir"])
            mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"
        except Exception as e:
            return jsonify({"error": f"failed to resolve paths: {e}"}), 400

        # If frame specified -> inspect instantaneous vector .mat for that frame
        if frame is not None:
            # determine vector format (support list or str)
            vec_fmt = config.vector_format
            if isinstance(vec_fmt, (list, tuple)):
                vec_fmt = vec_fmt[0] if len(vec_fmt) > 0 else "%05d.mat"
            try:
                mat_path = Path(data_dir) / (vec_fmt % frame)
            except Exception as e:
                return jsonify({"error": f"failed to format vector filename with frame={frame}: {e}"}), 400
            if not mat_path.exists():
                return jsonify({"error": f"instantaneous mat not found: {mat_path}"}), 404
            source_info = {"type": "instantaneous", "path": str(mat_path)}
        else:
            # No frame and no mat_path -> inspect mean stats .mat (legacy)
            out_file = mean_stats_dir / (f"{'merged' if use_merged else f'Cam{cam_num}'}_mean.mat")
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

    # Find element to inspect (handles array-of-structs or single struct)
    pr = None
    try:
        if isinstance(piv_result, np.ndarray) and getattr(piv_result, "dtype", None) == object:
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

    if pr is None:
        return jsonify({"vars": [], "source": source_info})

    # Preferred: structured dtype names
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
    """
    Check min/max of a given variable across up to 50 random .mat files in the data directory.
    Query params:
      - base_path (or path/full_path): base directory (required)
      - camera: camera number or 'CamN' (default '1')
      - merged: '1'/'true' to use Merged (default '0')
      - endpoint: optional endpoint string
      - var: variable name to inspect (e.g. 'ux') (required)
    Returns JSON:
      { "min": <float>, "max": <float>, "files_checked": int, "files_sampled": int, "files_total": int }
    """
    base_path = request.args.get("base_path") or request.args.get("path") or request.args.get("full_path")
    camera = request.args.get("camera", default="1", type=str)
    merged = request.args.get("merged", default="0", type=str)
    endpoint = request.args.get("endpoint", default="", type=str)
    var = request.args.get("var", type=str)

    if not base_path:
        return jsonify({"error": "base_path/path required"}), 400
    if not var:
        return jsonify({"error": "var parameter required"}), 400

    try:
        base = base_path
        cam_num = int(camera) if not str(camera).lower().startswith("cam") else int(str(camera).lstrip("Cam").lstrip("cam"))
        use_merged = merged in ("1", "true", "True")
    except Exception as e:
        return jsonify({"error": f"Invalid parameters: {e}"}), 400

    try:
        cam_folder_eff = "Merged" if use_merged else f"Cam{cam_num}"
        paths = get_data_paths(
            base_dir=base,
            num_images=config.num_images,
            cam_folder=cam_folder_eff,
            type_name="instantaneous",
            endpoint=endpoint,
            use_merged=use_merged,
        )
        data_dir = Path(paths["data_dir"])
    except Exception as e:
        return jsonify({"error": f"failed to resolve data_dir: {e}"}), 400

    if not data_dir.exists():
        return jsonify({"error": f"data directory not found: {data_dir}"}), 404

    # Collect .mat files, exclude coordinates/mean files and folders
    all_mats = [p for p in sorted(data_dir.glob("*.mat")) if not (p.name.lower().endswith("_coordinates.mat") or p.name.lower().endswith("_mean.mat") or p.name == "coordinates.mat")]
    files_total = len(all_mats)
    if files_total == 0:
        return jsonify({"error": f"No .mat files found in {data_dir}"}), 404

    # Sample up to 50 random files
    # sample 25% of total files (rounded up), at least 1
    sample_count = min(files_total, max(1, math.ceil(files_total * 0.25)))
    if files_total <= sample_count:
        sampled = all_mats
    else:
        sampled = random.sample(all_mats, sample_count)

    global_min = None
    global_max = None
    files_checked = 0

    for mat_path in sampled:
        try:
            mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        except Exception:
            # skip unreadable files
            continue

        if "piv_result" not in mat:
            continue
        piv_result = mat["piv_result"]

        # Collect numeric values for this file
        file_min = None
        file_max = None

        # If structured array of runs
        if isinstance(piv_result, np.ndarray) and getattr(piv_result, "dtype", None) == object:
            for el in piv_result:
                try:
                    arr = np.asarray(getattr(el, var))
                except Exception:
                    continue
                if arr is None or getattr(arr, "size", 0) == 0:
                    continue
                # flatten and filter finite values
                try:
                    vals = np.asarray(arr, dtype=float).ravel()
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    local_min = float(np.min(vals))
                    local_max = float(np.max(vals))
                except Exception:
                    continue
                file_min = local_min if file_min is None else min(file_min, local_min)
                file_max = local_max if file_max is None else max(file_max, local_max)
        else:
            # single run element
            try:
                arr = np.asarray(getattr(piv_result, var, None))
            except Exception:
                arr = None
            if arr is not None and getattr(arr, "size", 0) > 0:
                try:
                    vals = np.asarray(arr, dtype=float).ravel()
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        file_min = float(np.min(vals))
                        file_max = float(np.max(vals))
                except Exception:
                    pass

        if file_min is None:
            # nothing found in this file
            continue

        files_checked += 1
        global_min = file_min if global_min is None else min(global_min, file_min)
        global_max = file_max if global_max is None else max(global_max, file_max)

    if files_checked == 0:
        return jsonify({"error": f"No valid values found for var '{var}' in sampled files"}), 404

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

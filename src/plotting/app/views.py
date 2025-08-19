import base64
import random
import time
from io import BytesIO
from pathlib import Path

import matplotlib
from loguru import logger

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, jsonify, request
from scipy.io import loadmat

from common.utils import camera_folder, camera_number
from config import get_config
from paths import get_data_paths
from plotting.plot_maker import make_scalar_settings, plot_scalar_field
from vector_statistics.instantaneous_statistics import instantaneous_statistics

vector_plot_bp = Blueprint("vector_plot", __name__, url_prefix="/plot")


def _loadmat_safe(path: Path, max_wait=2.0, poll_interval=0.1, require_stable=True):
    """
    Robust .mat loading to avoid race with file creation.
    Waits until file exists (and optionally size is stable across two polls) or timeout.
    """
    deadline = time.time() + max_wait
    last_size = -1
    stable = False
    while time.time() < deadline:
        if path.exists():
            size = path.stat().st_size
            if not require_stable:
                try:
                    return loadmat(str(path), struct_as_record=False, squeeze_me=True)
                except Exception:
                    pass
            else:
                if size == last_size and size > 0:
                    stable = True
                else:
                    stable = False
                if stable:
                    try:
                        return loadmat(
                            str(path), struct_as_record=False, squeeze_me=True
                        )
                    except Exception:
                        # Possibly still being written; continue until timeout
                        pass
                last_size = size
        time.sleep(poll_interval)
    raise FileNotFoundError(f"Timed out waiting for stable .mat file: {path}")


def load_piv_result(mat_path: Path):
    """Helper: load a .mat and return its piv_result or raise ValueError with good message."""
    try:
        mat = _loadmat_safe(mat_path)
    except FileNotFoundError as e:
        raise ValueError(str(e))
    if "piv_result" not in mat:
        raise ValueError(f"Variable 'piv_result' not found in mat: {mat_path}")
    return mat["piv_result"]


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


def safe_get_data_paths(*, base, cam_num, params):
    """
    Support both possible signatures of get_data_paths (cam= or cam_folder=) for compatibility.
    """
    from inspect import signature

    sig = signature(get_data_paths)
    kw = dict(
        base_dir=base,
        num_images=get_config().num_images,
        type_name=params["type_name"],
        endpoint=params["endpoint"],
        use_merged=params["use_merged"],
    )
    if "cam" in sig.parameters:
        kw["cam"] = cam_num
    else:
        # fallback name
        kw["cam_folder"] = f"Cam{cam_num}" if not params["use_merged"] else "Merged"
    return get_data_paths(**kw)


def create_and_return_plot(var_arr, mask_arr, settings, raw=False):
    """
    raw=True -> marginless image (pixel grid == data grid). Always returns extra meta with
    grid_dims (nx=W, ny=H), raw flag, and a simple axes_bbox covering full PNG for legacy.
    """
    H, W = int(var_arr.shape[0]), int(var_arr.shape[1])
    if raw:
        dpi = 100
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        arr = np.asarray(var_arr).squeeze()
        vmin = (
            getattr(settings, "lower_limit", None)
            if hasattr(settings, "lower_limit")
            else None
        )
        vmax = (
            getattr(settings, "upper_limit", None)
            if hasattr(settings, "upper_limit")
            else None
        )
        cmap = getattr(settings, "cmap", None)
        if cmap in (None, "default"):
            cmap = "viridis"
        ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_axis_off()
        buf = BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="white",
        )
        plt.close(fig)
        buf.seek(0)
        from PIL import Image

        with Image.open(buf) as im:
            png_w, png_h = im.size
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        extra = {
            "grid_dims": {"nx": W, "ny": H},
            "raw": True,
            "axes_bbox": {
                "left": 0,
                "top": 0,
                "width": png_w,
                "height": png_h,
                "png_width": png_w,
                "png_height": png_h,
            },
        }
        return b64, W, H, extra

    # fallback to original (non-raw) path
    fig, ax, im = plot_scalar_field(var_arr, mask_arr, settings)
    # Minimal safeguard: ensure we have a valid mappable (first call edge case)
    if im is None or not hasattr(im, "get_array"):
        arr = np.asarray(var_arr).squeeze()
        vmin = (
            getattr(settings, "lower_limit", None)
            if hasattr(settings, "lower_limit")
            else None
        )
        vmax = (
            getattr(settings, "upper_limit", None)
            if hasattr(settings, "upper_limit")
            else None
        )
        cmap = getattr(settings, "cmap", None) if hasattr(settings, "cmap") else None
        if cmap in (None, "default"):
            cmap = "viridis"
        im = ax.imshow(
            arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
        )
        if hasattr(settings, "title"):
            ax.set_title(settings.title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Make sure figure is fully rendered
    fig.canvas.draw()

    # First get original figure dimensions and axes position
    fig_width_inches, fig_height_inches = fig.get_size_inches()
    dpi = fig.dpi
    fig_width_pixels = int(fig_width_inches * dpi)
    fig_height_pixels = int(fig_height_inches * dpi)

    # Get axes position in figure coordinates (0-1)
    bbox = ax.get_position()

    # Convert to pixel coordinates in the original figure
    axes_left_orig = int(bbox.x0 * fig_width_pixels)
    axes_bottom_orig = int(bbox.y0 * fig_height_pixels)
    axes_width_orig = int(bbox.width * fig_width_pixels)
    axes_height_orig = int(bbox.height * fig_height_pixels)

    # Save with tight_bbox to get the actual PNG
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)

    # Get dimensions of the PNG
    from PIL import Image

    with Image.open(buf) as img:
        png_width, png_height = img.size

    # Reset buffer position and encode
    buf.seek(0)
    b64_img = base64.b64encode(buf.read()).decode("utf-8")

    # Now we need to determine where the axes are in the PNG
    # The PNG with tight_bbox crops out some whitespace
    # We need to get the transformation from original figure to PNG

    # Redraw the figure and save again, this time with explicit bbox
    # Include the axes and a bit of padding to ensure it's not cropped
    fig.canvas.draw()
    extent = ax.get_tightbbox(fig.canvas.get_renderer())

    # Calculate the crop amount that tight_bbox would do
    left_crop = max(0, int(extent.x0 * dpi) - 10)  # Add some padding
    bottom_crop = max(0, int(extent.y0 * dpi) - 10)

    # Calculate the axes position in the PNG
    axes_left = axes_left_orig - left_crop
    axes_bottom = axes_bottom_orig - bottom_crop

    # In the PNG, y=0 is at the top, but in matplotlib y=0 is at the bottom
    # Convert the coordinates to have y=0 at the top for the frontend
    axes_top = png_height - (axes_bottom + axes_height_orig)

    # Create data region info - change the key to match what frontend expects
    axes_bbox = {
        "left": axes_left,
        "top": axes_top,
        "width": axes_width_orig,
        "height": axes_height_orig,
        "png_width": png_width,
        "png_height": png_height,
    }

    plt.close(fig)

    H, W = int(var_arr.shape[0]), int(var_arr.shape[1])
    return b64_img, W, H, {"axes_bbox": axes_bbox}


def parse_request_params(req, allow_baseidx=True):
    """
    Minimal, explicit parsing. camera expected as int (or string of int).
    Returns a dict with normalized fields.
    """
    # base_path: either explicit or by index into config.base_paths
    base_path = req.args.get("base_path", default=None, type=str)
    base_idx = req.args.get("base_path_idx", default=0, type=int)
    cfg = get_config()
    # Honor explicit base_path if provided; fallback to indexed list only if absent.
    if not base_path:
        try:
            base_path = cfg.base_paths[base_idx]
        except Exception:
            raise ValueError("Invalid base_path and base_path_idx fallback failed")
    # camera: prefer integer; if missing default to 1
    camera = req.args.get("camera", default=1, type=int)
    # merged flag
    merged_raw = req.args.get("merged", default="0", type=str)  # what is this?
    use_merged = merged_raw in ("1", "true", "True", "TRUE")
    # other params
    type_name = req.args.get("type_name", default="instantaneous", type=str)
    frame = req.args.get("frame", default=1, type=int)
    run = req.args.get("run", default=1, type=int)
    endpoint = req.args.get("endpoint", default="", type=str)
    var = req.args.get("var", default="ux", type=str)
    lower_limit = req.args.get("lower_limit", type=float)
    upper_limit = req.args.get("upper_limit", type=float)
    cmap = req.args.get("cmap", default=None, type=str)
    if cmap is not None and (cmap.strip() == "" or cmap.lower() == "default"):
        cmap = None
    raw_mode = req.args.get("raw", default="0", type=str) in (
        "1",
        "true",
        "True",
        "TRUE",
    )
    return dict(
        base_path=base_path,
        camera=camera_number(camera),
        frame=frame,
        run=run,
        endpoint=endpoint,
        var=var,
        use_merged=use_merged,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        cmap=cmap,
        type_name=type_name,
        raw=raw_mode,
    )


def plot_from_mat(
    mat_path: Path,
    coords_path: Path,
    var: str,
    run: int,
    save_basepath: Path,
    lower_limit=None,
    upper_limit=None,
    cmap=None,
    raw=False,
):
    """
    Load mat_path (expects piv_result), find non-empty run, extract var and mask,
    load coords if provided, build settings and return (b64_img, width, height, effective_run).
    Raises ValueError on missing items.
    """
    piv_result = load_piv_result(mat_path)

    pr, effective_run = find_non_empty_run(piv_result, var, run)
    if pr is None:
        raise ValueError(f"No non-empty run found for variable {var}")

    var_arr, mask_arr = extract_var_and_mask(pr, var)

    # coordinates optional
    cx = cy = None
    if coords_path is not None:
        try:
            coords_mat = _loadmat_safe(coords_path, max_wait=1.0)
        except FileNotFoundError as e:
            raise ValueError(f"coordinates file issue: {e}")
        if "coordinates" not in coords_mat:
            raise ValueError("Variable 'coordinates' not found in coords mat")
        coords = coords_mat["coordinates"]
        cx, cy = extract_coordinates(coords, effective_run)

    settings = make_scalar_settings(
        get_config(),
        variable=var,
        run_label=effective_run,
        save_basepath=save_basepath,
        variable_units="m/s",
        coords_x=cx,
        coords_y=cy,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        cmap=cmap,
    )

    b64_img, W, H, extra = create_and_return_plot(var_arr, mask_arr, settings, raw=raw)
    return b64_img, W, H, extra


@vector_plot_bp.route("/plot_vector", methods=["GET"])
def plot_vector():
    try:
        logger.debug("plot_vector: received request with args: %s", dict(request.args))
        params = parse_request_params(request)
        logger.debug("plot_vector: parsed params: %s", params)
        get_config()
        base = params["base_path"]
        cam_num = params["camera"]
        try:
            paths = safe_get_data_paths(base=base, cam_num=cam_num, params=params)
        except Exception as e:
            logger.exception("plot_vector: get_data_paths failed")
            return jsonify({"error": f"paths resolution failed: {e}"}), 400
        data_dir = Path(paths["data_dir"])
        vector_fmt = get_config().vector_format
        data_path = data_dir / (vector_fmt % params["frame"])
        coords_path = data_dir / "coordinates.mat"
        b64_img, W, H, extra = plot_from_mat(
            mat_path=data_path,
            coords_path=coords_path if coords_path.exists() else None,
            var=params["var"],
            run=params["run"],
            save_basepath=Path("plot_vector_tmp"),
            lower_limit=params["lower_limit"],
            upper_limit=params["upper_limit"],
            cmap=params["cmap"],
            raw=params["raw"],
        )
        meta = {"run": params["run"], "var": params["var"], "width": W, "height": H}
        if isinstance(extra, dict):
            meta.update(extra)
        return jsonify({"image": b64_img, "meta": meta})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("plot_vector: Exception")
        return jsonify({"error": str(e)}), 500


@vector_plot_bp.route("/plot_stats", methods=["GET"])
def plot_stats():
    """
    Run instantaneous_statistics if needed, then load the generated mean .mat and coordinates
    and return PNG (base64) of requested scalar.
    """
    params = parse_request_params(request)
    cam_num = params["camera"]
    base = params["base_path"]
    try:
        paths = safe_get_data_paths(base=base, cam_num=cam_num, params=params)
        mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"
        out_file = mean_stats_dir / "mean_stats.mat"
        coords_file = mean_stats_dir / "coordinates.mat"
    except Exception as e:
        return jsonify({"error": f"failed to resolve mean/coords paths: {e}"}), 400

    try:
        instantaneous_statistics(cam_num=cam_num, config=get_config(), base=base)
    except Exception as e:
        return jsonify({"error": f"instantaneous_statistics failed: {e}"}), 500

    try:
        b64_img, W, H, extra = plot_from_mat(
            mat_path=out_file,
            coords_path=coords_file,
            var=params["var"],
            run=params["run"],
            save_basepath=Path("plot_stats_tmp"),
            lower_limit=params["lower_limit"],
            upper_limit=params["upper_limit"],
            cmap=params["cmap"],
            raw=params["raw"],
        )
        meta = {"run": params["run"], "var": params["var"], "width": W, "height": H}
        if isinstance(extra, dict):
            meta.update(extra)
        return jsonify({"image": b64_img, "meta": meta})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"plotting failed: {e}"}), 500


@vector_plot_bp.route("/check_vars", methods=["GET"])
@vector_plot_bp.route("/check_stat_vars", methods=["GET"])  # legacy alias
def check_vars():
    """Inspect a .mat and return available variable names."""
    frame = request.args.get("frame", default=None, type=int)
    params = parse_request_params(request)
    cfg = get_config()
    paths = get_data_paths(
        base_dir=params["base_path"],
        num_images=cfg.num_images,
        cam=params["camera"],
        type_name=params["type_name"],
        endpoint=params["endpoint"],
        use_merged=params["use_merged"],
    )
    data_dir = Path(paths["data_dir"])
    mean_stats_dir = Path(paths["stats_dir"]) / "mean_stats"
    # If frame specified -> inspect instantaneous vector .mat
    if frame is not None:
        vec_fmt = get_config().vector_format
        mat_path = Path(data_dir) / (vec_fmt % frame)
    else:
        cam_part = "merged" if params["use_merged"] else camera_folder(params["camera"])
        mat_path = mean_stats_dir / f"{cam_part}_mean.mat"

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
    print("Data directory: ", data_dir)
    return jsonify({"vars": vars_list})


@vector_plot_bp.route("/check_limits", methods=["GET"])
def check_limits():
    params = parse_request_params(request)
    cfg = get_config()
    base = params["base_path"]
    cam_num = params["camera"]
    try:
        paths = get_data_paths(
            base_dir=base,
            num_images=cfg.num_images,
            cam=cam_num,
            type_name=params["type_name"],
            endpoint=params["endpoint"],
            use_merged=params["use_merged"],
        )
        data_dir = Path(paths["data_dir"])
    except Exception as e:
        return jsonify({"error": f"failed to resolve data_dir: {e}"}), 400

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

    sample_count = min(files_total, 50)
    sampled = (
        random.sample(all_mats, sample_count)
        if files_total > sample_count
        else all_mats
    )

    all_values = []
    files_checked = 0

    for mat_path in sampled:
        try:
            mat = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
            if "piv_result" not in mat:
                continue
            piv_result = mat["piv_result"]

            vals = []
            if isinstance(piv_result, np.ndarray):
                for el in piv_result:
                    try:
                        arr = np.asarray(getattr(el, params["var"]))
                        arr = np.asarray(arr, dtype=float).ravel()
                        arr = arr[np.isfinite(arr)]
                        if arr.size > 0:
                            vals.append(arr)
                    except Exception:
                        continue
            else:
                try:
                    arr = np.asarray(getattr(piv_result, params["var"], None))
                    if arr is not None and arr.size > 0:
                        arr = np.asarray(arr, dtype=float).ravel()
                        arr = arr[np.isfinite(arr)]
                        if arr.size > 0:
                            vals.append(arr)
                except Exception:
                    pass

            if vals:
                files_checked += 1
                all_values.extend(np.concatenate(vals))
        except Exception:
            continue

    if files_checked == 0 or len(all_values) == 0:
        return (
            jsonify(
                {
                    "error": f"No valid values found for var '{params['var']}' in sampled files"
                }
            ),
            404,
        )

    all_values = np.asarray(all_values)
    p5 = float(np.percentile(all_values, 5))
    p95 = float(np.percentile(all_values, 95))

    # Also provide min/max for frontend compatibility (top-level keys expected by frontend)
    min_val = float(np.min(all_values))
    max_val = float(np.max(all_values))

    return jsonify(
        {
            "min": min_val,
            "max": max_val,
            "p5": p5,
            "p95": p95,
            "files_checked": files_checked,
            "files_sampled": len(sampled),
            "files_total": files_total,
            "sampled_files": [p.name for p in sampled],
        }
    )


@vector_plot_bp.route("/get_uncalibrated_image", methods=["GET"])
def get_uncalibrated_image():
    """Return a single uncalibrated PNG by index if present."""
    # keep compatibility with previous param names
    params = parse_request_params(request)
    cfg = get_config()
    basepath_idx = request.args.get("basepath_idx", default=0, type=int)
    idx = request.args.get("index", type=int)

    # support legacy basepath_idx if user provided it; fallback to parsed base_path
    try:
        base = params["base_path"]
        # if explicit legacy index provided, prefer it
        if request.args.get("basepath_idx") is not None:
            base = cfg.base_paths[basepath_idx]
    except Exception as e:
        logger.debug("get_uncalibrated_image: failed to resolve base: %s", e)
        return jsonify({"error": f"Failed to resolve base: {e}"}), 400

    cam_num = params["camera"]

    try:
        paths = get_data_paths(
            base_dir=base,
            num_images=cfg.num_images,
            cam=cam_num,
            type_name=params["type_name"],
            use_uncalibrated=True,
        )
        data_dir = Path(paths["data_dir"])
    except Exception as e:
        logger.exception("get_uncalibrated_image: Failed to resolve paths")
        return jsonify({"error": f"Failed to resolve paths: {e}"}), 400

    vector_fmt = cfg.vector_format
    name = vector_fmt % idx
    mat_path = data_dir / name
    try:
        b64_img, W, H, extra = plot_from_mat(
            mat_path=mat_path,
            coords_path=None,
            var=params["var"],
            run=params["run"],
            save_basepath=Path("plot_vector_tmp"),
            lower_limit=params["lower_limit"],
            upper_limit=params["upper_limit"],
            cmap=params["cmap"],
        )
        meta = {"run": params["run"], "var": params["var"], "width": W, "height": H}
        if isinstance(extra, dict):
            meta.update(extra)
        return jsonify({"image": b64_img, "meta": meta})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("get_uncalibrated_image: plotting error")
        return jsonify({"error": str(e)}), 500


@vector_plot_bp.route("/get_coordinate_at_point", methods=["GET"])
def get_coordinate_at_point():
    """Get the real-world coordinate at a specific point in the image."""
    try:
        # Parse parameters
        base_path = request.args.get("base_path")
        camera = request.args.get("camera", "1")
        x_percent = float(request.args.get("x_percent", 0))
        y_percent = float(request.args.get("y_percent", 0))
        frame = request.args.get("frame", "1")

        # Get configuration
        get_config()

        # Validate parameters
        if not base_path:
            return jsonify({"error": "Base path is required"}), 400

        # Load vector data from file
        try:
            # Construct vector file path
            camera_dir = f"Camera_{camera}"
            vector_path = Path(base_path) / camera_dir / f"vec{int(frame):04d}.npz"

            if not vector_path.exists():
                return jsonify({"error": f"Vector file not found: {vector_path}"}), 404

            # Load vector data
            vector_data = np.load(vector_path, allow_pickle=True)
            x_coords = vector_data["x"]
            y_coords = vector_data["y"]

            # Calculate corner coordinates
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)

            # Interpolate the coordinate based on image percentage
            # Invert y_percent because image coordinates are top-down
            x_coord = x_min + x_percent * (x_max - x_min)
            y_coord = y_min + (1 - y_percent) * (y_max - y_min)

            return jsonify({"coordinate": {"x": float(x_coord), "y": float(y_coord)}})

        except Exception as e:
            logger.exception(f"Error loading vector data: {e}")
            return jsonify({"error": f"Error loading vector data: {str(e)}"}), 500

    except Exception as e:
        logger.exception(f"Error in get_coordinate_at_point: {e}")
        return jsonify({"error": f"Error: {str(e)}"}), 500


@vector_plot_bp.route("/get_vector_at_position", methods=["GET"])
def get_vector_at_position():
    """
    Given x_percent/y_percent in image (0..1), return the real-world coordinate
    and vector/scalar values at the nearest grid point.
    Response: { x: float, y: float, ux: float|null, uy: float|null, value: float|null, i: int, j: int }
    """
    try:
        logger.info("get_vector_at_position: request args = %s", dict(request.args))
        params = parse_request_params(request)
        cfg = get_config()
        base = params["base_path"]
        cam_num = params["camera"]
        # frame and percent coordinates
        frame = request.args.get("frame", default=1, type=int)
        try:
            x_percent = float(request.args.get("x_percent", type=float))
            y_percent = float(request.args.get("y_percent", type=float))
            logger.info(
                "get_vector_at_position: x_percent=%f, y_percent=%f",
                x_percent,
                y_percent,
            )
        except Exception as e:
            logger.error("get_vector_at_position: percent parse error: %s", e)
            return jsonify({"error": "x_percent and y_percent required (0..1)"}), 400

        # resolve paths to vector file
        paths = get_data_paths(
            base_dir=base,
            num_images=cfg.num_images,
            cam=cam_num,
            type_name=params["type_name"],
            endpoint=params["endpoint"],
            use_merged=params["use_merged"],
        )
        data_dir = Path(paths["data_dir"])
        vec_fmt = cfg.vector_format
        mat_path = data_dir / (vec_fmt % frame)
        if not mat_path.exists():
            return jsonify({"error": f"vector mat not found: {mat_path}"}), 404

        # load piv_result and pick element/run
        try:
            piv_result = load_piv_result(mat_path)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        pr, effective_run = find_non_empty_run(piv_result, params["var"], params["run"])
        if pr is None:
            return jsonify({"error": "no non-empty run found"}), 404

        # try to extract arrays
        def safe_get_array(obj, name):
            try:
                a = np.asarray(getattr(obj, name))
                return a
            except Exception:
                return None

        # try variable array (for shape)
        var_arr = safe_get_array(pr, params["var"])
        # if var_arr missing, try ux/uy to infer shape
        if var_arr is None:
            var_arr = safe_get_array(pr, "ux") or safe_get_array(pr, "uy")
        if var_arr is None:
            return (
                jsonify(
                    {"error": "no data array found for var or ux/uy in piv_result"}
                ),
                404,
            )

        # ensure 2D shape
        if var_arr.ndim < 2:
            # attempt to reshape or fail
            try:
                var_arr = np.asarray(var_arr).reshape(var_arr.shape[0], -1)
            except Exception:
                return jsonify({"error": "unexpected variable array shape"}), 500

        H, W = int(var_arr.shape[0]), int(var_arr.shape[1])
        # clamp percents to [0,1]
        xp = max(0.0, min(1.0, float(x_percent)))
        yp = max(0.0, min(1.0, float(y_percent)))

        # Map percent -> array indices
        j = int(round(xp * (W - 1)))
        # Convert y percent to array index - note that in the data array,
        # index 0 is at the bottom, but in the image, y=0 is at the top
        i = int(round((1.0 - yp) * (H - 1)))
        i = max(0, min(H - 1, i))
        j = max(0, min(W - 1, j))

        logger.info(
            "get_vector_at_position: Array indices i=%d, j=%d for array shape %s",
            i,
            j,
            var_arr.shape,
        )

        # coordinate arrays (if present)
        x_coords = safe_get_array(pr, "x")
        y_coords = safe_get_array(pr, "y")
        if (
            x_coords is not None
            and y_coords is not None
            and x_coords.shape == var_arr.shape
            and y_coords.shape == var_arr.shape
        ):
            coord_x = float(x_coords[i, j])
            coord_y = float(y_coords[i, j])
        else:
            # fallback: use min/max mapping if 1D or mismatched shapes
            try:
                if x_coords is not None:
                    x_min, x_max = float(np.nanmin(x_coords)), float(
                        np.nanmax(x_coords)
                    )
                    coord_x = float(x_min + xp * (x_max - x_min))
                else:
                    coord_x = float(j)
                if y_coords is not None:
                    y_min, y_max = float(np.nanmin(y_coords)), float(
                        np.nanmax(y_coords)
                    )
                    # Convert y percent to real-world coordinate
                    coord_y = float(y_max - yp * (y_max - y_min))
                else:
                    coord_y = float(i)
            except Exception:
                coord_x = float(j)
                coord_y = float(i)

        # try to fetch ux/uy
        ux_arr = safe_get_array(pr, "ux")
        uy_arr = safe_get_array(pr, "uy")
        ux_val = (
            float(ux_arr[i, j])
            if (ux_arr is not None and ux_arr.shape == var_arr.shape)
            else None
        )
        uy_val = (
            float(uy_arr[i, j])
            if (uy_arr is not None and uy_arr.shape == var_arr.shape)
            else None
        )

        # Also provide the requested variable value (could be same as ux)
        value_val = None
        try:
            v = getattr(pr, params["var"], None)
            if v is not None:
                v_arr = np.asarray(v)
                if v_arr.shape == var_arr.shape:
                    value_val = float(v_arr[i, j])
                else:
                    # flatten fallback
                    v_flat = np.asarray(v).ravel()
                    if v_flat.size > 0:
                        value_val = float(v_flat[0])
        except Exception:
            value_val = None

        # Improved logging to debug the issue
        logger.debug(
            f"get_vector_at_position: var_arr shape={var_arr.shape}, indices i={i}, j={j}"
        )

        # Add more logging before returning
        result = {
            "x": coord_x,
            "y": coord_y,
            "ux": ux_val,
            "uy": uy_val,
            "value": value_val,
            "i": int(i),
            "j": int(j),
        }
        logger.info("get_vector_at_position: returning data=%s", result)
        return jsonify(result)
    except Exception as e:
        logger.exception("get_vector_at_position error")
        return jsonify({"error": str(e)}), 500


@vector_plot_bp.route("/get_vector_at_index", methods=["GET"])
def get_vector_at_index():
    """
    Direct i,j lookup (zero-based) in current frame.
    Query: base_path, camera, frame, var, run, merged, i, j
    Returns: x,y,ux,uy,value,i,j (coordinates in same units as stored)
    """
    try:
        params = parse_request_params(request)
        cfg = get_config()
        base = params["base_path"]
        cam = params["camera"]
        frame = request.args.get("frame", default=1, type=int)
        i = request.args.get("i", type=int)
        j = request.args.get("j", type=int)
        if i is None or j is None:
            return jsonify({"error": "i and j required"}), 400
        paths = get_data_paths(
            base_dir=base,
            num_images=cfg.num_images,
            cam=cam,
            type_name=params["type_name"],
            endpoint=params["endpoint"],
            use_merged=params["use_merged"],
        )
        data_dir = Path(paths["data_dir"])
        vec_fmt = cfg.vector_format
        mat_path = data_dir / (vec_fmt % frame)
        if not mat_path.exists():
            return jsonify({"error": f"vector mat not found: {mat_path}"}), 404
        piv_result = load_piv_result(mat_path)
        pr, effective_run = find_non_empty_run(piv_result, params["var"], params["run"])
        if pr is None:
            return jsonify({"error": "no non-empty run found"}), 404

        def safe(name):
            try:
                return np.asarray(getattr(pr, name))
            except Exception:
                return None

        v_arr = safe(params["var"])
        if v_arr is None:
            # fallback to ux shape
            v_arr = safe("ux") or safe("uy")
        if v_arr is None:
            return jsonify({"error": "no data array available"}), 500
        if v_arr.ndim < 2:
            return jsonify({"error": "data array not 2D"}), 500
        H, W = v_arr.shape[0], v_arr.shape[1]
        if i < 0 or j < 0 or i >= H or j >= W:
            return jsonify({"error": "i,j out of bounds"}), 400
        x_arr = safe("x")
        y_arr = safe("y")
        if (
            x_arr is not None
            and y_arr is not None
            and x_arr.shape == v_arr.shape
            and y_arr.shape == v_arr.shape
        ):
            coord_x = float(x_arr[i, j])
            coord_y = float(y_arr[i, j])
        else:
            coord_x = float(j)
            coord_y = float(i)
        ux_arr = safe("ux")
        uy_arr = safe("uy")
        ux_val = (
            float(ux_arr[i, j])
            if ux_arr is not None and ux_arr.shape == v_arr.shape
            else None
        )
        uy_val = (
            float(uy_arr[i, j])
            if uy_arr is not None and uy_arr.shape == v_arr.shape
            else None
        )
        val = None
        targ = safe(params["var"])
        if targ is not None and targ.shape == v_arr.shape:
            val = float(targ[i, j])
        return jsonify(
            {
                "x": coord_x,
                "y": coord_y,
                "ux": ux_val,
                "uy": uy_val,
                "value": val,
                "i": int(i),
                "j": int(j),
            }
        )
    except Exception as e:
        logger.exception("get_vector_at_index error")
        return jsonify({"error": str(e)}), 500
